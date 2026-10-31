"""D-083 stage-1 evaluation: physics estimators versus a no-physics baseline.

Both models use the same ML-8 windows and the same session split. The physics model is a
robust-z score over the cooling, charging and gearing parameters/residuals identified per
window. The baseline is the ML-16 PCA reconstruction error and robust feature z-score.
Thresholds are chosen on **validation healthy sessions** only; test sessions are never
used for fitting or threshold selection.

The output is synthetic-data evidence only. It does not claim generalisation to real
rides (CLAUDE.md rule 4) and it never feeds a safety partition (invariant 3).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vehicle_ml.baseline import BaselineModel
from vehicle_ml.data.store import SessionStore
from vehicle_ml.dataset import build_dataset
from vehicle_ml.defs import defs_version
from vehicle_ml.physics.charging import fit_charging
from vehicle_ml.physics.config import RoleMap, load_roles
from vehicle_ml.physics.cooling import fit_cooling
from vehicle_ml.physics.errors import NotIdentifiableError
from vehicle_ml.physics.gearing import GearTable
from vehicle_ml.windows import align_frames

COOLING_FEATURES = (
    "cooling_a",
    "cooling_h",
    "cooling_ambient_c",
    "cooling_time_constant_s",
    "cooling_residual_mean",
    "cooling_residual_std",
    "cooling_residual_max_abs",
    "cooling_quality",
)
CHARGING_FEATURES = (
    "charging_base_v",
    "charging_plateau_v",
    "charging_cut_in_rpm",
    "charging_slope_v_per_rpm",
    "charging_residual_mean",
    "charging_residual_std",
    "charging_residual_max_abs",
    "charging_quality",
)
GEARING_FEATURES = (
    "gearing_residual_mean",
    "gearing_residual_std",
    "gearing_residual_max_abs",
    "gearing_residual_positive_fraction",
)
MODEL_NAMES = ("physics", "baseline_pca", "baseline_z")
SYNTHETIC_NOTE = "synthetic data; no generalisation claim (rule 4)"


@dataclass(frozen=True)
class EvaluationResult:
    data: dict
    json_path: Path
    markdown_path: Path


def _load_faults(data_dir: Path) -> dict[str, dict[str, object]]:
    path = data_dir / "faults.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    sessions = payload.get("sessions", {})
    if not isinstance(sessions, dict):
        raise ValueError("faults.json: sessions must be an object")
    result: dict[str, dict[str, object]] = {}
    for session_id, entry in sessions.items():
        if not isinstance(entry, dict):
            raise ValueError(f"faults.json: {session_id!r} must be an object")
        fault = entry.get("fault")
        onset = entry.get("onset_ms")
        if not isinstance(fault, str) or not isinstance(onset, (int, float)):
            raise ValueError(f"faults.json: {session_id!r} needs fault and onset_ms")
        result[str(session_id)] = {"fault": fault, "onset_ms": int(onset)}
    return result


def _load_role_frame(
    store: SessionStore,
    session_id: str,
    roles: RoleMap,
    max_gap_ms: float,
) -> tuple[pd.DataFrame | None, list[str]]:
    telemetry = store.load(session_id, "telemetry")
    if telemetry is None:
        return None, list(roles.roles)

    frame = telemetry[["session_id", "device_time_ms"]].copy()
    missing: list[str] = []
    for role in roles.roles:
        if roles.kind(role) != "did":
            continue
        column = roles.column(role)
        if column in telemetry.columns:
            frame[role] = telemetry[column].to_numpy()
        else:
            missing.append(role)
            frame[role] = np.nan

    derived = store.load(session_id, "derived")
    derived_roles = [role for role in roles.roles if roles.kind(role) == "derived"]
    if derived is None or derived.empty:
        missing.extend(derived_roles)
        for role in derived_roles:
            frame[role] = np.nan
        return frame, missing

    wide = derived.pivot_table(
        index="device_time_ms", columns="signal", values="value", aggfunc="last"
    ).reset_index()
    wide["session_id"] = session_id
    available = [role for role in derived_roles if roles.column(role) in wide.columns]
    unavailable = [role for role in derived_roles if role not in available]
    missing.extend(unavailable)
    for role in unavailable:
        frame[role] = np.nan
    if available:
        aligned = align_frames(frame, wide, available, tolerance_ms=max_gap_ms)
        for role in available:
            frame[role] = aligned[role].to_numpy()
    return frame, missing


def _healthy(ids: Sequence[str], faults: Mapping[str, dict[str, object]]) -> list[str]:
    return [session_id for session_id in ids if session_id not in faults]


def _fit_gear_table(
    frames: Mapping[str, pd.DataFrame | None],
    roles: RoleMap,
    train_ids: Sequence[str],
    faults: Mapping[str, dict[str, object]],
) -> GearTable | None:
    ratios: list[np.ndarray] = []
    gears: list[np.ndarray] = []
    for session_id in _healthy(train_ids, faults):
        frame = frames.get(session_id)
        if frame is None:
            continue
        if "gearbox_ratio_measured" not in frame.columns or "gear_estimate" not in frame.columns:
            continue
        ratios.append(frame["gearbox_ratio_measured"].to_numpy(dtype=float))
        gears.append(frame["gear_estimate"].to_numpy(dtype=float))
    if not ratios:
        return None
    try:
        return GearTable.fit(np.concatenate(ratios), np.concatenate(gears))
    except NotIdentifiableError:
        return None


def _residual_stats(residual: np.ndarray) -> dict[str, float]:
    finite = residual[np.isfinite(residual)]
    if finite.size == 0:
        return {"mean": math.nan, "std": math.nan, "max_abs": math.nan}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "max_abs": float(np.max(np.abs(finite))),
    }


def _physics_window(
    frame: pd.DataFrame, start_ms: int, end_ms: int, table: GearTable | None
) -> dict:
    window = frame.loc[(frame["device_time_ms"] >= start_ms) & (frame["device_time_ms"] <= end_ms)]
    values: dict[str, float] = {
        "cooling_ok": 0.0,
        "charging_ok": 0.0,
        "gearing_ok": 0.0,
    }
    if window.empty:
        return values

    times = window["device_time_ms"].to_numpy(dtype=float)
    if {"coolant_temp", "engine_speed"}.issubset(window.columns):
        try:
            fit = fit_cooling(
                times,
                window["coolant_temp"].to_numpy(dtype=float),
                window["engine_speed"].to_numpy(dtype=float),
            )
        except (NotIdentifiableError, ValueError):
            pass
        else:
            values.update(
                {
                    "cooling_ok": 1.0,
                    "cooling_a": fit.a,
                    "cooling_h": fit.h,
                    "cooling_ambient_c": fit.ambient_c,
                    "cooling_time_constant_s": fit.time_constant_s,
                    "cooling_quality": fit.quality,
                }
            )
            stats = _residual_stats(fit.residual)
            values.update(
                {
                    "cooling_residual_mean": stats["mean"],
                    "cooling_residual_std": stats["std"],
                    "cooling_residual_max_abs": stats["max_abs"],
                }
            )

    if {"battery_voltage", "engine_speed"}.issubset(window.columns):
        try:
            fit = fit_charging(
                window["engine_speed"].to_numpy(dtype=float),
                window["battery_voltage"].to_numpy(dtype=float),
            )
        except (NotIdentifiableError, ValueError):
            pass
        else:
            values.update(
                {
                    "charging_ok": 1.0,
                    "charging_base_v": fit.base_v,
                    "charging_plateau_v": fit.plateau_v,
                    "charging_cut_in_rpm": fit.cut_in_rpm,
                    "charging_slope_v_per_rpm": fit.slope_v_per_rpm,
                    "charging_quality": fit.quality,
                }
            )
            stats = _residual_stats(fit.residual)
            values.update(
                {
                    "charging_residual_mean": stats["mean"],
                    "charging_residual_std": stats["std"],
                    "charging_residual_max_abs": stats["max_abs"],
                }
            )

    if table is not None and {"gearbox_ratio_measured", "gear_estimate"}.issubset(window.columns):
        residual = table.residual(
            window["gearbox_ratio_measured"].to_numpy(dtype=float),
            window["gear_estimate"].to_numpy(dtype=float),
        )
        finite = residual[np.isfinite(residual)]
        if finite.size:
            values.update(
                {
                    "gearing_ok": 1.0,
                    "gearing_residual_mean": float(np.mean(finite)),
                    "gearing_residual_std": float(np.std(finite)),
                    "gearing_residual_max_abs": float(np.max(np.abs(finite))),
                    "gearing_residual_positive_fraction": float(np.mean(finite > 0.0)),
                }
            )
    return values


def _physics_features(
    features: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame | None],
    table: GearTable | None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for row in features.itertuples(index=False):
        frame = frames.get(str(row.session_id))
        if frame is None:
            rows.append({"cooling_ok": 0.0, "charging_ok": 0.0, "gearing_ok": 0.0})
        else:
            rows.append(_physics_window(frame, int(row.start_ms), int(row.end_ms), table))
    return pd.DataFrame(rows, index=features.index)


def _robust_z(values: pd.Series, reference: pd.Series) -> pd.Series:
    finite = reference.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return pd.Series(np.nan, index=values.index)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad
    if scale == 0.0:
        scale = 1.0
    return (values.astype(float) - median).abs() / scale


def _physics_scores(physics: pd.DataFrame, fit_mask: pd.Series) -> pd.DataFrame:
    reference = physics.loc[fit_mask]
    cooling = pd.concat(
        [_robust_z(physics[name], reference[name]) for name in COOLING_FEATURES], axis=1
    )
    charging = pd.concat(
        [_robust_z(physics[name], reference[name]) for name in CHARGING_FEATURES], axis=1
    )
    gearing = pd.concat(
        [_robust_z(physics[name], reference[name]) for name in GEARING_FEATURES], axis=1
    )
    result = pd.DataFrame(
        {
            "physics_cooling": cooling.max(axis=1, skipna=True),
            "physics_charging": charging.max(axis=1, skipna=True),
            "physics_gearing": gearing.max(axis=1, skipna=True),
        },
        index=physics.index,
    )
    result = result.fillna(0.0)
    result["physics"] = result.max(axis=1)
    return result


def _threshold(values: pd.Series, target_per_hour: float, stride_ms: int) -> float:
    if target_per_hour <= 0.0:
        raise ValueError("target false alarms per hour must be positive")
    windows_per_hour = 3_600_000.0 / stride_ms
    per_window = min(max(target_per_hour / windows_per_hour, 0.0), 1.0)
    finite = values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("no validation healthy scores for threshold selection")
    return float(np.quantile(finite, 1.0 - per_window))


def _episodes(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    return int(np.count_nonzero(mask & ~np.r_[False, mask[:-1]]))


def _false_alarms(
    scores: pd.DataFrame,
    model: str,
    threshold: float,
    session_ids: Sequence[str],
) -> dict[str, float]:
    episodes = 0
    hours = 0.0
    for session_id in session_ids:
        rows = scores.loc[scores["session_id"] == session_id].sort_values("start_ms")
        if rows.empty:
            continue
        mask = rows[model].to_numpy(dtype=float) > threshold
        episodes += _episodes(mask)
        hours += float(rows["end_ms"].max() - rows["start_ms"].min()) / 3_600_000.0
    return {
        "episodes": float(episodes),
        "hours": hours,
        "per_hour": episodes / hours if hours > 0.0 else 0.0,
    }


def _fault_results(
    scores: pd.DataFrame,
    model: str,
    threshold: float,
    test_ids: Sequence[str],
    faults: Mapping[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for session_id in test_ids:
        truth = faults.get(session_id)
        if truth is None:
            continue
        kind = str(truth["fault"])
        onset = int(truth["onset_ms"])
        rows = scores.loc[scores["session_id"] == session_id].sort_values("start_ms")
        rows = rows.loc[rows["start_ms"] >= onset]
        entry = grouped.setdefault(
            kind,
            {"sessions": 0, "detected": 0, "missed": 0, "delays_ms": [], "modalities": Counter()},
        )
        entry["sessions"] = int(entry["sessions"]) + 1
        alarm = rows.loc[rows[model].to_numpy(dtype=float) > threshold]
        if alarm.empty:
            entry["missed"] = int(entry["missed"]) + 1
            continue
        first = alarm.iloc[0]
        entry["detected"] = int(entry["detected"]) + 1
        entry["delays_ms"].append(int(first["start_ms"]) - onset)
        if model == "physics":
            choices = {
                "cooling": float(first["physics_cooling"]),
                "charging": float(first["physics_charging"]),
                "gearing": float(first["physics_gearing"]),
            }
            entry["modalities"][max(choices, key=choices.get)] += 1
        elif model == "baseline_pca":
            entry["modalities"]["features_pca"] += 1
        else:
            entry["modalities"]["features_robust_z"] += 1

    result: dict[str, dict[str, object]] = {}
    for kind, entry in sorted(grouped.items()):
        delays = [int(value) for value in entry["delays_ms"]]
        result[kind] = {
            "sessions": int(entry["sessions"]),
            "detected": int(entry["detected"]),
            "missed": int(entry["missed"]),
            "median_delay_ms": float(np.median(delays)) if delays else None,
            "modalities": dict(entry["modalities"]),
        }
    return result


def _markdown(data: dict) -> str:
    lines = [
        "# D-083 stage-1 evaluation",
        "",
        f"- defs: {data['defs_version']}",
        f"- seed: {data['seed']}",
        f"- windows: {data['window_ms']} ms / stride {data['stride_ms']} ms",
        f"- target false alarms: {data['target_false_alarms_per_hour']}/h",
        "",
        SYNTHETIC_NOTE + ".",
        "",
        "## Split",
        "",
        "| split | sessions |",
        "|---|---|",
    ]
    for part, sessions in data["split"].items():
        lines.append(f"| {part} | {len(sessions)} |")
    lines += ["", "## Thresholds", "", "| model | threshold |", "|---|---|"]
    for model in MODEL_NAMES:
        lines.append(f"| {model} | {data['thresholds'][model]:.6g} |")
    lines += [
        "",
        "## False alarms on test healthy sessions",
        "",
        "| model | episodes | hours | per hour |",
        "|---|---|---|---|",
    ]
    for model in MODEL_NAMES:
        alarm = data["models"][model]["test"]["false_alarms"]
        lines.append(
            f"| {model} | {alarm['episodes']:.0f} | {alarm['hours']:.4f} | "
            f"{alarm['per_hour']:.3f} |"
        )
    lines += ["", "## Faults", ""]
    if not data["faults_present"]:
        lines.append("No fault ground truth was found (`faults.json`); false alarms only.")
    else:
        lines += [
            "| model | fault kind | detected | missed | median delay ms | caught by |",
            "|---|---|---|---|---|---|",
        ]
        for model in MODEL_NAMES:
            faults = data["models"][model]["test"]["faults"]
            if not faults:
                lines.append(f"| {model} | (none) | 0 | 0 | - | - |")
                continue
            for kind, entry in faults.items():
                delay = (
                    "-" if entry["median_delay_ms"] is None else f"{entry['median_delay_ms']:.0f}"
                )
                caught = ", ".join(f"{name}:{count}" for name, count in entry["modalities"].items())
                lines.append(
                    f"| {model} | {kind} | {entry['detected']}/{entry['sessions']} | "
                    f"{entry['missed']} | {delay} | {caught or '-'} |"
                )
    lines.append("")
    return "\n".join(lines)


def evaluate(
    data_dir: str | Path,
    out_dir: str | Path,
    config_path: str | Path,
    *,
    seed: int = 0,
    target_false_alarms_per_hour: float = 1.0,
    window_ms: int = 10_000,
    stride_ms: int = 5_000,
    max_gap_ms: int = 500,
) -> EvaluationResult:
    """Run the stage-1 comparison and write ``evaluation.json`` and ``evaluation.md``."""
    if window_ms <= 0 or stride_ms <= 0 or max_gap_ms <= 0:
        raise ValueError("window_ms, stride_ms and max_gap_ms must be positive")
    data_path = Path(data_dir)
    out_path = Path(out_dir)
    roles = load_roles(config_path)
    store = SessionStore(data_path)
    result = build_dataset(
        data_path,
        out_path,
        kinds=("telemetry", "derived"),
        channels=None,
        window_ms=window_ms,
        stride_ms=stride_ms,
        max_gap_ms=max_gap_ms,
        seed=seed,
    )
    if result.features.empty:
        raise ValueError("no windows were built from the data directory")

    faults = _load_faults(data_path)
    all_ids = [*result.split["train"], *result.split["val"], *result.split["test"]]
    frames: dict[str, pd.DataFrame | None] = {}
    missing_roles: dict[str, list[str]] = {}
    for session_id in all_ids:
        frame, missing = _load_role_frame(store, session_id, roles, max_gap_ms)
        frames[session_id] = frame
        if missing:
            missing_roles[session_id] = missing

    table = _fit_gear_table(frames, roles, result.split["train"], faults)
    physics = _physics_features(result.features, frames, table)
    train_healthy = _healthy(result.split["train"], faults)
    physics_scores = _physics_scores(physics, result.features["session_id"].isin(train_healthy))

    train_features = result.features.loc[result.features["session_id"].isin(train_healthy)]
    baseline = BaselineModel.fit(train_features, result.split, result.scaler)
    baseline_scores = baseline.scores(result.features)

    scores = pd.DataFrame(
        {
            "session_id": result.features["session_id"].to_numpy(),
            "start_ms": result.features["start_ms"].to_numpy(),
            "end_ms": result.features["end_ms"].to_numpy(),
            "split": result.features["split"].to_numpy(),
            "physics": physics_scores["physics"].to_numpy(),
            "physics_cooling": physics_scores["physics_cooling"].to_numpy(),
            "physics_charging": physics_scores["physics_charging"].to_numpy(),
            "physics_gearing": physics_scores["physics_gearing"].to_numpy(),
            "baseline_pca": baseline_scores["pca_error"].to_numpy(),
            "baseline_z": baseline_scores["robust_z"].to_numpy(),
        }
    )

    val_healthy = _healthy(result.split["val"], faults)
    thresholds = {
        model: _threshold(
            scores.loc[scores["session_id"].isin(val_healthy), model],
            target_false_alarms_per_hour,
            stride_ms,
        )
        for model in MODEL_NAMES
    }

    test_ids = result.split["test"]
    test_healthy = _healthy(test_ids, faults)
    models: dict[str, dict[str, object]] = {}
    for model in MODEL_NAMES:
        models[model] = {
            "threshold": thresholds[model],
            "test": {
                "false_alarms": _false_alarms(scores, model, thresholds[model], test_healthy),
                "faults": _fault_results(scores, model, thresholds[model], test_ids, faults),
            },
        }

    fit_ids = set(result.split["train"]) | set(val_healthy)
    if fit_ids & set(test_ids):
        raise AssertionError("a test session leaked into fitting or threshold selection")

    data: dict[str, object] = {
        "defs_version": defs_version(),
        "seed": seed,
        "window_ms": window_ms,
        "stride_ms": stride_ms,
        "max_gap_ms": max_gap_ms,
        "target_false_alarms_per_hour": target_false_alarms_per_hour,
        "split": result.split,
        "fit_sessions": result.split["train"],
        "threshold_sessions": val_healthy,
        "thresholds": thresholds,
        "models": models,
        "faults_present": bool(faults),
        "missing_roles": missing_roles,
        "note": SYNTHETIC_NOTE,
    }

    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "evaluation.json"
    markdown_path = out_path / "evaluation.md"
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(data), encoding="utf-8")
    return EvaluationResult(data=data, json_path=json_path, markdown_path=markdown_path)


__all__ = ["EvaluationResult", "SYNTHETIC_NOTE", "evaluate"]
