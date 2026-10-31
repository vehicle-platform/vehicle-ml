"""D-083 stage-1 evaluation: physics estimators versus a no-physics baseline.

Both models use the same ML-8 windows and the same session split. The physics model is a
robust-z score over the cooling, charging and gearing parameters/residuals identified per
window. The baseline is the ML-16 PCA reconstruction error and robust feature z-score.
Thresholds are chosen on **validation healthy sessions** only; test sessions are never
used for fitting or threshold selection.

Every alarm also records the **modality** that raised it (CLAUDE.md rule 3): the CAN
physics estimators are modality ``can`` with their estimator group, the PCA baseline is
attributed to the modality with the largest ratio of its window's summed squared
reconstruction error to its healthy-training mean, and the robust-z baseline to the
modality of its largest |z|.

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
from vehicle_ml.data.sources import MODALITIES, feature_metadata, imu_columns
from vehicle_ml.data.store import SessionStore
from vehicle_ml.dataset import build_dataset
from vehicle_ml.defs import defs_version, vehicle_derived
from vehicle_ml.features import IDENTITY_COLUMNS
from vehicle_ml.normalize import Standardizer
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
PHYSICS_FEATURES_BY_ESTIMATOR = {
    "cooling": COOLING_FEATURES,
    "charging": CHARGING_FEATURES,
    "gearing": GEARING_FEATURES,
}
MODEL_NAMES = ("physics", "baseline_pca", "baseline_z")
PHYSICS_MODALITY = "can"
IMU_BAND_COUNT = 5  # equal-width FFT bands from 0 Hz to the IMU Nyquist frequency
MISS_NOTE = (
    "The CAN-only physics models are expected to miss `wheel_vibration`; misses are "
    "reported as misses and the models are not tuned to catch it (rule 4)."
)
SYNTHETIC_NOTE = "synthetic data; no generalisation claim (rule 4)"
NO_ALARM = "no alarm"
NOT_EVALUATED = "not evaluated"
MIXED = "mixed"
UNATTRIBUTED = "unattributed"  # an alarm whose window has no attributable modality
NOT_CALIBRATED = "not calibrated"


@dataclass(frozen=True)
class EvaluationResult:
    data: dict
    json_path: Path
    markdown_path: Path


def _imu_bands() -> dict[str, list[tuple[float, float]]]:
    """Equal-width FFT bands up to the IMU Nyquist limit, for every defs IMU column."""
    periods = {
        int(spec["sample_period_ms"])
        for name, spec in vehicle_derived.SENSORS.items()
        if name.startswith("imu_")
    }
    if len(periods) != 1:
        raise ValueError("the defs IMU sensors do not share one sample period")
    nyquist = 500.0 / periods.pop()
    edges = np.linspace(0.0, nyquist, IMU_BAND_COUNT + 1)
    bands = [(float(low), float(high)) for low, high in zip(edges[:-1], edges[1:], strict=True)]
    return {column: bands for column in imu_columns()}


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
        expected = entry.get("expected_modality")
        if expected is not None and expected not in MODALITIES:
            raise ValueError(f"faults.json: {session_id!r} has unknown expected_modality")
        result[str(session_id)] = {
            "fault": fault,
            "onset_ms": int(onset),
            "expected_modality": expected,
        }
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


def physics_feature_groups(roles: RoleMap) -> dict[str, str]:
    """Estimator group for every physics feature, from the validated role config."""
    return {
        feature: roles.group(estimator)
        for estimator, names in PHYSICS_FEATURES_BY_ESTIMATOR.items()
        for feature in names
    }


def _physics_features(
    features: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame | None],
    table: GearTable | None,
    groups: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for row in features.itertuples(index=False):
        frame = frames.get(str(row.session_id))
        if frame is None:
            rows.append({"cooling_ok": 0.0, "charging_ok": 0.0, "gearing_ok": 0.0})
        else:
            rows.append(_physics_window(frame, int(row.start_ms), int(row.end_ms), table))
    result = pd.DataFrame(rows, index=features.index)
    if groups:
        result.attrs["feature_groups"] = dict(groups)
    return result


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


def _scored_hours(scored: np.ndarray, start_ms: np.ndarray, end_ms: np.ndarray) -> float:
    """Hours covered by the contiguous runs of scored windows (unscored gaps excluded)."""
    if scored.size == 0:
        return 0.0
    first = np.flatnonzero(scored & ~np.r_[False, scored[:-1]])
    last = np.flatnonzero(scored & ~np.r_[scored[1:], False])
    total = sum(
        float(end_ms[a : b + 1].max() - start_ms[a : b + 1].min())
        for a, b in zip(first, last, strict=True)
    )
    return total / 3_600_000.0


def _false_alarms(
    scores: pd.DataFrame,
    model: str,
    threshold: float,
    session_ids: Sequence[str],
) -> dict[str, float]:
    """Alarm episodes and scored hours of healthy sessions.

    Episodes are counted over the full time-ordered windows with unscored windows as
    "no alarm", so two alarm runs separated by an unscored gap stay two episodes. Hours
    sum each contiguous scored run, so an unscored gap is not counted as healthy riding.
    """
    episodes = 0
    hours = 0.0
    for session_id in session_ids:
        rows = scores.loc[scores["session_id"] == session_id].sort_values("start_ms")
        values = rows[model].to_numpy(dtype=float)
        scored = np.isfinite(values)
        with np.errstate(invalid="ignore"):
            alarm = scored & (values > threshold)
        episodes += _episodes(alarm)
        hours += _scored_hours(
            scored,
            rows["start_ms"].to_numpy(dtype=float),
            rows["end_ms"].to_numpy(dtype=float),
        )
    return {
        "episodes": float(episodes),
        "hours": hours,
        "per_hour": episodes / hours if hours > 0.0 else 0.0,
    }


def _single_modality(weights: np.ndarray, modalities: Sequence[str]) -> str:
    """Modality of the largest single feature weight."""
    if weights.shape != (len(modalities),):
        raise ValueError("need exactly one modality per feature")
    if not np.isfinite(weights).all():
        return ""
    return modalities[int(np.argmax(weights))]


def pca_training_means(
    baseline: BaselineModel,
    features: pd.DataFrame,
    modalities: Sequence[str],
    train_ids: Sequence[str],
) -> dict[str, float | None]:
    """Per-modality mean summed squared PCA residual on the healthy training windows.

    Only rows whose session is in ``train_ids`` are read; the baseline itself was fitted
    on complete rows from the same sessions, so val/test rows can never change these
    means. A modality with a non-finite or zero mean is stored as ``None`` and skipped
    by the ratio attribution.
    """
    if len(modalities) != len(baseline.feature_names):
        raise ValueError("need exactly one modality per baseline feature")
    train = features.loc[features["session_id"].astype(str).isin(set(train_ids))]
    residuals = baseline.pca_residuals(train)
    complete = np.isfinite(residuals).all(axis=1)
    names = sorted(set(modalities))
    means: dict[str, float | None] = {}
    for name in names:
        selected = residuals[complete][:, [modality == name for modality in modalities]]
        if selected.shape[0] == 0 or selected.shape[1] == 0:
            means[name] = None
            continue
        value = float(np.mean(np.sum(selected, axis=1)))
        means[name] = value if np.isfinite(value) and value > 0.0 else None
    return means


def pca_alarm_modality(
    residual_sq: np.ndarray,
    modalities: Sequence[str],
    training_mean_sq: Mapping[str, float | None],
) -> str:
    """Modality with the largest normalised squared PCA residual.

    ``residual_sq`` is the per-feature squared residual and ``modalities`` gives each
    feature's modality. A modality's score is its summed residual divided by its mean
    summed residual on the healthy training windows. Normalising removes the structural
    advantage of a modality with more features (for example the six IMU axes times
    statistics and FFT bands). A modality with a zero or non-finite training mean is
    skipped; if every modality is skipped the window has no attribution (``""``). Ties
    go to the alphabetically first modality, and a non-finite window is also ``""``.
    """
    weights = np.asarray(residual_sq, dtype=float)
    if weights.shape != (len(modalities),):
        raise ValueError("need exactly one modality per feature")
    if not np.isfinite(weights).all():
        return ""
    ratios: dict[str, float] = {}
    for name in sorted(set(modalities)):
        mean = training_mean_sq.get(name)
        if mean is None:
            continue
        mean = float(mean)
        if not np.isfinite(mean) or mean <= 0.0:
            continue
        total = float(np.sum(weights[[modality == name for modality in modalities]]))
        ratios[name] = total / mean
    if not ratios:
        return ""
    return min(ratios, key=lambda name: (-ratios[name], name))


def z_alarm_modality(abs_z: np.ndarray, modalities: Sequence[str]) -> str:
    """Modality of the feature with the largest |z| (``""`` for a non-finite window).

    This stays unnormalised: |z| is already scale-free per feature, so the modality with
    the largest single deviation is the intended attribution.
    """
    return _single_modality(np.abs(np.asarray(abs_z, dtype=float)), modalities)


def _modalities_of(feature_names: Sequence[str]) -> list[str]:
    metadata = feature_metadata(feature_names, allow_unknown=True)
    return [metadata[name].modality for name in feature_names]


def _numeric_feature_columns(features: pd.DataFrame) -> list[str]:
    """Numeric feature columns (identity columns and the split label excluded)."""
    skipped = {*IDENTITY_COLUMNS, "split"}
    return [
        str(column)
        for column in features.select_dtypes(include=[np.number]).columns
        if str(column) not in skipped
    ]


def _baseline_feature_names(features: pd.DataFrame) -> list[str]:
    """Numeric feature columns that have at least one finite value in the fit rows."""
    return [
        column for column in _numeric_feature_columns(features) if features[column].notna().any()
    ]


def _baseline_unscored(
    features: pd.DataFrame,
    baseline: BaselineModel,
    modalities: Sequence[str],
    split: Mapping[str, Sequence[str]],
) -> dict[str, list[dict[str, object]]]:
    """Sessions/windows without every baseline feature, with the missing modalities."""
    matrix = features[baseline.feature_names].to_numpy(dtype=float)
    missing = ~np.isfinite(matrix)
    incomplete = missing.any(axis=1)
    session = features["session_id"].astype(str).to_numpy()
    modality_of = np.asarray(modalities, dtype=object)
    result: dict[str, list[dict[str, object]]] = {}
    for part, session_ids in split.items():
        rows = np.flatnonzero(incomplete & np.isin(session, list(session_ids)))
        groups = pd.DataFrame({"session_id": session[rows]}).groupby("session_id").indices
        result[part] = [
            {
                "session_id": str(session_id),
                "windows": int(len(positions)),
                "missing_modalities": sorted(
                    {str(name) for name in modality_of[missing[rows[positions]].any(axis=0)]}
                ),
            }
            for session_id, positions in sorted(groups.items())
        ]
    return result


def _fit_baseline(
    features: pd.DataFrame,
    split_manifest: Mapping[str, Sequence[str]],
    train_ids: Sequence[str],
) -> BaselineModel:
    """Fit the scaler and both baselines on complete train windows only.

    A window missing any baseline feature is excluded from the fit; it later receives
    no baseline score and is reported as unscored (ML-24).
    """
    train = features.loc[features["session_id"].astype(str).isin(set(train_ids))]
    names = _baseline_feature_names(train)
    if not names:
        raise ValueError("no numeric baseline features in the train windows")
    complete = np.isfinite(train[names].to_numpy(dtype=float)).all(axis=1)
    if not complete.any():
        raise ValueError("baseline needs at least one complete train window")
    complete_features = train.loc[complete, ["session_id", *names]].copy()
    scaler = Standardizer.fit(complete_features, split_manifest)
    return BaselineModel.fit(complete_features, split_manifest, scaler)


def _physics_attribution(scores: pd.DataFrame, group_of: Mapping[str, str]) -> pd.DataFrame:
    """Physics alarms are CAN; the group is the estimator group with the largest score."""
    estimators = list(group_of)
    stacked = np.column_stack(
        [scores[f"physics_{name}"].to_numpy(dtype=float) for name in estimators]
    )
    best = np.argmax(stacked, axis=1)
    return pd.DataFrame(
        {
            "physics_modality": PHYSICS_MODALITY,
            "physics_group": [group_of[estimators[index]] for index in best],
        },
        index=scores.index,
    )


def _modality_label(value: object) -> str:
    """Report label of an alarm modality; an empty attribution is ``unattributed``."""
    return str(value) or UNATTRIBUTED


def _fault_results(
    scores: pd.DataFrame,
    model: str,
    threshold: float,
    test_ids: Sequence[str],
    faults: Mapping[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Per fault kind: detections, delay and the modality of every alarm after onset.

    ``modalities`` counts all alarm windows after onset per modality (the detections);
    ``first_alarm_modalities`` counts, per session, the modality of its first alarm.
    ``groups`` is the physics estimator group of the first alarms (empty for the
    baselines).
    """
    grouped: dict[str, dict[str, object]] = {}
    for session_id in test_ids:
        truth = faults.get(session_id)
        if truth is None:
            continue
        kind = str(truth["fault"])
        onset = int(truth["onset_ms"])
        expected = truth.get("expected_modality")
        rows = scores.loc[scores["session_id"] == session_id].sort_values("start_ms")
        rows = rows.loc[rows["start_ms"] >= onset]
        entry = grouped.setdefault(
            kind,
            {
                "sessions": 0,
                "detected": 0,
                "missed": 0,
                "not_evaluated": 0,
                "delays_ms": [],
                "expected_modality": expected,
                "modalities": Counter(),
                "first_alarm_modalities": Counter(),
                "first_alarm_matches_expected": 0,
                "groups": Counter(),
            },
        )
        entry["sessions"] = int(entry["sessions"]) + 1
        scored = rows.loc[rows[model].notna()]
        if scored.empty:
            entry["not_evaluated"] = int(entry["not_evaluated"]) + 1
            continue
        alarm = scored.loc[scored[model].to_numpy(dtype=float) > threshold]
        if alarm.empty:
            entry["missed"] = int(entry["missed"]) + 1
            continue
        first = alarm.iloc[0]
        entry["detected"] = int(entry["detected"]) + 1
        entry["delays_ms"].append(int(first["start_ms"]) - onset)
        entry["modalities"].update(_modality_label(value) for value in alarm[f"{model}_modality"])
        first_modality = _modality_label(first[f"{model}_modality"])
        entry["first_alarm_modalities"][first_modality] += 1
        if expected is not None and first_modality == expected:
            entry["first_alarm_matches_expected"] = int(entry["first_alarm_matches_expected"]) + 1
        if model == "physics":
            entry["groups"][str(first["physics_group"])] += 1

    result: dict[str, dict[str, object]] = {}
    for kind, entry in sorted(grouped.items()):
        delays = [int(value) for value in entry["delays_ms"]]
        result[kind] = {
            "sessions": int(entry["sessions"]),
            "detected": int(entry["detected"]),
            "missed": int(entry["missed"]),
            "not_evaluated": int(entry["not_evaluated"]),
            "median_delay_ms": float(np.median(delays)) if delays else None,
            "expected_modality": entry["expected_modality"],
            "modalities": dict(entry["modalities"]),
            "first_alarm_modalities": dict(entry["first_alarm_modalities"]),
            "first_alarm_matches_expected": (
                None
                if entry["expected_modality"] is None
                else entry["first_alarm_matches_expected"]
            ),
            "groups": dict(entry["groups"]),
        }
    return result


def _counts(counter: Mapping[str, int]) -> str:
    return ", ".join(f"{name}:{count}" for name, count in sorted(counter.items())) or "-"


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
    lines += [
        "",
        "## Baseline coverage",
        "",
        "A window without every baseline feature is not scored (missing modality); it is "
        "never counted as a detection, a miss or a false alarm. The physics models still "
        "score it.",
        "",
    ]
    excluded = data.get("baseline_excluded_features", [])
    if excluded:
        lines.append(
            "Features without a finite value in the healthy training windows are excluded "
            f"from the baselines (modalities: {', '.join(data['baseline_excluded_modalities'])}): "
            + ", ".join(f"`{name}`" for name in excluded)
            + "."
        )
    else:
        lines.append("No feature was excluded from the baselines.")
    lines += [
        "",
        "| split | session | unscored windows | missing modalities |",
        "|---|---|---|---|",
    ]
    for part in data["split"]:
        entries = data.get("baseline_unscored", {}).get(part, [])
        if not entries:
            lines.append(f"| {part} | (none) | 0 | - |")
            continue
        for entry in entries:
            lines.append(
                f"| {part} | {entry['session_id']} | {entry['windows']} | "
                f"{', '.join(entry['missing_modalities'])} |"
            )
    lines += ["", "## Thresholds", "", "| model | threshold |", "|---|---|"]
    for model in MODEL_NAMES:
        threshold = data["thresholds"][model]
        lines.append(f"| {model} | {NOT_CALIBRATED if threshold is None else f'{threshold:.6g}'} |")
    uncalibrated = data.get("not_calibrated", [])
    if uncalibrated:
        lines += [
            "",
            f"Note: {', '.join(uncalibrated)} {NOT_CALIBRATED}: no validation healthy window "
            "has a score, so every window of the model is treated as unscored (its faults "
            "are counted as not evaluated, no false-alarm hours).",
        ]
    lines += [
        "",
        "## PCA attribution reference",
        "",
        "The PCA modality ratio divides a window's summed squared residual by these "
        "healthy-training means, so a modality with more features is not favoured.",
        "",
        "| modality | mean summed squared residual |",
        "|---|---|",
    ]
    for name, mean in data.get("pca_modality_training_mean_sq", {}).items():
        lines.append(f"| {name} | {'skipped' if mean is None else f'{mean:.6g}'} |")
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
            "Modality is the signal group that raised the alarm: CAN (`did.*` / `derived.*`), "
            "IMU or other. Alarm windows count every window at or after onset above the "
            "threshold. " + MISS_NOTE,
            "",
            "| model | fault kind | expected modality | detected | missed | not evaluated "
            "| median delay ms | alarm windows by modality | first alarm modality "
            "| first alarm = expected | first alarm group |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for model in MODEL_NAMES:
            faults = data["models"][model]["test"]["faults"]
            if not faults:
                lines.append(f"| {model} | (none) | - | 0 | 0 | 0 | - | - | - | - | - |")
                continue
            for kind, entry in faults.items():
                delay = (
                    "-" if entry["median_delay_ms"] is None else f"{entry['median_delay_ms']:.0f}"
                )
                matches = entry["first_alarm_matches_expected"]
                match_text = "unknown" if matches is None else f"{matches}/{entry['detected']}"
                lines.append(
                    f"| {model} | {kind} | {entry['expected_modality'] or 'unknown'} | "
                    f"{entry['detected']}/{entry['sessions']} | {entry['missed']} | "
                    f"{entry['not_evaluated']} | {delay} | "
                    f"{_counts(entry['modalities'])} | "
                    f"{_counts(entry['first_alarm_modalities'])} | {match_text} | "
                    f"{_counts(entry['groups'])} |"
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
        kinds=("telemetry", "derived", "imu"),
        channels=None,
        window_ms=window_ms,
        stride_ms=stride_ms,
        max_gap_ms=max_gap_ms,
        seed=seed,
        bands=_imu_bands(),
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
    physics = _physics_features(result.features, frames, table, physics_feature_groups(roles))
    train_healthy = _healthy(result.split["train"], faults)
    physics_scores = _physics_scores(physics, result.features["session_id"].isin(train_healthy))

    baseline = _fit_baseline(result.features, result.split, train_healthy)
    baseline_scores = baseline.scores(result.features)
    excluded_features = [
        name
        for name in _numeric_feature_columns(result.features)
        if name not in baseline.feature_names
    ]
    excluded_modalities = sorted(set(_modalities_of(excluded_features)))
    feature_modalities = _modalities_of(baseline.feature_names)
    baseline_unscored = _baseline_unscored(
        result.features, baseline, feature_modalities, result.split
    )
    pca_training_mean_sq = pca_training_means(
        baseline, result.features, feature_modalities, train_healthy
    )
    pca_modality = [
        pca_alarm_modality(row, feature_modalities, pca_training_mean_sq)
        for row in baseline.pca_residuals(result.features)
    ]
    z_modality = [
        z_alarm_modality(row, feature_modalities) for row in baseline.abs_z(result.features)
    ]

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
            "baseline_pca_modality": pca_modality,
            "baseline_z_modality": z_modality,
        }
    )
    attribution = _physics_attribution(
        scores, {name: roles.group(name) for name in PHYSICS_FEATURES_BY_ESTIMATOR}
    )
    scores["physics_modality"] = attribution["physics_modality"].to_numpy()
    scores["physics_group"] = attribution["physics_group"].to_numpy()

    val_healthy = _healthy(result.split["val"], faults)
    thresholds: dict[str, float | None] = {}
    for model in MODEL_NAMES:
        validation = scores.loc[scores["session_id"].isin(val_healthy), model]
        if np.isfinite(validation.to_numpy(dtype=float)).any():
            thresholds[model] = _threshold(validation, target_false_alarms_per_hour, stride_ms)
        else:
            # Nothing to calibrate on: the whole model is unscored, never a miss.
            thresholds[model] = None
            scores[model] = np.nan

    test_ids = result.split["test"]
    test_healthy = _healthy(test_ids, faults)
    models: dict[str, dict[str, object]] = {}
    for model in MODEL_NAMES:
        limit = math.nan if thresholds[model] is None else thresholds[model]
        models[model] = {
            "threshold": thresholds[model],
            "test": {
                "false_alarms": _false_alarms(scores, model, limit, test_healthy),
                "faults": _fault_results(scores, model, limit, test_ids, faults),
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
        "pca_modality_training_mean_sq": pca_training_mean_sq,
        "not_calibrated": [model for model in MODEL_NAMES if thresholds[model] is None],
        "baseline_excluded_features": excluded_features,
        "baseline_excluded_modalities": excluded_modalities,
        "baseline_unscored": baseline_unscored,
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


def parse_seeds(text: str) -> list[int]:
    """Parse ``--seeds 0,1,2``: at least one integer, no duplicates."""
    items = [item.strip() for item in text.split(",")]
    try:
        seeds = [int(item) for item in items]
    except ValueError:
        raise ValueError(f"invalid seed list {text!r}; expected integers like 0,1,2") from None
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seed in {text!r}")
    return seeds


def _seed_metrics(data: Mapping[str, object]) -> dict[str, float]:
    """The scalar metrics of one single-seed evaluation, keyed ``model.[kind.]metric``.

    A rate with nothing to divide by (no evaluated session of a fault kind, no scored
    healthy hour) is omitted, so an all-unscored seed never counts like a miss.
    """
    metrics: dict[str, float] = {}
    for model, entry in data["models"].items():
        test = entry["test"]
        hours = test["false_alarms"].get("hours")
        if hours is None or hours > 0.0:
            # No scored healthy hour (an uncalibrated model) leaves the rate undefined.
            metrics[f"{model}.false_alarms_per_hour"] = float(test["false_alarms"]["per_hour"])
        for kind, fault in test["faults"].items():
            sessions = int(fault["sessions"])
            not_evaluated = int(fault.get("not_evaluated", 0))
            evaluated = sessions - not_evaluated
            prefix = f"{model}.{kind}"
            if evaluated:
                metrics[f"{prefix}.detection_rate"] = fault["detected"] / evaluated
            metrics[f"{prefix}.not_evaluated_rate"] = not_evaluated / sessions if sessions else 0.0
            if fault["median_delay_ms"] is not None:
                metrics[f"{prefix}.median_delay_ms"] = float(fault["median_delay_ms"])
            if fault["first_alarm_matches_expected"] is not None and evaluated:
                metrics[f"{prefix}.first_alarm_expected_rate"] = (
                    fault["first_alarm_matches_expected"] / evaluated
                )
    return metrics


def _seed_first_alarm_modality(fault: Mapping[str, object]) -> str:
    """One value per seed: the first-alarm modality, ``no alarm`` or ``not evaluated``.

    A fault kind can have several test sessions in one seed. If their first alarms all
    agree, that modality is used; if they disagree, the seed is ``mixed``.
    """
    sessions = int(fault.get("sessions", 0))
    detected = int(fault.get("detected", 0))
    not_evaluated = int(fault.get("not_evaluated", 0))
    if detected == 0:
        if sessions > 0 and not_evaluated == sessions:
            return NOT_EVALUATED
        return NO_ALARM
    modalities = fault.get("first_alarm_modalities", {})
    if not isinstance(modalities, Mapping):
        raise ValueError("first_alarm_modalities must be an object")
    names = sorted(_modality_label(name) for name, count in modalities.items() if int(count) > 0)
    if len(set(names)) == 1:
        return names[0]
    return MIXED


def aggregate_attribution(
    per_seed: Mapping[int, Mapping[str, object]],
) -> dict[str, dict[str, dict[str, object]]]:
    """First-alarm modality stability per model and fault kind over the seeds.

    ``first_alarm_modalities`` has one entry per seed that has the fault kind in its test
    split. A seed with no alarm is ``no alarm`` (not a mismatch); an all-unscored seed is
    ``not evaluated``. ``fraction_matching_expected`` divides matching seeds by all seeds
    with the fault kind, or is ``None`` when the ground truth has no expected modality.
    """
    collected: dict[str, dict[str, dict[str, object]]] = {}
    for seed in sorted(per_seed):
        data = per_seed[seed]
        for model, entry in data["models"].items():
            for kind, fault in entry["test"]["faults"].items():
                bucket = collected.setdefault(model, {}).setdefault(
                    kind,
                    {
                        "expected_modality": fault.get("expected_modality"),
                        "seeds": [],
                        "labels": [],
                    },
                )
                bucket["seeds"].append(int(seed))
                bucket["labels"].append(_seed_first_alarm_modality(fault))

    result: dict[str, dict[str, dict[str, object]]] = {}
    for model, kinds in sorted(collected.items()):
        result[model] = {}
        for kind, bucket in sorted(kinds.items()):
            seeds = [int(seed) for seed in bucket["seeds"]]
            labels = Counter(bucket["labels"])
            expected = bucket["expected_modality"]
            matches = labels[expected] if expected is not None else 0
            result[model][kind] = {
                "expected_modality": expected,
                "seeds": seeds,
                "first_alarm_modalities": list(bucket["labels"]),
                "matches_expected": matches,
                "fraction_matching_expected": (
                    matches / len(seeds) if expected is not None else None
                ),
                "no_alarm_seeds": labels[NO_ALARM],
                "not_evaluated_seeds": labels[NOT_EVALUATED],
                "mixed_seeds": labels[MIXED],
            }
    return result


def aggregate_metrics(per_seed: Mapping[int, Mapping[str, object]]) -> dict[str, dict[str, float]]:
    """Mean/min/max of every metric over the seeds that produced it.

    A fault kind that falls into the test split for only some seeds is aggregated over
    those seeds; ``n_seeds`` says how many.
    """
    collected: dict[str, list[float]] = {}
    for data in per_seed.values():
        for name, value in _seed_metrics(data).items():
            collected.setdefault(name, []).append(value)
    return {
        name: {
            "n_seeds": len(values),
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for name, values in sorted(collected.items())
    }


def _aggregate_markdown(data: Mapping[str, object]) -> str:
    seeds = data["seeds"]
    lines = [
        "# D-083 stage-1 evaluation over several split seeds",
        "",
        f"- defs: {data['defs_version']}",
        f"- seeds: {', '.join(str(seed) for seed in seeds)}",
        "- a seed changes only the session split, never the data; per-seed reports are in "
        + ", ".join(f"`seed-{seed}/`" for seed in seeds),
        "",
        SYNTHETIC_NOTE + ".",
        "",
        MISS_NOTE,
        "",
        "## Metrics over seeds",
        "",
        "| metric | seeds | mean | min | max |",
        "|---|---|---|---|---|",
    ]
    for name, stats in data["aggregate"].items():
        lines.append(
            f"| {name} | {stats['n_seeds']} | {stats['mean']:.6g} | {stats['min']:.6g} | "
            f"{stats['max']:.6g} |"
        )
    lines += [
        "",
        "## Attribution stability",
        "",
        "One entry per seed that has the fault kind in its test split. `no alarm` is not a "
        "mismatch; `not evaluated` means every window was unscored; `mixed` means the "
        "seed's test sessions disagreed on the first-alarm modality.",
        "",
        "| model | fault kind | expected modality | first alarm modality per seed "
        "| matching seeds | fraction |",
        "|---|---|---|---|---|---|",
    ]
    attribution = data.get("attribution", {})
    if not attribution:
        lines.append("| (none) | - | - | - | 0/0 | - |")
    for model, kinds in attribution.items():
        for kind, entry in kinds.items():
            labels = ", ".join(
                f"seed {seed}: {modality}"
                for seed, modality in zip(
                    entry["seeds"], entry["first_alarm_modalities"], strict=True
                )
            )
            fraction = (
                "unknown"
                if entry["fraction_matching_expected"] is None
                else f"{entry['fraction_matching_expected']:.3f}"
            )
            lines.append(
                f"| {model} | {kind} | {entry['expected_modality'] or 'unknown'} | {labels} | "
                f"{entry['matches_expected']}/{len(entry['seeds'])} | {fraction} |"
            )
    lines.append("")
    return "\n".join(lines)


def evaluate_seeds(
    data_dir: str | Path,
    out_dir: str | Path,
    config_path: str | Path,
    seeds: Sequence[int],
    **kwargs: object,
) -> EvaluationResult:
    """Run ``evaluate`` once per split seed and aggregate the metrics.

    Per-seed reports go to ``<out>/seed-<n>/``; ``<out>/evaluation.json`` and
    ``evaluation.md`` hold every per-seed result plus the mean/min/max per metric.
    """
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    out_path = Path(out_dir)
    per_seed: dict[int, dict] = {}
    for seed in seeds:
        result = evaluate(data_dir, out_path / f"seed-{seed}", config_path, seed=seed, **kwargs)
        per_seed[seed] = result.data

    data: dict[str, object] = {
        "defs_version": defs_version(),
        "seeds": list(seeds),
        "per_seed": {str(seed): per_seed[seed] for seed in seeds},
        "aggregate": aggregate_metrics(per_seed),
        "attribution": aggregate_attribution(per_seed),
        "note": SYNTHETIC_NOTE,
    }
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "evaluation.json"
    markdown_path = out_path / "evaluation.md"
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_aggregate_markdown(data), encoding="utf-8")
    return EvaluationResult(data=data, json_path=json_path, markdown_path=markdown_path)


__all__ = [
    "EvaluationResult",
    "SYNTHETIC_NOTE",
    "aggregate_attribution",
    "aggregate_metrics",
    "evaluate",
    "evaluate_seeds",
    "parse_seeds",
    "pca_training_means",
]
