"""Dataset builder: sessions -> windows -> features, split and scaler.

The output directory is caller-given and never inside the server's data directory.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from vehicle_ml.data.store import KINDS, SessionStore
from vehicle_ml.features import features_frame
from vehicle_ml.normalize import Standardizer
from vehicle_ml.split import PARTS, split_sessions
from vehicle_ml.windows import TIME_COLUMN, DropReport, align_frames, make_windows

FRACTIONS: tuple[float, float, float] = (0.7, 0.15, 0.15)

# Version 4 tester statistics (vehicle-server decode.V4_EXTRA_HEADER).
_TESTER_STATS = frozenset(
    {
        "step_gap_max_ms",
        "step_gap_over_count",
        "rtt_did",
        "rtt_min_ms",
        "rtt_max_ms",
        "rtt_sum_ms",
        "rtt_count",
        "rtt_nrc78_count",
    }
)

# Columns that describe the storage, not a measured channel (vehicle-server/README.md
# "Storage layout"). They are skipped by --channels all and may still be requested by name.
_METADATA_COLUMNS = frozenset(
    {
        "session_id",
        "rx_utc",
        "rx_mono_ms",
        "seq",
        "packet_version",
        "decode_error",
        "block_seq",
        "block_flags",
        "block_missing_samples",
        "sample_index",
        "device_time_ms",
        "total_records",
        "first_record",
        "did",
        "state",
        "age_ms",
        "raw",
        "sample_device_time_ms",
        "signal",
        "unit",
        "value",
        "fix_type_name",
    }
)


def _is_metadata(name: str) -> bool:
    """True for columns --channels all skips; an explicit name still selects them.

    Beyond the exact names above, rules cover the server's per-signal and health
    columns: ``<signal>_age_ms`` / ``<signal>_valid`` (vehicle-server parquet.py
    telemetry docstring), ``can_*`` and ``unanswered_did_count`` (version 3 CAN health,
    vehicle-server decode.CAN_HEALTH_FIELDS) and the version 4 tester statistics
    (vehicle-server decode.V4_EXTRA_HEADER).
    """
    return (
        name in _METADATA_COLUMNS
        or name in _TESTER_STATS
        or name.endswith(("_age_ms", "_valid", "_raw"))
        or name.startswith("can_")
        or name == "unanswered_did_count"
    )


@dataclass
class DatasetResult:
    features: pd.DataFrame
    split: dict[str, list[str]]
    scaler: Standardizer
    seed: int
    drops: dict[str, int] = field(default_factory=dict)
    channels: dict[str, list[str]] = field(default_factory=dict)
    defs_versions: dict[str, str | None] = field(default_factory=dict)
    excluded: set[str] = field(default_factory=set)
    base_kinds: dict[str, int] = field(default_factory=dict)


@dataclass
class _SessionData:
    frame: pd.DataFrame
    report: DropReport
    channels: dict[str, list[str]] = field(default_factory=dict)
    excluded: set[str] = field(default_factory=set)
    base_kind: str | None = None


def _wide(
    frame: pd.DataFrame, kind: str, requested: Sequence[str] | None
) -> tuple[pd.DataFrame, set[str]]:
    """Wide numeric frame plus the columns ``--channels all`` auto-excluded."""
    if kind == "derived":
        frame = frame.pivot_table(
            index=["session_id", TIME_COLUMN],
            columns="signal",
            values="value",
            aggfunc="last",
        ).reset_index()
        frame.columns.name = None
    excluded: set[str] = set()
    if requested is None:
        numeric = [
            str(column) for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])
        ]
        excluded = {column for column in numeric if _is_metadata(column)}
        columns = [column for column in numeric if column not in excluded]
    else:
        columns = [column for column in requested if column in frame.columns]
    identity = [column for column in ("session_id", TIME_COLUMN) if column in frame.columns]
    return frame[[*identity, *columns]], excluded


def _kind_frames(
    store: SessionStore, session_id: str, kinds: Sequence[str], requested: Sequence[str] | None
) -> tuple[dict[str, pd.DataFrame], set[str]]:
    frames: dict[str, pd.DataFrame] = {}
    excluded: set[str] = set()
    for kind in kinds:
        frame = store.load(session_id, kind)
        if frame is None or frame.empty:
            continue
        wide, skipped = _wide(frame, kind, requested)
        excluded |= skipped
        if len(wide.columns) > (1 if "session_id" in wide.columns else 0) + 1:
            frames[kind] = wide
    return frames, excluded


def _session_windows(
    store: SessionStore,
    session_id: str,
    kinds: Sequence[str],
    channels: Sequence[str] | None,
    window_ms: int,
    stride_ms: int,
    max_gap_ms: int,
    bands: Mapping[str, Sequence[tuple[float, float]]],
) -> _SessionData:
    frames, excluded = _kind_frames(store, session_id, kinds, channels)
    if not frames:
        report = DropReport(candidates=0, kept=0, dropped={"no_data": 1})
        return _SessionData(pd.DataFrame(), report, excluded=excluded)

    # Time base: the kind with the most rows; ties go to the earlier kind in KINDS.
    base_kind = max(frames, key=lambda kind: (len(frames[kind]), -KINDS.index(kind)))
    base = frames[base_kind]
    combined = base.copy()
    used_channels = [
        str(column) for column in base.columns if column not in {"session_id", TIME_COLUMN}
    ]
    channels_by_kind = {base_kind: list(used_channels)}

    for kind, frame in frames.items():
        if kind == base_kind:
            continue
        extra = [
            str(column) for column in frame.columns if column not in {"session_id", TIME_COLUMN}
        ]
        duplicate = set(extra) & set(used_channels)
        if duplicate:
            names = ", ".join(sorted(duplicate))
            raise ValueError(f"channel appears in more than one kind: {names}")
        aligned = align_frames(combined, frame, extra, tolerance_ms=max_gap_ms)
        for column in extra:
            combined[column] = aligned[column]
        used_channels.extend(extra)
        channels_by_kind[kind] = extra

    windows, report = make_windows(
        combined,
        used_channels,
        window_ms=window_ms,
        stride_ms=stride_ms,
        max_gap_ms=max_gap_ms,
    )
    frame = (
        features_frame(windows, bands, report=report, allow_unknown=True)
        if windows
        else pd.DataFrame()
    )
    return _SessionData(frame, report, channels_by_kind, excluded, base_kind)


def _defs_version(session_dir: Path) -> str | None:
    report = session_dir / "report.json"
    if not report.exists():
        return None
    try:
        value = json.loads(report.read_text(encoding="utf-8")).get("defs_version")
    except (json.JSONDecodeError, OSError):
        return None
    return str(value) if value is not None else None


def build_dataset(
    data_dir: str | Path,
    out_dir: str | Path,
    kinds: Sequence[str] = ("telemetry", "imu"),
    channels: Sequence[str] | None = None,
    window_ms: int = 2_000,
    stride_ms: int = 1_000,
    max_gap_ms: int = 500,
    seed: int = 0,
    bands: Mapping[str, Sequence[tuple[float, float]]] | None = None,
) -> DatasetResult:
    """Build the dataset in memory; ``write_dataset`` persists it."""
    data = Path(data_dir).resolve()
    out = Path(out_dir).resolve()
    if out == data or data in out.parents:
        raise ValueError("--out must not be inside --data-dir")
    unknown = [kind for kind in kinds if kind not in KINDS]
    if unknown:
        raise ValueError(f"unknown kinds: {', '.join(unknown)}")
    if not kinds:
        raise ValueError("at least one kind is required")

    store = SessionStore(data)
    sessions = store.sessions()
    ids = [str(session_id) for session_id in sessions["session_id"]]
    split = split_sessions(ids, FRACTIONS, seed)
    split_of = {session_id: part for part in PARTS for session_id in split[part]}

    rows: list[pd.DataFrame] = []
    drops: dict[str, int] = {}
    used_channels: dict[str, set[str]] = {kind: set() for kind in kinds}
    excluded: set[str] = set()
    base_kinds: dict[str, int] = {}
    defs_versions: dict[str, str | None] = {}
    for session_id in ids:
        session = _session_windows(
            store,
            session_id,
            kinds,
            channels,
            window_ms,
            stride_ms,
            max_gap_ms,
            bands or {},
        )
        for reason, count in session.report.dropped.items():
            drops[reason] = drops.get(reason, 0) + count
        for kind, names in session.channels.items():
            used_channels[kind].update(names)
        excluded |= session.excluded
        if session.base_kind is not None:
            base_kinds[session.base_kind] = base_kinds.get(session.base_kind, 0) + 1
        defs_versions[session_id] = _defs_version(data / "sessions" / session_id)
        if not session.frame.empty:
            session.frame["split"] = split_of[session_id]
            rows.append(session.frame)

    if not any(used_channels.values()):
        raise ValueError("no requested channels found in the data")
    features = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["session_id", "start_ms", "end_ms", "split"])
    )
    train = features[features["split"] == "train"] if "split" in features.columns else features
    scaler = Standardizer.fit(train, {"train": split["train"]})
    return DatasetResult(
        features=features,
        split=split,
        scaler=scaler,
        seed=seed,
        drops=drops,
        channels={kind: sorted(names) for kind, names in used_channels.items() if names},
        defs_versions=defs_versions,
        excluded=excluded,
        base_kinds=base_kinds,
    )


def _dataset_card(result: DatasetResult, sessions: pd.DataFrame, kinds: Sequence[str]) -> str:
    lines = [
        "# Dataset card",
        "",
        f"Sessions: {len(sessions)}  ",
        f"Windows: {len(result.features)}  ",
        f"Kinds: {', '.join(kinds)}",
        "",
        "## Splits",
        "",
        "| split | sessions | windows |",
        "|---|---|---|",
    ]
    for part in PARTS:
        window_count = (
            int((result.features["split"] == part).sum())
            if "split" in result.features.columns
            else 0
        )
        lines.append(f"| {part} | {len(result.split[part])} | {window_count} |")
    lines += ["", "## Channels", ""]
    for kind in kinds:
        names = result.channels.get(kind, [])
        lines.append(f"- **{kind}**: {', '.join(names) if names else '(none)'}")
    if result.excluded:
        lines += [
            "",
            "Auto-excluded by `--channels all` (storage metadata, per-signal age/validity, "
            "CAN health, tester statistics): " + ", ".join(sorted(result.excluded)),
        ]
    lines += [
        "",
        "## Time base",
        "",
        "Time base per session: the kind with the most rows; other kinds are "
        "nearest-previous aligned within max_gap_ms.",
        "",
        "| base kind | sessions |",
        "|---|---|",
    ]
    for kind in KINDS:
        if kind in result.base_kinds:
            lines.append(f"| {kind} | {result.base_kinds[kind]} |")
    lines += [
        "",
        "## Modality availability",
        "",
        "| session | imu | gps | vehicle signals | derived |",
        "|---|---|---|---|---|",
    ]
    for _, row in sessions.iterrows():
        lines.append(
            f"| {row['session_id']} | {bool(row['has_imu'])} | {bool(row['has_gps'])} | "
            f"{bool(row['has_vehicle_signals'])} | {bool(row['has_derived'])} |"
        )
    lines += ["", "## Dropped windows", "", "| reason | count |", "|---|---|"]
    if result.drops:
        for reason, count in sorted(result.drops.items()):
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("| — | 0 |")
    lines += ["", "## Defs versions", "", "| session | defs_version |", "|---|---|"]
    for session_id, version in sorted(result.defs_versions.items()):
        lines.append(f"| {session_id} | {version or 'unknown'} |")
    lines.append("")
    return "\n".join(lines)


def write_dataset(
    result: DatasetResult, out_dir: str | Path, sessions: pd.DataFrame, kinds: Sequence[str]
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.features.to_parquet(out / "features.parquet", index=False)
    (out / "split.json").write_text(
        json.dumps(
            {
                "seed": result.seed,
                "fractions": list(FRACTIONS),
                "sessions": {
                    session_id: part for part in PARTS for session_id in result.split[part]
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "scaler.json").write_text(result.scaler.to_json(), encoding="utf-8")
    (out / "dataset.md").write_text(_dataset_card(result, sessions, kinds), encoding="utf-8")
