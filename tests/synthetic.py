"""Synthetic vehicle-server data directories for the vehicle-ml tests.

The layout mirrors the server's storage (vehicle-server/README.md "Storage layout"):
``index.sqlite`` plus ``sessions/<id>/parquet/*.parquet`` and an optional
``report.json``. Column names and types are copied from the server writers:

- ``vehicle-server/src/vehicle_server/index.py:13`` (sessions table)
- ``vehicle-server/src/vehicle_server/parquet.py:43`` (telemetry)
- ``vehicle-server/src/vehicle_server/parquet.py:99`` (imu)
- ``vehicle-server/src/vehicle_server/parquet.py:130`` (gps)
- ``vehicle-server/src/vehicle_server/parquet.py:167`` (signals)
- ``vehicle-server/src/vehicle_server/derived/__init__.py:94`` (derived, long format)

All values are synthetic; no real ride data, GPS position or vehicle-defs fact is used.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE_UTC = datetime(2026, 10, 1, 9, 0, 0, tzinfo=UTC)
DEFS_VERSION = "0.20.0"


@dataclass(frozen=True)
class SessionSpec:
    """One synthetic session; ``gap_after_ms`` starts a hole of ``gap_ms`` in telemetry."""

    session_id: str
    created_utc: str = "2026-10-01T09:00:00Z"
    status: str = "ok"
    packet_count: int = 101
    loss_percent: float = 0.0
    has_imu: bool = True
    has_gps: bool = True
    has_vehicle_signals: bool = True
    has_derived: bool = True
    duration_ms: int = 10_000
    gap_after_ms: int | None = None
    gap_ms: int = 0
    extra_position_column: bool = False
    write_report: bool = True


DEFAULT_SESSIONS: tuple[SessionSpec, ...] = (
    SessionSpec(session_id="s1"),
    SessionSpec(
        session_id="s2",
        status="warn",
        packet_count=80,
        loss_percent=2.5,
        has_imu=False,
        gap_after_ms=4_000,
        gap_ms=3_000,
        write_report=False,
    ),
    SessionSpec(
        session_id="s3",
        status="fail",
        packet_count=60,
        loss_percent=12.0,
        has_gps=False,
        has_derived=False,
        extra_position_column=True,
    ),
)


def _iso(ms: int) -> str:
    return (BASE_UTC + timedelta(milliseconds=ms)).isoformat().replace("+00:00", "Z")


def _times(duration_ms: int, step_ms: int) -> list[int]:
    return list(range(0, duration_ms + 1, step_ms))


def _telemetry(spec: SessionSpec) -> pd.DataFrame:
    times = _times(spec.duration_ms, 100)
    if spec.gap_after_ms is not None and spec.gap_ms:
        times = [t for t in times if not (spec.gap_after_ms < t <= spec.gap_after_ms + spec.gap_ms)]
    seconds = np.asarray(times, dtype=float) / 1000.0
    data: dict[str, list[object]] = {
        "session_id": [spec.session_id] * len(times),
        "rx_utc": [_iso(t) for t in times],
        "rx_mono_ms": times,
        "seq": list(range(len(times))),
        "packet_version": [4] * len(times),
        "device_time_ms": times,
        "decode_error": [None] * len(times),
        "engine_speed_rpm": (1000 + 500 * np.sin(seconds)).tolist(),
        "engine_speed_age_ms": [50] * len(times),
        "engine_speed_valid": [True] * len(times),
        "vehicle_speed_kmh": (50 + 20 * np.sin(seconds / 2)).tolist(),
        "vehicle_speed_age_ms": [50] * len(times),
        "vehicle_speed_valid": [True] * len(times),
        "coolant_temp_degc": (80 + 5 * np.cos(seconds / 3)).tolist(),
        "coolant_temp_age_ms": [50] * len(times),
        "coolant_temp_valid": [True] * len(times),
        "throttle_pos_pct": (20 + 10 * np.sin(seconds / 4)).tolist(),
        "throttle_pos_age_ms": [50] * len(times),
        "throttle_pos_valid": [True] * len(times),
        "battery_voltage_v": (12.4 + 0.2 * np.cos(seconds / 5)).tolist(),
        "battery_voltage_age_ms": [50] * len(times),
        "battery_voltage_valid": [True] * len(times),
        "step_gap_max_ms": [110] * len(times),
        "step_gap_over_count": [0] * len(times),
        "rtt_did": [0] * len(times),
        "rtt_min_ms": [10] * len(times),
        "rtt_max_ms": [20] * len(times),
        "rtt_sum_ms": [30] * len(times),
        "rtt_count": [2] * len(times),
        "rtt_nrc78_count": [0] * len(times),
        "can_bus_state": [0] * len(times),
        "can_tx_error_count": [0] * len(times),
        "can_rx_error_count": [0] * len(times),
        "can_bus_off_count": [0] * len(times),
        "unanswered_did_count": [0] * len(times),
        "can_flags": [0] * len(times),
    }
    if spec.extra_position_column:
        data["gps_latitude"] = [0.0] * len(times)
    return pd.DataFrame(data)


def _imu(spec: SessionSpec) -> pd.DataFrame:
    times = _times(spec.duration_ms, 10)
    seconds = np.asarray(times, dtype=float) / 1000.0
    n = len(times)
    return pd.DataFrame(
        {
            "session_id": [spec.session_id] * n,
            "rx_utc": [_iso(t) for t in times],
            "rx_mono_ms": times,
            "block_seq": [t // 100 for t in times],
            "block_flags": [0] * n,
            "block_missing_samples": [0] * n,
            "sample_index": [t // 10 for t in times],
            "device_time_ms": times,
            "ax_raw": [0] * n,
            "ay_raw": [0] * n,
            "az_raw": [0] * n,
            "gx_raw": [0] * n,
            "gy_raw": [0] * n,
            "gz_raw": [0] * n,
            "ax_g": np.sin(seconds).tolist(),
            "ay_g": np.cos(seconds).tolist(),
            "az_g": [1.0] * n,
            "gx_dps": np.sin(seconds / 2).tolist(),
            "gy_dps": np.cos(seconds / 2).tolist(),
            "gz_dps": [0.0] * n,
        }
    )


def _gps(spec: SessionSpec) -> pd.DataFrame:
    times = _times(spec.duration_ms, 1000)
    n = len(times)
    return pd.DataFrame(
        {
            "session_id": [spec.session_id] * n,
            "rx_utc": [_iso(t) for t in times],
            "rx_mono_ms": times,
            "seq": list(range(n)),
            "decode_error": [None] * n,
            "device_time_ms": times,
            "ground_speed": [500 + 10 * i for i in range(n)],
            "heading_of_motion": [9000 + 10 * i for i in range(n)],
            "speed_accuracy": [100] * n,
            "heading_accuracy": [200] * n,
            "fix_type": [3] * n,
            "num_sv": [12] * n,
            "flags": [1] * n,
            "ground_speed_mps": [5.0 + 0.1 * i for i in range(n)],
            "heading_of_motion_deg": [90.0 + 0.1 * i for i in range(n)],
            "speed_accuracy_mps": [1.0] * n,
            "heading_accuracy_deg": [2.0] * n,
            "fix_type_name": ["fix3d"] * n,
            "gnss_fix_ok": [True] * n,
            "parse_error": [False] * n,
            "uart_overflow": [False] * n,
        }
    )


def _signals(spec: SessionSpec) -> pd.DataFrame:
    times = _times(spec.duration_ms, 1000)
    seconds = np.asarray(times, dtype=float) / 1000.0
    n = len(times)
    return pd.DataFrame(
        {
            "session_id": [spec.session_id] * n,
            "rx_utc": [_iso(t) for t in times],
            "rx_mono_ms": times,
            "block_seq": list(range(n)),
            "device_time_ms": times,
            "total_records": [1] * n,
            "first_record": [0] * n,
            "did": [0] * n,
            "state": [1] * n,
            "state_name": ["valid"] * n,
            "age_ms": [50] * n,
            "raw": [0] * n,
            "sample_device_time_ms": [t - 50 for t in times],
            "signal": ["engine_speed_rpm"] * n,
            "unit": ["rpm"] * n,
            "value": (1000 + 500 * np.sin(seconds)).tolist(),
            "decode_error": [None] * n,
        }
    )


def _derived(spec: SessionSpec) -> pd.DataFrame:
    times = _times(spec.duration_ms, 500)
    seconds = np.asarray(times, dtype=float) / 1000.0
    rows: list[dict[str, object]] = []
    for t, s in zip(times, seconds, strict=True):
        rows.append(
            {
                "session_id": spec.session_id,
                "device_time_ms": t,
                "signal": "vehicle_speed_kmh",
                "value": 50 + 20 * np.sin(s / 2),
                "unit": "km/h",
                "valid": True,
                "flag": None,
            }
        )
    return pd.DataFrame(rows)


def _write_index(root: Path, specs: list[SessionSpec]) -> None:
    conn = sqlite3.connect(root / "index.sqlite")
    try:
        conn.execute(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                created_utc TEXT,
                imported_utc TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                app_version TEXT,
                ble_schema_version INTEGER,
                packet_count INTEGER,
                loss_percent REAL,
                status TEXT,
                has_imu INTEGER NOT NULL DEFAULT 0,
                defs_version TEXT,
                has_gps INTEGER NOT NULL DEFAULT 0,
                has_vehicle_signals INTEGER NOT NULL DEFAULT 0,
                has_derived INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        for spec in specs:
            conn.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    spec.session_id,
                    spec.created_utc,
                    spec.created_utc,
                    f"hash-{spec.session_id}",
                    "0.1.0",
                    4,
                    spec.packet_count,
                    spec.loss_percent,
                    spec.status,
                    int(spec.has_imu),
                    DEFS_VERSION,
                    int(spec.has_gps),
                    int(spec.has_vehicle_signals),
                    int(spec.has_derived),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def make_data_dir(root: Path, sessions: list[SessionSpec] | None = None) -> Path:
    """Create a vehicle-server-shaped data directory under ``root`` and return it."""
    specs = list(sessions) if sessions is not None else list(DEFAULT_SESSIONS)
    root.mkdir(parents=True, exist_ok=True)
    _write_index(root, specs)

    for spec in specs:
        parquet = root / "sessions" / spec.session_id / "parquet"
        parquet.mkdir(parents=True, exist_ok=True)
        _telemetry(spec).to_parquet(parquet / "telemetry.parquet", index=False)
        if spec.has_imu:
            _imu(spec).to_parquet(parquet / "imu.parquet", index=False)
        if spec.has_gps:
            _gps(spec).to_parquet(parquet / "gps.parquet", index=False)
        if spec.has_vehicle_signals:
            _signals(spec).to_parquet(parquet / "signals.parquet", index=False)
        if spec.has_derived:
            _derived(spec).to_parquet(parquet / "derived.parquet", index=False)
        if spec.write_report:
            (root / "sessions" / spec.session_id / "report.json").write_text(
                json.dumps({"defs_version": DEFS_VERSION}), encoding="utf-8"
            )
    return root
