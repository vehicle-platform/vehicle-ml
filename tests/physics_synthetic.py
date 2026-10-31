"""Synthetic physics sessions with known injected faults (ML-11).

The generated sessions use the same lumped equations as the stage-1 estimators:

- cooling: ``dT/dt = a * q(rpm) - h_eff * (T - T_amb)`` with ``q = rpm / 6000``;
- charging: a piecewise-linear voltage with a slope below the cut-in RPM and a plateau;
- gearing: one measured ratio per gear (the defs ``gear_ratio_<n>`` constants), with an
  optional clutch-slip multiplier.

The files mirror the vehicle-server storage layout used by ``tests/synthetic.py``.
Every value is synthetic; no real ride data or position is written. A fault changes
only its physical parameter after ``onset_ms``, so a healthy and a faulty session with
the same seed are identical before the onset.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from vehicle_ml.defs import vehicle_constants

BASE_UTC = datetime(2026, 10, 15, 9, 0, 0, tzinfo=UTC)
DEFS_VERSION = "0.20.0"
STEP_MS = 100

FAULT_KINDS = (
    "thermostat_stuck_open",
    "fan_failure",
    "plateau_sag",
    "clutch_slip",
)

# The gearbox ratios come from defs (invariant 2); only the speed bands are synthetic,
# one per gear so every gear appears at the generated speeds (at most ~70 km/h).
GEAR_RATIOS = tuple(
    vehicle_constants.CONSTANTS[f"gear_ratio_{number}"][0]
    for number in range(1, int(vehicle_constants.CONSTANTS["gear_count"][0]) + 1)
)
GEAR_THRESHOLDS_KMH = (15.0, 25.0, 35.0, 45.0, 55.0, 65.0)


@dataclass(frozen=True)
class PhysicsSessionSpec:
    session_id: str
    seed: int = 0
    duration_ms: int = 60_000
    ambient_c: float = 20.0
    load: float = 1.0
    fault: str | None = None
    onset_ms: int | None = None
    idle_only: bool = False

    def __post_init__(self) -> None:
        if self.fault is not None and self.fault not in FAULT_KINDS:
            raise ValueError(f"unknown fault kind: {self.fault!r}")
        if self.fault is not None and self.onset_ms is None:
            raise ValueError("a faulty session needs onset_ms")
        if self.onset_ms is not None and not 0 <= self.onset_ms <= self.duration_ms:
            raise ValueError("onset_ms must be inside the session")


@dataclass(frozen=True)
class FaultTruth:
    session_id: str
    kind: str | None
    onset_ms: int | None


@dataclass(frozen=True)
class PhysicsSession:
    spec: PhysicsSessionSpec
    telemetry: pd.DataFrame
    derived: pd.DataFrame
    truth: FaultTruth


@dataclass(frozen=True)
class PhysicsDataset:
    root: Path
    sessions: tuple[PhysicsSession, ...]

    def truths(self) -> dict[str, FaultTruth]:
        return {session.spec.session_id: session.truth for session in self.sessions}

    def write_faults(self, path: str | Path) -> Path:
        """Write the optional ground-truth file read by ``vehicle-ml evaluate``."""
        target = Path(path)
        payload = {
            "sessions": {
                truth.session_id: {"fault": truth.kind, "onset_ms": truth.onset_ms}
                for truth in self.truths().values()
                if truth.kind is not None
            }
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target


DEFAULT_PHYSICS_SESSIONS: tuple[PhysicsSessionSpec, ...] = (
    PhysicsSessionSpec("phys-00", seed=0, ambient_c=20.0, load=1.0),
    PhysicsSessionSpec("phys-01", seed=1, ambient_c=30.0, load=0.8),
    PhysicsSessionSpec("phys-02", seed=2, ambient_c=15.0, load=1.2),
    PhysicsSessionSpec(
        "phys-03", seed=3, ambient_c=21.0, load=1.0, fault="thermostat_stuck_open", onset_ms=30_000
    ),
    PhysicsSessionSpec(
        "phys-04", seed=4, ambient_c=24.0, load=0.9, fault="fan_failure", onset_ms=30_000
    ),
    PhysicsSessionSpec("phys-05", seed=5, ambient_c=25.0, load=1.1),
    PhysicsSessionSpec(
        "phys-06", seed=6, ambient_c=23.0, load=1.0, fault="plateau_sag", onset_ms=30_000
    ),
    PhysicsSessionSpec(
        "phys-07", seed=7, ambient_c=19.0, load=1.1, fault="clutch_slip", onset_ms=30_000
    ),
    PhysicsSessionSpec("phys-08", seed=8, ambient_c=18.0, load=0.9),
    PhysicsSessionSpec(
        "phys-09", seed=9, ambient_c=22.0, load=1.0, fault="plateau_sag", onset_ms=25_000
    ),
    PhysicsSessionSpec("phys-10", seed=10, ambient_c=28.0, load=1.3),
    PhysicsSessionSpec(
        "phys-11", seed=11, ambient_c=20.0, load=1.0, fault="clutch_slip", onset_ms=25_000
    ),
    PhysicsSessionSpec(
        "phys-12",
        seed=12,
        ambient_c=17.0,
        load=1.2,
        fault="thermostat_stuck_open",
        onset_ms=25_000,
    ),
    PhysicsSessionSpec("phys-13", seed=13, ambient_c=22.0, load=1.0),
    PhysicsSessionSpec(
        "phys-14", seed=14, ambient_c=26.0, load=0.8, fault="fan_failure", onset_ms=30_000
    ),
)


def _times(duration_ms: int) -> np.ndarray:
    return np.arange(0, duration_ms + 1, STEP_MS, dtype=np.int64)


def _gear_for_speed(speed: np.ndarray) -> np.ndarray:
    gear = np.zeros(speed.shape, dtype=np.int64)
    for number, threshold in enumerate(GEAR_THRESHOLDS_KMH, start=1):
        gear[speed >= threshold] = number
    return gear


def _operating_point(spec: PhysicsSessionSpec, times: np.ndarray) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(spec.seed)
    seconds = times.astype(float) / 1000.0
    phase = rng.uniform(0.0, 2.0 * np.pi)
    phase2 = rng.uniform(0.0, 2.0 * np.pi)

    if spec.idle_only:
        rpm = 1200.0 + 20.0 * np.sin(2.0 * np.pi * 0.05 * seconds + phase)
        speed = np.zeros_like(seconds)
    else:
        rpm = (
            1200.0
            + 2500.0 * spec.load
            + 400.0 * np.sin(2.0 * np.pi * 0.05 * seconds + phase)
            + 150.0 * np.sin(2.0 * np.pi * 0.23 * seconds + phase2)
        )
        speed = (
            12.0
            + 28.0 * spec.load
            + 18.0 * np.sin(2.0 * np.pi * 0.03 * seconds + phase2)
            + 3.0 * np.sin(2.0 * np.pi * 0.17 * seconds + phase)
        )
    rpm = np.clip(rpm, 800.0, 8000.0)
    speed = np.clip(speed, 0.0, 120.0)
    throttle = np.clip(5.0 + 35.0 * spec.load + 20.0 * np.sin(2.0 * np.pi * 0.07 * seconds), 0, 100)
    return rpm, speed, throttle


def _coolant(
    spec: PhysicsSessionSpec,
    times: np.ndarray,
    rpm: np.ndarray,
    speed: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    q = np.clip(rpm / 6000.0, 0.0, 1.0)
    heat_gain = 8.0
    base_h = 0.08
    fan_gain = np.where(speed < 25.0, 0.025, 0.0)
    onset = spec.onset_ms if spec.onset_ms is not None else spec.duration_ms + 1
    after = times >= onset

    h = np.full(times.shape, base_h, dtype=float) + fan_gain
    if spec.fault == "thermostat_stuck_open":
        h[after] += 0.08
    elif spec.fault == "fan_failure":
        h[after] -= fan_gain[after]

    temperature = np.empty(times.shape, dtype=float)
    temperature[0] = spec.ambient_c + 55.0
    for index in range(1, times.size):
        dt = (times[index] - times[index - 1]) / 1000.0
        derivative = heat_gain * q[index - 1] - h[index - 1] * (
            temperature[index - 1] - spec.ambient_c
        )
        temperature[index] = temperature[index - 1] + dt * derivative
    return temperature + rng.normal(0.0, 0.05, temperature.shape)


def _charging(
    spec: PhysicsSessionSpec, times: np.ndarray, rpm: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    base = 12.2
    slope = 0.0015
    cut_in = 1800.0
    plateau = base + slope * cut_in
    voltage = np.minimum(base + slope * rpm, plateau)
    onset = spec.onset_ms if spec.onset_ms is not None else spec.duration_ms + 1
    if spec.fault == "plateau_sag":
        voltage = np.where((times >= onset) & (rpm > cut_in), voltage - 0.7, voltage)
    return voltage + rng.normal(0.0, 0.02, voltage.shape)


def _gearing(
    spec: PhysicsSessionSpec,
    times: np.ndarray,
    speed: np.ndarray,
    throttle: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    gear = _gear_for_speed(speed)
    ratio = np.zeros(times.shape, dtype=float)
    for number, value in enumerate(GEAR_RATIOS, start=1):
        ratio[gear == number] = value
    onset = spec.onset_ms if spec.onset_ms is not None else spec.duration_ms + 1
    if spec.fault == "clutch_slip":
        loaded = throttle > 45.0
        ratio = np.where((times >= onset) & loaded & (gear > 0), ratio * 1.12, ratio)
    return gear, ratio + rng.normal(0.0, 0.005, ratio.shape)


def generate_session(spec: PhysicsSessionSpec) -> PhysicsSession:
    """Generate one session from its physical parameters and injected fault."""
    times = _times(spec.duration_ms)
    rng = np.random.default_rng(spec.seed)
    rpm, speed, throttle = _operating_point(spec, times)
    temperature = _coolant(spec, times, rpm, speed, rng)
    voltage = _charging(spec, times, rpm, rng)
    gear, ratio = _gearing(spec, times, speed, throttle, rng)

    telemetry = pd.DataFrame(
        {
            "session_id": [spec.session_id] * times.size,
            "device_time_ms": times,
            "engine_speed_rpm": rpm,
            "vehicle_speed_kmh": speed,
            "coolant_temp_degc": temperature,
            "battery_voltage_v": voltage,
            "throttle_pos_pct": throttle,
        }
    )
    rows: list[dict[str, object]] = []
    for t, gear_value, ratio_value in zip(times, gear, ratio, strict=True):
        rows.append(
            {
                "session_id": spec.session_id,
                "device_time_ms": int(t),
                "signal": "gear_estimate",
                "value": float(gear_value),
                "unit": "enum",
                "valid": True,
                "flag": None,
            }
        )
        rows.append(
            {
                "session_id": spec.session_id,
                "device_time_ms": int(t),
                "signal": "gearbox_ratio_measured",
                "value": float(ratio_value),
                "unit": "ratio",
                "valid": True,
                "flag": None,
            }
        )
    derived = pd.DataFrame(rows)
    truth = FaultTruth(spec.session_id, spec.fault, spec.onset_ms)
    return PhysicsSession(spec, telemetry, derived, truth)


def _iso(ms: int) -> str:
    return (BASE_UTC + timedelta(milliseconds=ms)).isoformat().replace("+00:00", "Z")


def _write_index(root: Path, specs: tuple[PhysicsSessionSpec, ...]) -> None:
    connection = sqlite3.connect(root / "index.sqlite")
    try:
        connection.execute(
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
            packet_count = len(_times(spec.duration_ms))
            connection.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    spec.session_id,
                    _iso(0),
                    _iso(0),
                    f"hash-{spec.session_id}",
                    "0.1.0",
                    4,
                    packet_count,
                    0.0,
                    "ok",
                    0,
                    DEFS_VERSION,
                    0,
                    1,
                    1,
                ),
            )
        connection.commit()
    finally:
        connection.close()


def make_physics_data_dir(
    root: Path, specs: tuple[PhysicsSessionSpec, ...] | None = None
) -> PhysicsDataset:
    """Create a vehicle-server-shaped data directory with physics sessions."""
    chosen = DEFAULT_PHYSICS_SESSIONS if specs is None else specs
    root.mkdir(parents=True, exist_ok=True)
    _write_index(root, chosen)

    sessions: list[PhysicsSession] = []
    for spec in chosen:
        session = generate_session(spec)
        session_dir = root / "sessions" / spec.session_id
        parquet = session_dir / "parquet"
        parquet.mkdir(parents=True, exist_ok=True)
        session.telemetry.to_parquet(parquet / "telemetry.parquet", index=False)
        session.derived.to_parquet(parquet / "derived.parquet", index=False)
        (session_dir / "report.json").write_text(
            json.dumps({"defs_version": DEFS_VERSION}), encoding="utf-8"
        )
        sessions.append(session)
    return PhysicsDataset(root, tuple(sessions))


def healthy_variant(spec: PhysicsSessionSpec) -> PhysicsSessionSpec:
    """The same session without its fault (used by the before-onset tests)."""
    return replace(spec, fault=None, onset_ms=None)
