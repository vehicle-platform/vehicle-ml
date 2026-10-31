"""Synthetic physics sessions with known injected faults (ML-11).

The generated sessions use the same lumped equations as the stage-1 estimators:

- cooling: ``dT/dt = a * q(rpm) - h_eff * (T - T_amb)`` with ``q = rpm / 6000``;
- charging: a piecewise-linear voltage with a slope below the cut-in RPM and a plateau;
- gearing: one measured ratio per gear (the defs ``gear_ratio_<n>`` constants), with an
  optional clutch-slip multiplier;
- IMU: broadband road noise on all six axes, plus (``wheel_vibration`` only) a
  narrow-band line in ``imu_accel_z`` locked to the wheel rotation frequency.

The files mirror the vehicle-server storage layout used by ``tests/synthetic.py``.
Every value is synthetic; no real ride data or position is written. A fault changes
only its physical parameter after ``onset_ms``, so a healthy and a faulty session with
the same seed are identical before the onset. ``wheel_vibration`` is the rule-3
weak-CAN case: it changes only the IMU, so every CAN and derived column equals the
healthy twin with the same seed (the IMU noise has its own random stream).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from vehicle_ml.data.sources import imu_columns, sensor_column
from vehicle_ml.defs import ble, vehicle_constants, vehicle_derived

BASE_UTC = datetime(2026, 10, 15, 9, 0, 0, tzinfo=UTC)
DEFS_VERSION = "0.20.0"
STEP_MS = 100

FAULT_KINDS = (
    "thermostat_stuck_open",
    "fan_failure",
    "plateau_sag",
    "clutch_slip",
    "wheel_vibration",
)

# What each fault kind is expected to show up on (rule 3); written to faults.json.
EXPECTED_MODALITY = {
    "thermostat_stuck_open": "can",
    "fan_failure": "can",
    "plateau_sag": "can",
    "clutch_slip": "can",
    "wheel_vibration": "imu",
}

# IMU layout and rate come from defs (invariant 2): sensor names from
# ``vehicle_derived.SENSORS``, the 10 ms period from their ``sample_period_ms``.
IMU_ACCEL_Z = "imu_accel_z"
IMU_PERIOD_MS = int(vehicle_derived.SENSORS[IMU_ACCEL_Z]["sample_period_ms"])
IMU_NYQUIST_HZ = 500.0 / IMU_PERIOD_MS
# The wheel_vibration line sits at this multiple of the rear-wheel rotation frequency
# (4th order, e.g. a four-lobe out-of-round wheel or tyre). At the generated speeds
# (at most ~70 km/h, wheel rotation <= ~10 Hz) the line stays at ~8-40 Hz, below the
# 50 Hz Nyquist limit of the 10 ms IMU period; a larger multiple would alias.
WHEEL_ORDER = 4
VIBRATION_AMPLITUDE_G = 0.15
ROAD_NOISE_ACCEL_G = 0.05
ROAD_NOISE_GYRO_DPS = 1.0
IMU_STREAM = 1  # SeedSequence entropy word: the IMU noise never shares the CAN stream

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
    has_imu: bool = True

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
    imu: pd.DataFrame
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
                truth.session_id: {
                    "fault": truth.kind,
                    "onset_ms": truth.onset_ms,
                    "expected_modality": EXPECTED_MODALITY[truth.kind],
                }
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
    # Sessions 15-27 add the weak-CAN fault; the ids put one wheel_vibration session in the
    # test split for split seeds 0, 1 and 2.
    PhysicsSessionSpec("phys-15", seed=15, ambient_c=23.0, load=1.1),
    PhysicsSessionSpec(
        "phys-16", seed=16, ambient_c=16.0, load=0.8, fault="wheel_vibration", onset_ms=30000
    ),
    PhysicsSessionSpec("phys-17", seed=17, ambient_c=23.0, load=1.1),
    PhysicsSessionSpec("phys-18", seed=18, ambient_c=16.0, load=0.8),
    PhysicsSessionSpec(
        "phys-19", seed=19, ambient_c=23.0, load=1.1, fault="wheel_vibration", onset_ms=25000
    ),
    PhysicsSessionSpec("phys-20", seed=20, ambient_c=16.0, load=0.8),
    PhysicsSessionSpec("phys-21", seed=21, ambient_c=23.0, load=1.1),
    PhysicsSessionSpec(
        "phys-22", seed=22, ambient_c=16.0, load=0.8, fault="wheel_vibration", onset_ms=30000
    ),
    PhysicsSessionSpec("phys-23", seed=23, ambient_c=23.0, load=1.1),
    PhysicsSessionSpec("phys-24", seed=24, ambient_c=16.0, load=0.8),
    PhysicsSessionSpec("phys-25", seed=25, ambient_c=23.0, load=1.1),
    PhysicsSessionSpec(
        "phys-26", seed=26, ambient_c=16.0, load=0.8, fault="wheel_vibration", onset_ms=25000
    ),
    PhysicsSessionSpec(
        "phys-27", seed=27, ambient_c=23.0, load=1.1, fault="wheel_vibration", onset_ms=30000
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


def wheel_rotation_hz(speed_kmh: np.ndarray) -> np.ndarray:
    """Rear-wheel rotation frequency ``speed / (2 * pi * r)`` from the defs radius."""
    radius_m = vehicle_constants.CONSTANTS["rear_wheel_radius_dynamic_m"][0]
    return (speed_kmh / 3.6) / (2.0 * np.pi * radius_m)


def _imu(spec: PhysicsSessionSpec, times: np.ndarray, speed: np.ndarray) -> pd.DataFrame:
    """IMU frame in the server layout of ``tests/synthetic.py``."""
    imu_times = np.arange(0, spec.duration_ms + 1, IMU_PERIOD_MS, dtype=np.int64)
    rng = np.random.default_rng([spec.seed, IMU_STREAM])
    columns = imu_columns()
    values: dict[str, np.ndarray] = {}
    for column in columns:
        sigma = ROAD_NOISE_ACCEL_G if column.endswith("_g") else ROAD_NOISE_GYRO_DPS
        values[column] = rng.normal(0.0, sigma, imu_times.shape)
    values[sensor_column(IMU_ACCEL_Z)] += 1.0  # gravity on the vertical axis

    if spec.fault == "wheel_vibration":
        assert spec.onset_ms is not None
        line_hz = WHEEL_ORDER * wheel_rotation_hz(np.interp(imu_times, times, speed))
        if float(np.max(line_hz)) >= IMU_NYQUIST_HZ:
            raise ValueError("wheel_vibration line would alias above the IMU Nyquist limit")
        phase = 2.0 * np.pi * np.cumsum(line_hz) * (IMU_PERIOD_MS / 1000.0)
        line = VIBRATION_AMPLITUDE_G * np.sin(phase)
        values[sensor_column(IMU_ACCEL_Z)] += np.where(imu_times >= spec.onset_ms, line, 0.0)

    scale = ble.SCHEMA["imuBlock"]["scale"]
    count = imu_times.size
    data: dict[str, object] = {
        "session_id": [spec.session_id] * count,
        "rx_utc": [_iso(int(t)) for t in imu_times],
        "rx_mono_ms": imu_times,
        "block_seq": imu_times // 100,
        "block_flags": np.zeros(count, dtype=np.int64),
        "block_missing_samples": np.zeros(count, dtype=np.int64),
        "sample_index": imu_times // IMU_PERIOD_MS,
        "device_time_ms": imu_times,
    }
    for column in columns:
        field = column.split("_")[0]
        kind = "accel" if field.startswith("a") else "gyro"
        data[f"{field}_raw"] = np.round(values[column] * scale[kind]["lsbPerUnit"]).astype(np.int64)
    for column in columns:
        data[column] = values[column]
    return pd.DataFrame(data)


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
    imu = (
        _imu(spec, times, speed)
        if spec.has_imu
        else pd.DataFrame(columns=["session_id", "device_time_ms", *imu_columns()])
    )
    truth = FaultTruth(spec.session_id, spec.fault, spec.onset_ms)
    return PhysicsSession(spec, telemetry, derived, imu, truth)


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
                    int(spec.has_imu),
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
        if spec.has_imu:
            session.imu.to_parquet(parquet / "imu.parquet", index=False)
        (session_dir / "report.json").write_text(
            json.dumps({"defs_version": DEFS_VERSION}), encoding="utf-8"
        )
        sessions.append(session)
    return PhysicsDataset(root, tuple(sessions))


def healthy_variant(spec: PhysicsSessionSpec) -> PhysicsSessionSpec:
    """The same session without its fault (used by the before-onset tests)."""
    return replace(spec, fault=None, onset_ms=None)
