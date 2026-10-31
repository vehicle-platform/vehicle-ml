"""Synthetic physics sessions: determinism, onset isolation and no position data (ML-11)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tests.physics_synthetic import (
    FAULT_KINDS,
    IMU_NYQUIST_HZ,
    IMU_PERIOD_MS,
    WHEEL_ORDER,
    PhysicsSessionSpec,
    generate_session,
    healthy_variant,
    make_physics_data_dir,
    wheel_rotation_hz,
)
from vehicle_ml.data.position import is_position_name
from vehicle_ml.data.sources import imu_columns, sensor_column
from vehicle_ml.data.store import SessionStore

CAN_FAULT_KINDS = tuple(kind for kind in FAULT_KINDS if kind != "wheel_vibration")


def _before_onset(frame: pd.DataFrame, onset_ms: int) -> pd.DataFrame:
    return frame.loc[frame["device_time_ms"] < onset_ms].reset_index(drop=True)


def test_every_can_fault_differs_only_after_onset() -> None:
    for fault in CAN_FAULT_KINDS:
        faulty_spec = PhysicsSessionSpec("phys-test", seed=42, fault=fault, onset_ms=30_000)
        healthy = generate_session(healthy_variant(faulty_spec))
        faulty = generate_session(faulty_spec)

        assert_frame_equal(
            _before_onset(healthy.telemetry, 30_000),
            _before_onset(faulty.telemetry, 30_000),
        )
        assert_frame_equal(
            _before_onset(healthy.derived, 30_000),
            _before_onset(faulty.derived, 30_000),
        )
        assert not (
            faulty.telemetry.equals(healthy.telemetry) and faulty.derived.equals(healthy.derived)
        )
        assert_frame_equal(healthy.imu, faulty.imu)


def test_generator_is_deterministic_per_seed() -> None:
    spec = PhysicsSessionSpec("phys-test", seed=7)
    first = generate_session(spec)
    second = generate_session(spec)
    assert_frame_equal(first.telemetry, second.telemetry)
    assert_frame_equal(first.derived, second.derived)
    assert_frame_equal(first.imu, second.imu)
    assert first.truth == second.truth
    vibrating = PhysicsSessionSpec("phys-test", seed=7, fault="wheel_vibration", onset_ms=30_000)
    assert_frame_equal(generate_session(vibrating).imu, generate_session(vibrating).imu)


def test_generated_frames_have_no_position_columns() -> None:
    session = generate_session(PhysicsSessionSpec("phys-test", seed=1))
    for frame in (session.telemetry, session.derived, session.imu):
        assert not any(is_position_name(str(column)) for column in frame.columns)


def test_data_dir_has_the_server_layout_and_truth(tmp_path: Path) -> None:
    dataset = make_physics_data_dir(tmp_path / "data")
    store = SessionStore(dataset.root)
    sessions = store.sessions()
    assert set(sessions["session_id"]) == {session.spec.session_id for session in dataset.sessions}
    assert sessions["has_derived"].all()

    telemetry = store.load("phys-03", "telemetry")
    derived = store.load("phys-03", "derived")
    assert telemetry is not None and derived is not None
    assert {"engine_speed_rpm", "coolant_temp_degc", "battery_voltage_v"} <= set(telemetry.columns)
    assert set(derived["signal"]) == {"gear_estimate", "gearbox_ratio_measured"}
    assert sessions["has_imu"].all()
    imu = store.load("phys-03", "imu")
    assert imu is not None
    assert set(imu_columns()) <= set(imu.columns)
    assert dataset.truths()["phys-03"].kind == "thermostat_stuck_open"
    assert dataset.truths()["phys-00"].kind is None


def test_session_without_imu_writes_no_imu_file(tmp_path: Path) -> None:
    spec = PhysicsSessionSpec("phys-no-imu", seed=1, has_imu=False)
    dataset = make_physics_data_dir(tmp_path / "data", (spec,))
    store = SessionStore(dataset.root)

    assert dataset.sessions[0].imu.empty
    assert not store.sessions()["has_imu"].iloc[0]
    assert store.load("phys-no-imu", "imu") is None


def test_wheel_vibration_leaves_can_and_derived_identical_to_the_healthy_twin() -> None:
    spec = PhysicsSessionSpec("phys-test", seed=5, fault="wheel_vibration", onset_ms=30_000)
    faulty = generate_session(spec)
    healthy = generate_session(healthy_variant(spec))
    assert_frame_equal(faulty.telemetry, healthy.telemetry)
    assert_frame_equal(faulty.derived, healthy.derived)


def test_wheel_vibration_changes_only_accel_z_after_onset() -> None:
    spec = PhysicsSessionSpec("phys-test", seed=5, fault="wheel_vibration", onset_ms=30_000)
    faulty = generate_session(spec).imu
    healthy = generate_session(healthy_variant(spec)).imu
    z = sensor_column("imu_accel_z")
    other = [column for column in imu_columns() if column != z]
    assert_frame_equal(faulty[other], healthy[other])
    before = faulty["device_time_ms"] < 30_000
    assert_frame_equal(faulty.loc[before], healthy.loc[before])
    assert not faulty.loc[~before, z].equals(healthy.loc[~before, z])


def test_healthy_sessions_have_broadband_noise_on_all_six_axes() -> None:
    imu = generate_session(PhysicsSessionSpec("phys-test", seed=3)).imu
    assert len(imu_columns()) == 6
    assert imu["device_time_ms"].diff().dropna().eq(IMU_PERIOD_MS).all()
    for column in imu_columns():
        assert imu[column].std() > 0.0
        spectrum = np.abs(np.fft.rfft(imu[column] - imu[column].mean())) ** 2
        # No line: the strongest bin holds only a small share of the total power.
        assert spectrum.max() < 0.05 * spectrum.sum()


@pytest.mark.parametrize("seed", [0, 5, 9])
def test_wheel_vibration_peaks_at_the_wheel_order_frequency(seed: int) -> None:
    spec = PhysicsSessionSpec("phys-test", seed=seed, fault="wheel_vibration", onset_ms=30_000)
    session = generate_session(spec)
    imu = session.imu
    z = sensor_column("imu_accel_z")
    # 1 s window after onset (1 Hz bins). The line follows the wheel speed, so it sweeps a
    # few Hz at most: the peak must lie in the swept range, within one bin, and the swept
    # range must hold most of the window's power.
    window = imu.loc[(imu["device_time_ms"] >= 40_000) & (imu["device_time_ms"] < 41_000)]
    samples = window[z].to_numpy() - window[z].mean()
    spectrum = np.abs(np.fft.rfft(samples)) ** 2
    frequencies = np.fft.rfftfreq(samples.size, d=IMU_PERIOD_MS / 1000.0)
    speed = session.telemetry["vehicle_speed_kmh"].to_numpy()
    times = session.telemetry["device_time_ms"].to_numpy()
    in_window = (times >= 40_000) & (times <= 41_000)
    line_hz = WHEEL_ORDER * wheel_rotation_hz(speed[in_window])
    low, high = line_hz.min() - 1.0, line_hz.max() + 1.0

    assert line_hz.max() < IMU_NYQUIST_HZ
    assert low <= frequencies[int(np.argmax(spectrum))] <= high
    assert spectrum[(frequencies >= low) & (frequencies <= high)].sum() > 0.5 * spectrum.sum()
