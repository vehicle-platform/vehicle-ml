"""Synthetic physics sessions: determinism, onset isolation and no position data (ML-11)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from tests.physics_synthetic import (
    FAULT_KINDS,
    PhysicsSessionSpec,
    generate_session,
    healthy_variant,
    make_physics_data_dir,
)
from vehicle_ml.data.position import is_position_name
from vehicle_ml.data.store import SessionStore


def _before_onset(frame: pd.DataFrame, onset_ms: int) -> pd.DataFrame:
    return frame.loc[frame["device_time_ms"] < onset_ms].reset_index(drop=True)


def test_every_fault_differs_only_after_onset() -> None:
    for fault in FAULT_KINDS:
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


def test_generator_is_deterministic_per_seed() -> None:
    spec = PhysicsSessionSpec("phys-test", seed=7)
    first = generate_session(spec)
    second = generate_session(spec)
    assert_frame_equal(first.telemetry, second.telemetry)
    assert_frame_equal(first.derived, second.derived)
    assert first.truth == second.truth


def test_generated_frames_have_no_position_columns() -> None:
    session = generate_session(PhysicsSessionSpec("phys-test", seed=1))
    for frame in (session.telemetry, session.derived):
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
    assert dataset.truths()["phys-03"].kind == "thermostat_stuck_open"
    assert dataset.truths()["phys-00"].kind is None
