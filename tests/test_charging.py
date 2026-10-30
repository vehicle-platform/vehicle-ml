"""Charging estimator: parameter recovery, plateau sag and refusal (ML-13)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.physics_synthetic import PhysicsSessionSpec, generate_session, healthy_variant
from vehicle_ml.physics.charging import NotIdentifiableError, fit_charging


def _clean(
    *, base: float = 12.2, slope: float = 0.0015, cut_in: float = 1800.0
) -> tuple[np.ndarray, np.ndarray]:
    rpm = np.linspace(1000.0, 5000.0, 601)
    voltage = np.minimum(base + slope * rpm, base + slope * cut_in)
    return rpm, voltage


def test_recovers_plateau_cut_in_and_slope() -> None:
    rpm, voltage = _clean()
    fit = fit_charging(rpm, voltage)
    assert fit.plateau_v == pytest.approx(14.9, abs=0.02)
    assert fit.cut_in_rpm == pytest.approx(1800.0, abs=20.0)
    assert fit.slope_v_per_rpm == pytest.approx(0.0015, rel=0.03)
    assert fit.quality > 0.999
    assert fit.residual.size == rpm.size


def test_plateau_sag_is_visible_in_the_parameters() -> None:
    spec = PhysicsSessionSpec("charge-test", seed=6, fault="plateau_sag", onset_ms=0)
    healthy = generate_session(healthy_variant(spec))
    faulty = generate_session(spec)
    healthy_fit = fit_charging(
        healthy.telemetry["engine_speed_rpm"].to_numpy(),
        healthy.telemetry["battery_voltage_v"].to_numpy(),
    )
    faulty_fit = fit_charging(
        faulty.telemetry["engine_speed_rpm"].to_numpy(),
        faulty.telemetry["battery_voltage_v"].to_numpy(),
    )
    assert healthy_fit.plateau_v - faulty_fit.plateau_v == pytest.approx(0.7, abs=0.15)


def test_idle_only_data_is_not_identifiable() -> None:
    session = generate_session(PhysicsSessionSpec("idle", seed=1, idle_only=True))
    with pytest.raises(NotIdentifiableError, match="not identifiable"):
        fit_charging(
            session.telemetry["engine_speed_rpm"].to_numpy(),
            session.telemetry["battery_voltage_v"].to_numpy(),
        )
