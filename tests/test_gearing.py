"""Gearing estimator: table recovery, clutch-slip residual and gear 0 (ML-14)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.physics_synthetic import PhysicsSessionSpec, generate_session, healthy_variant
from vehicle_ml.physics.gearing import GearTable


def test_recovers_the_per_gear_median() -> None:
    rng = np.random.default_rng(0)
    gears = np.repeat(np.arange(1, 7), 40)
    ratios = np.repeat(np.array([3.0, 2.1, 1.6, 1.3, 1.1, 0.95]), 40)
    measured = ratios + rng.normal(0.0, 0.01, ratios.shape)
    table = GearTable.fit(measured, gears)
    assert table.ratios.keys() == set(range(1, 7))
    for number, expected in enumerate([3.0, 2.1, 1.6, 1.3, 1.1, 0.95], start=1):
        assert table.ratios[number] == pytest.approx(expected, abs=0.01)


def test_clutch_slip_is_positive_under_load_only() -> None:
    spec = PhysicsSessionSpec("gear-test", seed=11, fault="clutch_slip", onset_ms=0)
    healthy = generate_session(healthy_variant(spec))
    faulty = generate_session(spec)
    table = GearTable.fit(
        healthy.derived.loc[
            healthy.derived["signal"] == "gearbox_ratio_measured", "value"
        ].to_numpy(),
        healthy.derived.loc[healthy.derived["signal"] == "gear_estimate", "value"].to_numpy(),
    )
    ratio = faulty.derived.loc[
        faulty.derived["signal"] == "gearbox_ratio_measured", "value"
    ].to_numpy()
    gear = faulty.derived.loc[faulty.derived["signal"] == "gear_estimate", "value"].to_numpy()
    residual = table.residual(ratio, gear)
    throttle = faulty.telemetry["throttle_pos_pct"].to_numpy()
    loaded = (throttle > 45.0) & (gear > 0) & np.isfinite(residual)
    unloaded = (throttle <= 45.0) & (gear > 0) & np.isfinite(residual)
    assert float(np.mean(residual[loaded])) > 0.05
    assert abs(float(np.mean(residual[unloaded]))) < 0.01


def test_gear_zero_is_ignored() -> None:
    gears = np.array([0, 0, 1, 1, 1, 1, 1])
    ratios = np.array([99.0, 98.0, 3.0, 3.1, 2.9, 3.05, 3.0])
    table = GearTable.fit(ratios, gears, min_samples=3)
    assert 0 not in table.ratios
    residual = table.residual(ratios, gears)
    assert np.isnan(residual[0])
    assert np.isnan(residual[1])
    assert np.isfinite(residual[2])
