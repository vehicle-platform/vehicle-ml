"""Cooling estimator: parameter recovery, fault direction, gaps and refusal (ML-12)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.physics_synthetic import PhysicsSessionSpec, generate_session, healthy_variant
from vehicle_ml.physics.cooling import NotIdentifiableError, fit_cooling


def _simulate(
    *,
    a: float = 8.0,
    h: float = 0.08,
    ambient: float = 20.0,
    duration_ms: int = 60_000,
    step_ms: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.arange(0, duration_ms + 1, step_ms, dtype=float)
    seconds = times / 1000.0
    rpm = 2500.0 + 1500.0 * np.sin(2.0 * np.pi * 0.04 * seconds)
    q = np.clip(rpm / 6000.0, 0.0, 1.0)
    temperature = np.empty(times.size, dtype=float)
    temperature[0] = ambient + 55.0
    for index in range(1, times.size):
        dt = (times[index] - times[index - 1]) / 1000.0
        q_mid = (q[index - 1] + q[index]) / 2.0
        temperature_mid = temperature[index - 1] + 0.5 * dt * (
            a * q[index - 1] - h * (temperature[index - 1] - ambient)
        )
        derivative = a * q_mid - h * (temperature_mid - ambient)
        temperature[index] = temperature[index - 1] + dt * derivative
    return times, temperature, rpm


def test_recovers_known_parameters() -> None:
    times, temperature, rpm = _simulate()
    fit = fit_cooling(times, temperature, rpm)
    assert fit.a == pytest.approx(8.0, rel=0.02)
    assert fit.h == pytest.approx(0.08, rel=0.02)
    assert fit.ambient_c == pytest.approx(20.0, abs=0.2)
    assert fit.time_constant_s == pytest.approx(12.5, rel=0.02)
    assert fit.quality > 0.999
    assert fit.stretches == 1


def test_stuck_thermostat_raises_the_heat_transfer_gain() -> None:
    spec = PhysicsSessionSpec("cool-test", seed=3, fault="thermostat_stuck_open", onset_ms=0)
    healthy = generate_session(healthy_variant(spec))
    faulty = generate_session(spec)
    healthy_fit = fit_cooling(
        healthy.telemetry["device_time_ms"].to_numpy(),
        healthy.telemetry["coolant_temp_degc"].to_numpy(),
        healthy.telemetry["engine_speed_rpm"].to_numpy(),
    )
    faulty_fit = fit_cooling(
        faulty.telemetry["device_time_ms"].to_numpy(),
        faulty.telemetry["coolant_temp_degc"].to_numpy(),
        faulty.telemetry["engine_speed_rpm"].to_numpy(),
    )
    assert faulty_fit.h > healthy_fit.h * 1.5


def test_gap_is_never_bridged() -> None:
    times, temperature, rpm = _simulate()
    keep = (times < 10_000) | (times > 20_000)
    times = times[keep]
    temperature = temperature[keep]
    rpm = rpm[keep]
    fit = fit_cooling(times, temperature, rpm, max_gap_ms=500.0)
    assert fit.stretches == 2
    first = np.count_nonzero(times < 10_000)
    second = np.count_nonzero(times > 20_000)
    assert fit.samples == (first - 1) + (second - 1)


def test_too_short_data_is_not_identifiable() -> None:
    times, temperature, rpm = _simulate(duration_ms=1_000)
    with pytest.raises(NotIdentifiableError, match="not identifiable"):
        fit_cooling(times, temperature, rpm)


def test_idle_only_data_is_not_identifiable() -> None:
    times = np.arange(0, 60_000, 100, dtype=float)
    rpm = np.full(times.size, 1200.0)
    temperature = 70.0 + 0.002 * times
    with pytest.raises(NotIdentifiableError, match="not identifiable"):
        fit_cooling(times, temperature, rpm)
