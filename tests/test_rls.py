"""Recursive least squares: convergence, forgetting and the batch limit (ML-15)."""

from __future__ import annotations

import numpy as np
import pytest

from vehicle_ml.physics.rls import RLS


def _system(seed: int, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    truth = np.array([1.5, -2.0, 0.5])
    return x, x @ truth


def test_converges_to_a_known_linear_system() -> None:
    x, y = _system(0)
    estimator = RLS(3)
    for row, target in zip(x, y, strict=True):
        estimator.update(row, target)
    assert estimator.theta == pytest.approx([1.5, -2.0, 0.5], abs=1e-4)


def test_forgetting_tracks_a_step_change() -> None:
    rng = np.random.default_rng(1)
    first = np.array([1.0, -1.0, 0.5])
    second = np.array([-2.0, 1.5, 1.0])
    estimator = RLS(3, forgetting=0.98)
    for index in range(800):
        row = rng.normal(size=3)
        truth = first if index < 400 else second
        estimator.update(row, float(row @ truth))
    assert np.linalg.norm(estimator.theta - second) < np.linalg.norm(estimator.theta - first)


def test_forgetting_one_matches_batch_least_squares() -> None:
    x, y = _system(2)
    estimator = RLS(3, forgetting=1.0)
    for row, target in zip(x, y, strict=True):
        estimator.update(row, target)
    batch, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    assert estimator.theta == pytest.approx(batch, abs=1e-5)


def test_covariance_guard_restores_the_initial_covariance() -> None:
    estimator = RLS(1, initial_covariance=1.0, min_covariance=10.0)
    estimator.update(np.array([1.0]), 1.0)
    assert float(np.trace(estimator.covariance)) == pytest.approx(1.0)


def test_invalid_forgetting_is_rejected() -> None:
    with pytest.raises(ValueError, match="forgetting"):
        RLS(2, forgetting=0.0)
    with pytest.raises(ValueError, match="forgetting"):
        RLS(2, forgetting=1.5)
