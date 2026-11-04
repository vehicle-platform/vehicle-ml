"""Hybrid score: fixed weight grid, robust normalisation and unscored windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vehicle_ml.hybrid import (
    HYBRID_WEIGHTS,
    HybridScorer,
    normalise_score,
    validate_weight,
)


def test_weight_grid_never_contains_zero() -> None:
    assert HYBRID_WEIGHTS == (0.25, 0.5, 0.75, 1.0)
    assert 0.0 not in HYBRID_WEIGHTS
    for weight in HYBRID_WEIGHTS:
        assert validate_weight(weight) == weight


@pytest.mark.parametrize("weight", [0.0, -0.1, 1.1, np.nan, np.inf])
def test_zero_and_invalid_weights_are_refused(weight: float) -> None:
    with pytest.raises(ValueError, match="weight"):
        validate_weight(weight)
    with pytest.raises(ValueError, match="weight"):
        HybridScorer(weight)


def test_normalise_uses_the_reference_median_and_mad() -> None:
    reference = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    values = pd.Series([3.0, 4.0, np.nan])
    result = normalise_score(values, reference)
    assert result.iloc[0] == pytest.approx(0.0)
    assert result.iloc[1] == pytest.approx(1.0 / 1.4826)
    assert np.isnan(result.iloc[2])


def test_normalise_with_a_zero_mad_uses_unit_scale() -> None:
    result = normalise_score(pd.Series([2.0]), pd.Series([2.0, 2.0, 2.0]))
    assert result.iloc[0] == 0.0


def test_hybrid_is_the_weighted_sum_of_the_normalised_scores() -> None:
    index = pd.RangeIndex(3)
    physics = pd.Series([1.0, 2.0, 3.0], index=index)
    baseline = pd.Series([3.0, 2.0, 1.0], index=index)
    reference = pd.Series([True, True, True], index=index)
    scored = pd.Series([True, True, True], index=index)
    result = HybridScorer(0.75).score(
        physics,
        baseline,
        reference=reference,
        physics_scored=scored,
        baseline_scored=scored,
    )

    expected_physics = normalise_score(physics, physics)
    expected_baseline = normalise_score(baseline, baseline)
    expected = 0.75 * expected_physics + 0.25 * expected_baseline
    assert result["hybrid"].to_numpy() == pytest.approx(expected.to_numpy())
    assert result["hybrid_physics_contribution"].to_numpy() == pytest.approx(
        (0.75 * expected_physics).to_numpy()
    )
    assert result["hybrid_baseline_contribution"].to_numpy() == pytest.approx(
        (0.25 * expected_baseline).to_numpy()
    )
    assert set(result["hybrid_dominant"]) == {"physics", "baseline"}


def test_reference_mask_does_not_read_validation_or_test_scores() -> None:
    index = pd.RangeIndex(4)
    physics = pd.Series([1.0, 2.0, 3.0, 100.0], index=index)
    baseline = pd.Series([1.0, 2.0, 3.0, 100.0], index=index)
    reference = pd.Series([True, True, True, False], index=index)
    scored = pd.Series([True, True, True, True], index=index)
    first = HybridScorer(0.5).score(
        physics,
        baseline,
        reference=reference,
        physics_scored=scored,
        baseline_scored=scored,
    )
    poisoned = physics.copy()
    poisoned.iloc[3] = 1e9
    second = HybridScorer(0.5).score(
        poisoned,
        baseline,
        reference=reference,
        physics_scored=scored,
        baseline_scored=scored,
    )
    assert first["hybrid"].iloc[:3].tolist() == pytest.approx(second["hybrid"].iloc[:3].tolist())
    assert first["hybrid"].iloc[3] != second["hybrid"].iloc[3]


def test_a_window_physics_could_not_score_is_unscored_not_baseline_only() -> None:
    index = pd.RangeIndex(3)
    physics = pd.Series([1.0, 2.0, 3.0], index=index)
    baseline = pd.Series([100.0, 100.0, 100.0], index=index)
    reference = pd.Series([True, True, True], index=index)
    result = HybridScorer(0.5).score(
        physics,
        baseline,
        reference=reference,
        physics_scored=pd.Series([True, False, True], index=index),
        baseline_scored=pd.Series([True, True, True], index=index),
    )
    assert np.isnan(result["hybrid"].iloc[1])
    assert np.isnan(result["hybrid_physics_contribution"].iloc[1])
    assert result["hybrid_dominant"].iloc[1] == ""
