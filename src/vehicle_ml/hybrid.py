"""Physics-weighted hybrid anomaly score (D-083 item 2).

The stage-1 physics score (robust-z over cooling/charging/gearing residuals) and the
PCA reconstruction error are not on one scale. Each is first normalised by the median
and MAD of the **train healthy** windows, then combined per window as

``hybrid = w * physics_norm + (1 - w) * baseline_norm``

with ``w`` in a fixed grid and never zero: the physics residual must always be part of
the score. A window that either model could not score is unscored for the hybrid too;
the code never falls back to the baseline alone.

The score and its threshold are offline evidence only. Nothing here is a safety value
(invariant 3).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

HYBRID_WEIGHTS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
DEFAULT_HYBRID_WEIGHT = 0.5
HYBRID_MODEL = "hybrid"


def validate_weight(weight: float) -> float:
    """Return ``weight`` if it is a finite value in ``(0, 1]``; zero is never allowed."""
    value = float(weight)
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError("hybrid weight must be finite and in (0, 1]; w=0 is not allowed")
    return value


def normalise_score(values: pd.Series, reference: pd.Series) -> pd.Series:
    """Median/MAD-normalise ``values`` against the finite reference scores."""
    finite = reference.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return pd.Series(np.nan, index=values.index)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad
    if scale == 0.0:
        scale = 1.0
    return (values.astype(float) - median) / scale


@dataclass(frozen=True)
class HybridScorer:
    """A validated hybrid weight and its per-window scoring rule."""

    weight: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight", validate_weight(self.weight))

    def score(
        self,
        physics: pd.Series,
        baseline: pd.Series,
        *,
        reference: Sequence[bool] | pd.Series,
        physics_scored: Sequence[bool] | pd.Series,
        baseline_scored: Sequence[bool] | pd.Series,
    ) -> pd.DataFrame:
        """Return hybrid scores, weighted contributions and the dominant term.

        ``reference`` marks the train-healthy windows used for the median/MAD scale.
        ``physics_scored`` and ``baseline_scored`` mark windows each model scored; the
        hybrid is NaN wherever either is false.
        """
        if not physics.index.equals(baseline.index):
            raise ValueError("physics and baseline scores must share one index")
        index = physics.index
        ref = pd.Series(reference, index=index, dtype=bool)
        physics_ok = pd.Series(physics_scored, index=index, dtype=bool)
        baseline_ok = pd.Series(baseline_scored, index=index, dtype=bool)
        scored = physics_ok & baseline_ok

        physics_norm = normalise_score(physics, physics.loc[ref & physics_ok])
        baseline_norm = normalise_score(baseline, baseline.loc[ref & baseline_ok])
        physics_contribution = self.weight * physics_norm
        baseline_contribution = (1.0 - self.weight) * baseline_norm
        hybrid = (physics_contribution + baseline_contribution).where(scored)

        dominant = pd.Series("", index=index, dtype=object)
        dominant.loc[scored & (physics_contribution >= baseline_contribution)] = "physics"
        dominant.loc[scored & (physics_contribution < baseline_contribution)] = "baseline"
        return pd.DataFrame(
            {
                "hybrid": hybrid,
                "hybrid_physics_contribution": physics_contribution.where(scored),
                "hybrid_baseline_contribution": baseline_contribution.where(scored),
                "hybrid_dominant": dominant,
            },
            index=index,
        )


__all__ = [
    "DEFAULT_HYBRID_WEIGHT",
    "HYBRID_MODEL",
    "HYBRID_WEIGHTS",
    "HybridScorer",
    "normalise_score",
    "validate_weight",
]
