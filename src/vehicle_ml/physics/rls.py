"""Recursive least squares with a forgetting factor and a covariance guard.

The update is the standard exponentially weighted RLS:

``k = P x / (lambda + x' P x)``, ``theta += k (y - x'theta)``,
``P = (P - k x' P) / lambda``.

``lambda = 1`` gives the ordinary least-squares limit. A covariance trace below
``min_covariance`` or a non-finite update resets only ``P`` (the parameter estimate is
kept), so a long run cannot collapse to a singular covariance matrix.
"""

from __future__ import annotations

import numpy as np


class RLS:
    """Online linear least squares for a vector of parameters."""

    def __init__(
        self,
        n_params: int,
        *,
        forgetting: float = 1.0,
        initial_covariance: float = 1e6,
        min_covariance: float = 1e-12,
    ) -> None:
        if n_params <= 0:
            raise ValueError("n_params must be positive")
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting must be in (0, 1]")
        if initial_covariance <= 0.0:
            raise ValueError("initial_covariance must be positive")
        self.n_params = int(n_params)
        self.forgetting = float(forgetting)
        self.initial_covariance = float(initial_covariance)
        self.min_covariance = float(min_covariance)
        self.theta = np.zeros(self.n_params, dtype=float)
        self.covariance = np.eye(self.n_params, dtype=float) * self.initial_covariance

    def reset(self) -> None:
        """Reset both the parameter estimate and the covariance."""
        self.theta = np.zeros(self.n_params, dtype=float)
        self.covariance = np.eye(self.n_params, dtype=float) * self.initial_covariance

    def _reset_covariance(self) -> None:
        self.covariance = np.eye(self.n_params, dtype=float) * self.initial_covariance

    def predict(self, x: np.ndarray) -> float:
        row = np.asarray(x, dtype=float).reshape(-1)
        if row.size != self.n_params:
            raise ValueError(f"x must have {self.n_params} entries")
        return float(row @ self.theta)

    def update(self, x: np.ndarray, y: float) -> np.ndarray:
        """Process one sample and return the updated parameter vector."""
        row = np.asarray(x, dtype=float).reshape(-1)
        if row.size != self.n_params:
            raise ValueError(f"x must have {self.n_params} entries")
        target = float(y)
        if not np.all(np.isfinite(row)) or not np.isfinite(target):
            return self.theta

        covariance_x = self.covariance @ row
        denominator = self.forgetting + float(row @ covariance_x)
        if not np.isfinite(denominator) or denominator <= np.finfo(float).eps:
            self._reset_covariance()
            return self.theta

        gain = covariance_x / denominator
        error = target - float(row @ self.theta)
        self.theta = self.theta + gain * error
        self.covariance = (self.covariance - np.outer(gain, covariance_x)) / self.forgetting
        self.covariance = (self.covariance + self.covariance.T) / 2.0

        if (
            not np.all(np.isfinite(self.theta))
            or not np.all(np.isfinite(self.covariance))
            or float(np.trace(self.covariance)) < self.min_covariance
        ):
            self._reset_covariance()
        return self.theta


__all__ = ["RLS"]
