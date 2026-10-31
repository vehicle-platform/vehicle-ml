"""Baseline without physics: PCA reconstruction error and robust z-scores.

D-083 item 3 asks for a baseline **without physics** on the same window features.
This module provides two distribution-only detectors:

- PCA reconstruction error: a linear autoencoder fitted with numpy SVD; a window that
  the train subspace cannot reconstruct scores high.
- Robust per-feature z-score: median/MAD distance from the train distribution, which
  does not assume Gaussian tails.

Neither detector contains a physical equation, so together they are the "without
physics" comparison. An isolation forest or a nonlinear autoencoder would need a new
runtime dependency (scikit-learn or torch), which needs the user's approval and a
licence check (D-091); this module deliberately uses numpy only.

The feature scaler is the ML-7 ``Standardizer``; it is fitted on train sessions before
this model and is stored with it so scoring uses exactly the same transform.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vehicle_ml.normalize import Standardizer


@dataclass
class BaselineModel:
    """Fitted PCA and robust-z detectors over the standardised window features."""

    scaler: Standardizer
    feature_names: list[str]
    pca_mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    median: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        split_manifest: Mapping[str, Sequence[str]],
        scaler: Standardizer,
        *,
        variance: float = 0.95,
    ) -> BaselineModel:
        """Fit on train rows only; reject a row whose session is outside the train split."""
        if "session_id" not in features.columns:
            raise ValueError("features needs a session_id column")
        if not 0.0 < variance <= 1.0:
            raise ValueError("variance must be in (0, 1]")
        train_ids = set(split_manifest.get("train", ()))
        leaked = sorted(set(features["session_id"].astype(str)) - train_ids)
        if leaked:
            raise ValueError(f"rows from non-train sessions: {', '.join(leaked)}")
        if "split" in features.columns and not (features["split"] == "train").all():
            raise ValueError("features contains non-train rows")

        feature_names = list(scaler.means)
        if not feature_names:
            raise ValueError("scaler has no fitted features")
        matrix = scaler.transform(features)[feature_names].to_numpy(dtype=float)
        finite = np.isfinite(matrix).all(axis=1)
        train = matrix[finite]
        if train.shape[0] < 2:
            raise ValueError("baseline needs at least two complete train rows")

        pca_mean = np.mean(train, axis=0)
        centred = train - pca_mean
        _, singular_values, right = np.linalg.svd(centred, full_matrices=False)
        total_variance = float(np.sum(singular_values**2))
        if total_variance == 0.0:
            explained = np.zeros(singular_values.shape, dtype=float)
            components = right[:1]
        else:
            explained = singular_values**2 / total_variance
            cumulative = np.cumsum(explained)
            count = int(np.searchsorted(cumulative, variance) + 1)
            count = min(max(count, 1), right.shape[0])
            components = right[:count]

        median = np.median(train, axis=0)
        mad = np.median(np.abs(train - median), axis=0)
        scale = 1.4826 * mad
        scale[scale == 0.0] = 1.0
        return cls(
            scaler=scaler,
            feature_names=feature_names,
            pca_mean=pca_mean,
            components=components,
            explained_variance_ratio=explained,
            median=median,
            scale=scale,
        )

    def _matrix(self, features: pd.DataFrame) -> np.ndarray:
        missing = [name for name in self.feature_names if name not in features.columns]
        if missing:
            raise ValueError(f"missing feature columns: {', '.join(missing)}")
        return self.scaler.transform(features)[self.feature_names].to_numpy(dtype=float)

    def scores(self, features: pd.DataFrame) -> pd.DataFrame:
        """Per-window PCA reconstruction error and maximum robust z-score."""
        matrix = self._matrix(features)
        finite = np.isfinite(matrix).all(axis=1)
        pca_error = np.full(matrix.shape[0], np.nan, dtype=float)
        robust_z = np.full(matrix.shape[0], np.nan, dtype=float)

        if finite.any():
            centred = matrix[finite] - self.pca_mean
            reconstructed = centred @ self.components.T @ self.components
            pca_error[finite] = np.mean((centred - reconstructed) ** 2, axis=1)

            z = np.abs((matrix[finite] - self.median) / self.scale)
            robust_z[finite] = np.max(z, axis=1)

        return pd.DataFrame({"pca_error": pca_error, "robust_z": robust_z}, index=features.index)

    def to_json(self) -> str:
        return json.dumps(
            {
                "scaler": json.loads(self.scaler.to_json()),
                "feature_names": self.feature_names,
                "pca_mean": self.pca_mean.tolist(),
                "components": self.components.tolist(),
                "explained_variance_ratio": self.explained_variance_ratio.tolist(),
                "median": self.median.tolist(),
                "scale": self.scale.tolist(),
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> BaselineModel:
        data = json.loads(text)
        return cls(
            scaler=Standardizer.from_json(json.dumps(data["scaler"])),
            feature_names=[str(name) for name in data["feature_names"]],
            pca_mean=np.asarray(data["pca_mean"], dtype=float),
            components=np.asarray(data["components"], dtype=float),
            explained_variance_ratio=np.asarray(data["explained_variance_ratio"], dtype=float),
            median=np.asarray(data["median"], dtype=float),
            scale=np.asarray(data["scale"], dtype=float),
        )


__all__ = ["BaselineModel"]
