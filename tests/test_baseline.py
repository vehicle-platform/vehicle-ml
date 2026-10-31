"""Baseline detectors: train-only fitting, outlier scores and JSON round trip (ML-16)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from vehicle_ml.baseline import BaselineModel
from vehicle_ml.normalize import Standardizer


def _features(seed: int = 0, n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n, 3))
    return pd.DataFrame(
        {
            "session_id": ["train"] * n,
            "start_ms": np.arange(n) * 1000,
            "end_ms": np.arange(n) * 1000 + 1000,
            "split": ["train"] * n,
            "a": values[:, 0],
            "b": values[:, 1],
            "c": values[:, 2],
        }
    )


def test_fit_rejects_rows_outside_train() -> None:
    train = _features()
    scaler = Standardizer.fit(train, {"train": ["train"]})
    other = train.copy()
    other["session_id"] = "val"
    other["split"] = "val"
    with pytest.raises(ValueError, match="non-train"):
        BaselineModel.fit(
            pd.concat([train, other], ignore_index=True), {"train": ["train"]}, scaler
        )


def test_feature_outlier_scores_high() -> None:
    train = _features()
    scaler = Standardizer.fit(train, {"train": ["train"]})
    model = BaselineModel.fit(train, {"train": ["train"]}, scaler)
    scored = model.scores(train)
    outlier = train.iloc[[0]].copy()
    outlier[["a", "b", "c"]] = [[100.0, 100.0, 100.0]]
    outlier_scores = model.scores(outlier)
    assert float(outlier_scores["robust_z"].iloc[0]) > float(scored["robust_z"].median()) * 10
    assert float(outlier_scores["pca_error"].iloc[0]) > float(scored["pca_error"].median()) * 10


def test_json_round_trip_preserves_scores() -> None:
    train = _features()
    scaler = Standardizer.fit(train, {"train": ["train"]})
    model = BaselineModel.fit(train, {"train": ["train"]}, scaler)
    restored = BaselineModel.from_json(model.to_json())
    assert_frame_equal(model.scores(train), restored.scores(train))
    assert restored.feature_names == model.feature_names
    assert restored.components.shape == model.components.shape
