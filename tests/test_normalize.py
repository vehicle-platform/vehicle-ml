"""Standardizer: train-only fit, constant features and JSON round-trip."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vehicle_ml.normalize import Standardizer

_TRAIN = {"train": ["train-1", "train-2"], "val": [], "test": []}


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": ["train-1", "train-1", "train-2"],
            "engine_speed_rpm": [1000.0, 2000.0, 3000.0],
            "constant_channel": [5.0, 5.0, 5.0],
        }
    )


def test_fit_and_transform_standardise_numeric_features() -> None:
    standardizer = Standardizer.fit(_frame(), _TRAIN)
    transformed = standardizer.transform(_frame())

    assert transformed["engine_speed_rpm"].mean() == pytest.approx(0.0, abs=1e-12)
    assert transformed["engine_speed_rpm"].std(ddof=0) == pytest.approx(1.0)
    assert transformed["constant_channel"].tolist() == [0.0, 0.0, 0.0]
    assert "constant_channel" in standardizer.constant_features


def test_json_round_trip() -> None:
    standardizer = Standardizer.fit(_frame(), _TRAIN)
    restored = Standardizer.from_json(standardizer.to_json())
    pd.testing.assert_frame_equal(restored.transform(_frame()), standardizer.transform(_frame()))
    assert restored.constant_features == standardizer.constant_features


def test_fit_rejects_rows_from_a_non_train_session() -> None:
    manifest = {"train": ["train-1"], "val": ["train-2"], "test": []}
    with pytest.raises(ValueError, match="non-train sessions"):
        Standardizer.fit(_frame(), manifest)


def test_fit_accepts_train_rows_only() -> None:
    manifest = {"train": ["train-1", "train-2"], "val": [], "test": []}
    standardizer = Standardizer.fit(_frame(), manifest)
    assert set(standardizer.means) == {"engine_speed_rpm", "constant_channel"}


def test_transform_requires_the_fitted_columns() -> None:
    standardizer = Standardizer.fit(_frame(), _TRAIN)
    with pytest.raises(ValueError, match="missing feature columns"):
        standardizer.transform(pd.DataFrame({"engine_speed_rpm": [1.0]}))


def test_all_nan_feature_is_constant() -> None:
    frame = pd.DataFrame(
        {"session_id": ["train-1", "train-1"], "engine_speed_rpm": [np.nan, np.nan]}
    )
    standardizer = Standardizer.fit(frame, _TRAIN)
    assert standardizer.stds["engine_speed_rpm"] == 1.0
    assert standardizer.means["engine_speed_rpm"] == 0.0
    assert "engine_speed_rpm" in standardizer.constant_features


def test_fit_skips_window_identity_and_split_columns() -> None:
    frame = _frame().assign(start_ms=[0, 1000, 2000], end_ms=[2000, 3000, 4000], split="train")
    standardizer = Standardizer.fit(frame, _TRAIN)
    assert set(standardizer.means) == {"engine_speed_rpm", "constant_channel"}
    assert not {"start_ms", "end_ms"} & set(standardizer.means)


def test_fit_requires_a_session_id_column() -> None:
    with pytest.raises(ValueError, match="session_id"):
        Standardizer.fit(_frame().drop(columns="session_id"), _TRAIN)
