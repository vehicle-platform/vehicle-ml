"""Feature standardisation fitted on the train split only (CLAUDE.md rule 1)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vehicle_ml.features import IDENTITY_COLUMNS


@dataclass
class Standardizer:
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    constant_features: list[str] = field(default_factory=list)

    @classmethod
    def fit(
        cls,
        train_features: pd.DataFrame,
        split_manifest: Mapping[str, Sequence[str]],
    ) -> Standardizer:
        """Fit on train rows; reject rows whose session is not in the train split.

        Needs a ``session_id`` column. Window identity columns and ``split`` are not
        features and are never fitted.
        """
        if "session_id" not in train_features.columns:
            raise ValueError("train_features needs a session_id column")
        train_ids = set(split_manifest.get("train", ()))
        leaked = sorted(set(train_features["session_id"].astype(str)) - train_ids)
        if leaked:
            raise ValueError(f"rows from non-train sessions: {', '.join(leaked)}")

        skipped = {*IDENTITY_COLUMNS, "split"}
        numeric = train_features.select_dtypes(include=[np.number]).drop(
            columns=[c for c in train_features.columns if c in skipped], errors="ignore"
        )
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        constant: list[str] = []
        for column in numeric.columns:
            values = numeric[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                means[str(column)] = 0.0
                stds[str(column)] = 1.0
                constant.append(str(column))
                continue
            means[str(column)] = float(np.mean(finite))
            std = float(np.std(finite))
            if std == 0.0:
                std = 1.0
                constant.append(str(column))
            stds[str(column)] = std
        return cls(means=means, stds=stds, constant_features=constant)

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Standardise the fitted feature columns; other columns pass through."""
        missing = [column for column in self.means if column not in features.columns]
        if missing:
            raise ValueError(f"missing feature columns: {', '.join(missing)}")
        result = features.copy()
        for column, mean in self.means.items():
            result[column] = (result[column].astype(float) - mean) / self.stds[column]
        return result

    def to_json(self) -> str:
        return json.dumps(
            {
                "means": dict(sorted(self.means.items())),
                "stds": dict(sorted(self.stds.items())),
                "constant_features": sorted(self.constant_features),
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> Standardizer:
        data = json.loads(text)
        return cls(
            means={str(key): float(value) for key, value in data["means"].items()},
            stds={str(key): float(value) for key, value in data["stds"].items()},
            constant_features=[str(name) for name in data["constant_features"]],
        )
