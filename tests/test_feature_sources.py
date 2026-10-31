"""Feature provenance from the pinned defs (ML-19)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vehicle_ml.data.sources import (
    feature_metadata,
    feature_modality,
    feature_source,
)
from vehicle_ml.evaluate import physics_feature_groups
from vehicle_ml.features import features_frame
from vehicle_ml.physics.config import load_roles
from vehicle_ml.windows import Window

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "physics_cl250.toml"


def test_source_and_modality_come_from_the_defs_tables() -> None:
    assert feature_source("engine_speed_rpm.mean") == "did.ENGINE_SPEED"
    assert feature_source("gear_estimate.band_0_5hz") == "derived.gear_estimate"
    assert feature_source("ax_g.mean") == "sensor.imu_accel_x"
    assert feature_source("gx_dps.rms") == "sensor.imu_gyro_x"
    assert feature_modality("did.ENGINE_SPEED") == "can"
    assert feature_modality("derived.gear_estimate") == "can"
    assert feature_modality("sensor.imu_accel_x") == "imu"
    assert feature_modality("sensor.gps_ground_speed") == "other"


def test_unknown_channel_fails_with_its_name() -> None:
    with pytest.raises(ValueError, match="mystery.mean"):
        feature_source("mystery.mean")


def test_every_feature_column_has_source_and_modality() -> None:
    times = np.arange(0, 1000, 100)
    window = Window(
        "s1",
        0,
        900,
        times,
        {
            "engine_speed_rpm": np.linspace(1000.0, 2000.0, times.size),
            "gear_estimate": np.ones(times.size),
            "ax_g": np.sin(times / 100.0),
        },
    )
    frame = features_frame([window])
    metadata = feature_metadata(frame.columns)
    assert set(frame.attrs["feature_sources"]) == set(metadata)
    assert set(frame.attrs["feature_modalities"]) == set(metadata)
    for column in frame.columns:
        if column in {"session_id", "start_ms", "end_ms"}:
            continue
        assert column in metadata
        assert metadata[column].source
        assert metadata[column].modality in {"can", "imu", "other"}
    assert frame.attrs["feature_sources"]["ax_g.mean"] == "sensor.imu_accel_x"
    assert frame.attrs["feature_modalities"]["engine_speed_rpm.mean"] == "can"


def test_physics_features_carry_their_estimator_group() -> None:
    groups = physics_feature_groups(load_roles(CONFIG))
    assert groups["cooling_a"] == "cooling"
    assert groups["charging_plateau_v"] == "charging"
    assert groups["gearing_residual_mean"] == "longitudinal"
    assert all(name.startswith(("cooling_", "charging_", "gearing_")) for name in groups)
