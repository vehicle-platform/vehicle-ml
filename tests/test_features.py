"""Feature statistics and FFT band energy on small hand-computed arrays."""

from __future__ import annotations

import numpy as np
import pytest

from vehicle_ml.features import STATS, features_frame, window_features
from vehicle_ml.windows import Window


def _window(values: list[float], step_ms: int = 100) -> Window:
    times = np.arange(len(values)) * step_ms
    return Window(
        session_id="s1",
        start_ms=0,
        end_ms=int(times[-1]) if len(times) else 0,
        times_ms=times,
        channels={"engine_speed_rpm": np.asarray(values, dtype=float)},
    )


def test_hand_computed_statistics() -> None:
    features = window_features(_window([1.0, 2.0, 3.0, 4.0]))
    assert features["engine_speed_rpm.count"] == 4
    assert features["engine_speed_rpm.mean"] == pytest.approx(2.5)
    assert features["engine_speed_rpm.std"] == pytest.approx(np.sqrt(1.25))
    assert features["engine_speed_rpm.min"] == 1.0
    assert features["engine_speed_rpm.max"] == 4.0
    assert features["engine_speed_rpm.rms"] == pytest.approx(np.sqrt(7.5))
    assert features["engine_speed_rpm.p05"] == pytest.approx(1.15)
    assert features["engine_speed_rpm.p50"] == pytest.approx(2.5)
    assert features["engine_speed_rpm.p95"] == pytest.approx(3.85)


def test_statistics_are_nan_safe() -> None:
    features = window_features(_window([1.0, np.nan, 3.0, 4.0]))
    assert features["engine_speed_rpm.count"] == 3
    assert features["engine_speed_rpm.mean"] == pytest.approx(8.0 / 3.0)
    assert features["engine_speed_rpm.min"] == 1.0
    assert features["engine_speed_rpm.max"] == 4.0


def test_all_nan_channel_has_no_statistics() -> None:
    features = window_features(_window([np.nan, np.nan]))
    assert features["engine_speed_rpm.count"] == 0
    for name in STATS:
        if name != "count":
            assert features[f"engine_speed_rpm.{name}"] is None


def test_pure_sine_lands_in_the_right_band() -> None:
    times = np.arange(0, 1000, 10)
    values = np.sin(2 * np.pi * 10.0 * times / 1000.0)
    window = Window(
        session_id="s1",
        start_ms=0,
        end_ms=990,
        times_ms=times,
        channels={"engine_speed_rpm": values},
    )
    features = window_features(window, bands={"engine_speed_rpm": [(8.0, 12.0), (30.0, 40.0)]})

    assert features["engine_speed_rpm.band_8_12hz"] > features["engine_speed_rpm.band_30_40hz"]
    assert features["engine_speed_rpm.band_8_12hz"] > 1.0


def test_varying_rate_is_rejected() -> None:
    window = Window(
        session_id="s1",
        start_ms=0,
        end_ms=300,
        times_ms=np.asarray([0, 100, 250, 300]),
        channels={"engine_speed_rpm": np.asarray([1.0, 2.0, 3.0, 4.0])},
    )
    with pytest.raises(ValueError, match="rate varies"):
        window_features(window, bands={"engine_speed_rpm": [(1.0, 5.0)]})


def test_features_frame_has_one_row_per_window() -> None:
    frame = features_frame([_window([1.0, 2.0]), _window([3.0, 4.0])])
    assert len(frame) == 2
    assert list(frame.columns[:3]) == ["session_id", "start_ms", "end_ms"]
    assert "engine_speed_rpm.mean" in frame.columns
