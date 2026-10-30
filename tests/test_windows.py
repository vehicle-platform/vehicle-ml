"""Windows: gap and missing-fraction drops, and cross-kind alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.synthetic import make_data_dir
from vehicle_ml.data.store import SessionStore
from vehicle_ml.windows import align_frames, make_windows


@pytest.fixture
def store(tmp_path) -> SessionStore:
    return SessionStore(make_data_dir(tmp_path / "data"))


def _load(store: SessionStore, session_id: str, kind: str) -> pd.DataFrame:
    frame = store.load(session_id, kind)
    assert frame is not None
    return frame


def test_full_session_is_cut_into_windows(store: SessionStore) -> None:
    telemetry = _load(store, "s1", "telemetry")
    windows, report = make_windows(
        telemetry,
        ["engine_speed_rpm", "vehicle_speed_kmh"],
        window_ms=2_000,
        stride_ms=1_000,
        max_gap_ms=500,
    )

    assert report.candidates == 9
    assert report.kept == 9
    assert report.dropped == {}
    assert windows[0].session_id == "s1"
    assert (windows[0].start_ms, windows[0].end_ms) == (0, 2_000)
    assert len(windows[0].channels["engine_speed_rpm"]) == 21
    assert windows[-1].end_ms == 10_000


def test_gap_session_drops_windows_that_span_the_gap(store: SessionStore) -> None:
    telemetry = _load(store, "s2", "telemetry")
    windows, report = make_windows(
        telemetry,
        ["engine_speed_rpm"],
        window_ms=2_000,
        stride_ms=1_000,
        max_gap_ms=500,
    )

    assert report.kept < report.candidates
    assert report.dropped.get("gap", 0) + report.dropped.get("missing:engine_speed_rpm", 0) > 0
    for window in windows:
        times = window.channels["engine_speed_rpm"]
        assert len(times) > 0


def test_missing_channel_drops_the_window(store: SessionStore) -> None:
    telemetry = _load(store, "s1", "telemetry")
    telemetry.loc[telemetry.index[:60], "engine_speed_rpm"] = np.nan
    windows, report = make_windows(
        telemetry, ["engine_speed_rpm"], window_ms=2_000, stride_ms=1_000, max_gap_ms=500
    )

    assert report.kept < 9
    assert report.dropped["missing:engine_speed_rpm"] > 0
    assert all(not np.isnan(window.channels["engine_speed_rpm"]).any() for window in windows)


def test_empty_session_returns_no_windows(store: SessionStore) -> None:
    telemetry = _load(store, "s1", "telemetry").iloc[0:0]
    windows, report = make_windows(
        telemetry, ["engine_speed_rpm"], window_ms=2_000, stride_ms=1_000, max_gap_ms=500
    )
    assert windows == []
    assert report.candidates == 0
    assert report.dropped == {"empty": 1}


def test_unknown_channel_is_rejected(store: SessionStore) -> None:
    telemetry = _load(store, "s1", "telemetry")
    with pytest.raises(ValueError, match="unknown channels"):
        make_windows(telemetry, ["nope"], window_ms=2_000, stride_ms=1_000, max_gap_ms=500)


def test_align_frames_uses_nearest_previous_within_tolerance(store: SessionStore) -> None:
    telemetry = _load(store, "s1", "telemetry")
    imu = _load(store, "s1", "imu")
    aligned = align_frames(telemetry, imu, ["ax_g"], tolerance_ms=15)

    assert len(aligned) == len(telemetry)
    assert "ax_g" in aligned.columns
    assert not aligned["ax_g"].isna().any()


def test_align_frames_does_not_cross_a_gap(tmp_path) -> None:
    grid = pd.DataFrame({"session_id": ["s1", "s1", "s1"], "device_time_ms": [0, 200, 5_000]})
    other = pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s1"],
            "device_time_ms": [0, 100, 5_100],
            "ax_g": [1.0, 2.0, 3.0],
        }
    )
    aligned = align_frames(grid, other, ["ax_g"], tolerance_ms=150)

    assert aligned["ax_g"].tolist()[0] == 1.0
    assert aligned["ax_g"].tolist()[1] == 2.0
    assert np.isnan(aligned["ax_g"].tolist()[2])
