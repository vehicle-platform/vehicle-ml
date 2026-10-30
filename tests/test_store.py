"""SessionStore: index, Parquet kinds, position filtering and read-only access."""

from __future__ import annotations

import os
import stat

import pytest

from tests.synthetic import make_data_dir
from vehicle_ml.data.position import drop_position_columns, is_position_name
from vehicle_ml.data.store import KINDS, SessionStore


@pytest.fixture
def store(tmp_path) -> SessionStore:
    return SessionStore(make_data_dir(tmp_path / "data"))


def test_sessions_returns_the_index_columns(store: SessionStore) -> None:
    frame = store.sessions()
    assert list(frame.columns) == [
        "session_id",
        "created_utc",
        "status",
        "packet_count",
        "loss_percent",
        "has_imu",
        "has_gps",
        "has_vehicle_signals",
        "has_derived",
    ]
    assert set(frame["session_id"]) == {"s1", "s2", "s3"}
    by_id = frame.set_index("session_id")
    assert not by_id.loc["s2", "has_imu"]
    assert by_id.loc["s1", "has_imu"]


@pytest.mark.parametrize("kind", KINDS)
def test_load_every_kind_adds_the_session_id(store: SessionStore, kind: str) -> None:
    frame = store.load("s1", kind)
    assert frame is not None
    assert set(frame["session_id"]) == {"s1"}
    assert not frame.empty


def test_missing_files_return_none(store: SessionStore) -> None:
    assert store.load("s2", "imu") is None
    assert store.load("s3", "gps") is None
    assert store.load("s3", "derived") is None


def test_unknown_session_is_a_clear_error(store: SessionStore) -> None:
    with pytest.raises(ValueError, match="unknown session"):
        store.load("nope", "telemetry")
    with pytest.raises(ValueError, match="unknown session"):
        store.channels("nope")


def test_unknown_kind_is_rejected(store: SessionStore) -> None:
    with pytest.raises(ValueError, match="unknown kind"):
        store.load("s1", "video")


def test_position_columns_are_dropped(store: SessionStore) -> None:
    frame = store.load("s3", "telemetry")
    assert frame is not None
    assert "gps_latitude" not in frame.columns
    assert "engine_speed_rpm" in frame.columns


def test_channels_reports_available_columns_per_kind(store: SessionStore) -> None:
    channels = store.channels("s1")
    assert set(channels) == set(KINDS)
    assert "device_time_ms" in channels["telemetry"]
    assert channels["imu"]
    assert channels["gps"]
    assert channels["signals"]
    assert channels["derived"]


def test_read_only_data_dir_still_loads(tmp_path) -> None:
    data = make_data_dir(tmp_path / "data")
    store = SessionStore(data)
    modes: list[tuple[str, int]] = []
    for path in [data, *[p for p in data.rglob("*") if p.is_dir()]]:
        modes.append((str(path), stat.S_IMODE(path.stat().st_mode)))
        os.chmod(path, 0o555)
    try:
        frame = store.load("s1", "telemetry")
        assert frame is not None
        assert not frame.empty
    finally:
        for path, mode in modes:
            os.chmod(path, mode)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("sensor.gps_latitude", True),
        ("did.LAT", True),
        ("derived.longitude", True),
        ("vehicle.height", True),
        ("gpsLatitude", True),
        ("GPSLongitude", True),
        ("did.ALTITUDE", True),
        ("sensor.alt", True),
        ("sensor.gps_height", True),
        ("did.ENGINE_SPEED", False),
        ("derived.LEAN_ANGLE", False),
        ("sensor.related", False),
        ("did.PLATE", False),
        ("sensor.velocity", False),
        ("did.ALTIMETER_OK", False),
        ("derived.altitude_change", False),
        ("derived.road_grade", False),
    ],
)
def test_position_rule_matches_the_mcp_cases(name: str, expected: bool) -> None:
    assert is_position_name(name) is expected


def test_drop_position_columns_keeps_the_rest() -> None:
    assert drop_position_columns(["engine_speed_rpm", "gps_latitude", "gx_dps"]) == [
        "engine_speed_rpm",
        "gx_dps",
    ]
