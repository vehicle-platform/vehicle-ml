"""The synthetic directory matches the server layout the data layer reads."""

from __future__ import annotations

import sqlite3

import pandas as pd

from tests.synthetic import DEFAULT_SESSIONS, make_data_dir


def test_make_data_dir_writes_the_server_layout(tmp_path) -> None:
    root = make_data_dir(tmp_path / "data")

    assert (root / "index.sqlite").exists()
    with sqlite3.connect(root / "index.sqlite") as conn:
        query = "SELECT session_id FROM sessions ORDER BY session_id"
        ids = [row[0] for row in conn.execute(query)]
    assert ids == [spec.session_id for spec in DEFAULT_SESSIONS]

    for spec in DEFAULT_SESSIONS:
        parquet = root / "sessions" / spec.session_id / "parquet"
        telemetry = pd.read_parquet(parquet / "telemetry.parquet")
        assert "device_time_ms" in telemetry.columns
        assert (parquet / "imu.parquet").exists() is spec.has_imu
        assert (parquet / "gps.parquet").exists() is spec.has_gps
        assert (parquet / "signals.parquet").exists() is spec.has_vehicle_signals
        assert (parquet / "derived.parquet").exists() is spec.has_derived


def test_gap_session_has_a_hole_and_position_session_has_a_position_column(tmp_path) -> None:
    root = make_data_dir(tmp_path / "data")
    gap = pd.read_parquet(root / "sessions" / "s2" / "parquet" / "telemetry.parquet")
    assert gap["device_time_ms"].diff().max() > 100

    with_position = pd.read_parquet(root / "sessions" / "s3" / "parquet" / "telemetry.parquet")
    assert "gps_latitude" in with_position.columns
