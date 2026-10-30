"""Read-only access to a vehicle-server data directory (index.sqlite + Parquet).

The server owns the files (vehicle-server/src/vehicle_server/index.py and
``parquet.py``); this module never writes next to them. Outputs of the pipeline go to
a caller-given directory instead.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from vehicle_ml.data.position import drop_position_columns

KINDS = ("telemetry", "imu", "gps", "signals", "derived")

_SESSION_COLUMNS = [
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


class SessionStore:
    """Session index and Parquet reader for one vehicle-server data directory."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.sqlite"

    def _connect(self) -> sqlite3.Connection:
        # D-033: the server's directory is read-only for this process.
        return sqlite3.connect(f"file:{self.index_path}?mode=ro", uri=True)

    def _session_dir(self, session_id: str) -> Path:
        return self.data_dir / "sessions" / session_id

    def _require_session(self, session_id: str) -> None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise ValueError(f"unknown session: {session_id!r}")

    def sessions(self) -> pd.DataFrame:
        """The session index: one row per ingested session."""
        with closing(self._connect()) as conn:
            frame = pd.read_sql_query(f"SELECT {', '.join(_SESSION_COLUMNS)} FROM sessions", conn)
        for column in ("has_imu", "has_gps", "has_vehicle_signals", "has_derived"):
            frame[column] = frame[column].astype(bool)
        return frame

    def load(self, session_id: str, kind: str) -> pd.DataFrame | None:
        """Load one Parquet kind for a session, or ``None`` when the file is absent.

        Position-like columns are dropped from every kind. The returned frame always
        carries ``session_id``.
        """
        if kind not in KINDS:
            raise ValueError(f"unknown kind: {kind!r} (expected one of {', '.join(KINDS)})")
        self._require_session(session_id)

        path = self._session_dir(session_id) / "parquet" / f"{kind}.parquet"
        if not path.exists():
            return None

        frame = pd.read_parquet(path)
        frame = frame[drop_position_columns([str(column) for column in frame.columns])]
        if "session_id" in frame.columns:
            frame["session_id"] = session_id
        else:
            frame.insert(0, "session_id", session_id)
        return frame

    def channels(self, session_id: str) -> dict[str, list[str]]:
        """Available (position-filtered) column names per kind, missing kinds empty."""
        self._require_session(session_id)
        result: dict[str, list[str]] = {}
        for kind in KINDS:
            frame = self.load(session_id, kind)
            result[kind] = [] if frame is None else [str(column) for column in frame.columns]
        return result
