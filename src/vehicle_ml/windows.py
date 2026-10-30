"""Window cutting and cross-kind alignment on the ``device_time_ms`` axis.

A window belongs to exactly one session. It is dropped when its samples span a gap
larger than ``max_gap_ms`` or when any requested channel misses more than
``max_missing`` of the expected samples. Alignment is nearest-previous within a
tolerance; nothing is interpolated across a gap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TIME_COLUMN = "device_time_ms"


@dataclass(frozen=True)
class Window:
    session_id: str
    start_ms: int
    end_ms: int
    channels: dict[str, np.ndarray]


@dataclass
class DropReport:
    candidates: int = 0
    kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1


def _nominal_step(times: np.ndarray) -> int:
    diffs = np.diff(times)
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return 0
    return int(np.median(positive))


def _window_rows(frame: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    times = frame[TIME_COLUMN].to_numpy()
    return frame.loc[(times >= start) & (times <= end)]


def make_windows(
    frame: pd.DataFrame,
    channels: Sequence[str],
    window_ms: int,
    stride_ms: int,
    max_gap_ms: int,
    max_missing: float = 0.2,
) -> tuple[list[Window], DropReport]:
    """Cut one session into windows and report why candidates were dropped."""
    if window_ms <= 0 or stride_ms <= 0:
        raise ValueError("window_ms and stride_ms must be positive")
    if not 0.0 <= max_missing <= 1.0:
        raise ValueError("max_missing must be between 0 and 1")
    if TIME_COLUMN not in frame.columns:
        raise ValueError(f"frame has no {TIME_COLUMN} column")
    missing_channels = [channel for channel in channels if channel not in frame.columns]
    if missing_channels:
        raise ValueError(f"unknown channels: {', '.join(missing_channels)}")

    report = DropReport()
    if frame.empty:
        report.drop("empty")
        return [], report

    session_ids = frame["session_id"].unique() if "session_id" in frame.columns else []
    if len(session_ids) != 1:
        raise ValueError("make_windows expects exactly one session")
    session_id = str(session_ids[0])

    ordered = frame.sort_values(TIME_COLUMN)
    times = ordered[TIME_COLUMN].to_numpy()
    step = _nominal_step(times)
    if step == 0:
        report.drop("empty")
        return [], report
    expected = max(1, int(round(window_ms / step)) + 1)

    windows: list[Window] = []
    last_start = int(times[-1]) - window_ms
    for start in range(int(times[0]), last_start + 1, stride_ms):
        end = start + window_ms
        report.candidates += 1
        rows = _window_rows(ordered, start, end)
        if rows.empty:
            report.drop("empty")
            continue

        window_times = rows[TIME_COLUMN].to_numpy()
        if window_times.size > 1 and np.diff(window_times).max() > max_gap_ms:
            report.drop("gap")
            continue

        values: dict[str, np.ndarray] = {}
        reason = None
        for channel in channels:
            column = rows[channel].to_numpy(dtype=float)
            missing_fraction = 1.0 - (np.count_nonzero(~np.isnan(column)) / expected)
            if missing_fraction > max_missing:
                reason = f"missing:{channel}"
                break
            values[channel] = column
        if reason is not None:
            report.drop(reason)
            continue

        windows.append(Window(session_id=session_id, start_ms=start, end_ms=end, channels=values))

    report.kept = len(windows)
    return windows, report


def align_frames(
    grid: pd.DataFrame, other: pd.DataFrame, columns: Sequence[str], tolerance_ms: int
) -> pd.DataFrame:
    """Add ``columns`` from ``other`` to ``grid`` by nearest-previous sample.

    A grid row gets NaN when the latest sample is older than ``tolerance_ms`` (or there
    is none), so a gap never becomes an interpolated value. Both frames must carry
    ``device_time_ms`` and, when present, the same ``session_id``.
    """
    for frame in (grid, other):
        if TIME_COLUMN not in frame.columns:
            raise ValueError(f"frame has no {TIME_COLUMN} column")
    if "session_id" in grid.columns and "session_id" in other.columns:
        if set(grid["session_id"]) != set(other["session_id"]):
            raise ValueError("align_frames cannot mix sessions")

    ordered = other.sort_values(TIME_COLUMN)
    other_times = ordered[TIME_COLUMN].to_numpy()
    grid_times = grid[TIME_COLUMN].to_numpy()
    result = grid.copy()

    if other_times.size == 0:
        for column in columns:
            result[column] = np.nan
        return result

    previous = np.searchsorted(other_times, grid_times, side="right") - 1
    in_range = previous >= 0
    safe_previous = np.clip(previous, 0, len(other_times) - 1)
    within = grid_times - other_times[safe_previous] <= tolerance_ms
    usable = in_range & within

    for column in columns:
        values = ordered[column].to_numpy(dtype=float)[safe_previous]
        result[column] = np.where(usable, values, np.nan)
    return result
