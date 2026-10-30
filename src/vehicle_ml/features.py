"""Per-window channel statistics and optional FFT band energy.

Feature values keep the channel's units. NaN samples are ignored by the statistics.
A window whose sampling interval varies more than ``rate_tolerance`` is rejected
instead of being resampled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from vehicle_ml.windows import Window

STATS = ("count", "mean", "std", "min", "max", "rms", "p05", "p50", "p95")


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = _finite(values)
    count = int(finite.size)
    if count == 0:
        return {name: (0 if name == "count" else None) for name in STATS}
    return {
        "count": count,
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "rms": float(np.sqrt(np.mean(finite**2))),
        "p05": float(np.percentile(finite, 5)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
    }


def _sample_rate(times_ms: np.ndarray, rate_tolerance: float) -> float:
    if times_ms.size < 2:
        raise ValueError("a window needs at least two samples to measure its rate")
    steps = np.diff(times_ms.astype(float))
    mean_step = float(np.mean(steps))
    if mean_step <= 0 or np.std(steps) / mean_step > rate_tolerance:
        raise ValueError("window sample rate varies more than the tolerance")
    return 1000.0 / mean_step


def _band_name(low: float, high: float) -> str:
    return f"band_{low:g}_{high:g}hz"


def window_features(
    window: Window,
    bands: Mapping[str, Sequence[tuple[float, float]]] | None = None,
    rate_tolerance: float = 0.05,
) -> dict[str, float | int | None]:
    """Features for every channel of one window, keyed ``<channel>.<stat>``."""
    result: dict[str, float | int | None] = {}
    rate: float | None = None
    for channel, values in window.channels.items():
        for name, value in _stats(np.asarray(values, dtype=float)).items():
            result[f"{channel}.{name}"] = value

        channel_bands = (bands or {}).get(channel, ())
        if not channel_bands:
            continue
        if rate is None:
            rate = _sample_rate(window.times_ms, rate_tolerance)
        spectrum = np.fft.rfft(np.nan_to_num(np.asarray(values, dtype=float)))
        frequencies = np.fft.rfftfreq(len(values), d=1.0 / rate)
        power = np.abs(spectrum) ** 2
        for low, high in channel_bands:
            selected = (frequencies >= low) & (frequencies < high)
            result[f"{channel}.{_band_name(low, high)}"] = float(np.sum(power[selected]))
    return result


def features_frame(
    windows: Sequence[Window],
    bands: Mapping[str, Sequence[tuple[float, float]]] | None = None,
    rate_tolerance: float = 0.05,
) -> pd.DataFrame:
    """One row per window: identity columns plus the feature columns."""
    rows: list[dict[str, object]] = []
    for window in windows:
        row: dict[str, object] = {
            "session_id": window.session_id,
            "start_ms": window.start_ms,
            "end_ms": window.end_ms,
        }
        row.update(window_features(window, bands, rate_tolerance))
        rows.append(row)
    return pd.DataFrame(rows)
