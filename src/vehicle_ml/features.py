"""Per-window channel statistics and optional FFT band energy.

Feature values keep the channel's units. NaN samples are ignored by the statistics.
A window whose sampling interval varies more than ``rate_tolerance`` is rejected
instead of being resampled: ``window_features`` raises ``RateError`` and
``features_frame`` skips the window and counts it as dropped reason ``"rate"``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from vehicle_ml.data.sources import feature_metadata
from vehicle_ml.windows import DropReport, Window

IDENTITY_COLUMNS = ("session_id", "start_ms", "end_ms")
STATS = ("count", "mean", "std", "min", "max", "rms", "p05", "p50", "p95")


class RateError(ValueError):
    """The window's sampling interval is too uneven for an FFT."""


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
        raise RateError("window sample rate varies more than the tolerance")
    return 1000.0 / mean_step


def _band_name(low: float, high: float) -> str:
    return f"band_{low:g}_{high:g}hz"


def window_features(
    window: Window,
    bands: Mapping[str, Sequence[tuple[float, float]]] | None = None,
    rate_tolerance: float = 0.05,
) -> dict[str, float | int | None]:
    """Features for every channel of one window, keyed ``<channel>.<stat>``.

    Band energy is taken from the FFT of the channel after subtracting its finite mean
    and then filling NaN with 0, so the fill does not bias the spectrum and the DC
    offset does not land in the lowest band. Raises ``RateError`` on an uneven rate.
    """
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
        samples = np.asarray(values, dtype=float)
        finite = _finite(samples)
        centred = samples - (float(np.mean(finite)) if finite.size else 0.0)
        spectrum = np.fft.rfft(np.nan_to_num(centred, nan=0.0, posinf=0.0, neginf=0.0))
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
    report: DropReport | None = None,
    *,
    allow_unknown: bool = False,
) -> pd.DataFrame:
    """One row per window: ``IDENTITY_COLUMNS`` plus the feature columns.

    A window with an uneven sample rate is skipped; when ``report`` is given it is
    counted as dropped reason ``"rate"`` and removed from ``report.kept``.
    """
    rows: list[dict[str, object]] = []
    for window in windows:
        try:
            features = window_features(window, bands, rate_tolerance)
        except RateError:
            if report is not None:
                report.drop("rate")
                report.kept -= 1
            continue
        row: dict[str, object] = {
            "session_id": window.session_id,
            "start_ms": window.start_ms,
            "end_ms": window.end_ms,
        }
        row.update(features)
        rows.append(row)
    frame = pd.DataFrame(rows)
    metadata = feature_metadata(frame.columns, allow_unknown=allow_unknown)
    frame.attrs["feature_metadata"] = {
        column: {
            "source": entry.source,
            "modality": entry.modality,
            "group": entry.group,
        }
        for column, entry in metadata.items()
    }
    frame.attrs["feature_sources"] = {column: entry.source for column, entry in metadata.items()}
    frame.attrs["feature_modalities"] = {
        column: entry.modality for column, entry in metadata.items()
    }
    return frame
