"""Charging model: a slope below cut-in RPM and a regulated plateau above it.

The model is ``V = min(base + slope * rpm, plateau)``. For a candidate cut-in RPM the
model is linear in ``base`` and ``slope``, so the estimator scans a documented grid of
cut-in candidates between the observed RPM limits, solves the two-parameter least-squares
problem for each, and keeps the candidate with the smallest residual sum of squares.
No scipy is used.

The 14.0-15.0 V plateau in the D-083 idea note is CONFIRM, not a hard limit; this module
never clamps to it. A session needs samples both below and above the cut-in, otherwise
the plateau and the cut-in are not identifiable (for example idle-only data).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vehicle_ml.physics.errors import NotIdentifiableError

CUT_IN_CANDIDATES = 400


@dataclass(frozen=True)
class ChargingFit:
    """Identified charging parameters and the measured-minus-model residual."""

    base_v: float
    plateau_v: float
    cut_in_rpm: float
    slope_v_per_rpm: float
    residual: np.ndarray
    quality: float
    samples: int


def fit_charging(
    engine_speed_rpm: np.ndarray,
    battery_voltage_v: np.ndarray,
    *,
    min_samples: int = 30,
    min_side_samples: int = 10,
) -> ChargingFit:
    """Fit plateau, cut-in and below-cut-in slope from RPM and battery voltage."""
    rpm = np.asarray(engine_speed_rpm, dtype=float)
    voltage = np.asarray(battery_voltage_v, dtype=float)
    if rpm.ndim != 1 or rpm.size != voltage.size:
        raise ValueError("RPM and battery voltage must be 1-D arrays of equal length")
    usable = np.isfinite(rpm) & np.isfinite(voltage)
    rpm = rpm[usable]
    voltage = voltage[usable]
    if rpm.size < min_samples:
        raise NotIdentifiableError(
            f"charging model is not identifiable: {rpm.size} samples (minimum {min_samples})"
        )

    low = float(np.min(rpm))
    high = float(np.max(rpm))
    if high - low < 100.0:
        raise NotIdentifiableError(
            "charging model is not identifiable: engine speed has too little range"
        )

    best: tuple[float, float, float, np.ndarray] | None = None
    for cut_in in np.linspace(low, high, CUT_IN_CANDIDATES):
        above = int(np.count_nonzero(rpm > cut_in))
        below = int(np.count_nonzero(rpm <= cut_in))
        if above < min_side_samples or below < min_side_samples:
            continue
        design = np.column_stack((np.ones(rpm.size), np.minimum(rpm, cut_in)))
        parameters, _, rank, _ = np.linalg.lstsq(design, voltage, rcond=None)
        if rank < 2:
            continue
        residual = voltage - design @ parameters
        error = float(np.sum(residual**2))
        if best is None or error < best[0]:
            best = (error, float(cut_in), parameters[0], parameters)

    if best is None:
        raise NotIdentifiableError(
            "charging model is not identifiable: no cut-in has samples on both sides"
        )

    _, cut_in, base, parameters = best
    slope = float(parameters[1])
    if not np.isfinite(slope) or slope <= 0.0:
        raise NotIdentifiableError(
            "charging model is not identifiable: below-cut-in slope is not positive"
        )
    plateau = base + slope * cut_in
    if not np.isfinite(plateau):
        raise NotIdentifiableError("charging model is not identifiable: plateau is not finite")

    design = np.column_stack((np.ones(rpm.size), np.minimum(rpm, cut_in)))
    residual = voltage - design @ parameters
    total = float(np.sum((voltage - np.mean(voltage)) ** 2))
    quality = 1.0 if total == 0.0 and float(np.sum(residual**2)) == 0.0 else 0.0
    if total > 0.0:
        quality = 1.0 - float(np.sum(residual**2)) / total

    return ChargingFit(
        base_v=float(base),
        plateau_v=float(plateau),
        cut_in_rpm=cut_in,
        slope_v_per_rpm=slope,
        residual=residual,
        quality=quality,
        samples=int(rpm.size),
    )


__all__ = ["ChargingFit", "NotIdentifiableError", "fit_charging"]
