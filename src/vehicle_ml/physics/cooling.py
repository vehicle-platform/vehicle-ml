"""First-order cooling model: ``dT/dt = a * q(rpm) - h * (T - T_amb)``.

The engine heat input uses a fixed, documented proxy ``q = clip(rpm / 6000, 0, 1)``.
``T_amb`` is unknown in phase 0 (the ambient sensor is pending), so it is fitted as a
per-session constant. The fit is a linear least-squares problem in
``(a, h, h * T_amb)``; no scipy is used.

Only gap-free stretches are used. A stretch is kept when its consecutive samples are at
most ``max_gap_ms`` apart; a gap is never bridged by interpolation. The model is
refused as "not identifiable" when it has fewer than ``min_samples`` derivative
samples, when the RPM proxy barely varies (``q`` standard deviation below
``min_q_std``), or when the coolant temperature does not move by at least
``min_temp_range_c``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vehicle_ml.physics.errors import NotIdentifiableError
from vehicle_ml.physics.rls import RLS

RPM_PROXY_MAX = 6000.0


@dataclass(frozen=True)
class CoolingFit:
    """Identified cooling parameters and the derivative residual of the fit."""

    a: float
    h: float
    ambient_c: float
    time_constant_s: float
    residual: np.ndarray
    quality: float
    samples: int
    stretches: int


def _stretch_indices(times: np.ndarray, max_gap_ms: float) -> list[np.ndarray]:
    if times.size < 2:
        return []
    breaks = np.where(np.diff(times) > max_gap_ms)[0] + 1
    return [part for part in np.split(np.arange(times.size), breaks) if part.size >= 2]


def _design(
    times: np.ndarray, temperature: np.ndarray, rpm: np.ndarray, max_gap_ms: float
) -> tuple[np.ndarray, np.ndarray, int]:
    rows: list[np.ndarray] = []
    targets: list[float] = []
    stretches = 0
    for indices in _stretch_indices(times, max_gap_ms):
        local_t = times[indices].astype(float)
        local_temp = temperature[indices]
        local_rpm = rpm[indices]
        usable = np.isfinite(local_t) & np.isfinite(local_temp) & np.isfinite(local_rpm)
        if not usable.all():
            local_t = local_t[usable]
            local_temp = local_temp[usable]
            local_rpm = local_rpm[usable]
        if local_t.size < 2:
            continue
        steps = np.diff(local_t) / 1000.0
        keep = steps > 0
        if not keep.any():
            continue
        stretches += 1
        temperature_mid = (local_temp[:-1] + local_temp[1:]) / 2.0
        rpm_mid = (local_rpm[:-1] + local_rpm[1:]) / 2.0
        q_mid = np.clip(rpm_mid / RPM_PROXY_MAX, 0.0, 1.0)
        derivative = (local_temp[1:] - local_temp[:-1]) / steps
        rows.append(
            np.column_stack((q_mid[keep], -temperature_mid[keep], np.ones(np.count_nonzero(keep))))
        )
        targets.extend(derivative[keep].tolist())
    if not rows:
        return np.empty((0, 3)), np.empty(0), stretches
    return np.vstack(rows), np.asarray(targets, dtype=float), stretches


def fit_cooling(
    times_ms: np.ndarray,
    coolant_temp_c: np.ndarray,
    engine_speed_rpm: np.ndarray,
    *,
    max_gap_ms: float = 500.0,
    min_samples: int = 30,
    min_q_std: float = 0.01,
    min_temp_range_c: float = 0.5,
) -> CoolingFit:
    """Fit the per-session cooling parameters on gap-free stretches."""
    times = np.asarray(times_ms, dtype=float)
    temperature = np.asarray(coolant_temp_c, dtype=float)
    rpm = np.asarray(engine_speed_rpm, dtype=float)
    if times.ndim != 1 or not (times.size == temperature.size == rpm.size):
        raise ValueError("times, coolant temperature and RPM must be 1-D arrays of equal length")
    if times.size < 2:
        raise NotIdentifiableError("cooling model is not identifiable: fewer than two samples")

    order = np.argsort(times)
    times = times[order]
    temperature = temperature[order]
    rpm = rpm[order]

    design, target, stretches = _design(times, temperature, rpm, max_gap_ms)
    if design.shape[0] < min_samples:
        raise NotIdentifiableError(
            f"cooling model is not identifiable: {design.shape[0]} samples (minimum {min_samples})"
        )
    q = design[:, 0]
    if float(np.std(q)) < min_q_std:
        raise NotIdentifiableError(
            "cooling model is not identifiable: engine speed has too little variation"
        )
    finite_temperature = temperature[np.isfinite(temperature)]
    if (
        finite_temperature.size == 0
        or float(np.max(finite_temperature) - np.min(finite_temperature)) < min_temp_range_c
    ):
        raise NotIdentifiableError(
            "cooling model is not identifiable: coolant temperature has too little variation"
        )

    parameters, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        raise NotIdentifiableError("cooling model is not identifiable: rank-deficient fit")
    a, h, h_ambient = (float(value) for value in parameters)
    if not np.isfinite(h) or h <= 1e-9:
        raise NotIdentifiableError(
            "cooling model is not identifiable: heat-transfer gain is not positive"
        )
    ambient = h_ambient / h
    if not np.isfinite(ambient):
        raise NotIdentifiableError("cooling model is not identifiable: ambient is not finite")

    residual = target - design @ parameters
    total = float(np.sum((target - np.mean(target)) ** 2))
    if total == 0.0:
        quality = 1.0 if float(np.sum(residual**2)) == 0.0 else 0.0
    else:
        quality = 1.0 - float(np.sum(residual**2)) / total

    return CoolingFit(
        a=a,
        h=h,
        ambient_c=ambient,
        time_constant_s=1.0 / h,
        residual=residual,
        quality=quality,
        samples=int(design.shape[0]),
        stretches=stretches,
    )


def fit_cooling_online(
    times_ms: np.ndarray,
    coolant_temp_c: np.ndarray,
    engine_speed_rpm: np.ndarray,
    *,
    forgetting: float = 1.0,
    initial_covariance: float = 1e6,
    max_gap_ms: float = 500.0,
    min_samples: int = 30,
    min_q_std: float = 0.01,
    min_temp_range_c: float = 0.5,
) -> CoolingFit:
    """Online RLS variant of :func:`fit_cooling`; ``forgetting=1`` matches the batch fit."""
    times = np.asarray(times_ms, dtype=float)
    temperature = np.asarray(coolant_temp_c, dtype=float)
    rpm = np.asarray(engine_speed_rpm, dtype=float)
    if times.ndim != 1 or not (times.size == temperature.size == rpm.size):
        raise ValueError("times, coolant temperature and RPM must be 1-D arrays of equal length")
    if times.size < 2:
        raise NotIdentifiableError("cooling model is not identifiable: fewer than two samples")

    order = np.argsort(times)
    times = times[order]
    temperature = temperature[order]
    rpm = rpm[order]
    design, target, stretches = _design(times, temperature, rpm, max_gap_ms)
    if design.shape[0] < min_samples:
        raise NotIdentifiableError(
            f"cooling model is not identifiable: {design.shape[0]} samples (minimum {min_samples})"
        )
    if float(np.std(design[:, 0])) < min_q_std:
        raise NotIdentifiableError(
            "cooling model is not identifiable: engine speed has too little variation"
        )
    finite_temperature = temperature[np.isfinite(temperature)]
    if (
        finite_temperature.size == 0
        or float(np.max(finite_temperature) - np.min(finite_temperature)) < min_temp_range_c
    ):
        raise NotIdentifiableError(
            "cooling model is not identifiable: coolant temperature has too little variation"
        )

    estimator = RLS(
        3,
        forgetting=forgetting,
        initial_covariance=initial_covariance,
    )
    for indices in _stretch_indices(times, max_gap_ms):
        local_t = times[indices]
        local_temp = temperature[indices]
        local_rpm = rpm[indices]
        usable = np.isfinite(local_t) & np.isfinite(local_temp) & np.isfinite(local_rpm)
        if not usable.all():
            local_t = local_t[usable]
            local_temp = local_temp[usable]
            local_rpm = local_rpm[usable]
        if local_t.size < 2:
            continue
        steps = np.diff(local_t) / 1000.0
        keep = steps > 0
        temperature_mid = (local_temp[:-1] + local_temp[1:]) / 2.0
        rpm_mid = (local_rpm[:-1] + local_rpm[1:]) / 2.0
        q_mid = np.clip(rpm_mid / RPM_PROXY_MAX, 0.0, 1.0)
        derivative = (local_temp[1:] - local_temp[:-1]) / steps
        for q_value, temp_value, y_value, use in zip(
            q_mid, temperature_mid, derivative, keep, strict=True
        ):
            if use:
                estimator.update(np.array([q_value, -temp_value, 1.0]), y_value)

    a, h, h_ambient = (float(value) for value in estimator.theta)
    if not np.isfinite(h) or h <= 1e-9:
        raise NotIdentifiableError(
            "cooling model is not identifiable: heat-transfer gain is not positive"
        )
    ambient = h_ambient / h
    if not np.isfinite(ambient):
        raise NotIdentifiableError("cooling model is not identifiable: ambient is not finite")

    residual = target - design @ estimator.theta
    total = float(np.sum((target - np.mean(target)) ** 2))
    quality = 1.0 if total == 0.0 and float(np.sum(residual**2)) == 0.0 else 0.0
    if total > 0.0:
        quality = 1.0 - float(np.sum(residual**2)) / total
    return CoolingFit(
        a=a,
        h=h,
        ambient_c=ambient,
        time_constant_s=1.0 / h,
        residual=residual,
        quality=quality,
        samples=int(design.shape[0]),
        stretches=stretches,
    )


__all__ = ["CoolingFit", "NotIdentifiableError", "fit_cooling", "fit_cooling_online"]
