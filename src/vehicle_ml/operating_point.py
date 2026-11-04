"""Data-derived operating points for the stage-2 holdout split (D-083 item 3).

The role map has no ambient-temperature signal, so the two operating-point axes are
**proxies** measured from the recorded session itself:

- ``temperature``: the median coolant temperature, used as an ambient-temperature proxy;
- ``load``: the median engine speed, used as a load proxy.

The median is taken over the finite samples of each configured role. A session without a
valid sample on an axis gets ``None`` for that axis and is reported by the caller; it is
never filled from the synthetic spec or from ``faults.json``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vehicle_ml.data.store import SessionStore
from vehicle_ml.physics.config import RoleMap
from vehicle_ml.windows import align_frames


@dataclass(frozen=True)
class OperatingPoint:
    """Median operating-point proxy values and valid sample counts for one session."""

    session_id: str
    values: Mapping[str, float | None]
    samples: Mapping[str, int]

    def value(self, axis: str) -> float | None:
        return self.values.get(axis)


def load_role_frame(
    store: SessionStore,
    session_id: str,
    roles: RoleMap,
    max_gap_ms: float,
) -> tuple[pd.DataFrame | None, list[str]]:
    """Load one session's telemetry/derived roles on its telemetry time base."""
    telemetry = store.load(session_id, "telemetry")
    if telemetry is None:
        return None, list(roles.roles)

    frame = telemetry[["session_id", "device_time_ms"]].copy()
    missing: list[str] = []
    for role in roles.roles:
        if roles.kind(role) != "did":
            continue
        column = roles.column(role)
        if column in telemetry.columns:
            frame[role] = telemetry[column].to_numpy()
        else:
            missing.append(role)
            frame[role] = np.nan

    derived = store.load(session_id, "derived")
    derived_roles = [role for role in roles.roles if roles.kind(role) == "derived"]
    if derived is None or derived.empty:
        missing.extend(derived_roles)
        for role in derived_roles:
            frame[role] = np.nan
        return frame, missing

    wide = derived.pivot_table(
        index="device_time_ms", columns="signal", values="value", aggfunc="last"
    ).reset_index()
    wide["session_id"] = session_id
    available = [role for role in derived_roles if roles.column(role) in wide.columns]
    unavailable = [role for role in derived_roles if role not in available]
    missing.extend(unavailable)
    for role in unavailable:
        frame[role] = np.nan
    if available:
        aligned = align_frames(frame, wide, available, tolerance_ms=max_gap_ms)
        for role in available:
            frame[role] = aligned[role].to_numpy()
    return frame, missing


def session_operating_points(
    store: SessionStore,
    session_ids: Sequence[str],
    roles: RoleMap,
    *,
    max_gap_ms: float = 500,
    frames: Mapping[str, pd.DataFrame | None] | None = None,
) -> dict[str, OperatingPoint]:
    """Return the median of each configured operating-point role over valid samples.

    ``frames`` may supply already-loaded role frames from :func:`load_role_frame`; when
    omitted they are loaded from ``store``. Missing role columns or a session with no
    finite sample leave that axis ``None`` and its sample count at zero.
    """
    if not roles.operating_point:
        raise ValueError("role config has no [operating_point] table")
    result: dict[str, OperatingPoint] = {}
    for session_id in session_ids:
        frame = frames.get(session_id) if frames is not None else None
        if frames is None or session_id not in frames:
            frame, _ = load_role_frame(store, session_id, roles, max_gap_ms)
        values: dict[str, float | None] = {}
        samples: dict[str, int] = {}
        for axis, role in roles.operating_point.items():
            if frame is None or role not in frame.columns:
                values[axis] = None
                samples[axis] = 0
                continue
            column = frame[role].to_numpy(dtype=float)
            finite = column[np.isfinite(column)]
            samples[axis] = int(finite.size)
            values[axis] = float(np.median(finite)) if finite.size else None
        result[str(session_id)] = OperatingPoint(
            session_id=str(session_id), values=values, samples=samples
        )
    return result


__all__ = ["OperatingPoint", "load_role_frame", "session_operating_points"]
