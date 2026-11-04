"""Session-based train/val/test splits (CLAUDE.md rule 1).

Nothing here splits rows: the unit is always a session, and the assignment is a stable
hash of ``(seed, session_id)`` so adding a session never moves an existing one.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Collection, Iterable, Mapping, Sequence

import pandas as pd

PARTS = ("train", "val", "test")
SPLIT_PARTS = (*PARTS, "test_holdout")


def _validate_fractions(fractions: Sequence[float]) -> tuple[float, float, float]:
    if len(fractions) != len(PARTS):
        raise ValueError(f"fractions must have {len(PARTS)} entries")
    values = tuple(float(value) for value in fractions)
    if any(value < 0 for value in values):
        raise ValueError("fractions must be non-negative")
    if abs(sum(values) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    return values


def _score(seed: int, session_id: str) -> float:
    digest = hashlib.sha256(f"{seed}:{session_id}".encode()).digest()
    return int.from_bytes(digest, "big") / 2**256


def assign_split(
    session_id: str, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 0
) -> str:
    """Return the split of one session, stable for a given seed."""
    train, val, _ = _validate_fractions(fractions)
    score = _score(seed, session_id)
    if score < train:
        return "train"
    if score < train + val:
        return "val"
    return "test"


def split_sessions(
    ids: Iterable[str],
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, list[str]]:
    """Assign every session id to exactly one split, preserving the input order."""
    _validate_fractions(fractions)
    result: dict[str, list[str]] = {part: [] for part in PARTS}
    for session_id in ids:
        result[assign_split(session_id, fractions, seed)].append(session_id)
    return result


def _operating_value(point: object, axis: str) -> float | None:
    """Read one axis from an ``OperatingPoint``, an axis mapping or a scalar."""
    values = getattr(point, "values", None)
    if values is None and isinstance(point, Mapping):
        values = point.get("values", point)
    if values is not None:
        if not hasattr(values, "get"):
            raise TypeError("operating point values must be a mapping")
        value = values.get(axis)
    else:
        value = point
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"operating point on axis {axis!r} is not numeric") from None
    return number if math.isfinite(number) else None


def split_by_band(
    points: Mapping[str, object],
    axis: str,
    band: tuple[float, float],
    *,
    seed: int = 0,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    healthy: Collection[str] | None = None,
) -> dict[str, list[str]]:
    """Hold out a closed operating-point band and split the rest by session.

    Every session whose ``axis`` value lies in ``[low, high]`` goes to
    ``test_holdout``. Sessions with no finite value on the axis are excluded and listed
    under ``excluded``; they are never assigned to train, val, test or the holdout. The
    remaining sessions use :func:`split_sessions` with the same seed.

    When ``healthy`` is given, train and val must each contain at least one healthy
    session; otherwise they must at least be non-empty. A band without any session or a
    band that empties train/val raises ``ValueError``.
    """
    if not axis:
        raise ValueError("axis must be a non-empty operating-point axis")
    low, high = (float(band[0]), float(band[1]))
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("band edges must be finite")
    if low > high:
        raise ValueError("band low edge must not exceed its high edge")
    _validate_fractions(fractions)

    held: list[str] = []
    remaining: list[str] = []
    excluded: list[str] = []
    for session_id, point in points.items():
        value = _operating_value(point, axis)
        if value is None:
            excluded.append(str(session_id))
        elif low <= value <= high:
            held.append(str(session_id))
        else:
            remaining.append(str(session_id))

    if not held:
        raise ValueError(f"holdout band [{low:g}, {high:g}] on {axis!r} has no session")
    result = split_sessions(remaining, fractions, seed)
    result["test_holdout"] = held
    result["excluded"] = excluded

    healthy_ids = set(healthy) if healthy is not None else None
    for part in ("train", "val"):
        if not result[part]:
            raise ValueError(f"holdout band leaves {part} without a session")
        if healthy_ids is not None and not healthy_ids.intersection(result[part]):
            raise ValueError(f"holdout band leaves {part} without a healthy session")

    assert_no_leakage(*(result[part] for part in SPLIT_PARTS))
    return result


def group_kfold(ids: Iterable[str], k: int, seed: int = 0) -> list[tuple[list[str], list[str]]]:
    """Return ``k`` (train_ids, test_ids) folds; each session is tested exactly once."""
    ordered = sorted(set(ids), key=lambda session_id: (_score(seed, session_id), session_id))
    if k < 2:
        raise ValueError("k must be at least 2")
    if k > len(ordered):
        raise ValueError("k cannot exceed the number of sessions")
    folds: list[tuple[list[str], list[str]]] = []
    for fold in range(k):
        test = [session_id for index, session_id in enumerate(ordered) if index % k == fold]
        test_set = set(test)
        train = [session_id for session_id in ordered if session_id not in test_set]
        folds.append((train, test))
    return folds


def _part_ids(part: object) -> set[str]:
    if isinstance(part, pd.DataFrame):
        if "session_id" not in part.columns:
            raise ValueError("a frame part must have a session_id column")
        return {str(value) for value in part["session_id"]}
    if isinstance(part, pd.Series):
        return {str(value) for value in part}
    if isinstance(part, str):
        return {part}
    if isinstance(part, Iterable):
        return {str(value) for value in part}
    raise TypeError(f"cannot read session ids from {type(part).__name__}")


def assert_no_leakage(*parts: object) -> None:
    """Raise if one session appears in more than one part."""
    seen: dict[str, int] = {}
    for index, part in enumerate(parts):
        for session_id in _part_ids(part):
            if session_id in seen:
                raise ValueError(
                    f"session {session_id!r} leaks between parts {seen[session_id]} and {index}"
                )
            seen[session_id] = index
