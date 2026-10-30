"""Session-based train/val/test splits (CLAUDE.md rule 1).

Nothing here splits rows: the unit is always a session, and the assignment is a stable
hash of ``(seed, session_id)`` so adding a session never moves an existing one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

import pandas as pd

PARTS = ("train", "val", "test")


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
