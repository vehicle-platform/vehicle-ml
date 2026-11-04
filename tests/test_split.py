"""Session split: determinism, stability, fractions, folds and leakage."""

from __future__ import annotations

import pandas as pd
import pytest

from vehicle_ml.operating_point import OperatingPoint
from vehicle_ml.split import (
    PARTS,
    SPLIT_PARTS,
    assert_no_leakage,
    assign_split,
    group_kfold,
    split_by_band,
    split_sessions,
)

IDS = [f"session-{index}" for index in range(1000)]


def test_assignment_is_deterministic() -> None:
    first = {session_id: assign_split(session_id, seed=7) for session_id in IDS}
    second = {session_id: assign_split(session_id, seed=7) for session_id in IDS}
    assert first == second


def test_adding_a_session_does_not_move_existing_ones() -> None:
    before = split_sessions(IDS, seed=3)
    after = split_sessions([*IDS, "session-new"], seed=3)
    for part in PARTS:
        kept = [session_id for session_id in after[part] if session_id in before[part]]
        assert before[part] == kept


def test_fractions_are_within_tolerance_on_many_ids() -> None:
    split = split_sessions(IDS, seed=0)
    shares = {part: len(ids) / len(IDS) for part, ids in split.items()}
    assert shares["train"] == pytest.approx(0.7, abs=0.03)
    assert shares["val"] == pytest.approx(0.15, abs=0.03)
    assert shares["test"] == pytest.approx(0.15, abs=0.03)
    assert sum(len(ids) for ids in split.values()) == len(IDS)


def test_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        assign_split("a", (0.5, 0.2, 0.2))
    with pytest.raises(ValueError, match="sum to 1"):
        split_sessions(IDS, (0.5, 0.2, 0.2))


def test_negative_fractions_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        assign_split("a", (1.1, -0.05, -0.05))


def test_group_kfold_covers_every_session_once() -> None:
    folds = group_kfold(IDS, k=5, seed=0)
    tested = [session_id for _, test in folds for session_id in test]
    assert len(tested) == len(set(tested)) == len(IDS)
    for train, test in folds:
        assert not set(train) & set(test)
        assert len(train) + len(test) == len(IDS)


def test_group_kfold_is_deterministic_and_rejects_bad_k() -> None:
    assert group_kfold(IDS, k=3, seed=1) == group_kfold(IDS, k=3, seed=1)
    with pytest.raises(ValueError, match="at least 2"):
        group_kfold(IDS, k=1)
    with pytest.raises(ValueError, match="cannot exceed"):
        group_kfold(["a", "b"], k=3)


def test_leakage_is_detected_across_frames_and_lists() -> None:
    train = pd.DataFrame({"session_id": ["a", "b"], "value": [1, 2]})
    test = pd.DataFrame({"session_id": ["c", "b"], "value": [3, 4]})
    with pytest.raises(ValueError, match="leaks"):
        assert_no_leakage(train, test)
    with pytest.raises(ValueError, match="leaks"):
        assert_no_leakage(["a", "b"], ["c", "a"])


def test_clean_parts_pass() -> None:
    assert_no_leakage(["a", "b"], ["c"], pd.DataFrame({"session_id": ["d"]}))


def _points(count: int) -> dict[str, OperatingPoint]:
    return {
        f"session-{index:02d}": OperatingPoint(
            session_id=f"session-{index:02d}",
            values={"temperature": float(index)},
            samples={"temperature": 1},
        )
        for index in range(count)
    }


def test_split_by_band_is_deterministic_and_holds_out_the_closed_band() -> None:
    points = _points(30)
    first = split_by_band(points, "temperature", (10.0, 19.0), seed=4)
    second = split_by_band(points, "temperature", (10.0, 19.0), seed=4)

    assert first == second
    assert first["test_holdout"] == [f"session-{index:02d}" for index in range(10, 20)]
    assert first["excluded"] == []
    assert sum(len(first[part]) for part in SPLIT_PARTS) == 30
    assert_no_leakage(*(first[part] for part in SPLIT_PARTS))
    for part in ("train", "val", "test"):
        values = [points[session_id].value("temperature") for session_id in first[part]]
        assert all(value is not None and not 10.0 <= value <= 19.0 for value in values)


def test_split_by_band_edges_are_inclusive_and_none_is_excluded() -> None:
    points = _points(12)
    points["session-none"] = OperatingPoint(
        session_id="session-none", values={"temperature": None}, samples={"temperature": 0}
    )
    split = split_by_band(points, "temperature", (3.0, 5.0), seed=0)

    assert set(split["test_holdout"]) == {"session-03", "session-04", "session-05"}
    assert split["excluded"] == ["session-none"]
    assert "session-none" not in {sid for part in SPLIT_PARTS for sid in split[part]}
    assert_no_leakage(*(split[part] for part in SPLIT_PARTS))


def test_split_by_band_seed_changes_only_the_remaining_split() -> None:
    points = _points(40)
    first = split_by_band(points, "temperature", (10.0, 19.0), seed=0)
    second = split_by_band(points, "temperature", (10.0, 19.0), seed=1)

    assert first["test_holdout"] == second["test_holdout"]
    assert first["train"] != second["train"]


def test_split_by_band_refuses_an_empty_holdout_or_an_empty_part() -> None:
    points = _points(10)
    with pytest.raises(ValueError, match="has no session"):
        split_by_band(points, "temperature", (100.0, 200.0), seed=0)
    with pytest.raises(ValueError, match="train without a session"):
        split_by_band(points, "temperature", (0.0, 20.0), seed=0)
    with pytest.raises(ValueError, match="low edge"):
        split_by_band(points, "temperature", (5.0, 1.0), seed=0)


def test_split_by_band_refuses_train_or_val_without_a_healthy_session() -> None:
    points = _points(40)
    held = {"session-00"}
    remaining = [session_id for session_id in points if session_id not in held]
    base = split_sessions(remaining, seed=3)
    assert base["train"] and base["val"]

    with pytest.raises(ValueError, match="val without a healthy session"):
        split_by_band(
            points,
            "temperature",
            (0.0, 0.0),
            seed=3,
            healthy=set(base["train"]),
        )
    with pytest.raises(ValueError, match="train without a healthy session"):
        split_by_band(
            points,
            "temperature",
            (0.0, 0.0),
            seed=3,
            healthy=set(base["val"]),
        )
