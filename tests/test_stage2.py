"""Stage-2 synthetic evaluation: holdout band, hybrid weighting and default regression."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tests.physics_synthetic import (
    EXPECTED_MODALITY,
    FAULT_KINDS,
    make_generalisation_data_dir,
    make_physics_data_dir,
)
from vehicle_ml.cli import main
from vehicle_ml.evaluate import (
    HYBRID_MODEL,
    SYNTHETIC_NOTE,
    _tune_hybrid,
    evaluate,
    evaluate_seeds,
)
from vehicle_ml.hybrid import HYBRID_WEIGHTS
from vehicle_ml.split import SPLIT_PARTS, assert_no_leakage

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "physics_cl250.toml"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "evaluation_stage1_default.json"
CAN_FAULTS = tuple(kind for kind in FAULT_KINDS if EXPECTED_MODALITY[kind] == "can")


def _stage2_dataset(tmp_path: Path):
    root = tmp_path / "data"
    dataset = make_generalisation_data_dir(root)
    dataset.write_faults(root / "faults.json")
    return root, dataset


def test_default_evaluation_json_matches_the_stage1_fixture(tmp_path: Path) -> None:
    root = tmp_path / "data"
    dataset = make_physics_data_dir(root)
    dataset.write_faults(root / "faults.json")

    result = evaluate(root, tmp_path / "out", CONFIG)
    assert result.json_path.read_text(encoding="utf-8") == FIXTURE.read_text(encoding="utf-8")


def test_holdout_hybrid_cli_end_to_end(tmp_path: Path) -> None:
    root, dataset = _stage2_dataset(tmp_path)
    out = tmp_path / "out"
    assert (
        main(
            [
                "evaluate",
                "--data-dir",
                str(root),
                "--out",
                str(out),
                "--config",
                str(CONFIG),
                "--holdout",
                "temperature:30:45",
                "--hybrid",
            ]
        )
        == 0
    )
    data = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
    markdown = (out / "evaluation.md").read_text(encoding="utf-8")

    assert set(data["split"]) == set(SPLIT_PARTS)
    assert data["note"] == SYNTHETIC_NOTE
    assert data["generalisation"]["axis"] == "temperature"
    assert data["generalisation"]["band"] == [30.0, 45.0]
    assert data["generalisation"]["proxy_role"] == "coolant_temp"
    assert data["generalisation"]["threshold_source"].startswith("validation healthy")
    assert data["generalisation"]["excluded_sessions"] == []
    assert set(data["models"]) == {"physics", "baseline_pca", "baseline_z", HYBRID_MODEL}
    assert set(data["generalisation"]["models"]) == set(data["models"])
    assert data["hybrid"]["tuned"] is True
    assert data["hybrid"]["weight"] in HYBRID_WEIGHTS
    assert data["hybrid"]["tuning_split"] == "val"
    assert data["hybrid"]["threshold"] == data["thresholds"][HYBRID_MODEL]
    assert data["hybrid"]["unscored_windows"] == {
        "train": 0,
        "val": 0,
        "test": 0,
        "test_holdout": 0,
    }
    assert data["hybrid"]["physics_unscored_windows"] == data["hybrid"]["unscored_windows"]
    assert set(data["split"]["test_holdout"]) == set(data["generalisation"]["holdout_sessions"])
    assert_no_leakage(*(data["split"][part] for part in SPLIT_PARTS))

    truths = dataset.truths()
    for part in SPLIT_PARTS:
        part_ids = set(data["split"][part])
        assert any(truths[session_id].kind is None for session_id in part_ids)
        present = {truths[session_id].kind for session_id in part_ids}
        assert set(CAN_FAULTS) <= present

    for model in data["models"]:
        test = data["models"][model]["test"]
        held = data["generalisation"]["models"][model]["test_holdout"]
        for result in (test, held):
            assert {"episodes", "hours", "per_hour"} <= set(result["false_alarms"])
            for entry in result["faults"].values():
                assert (
                    entry["detected"] + entry["missed"] + entry["not_evaluated"]
                    == entry["sessions"]
                )

    assert "## Generalisation (synthetic)" in markdown
    assert "## Hybrid weighting (synthetic)" in markdown
    assert "## Hybrid term dominance" in markdown
    assert SYNTHETIC_NOTE + "." in markdown
    assert "axis proxy role: coolant_temp" in markdown
    assert "w=0 is never allowed" in markdown


def test_cli_rejects_a_malformed_holdout(tmp_path: Path) -> None:
    assert (
        main(
            [
                "evaluate",
                "--data-dir",
                str(tmp_path / "missing"),
                "--out",
                str(tmp_path / "out"),
                "--config",
                str(CONFIG),
                "--holdout",
                "temperature:30",
            ]
        )
        == 2
    )


def test_holdout_seeds_share_the_band_but_change_the_remaining_split(tmp_path: Path) -> None:
    root, dataset = _stage2_dataset(tmp_path)
    result = evaluate_seeds(
        root,
        tmp_path / "seeds",
        CONFIG,
        [0, 1],
        holdout_axis="temperature",
        holdout_band=(30.0, 45.0),
        hybrid=True,
    )
    per_seed = result.data["per_seed"]
    assert set(per_seed) == {"0", "1"}
    assert (
        per_seed["0"]["generalisation"]["holdout_sessions"]
        == per_seed["1"]["generalisation"]["holdout_sessions"]
    )
    assert per_seed["0"]["split"]["train"] != per_seed["1"]["split"]["train"]
    assert set(per_seed["0"]["split"]["test_holdout"]) == set(
        per_seed["1"]["split"]["test_holdout"]
    )
    assert "physics.holdout.false_alarms_per_hour" in result.data["aggregate"]
    assert "hybrid.holdout.false_alarms_per_hour" in result.data["aggregate"]
    for seed in ("0", "1"):
        assert per_seed[seed]["generalisation"]["note"] == SYNTHETIC_NOTE


def _tuning_frame() -> tuple[pd.DataFrame, dict[float, pd.DataFrame]]:
    sessions = ["val-h", "val-h", "val-f", "val-f", "val-f", "val-f", "val-f"]
    starts = [0, 5000, 0, 5000, 10000, 15000, 20000]
    scores = pd.DataFrame({"session_id": sessions, "start_ms": starts})
    values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]
    candidates = {weight: pd.DataFrame({"hybrid": values}) for weight in HYBRID_WEIGHTS}
    return scores, candidates


def test_tuning_does_not_read_the_test_split() -> None:
    scores, candidates = _tuning_frame()
    faults = {"val-f": {"fault": "k", "onset_ms": 20000}}
    first = _tune_hybrid(candidates, scores, ["val-h"], ["val-h", "val-f"], faults, 1.0, 5000)

    test_rows = 5
    with_test = pd.concat(
        [
            scores,
            pd.DataFrame(
                {
                    "session_id": ["test-f"] * test_rows,
                    "start_ms": list(range(0, test_rows * 5000, 5000)),
                }
            ),
        ],
        ignore_index=True,
    )
    poisoned = {
        weight: pd.concat(
            [
                candidates[weight],
                pd.DataFrame({"hybrid": [1e9] * test_rows}, index=range(test_rows)),
            ],
            ignore_index=True,
        )
        for weight in HYBRID_WEIGHTS
    }
    test_faults = {
        **faults,
        "test-f": {"fault": "k", "onset_ms": 0},
    }
    second = _tune_hybrid(
        poisoned,
        with_test,
        ["val-h"],
        ["val-h", "val-f"],
        test_faults,
        1.0,
        5000,
    )
    assert first[0] == second[0]
    assert first[1] is True
    assert second[1] is True


def test_hybrid_falls_back_to_not_tuned_without_validation_faults() -> None:
    scores, candidates = _tuning_frame()
    only_test_fault = {"test-f": {"fault": "k", "onset_ms": 20000}}
    weight, tuned, tuning, threshold, reason = _tune_hybrid(
        candidates,
        scores,
        ["val-h"],
        ["val-h", "val-f"],
        only_test_fault,
        1.0,
        5000,
    )
    assert weight == 0.5
    assert tuned is False
    assert "no validation fault session" in reason
    assert threshold is not None
    assert {entry["weight"] for entry in tuning} == set(HYBRID_WEIGHTS)


def test_unknown_holdout_axis_is_refused(tmp_path: Path) -> None:
    root, _ = _stage2_dataset(tmp_path)
    with pytest.raises(ValueError, match="not in the role config"):
        evaluate(
            root,
            tmp_path / "out",
            CONFIG,
            holdout_axis="humidity",
            holdout_band=(0.0, 1.0),
        )
