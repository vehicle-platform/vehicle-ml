"""Evaluation harness: end-to-end, split isolation and CLI errors (ML-17)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.physics_synthetic import PhysicsDataset, make_physics_data_dir
from vehicle_ml.cli import main
from vehicle_ml.evaluate import MODEL_NAMES, SYNTHETIC_NOTE, evaluate

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "physics_cl250.toml"


def _dataset(tmp_path: Path) -> tuple[Path, PhysicsDataset]:
    root = tmp_path / "data"
    dataset = make_physics_data_dir(root)
    dataset.write_faults(root / "faults.json")
    return root, dataset


def _expected_fault_kinds(dataset: PhysicsDataset, split: dict[str, list[str]]) -> set[str]:
    test_ids = set(split["test"])
    return {
        str(truth.kind)
        for session_id, truth in dataset.truths().items()
        if session_id in test_ids and truth.kind is not None
    }


def test_evaluate_writes_reports_and_per_fault_metrics(tmp_path: Path) -> None:
    root, dataset = _dataset(tmp_path)
    out = tmp_path / "out"
    result = evaluate(root, out, CONFIG)

    assert result.json_path.exists()
    assert result.markdown_path.exists()
    data = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert data["seed"] == 0
    assert data["defs_version"].startswith("v0.20.0")
    assert data["note"] == SYNTHETIC_NOTE
    assert SYNTHETIC_NOTE in result.markdown_path.read_text(encoding="utf-8")
    assert set(data["thresholds"]) == set(MODEL_NAMES)
    assert set(data["models"]) == set(MODEL_NAMES)

    expected = _expected_fault_kinds(dataset, data["split"])
    for model in MODEL_NAMES:
        test = data["models"][model]["test"]
        assert {"episodes", "hours", "per_hour"} <= set(test["false_alarms"])
        assert set(test["faults"]) == expected
        for entry in test["faults"].values():
            assert {"sessions", "detected", "missed", "median_delay_ms", "modalities"} <= set(entry)
            assert entry["detected"] + entry["missed"] == entry["sessions"]
    assert data["models"]["physics"]["test"]["faults"]


def test_no_test_session_is_used_for_fit_or_thresholds(tmp_path: Path) -> None:
    root, dataset = _dataset(tmp_path)
    result = evaluate(root, tmp_path / "out", CONFIG)
    data = result.data

    test_ids = set(data["split"]["test"])
    assert not test_ids & set(data["fit_sessions"])
    assert not test_ids & set(data["threshold_sessions"])
    assert set(data["threshold_sessions"]) <= set(data["split"]["val"])
    faulty = {
        session_id for session_id, truth in dataset.truths().items() if truth.kind is not None
    }
    assert not faulty & set(data["threshold_sessions"])


def test_without_faults_json_reports_false_alarms_only(tmp_path: Path) -> None:
    root = tmp_path / "data"
    make_physics_data_dir(root)
    result = evaluate(root, tmp_path / "out", CONFIG)

    assert result.data["faults_present"] is False
    assert all(not result.data["models"][model]["test"]["faults"] for model in MODEL_NAMES)
    assert "No fault ground truth" in result.markdown_path.read_text(encoding="utf-8")


def test_cli_evaluate_end_to_end(tmp_path: Path) -> None:
    root, _ = _dataset(tmp_path)
    out = tmp_path / "cli-out"
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
            ]
        )
        == 0
    )
    assert (out / "evaluation.json").exists()
    assert (out / "evaluation.md").exists()


def test_cli_bad_arguments_exit_non_zero(tmp_path: Path) -> None:
    root, _ = _dataset(tmp_path)
    out = tmp_path / "bad-out"
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
                "--window-ms",
                "0",
            ]
        )
        == 2
    )
    with pytest.raises(SystemExit) as excinfo:
        main(["evaluate", "--data-dir", str(root), "--out", str(out)])
    assert excinfo.value.code != 0
