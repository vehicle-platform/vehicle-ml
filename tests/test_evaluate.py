"""Evaluation harness: end-to-end, split isolation and CLI errors (ML-17)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.physics_synthetic import IMU_PERIOD_MS, PhysicsDataset, make_physics_data_dir
from vehicle_ml.cli import main
from vehicle_ml.data.sources import imu_columns
from vehicle_ml.evaluate import (
    MISS_NOTE,
    MODEL_NAMES,
    SYNTHETIC_NOTE,
    _imu_bands,
    aggregate_metrics,
    evaluate,
    evaluate_seeds,
    pca_alarm_modality,
    z_alarm_modality,
)
from vehicle_ml.physics.config import PHYSICS_GROUPS

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


def test_alarm_modality_of_a_hand_built_pca_residual() -> None:
    modalities = ["can", "can", "imu", "can", "imu"]
    # The IMU features hold the largest share (5.1 against 0.6 on CAN).
    assert pca_alarm_modality(np.array([0.1, 0.2, 5.0, 0.3, 0.1]), modalities) == "imu"
    # Per-modality sums, not the single largest entry: three CAN features of 2.0 beat one 5.0.
    assert pca_alarm_modality(np.array([2.0, 2.0, 5.0, 2.0, 0.0]), modalities) == "can"
    assert pca_alarm_modality(np.array([0.1, np.nan, 5.0, 0.3, 0.1]), modalities) == ""
    with pytest.raises(ValueError, match="one modality per feature"):
        pca_alarm_modality(np.array([1.0, 2.0]), modalities)


def test_alarm_modality_of_a_hand_built_z_vector() -> None:
    modalities = ["can", "can", "imu", "can", "imu"]
    # The single largest |z| decides, even though CAN holds more total z here.
    assert z_alarm_modality(np.array([3.0, 3.0, 6.0, 3.0, 1.0]), modalities) == "imu"
    assert z_alarm_modality(np.array([1.0, 9.0, 6.0, 3.0, 1.0]), modalities) == "can"
    assert z_alarm_modality(np.array([1.0, -9.0, 6.0, 3.0, 1.0]), modalities) == "can"
    assert z_alarm_modality(np.array([1.0, np.nan, 6.0, 3.0, 1.0]), modalities) == ""


def test_imu_bands_cover_zero_to_nyquist_for_every_imu_column() -> None:
    bands = _imu_bands()
    assert set(bands) == set(imu_columns())
    for edges in bands.values():
        assert edges[0][0] == 0.0
        assert edges[-1][1] == 500.0 / IMU_PERIOD_MS
        assert all(low < high for low, high in edges)


def test_attribution_per_modality_and_group_on_the_weak_can_data(tmp_path: Path) -> None:
    root, dataset = _dataset(tmp_path)
    data = evaluate(root, tmp_path / "out", CONFIG).data

    truths = dataset.truths()
    kinds = {
        truths[session_id].kind
        for session_id in data["split"]["test"]
        if truths[session_id].kind is not None
    }
    assert "wheel_vibration" in kinds

    for model in MODEL_NAMES:
        for kind, entry in data["models"][model]["test"]["faults"].items():
            expected = "imu" if kind == "wheel_vibration" else "can"
            assert entry["expected_modality"] == expected
            assert set(entry["modalities"]) <= {"can", "imu", "other"}
            assert sum(entry["first_alarm_modalities"].values()) == entry["detected"]
            matches = entry["first_alarm_matches_expected"]
            assert matches == entry["first_alarm_modalities"].get(expected, 0)

    # The physics estimators read CAN only: they can never attribute to the IMU.
    for entry in data["models"]["physics"]["test"]["faults"].values():
        assert set(entry["modalities"]) <= {"can"}
        assert set(entry["first_alarm_modalities"]) <= {"can"}
        assert set(entry["groups"]) <= set(PHYSICS_GROUPS)
        assert sum(entry["groups"].values()) == entry["detected"]
    for model in ("baseline_pca", "baseline_z"):
        for entry in data["models"][model]["test"]["faults"].values():
            assert entry["groups"] == {}


def test_report_has_the_per_modality_table_and_the_synthetic_sentence(tmp_path: Path) -> None:
    root, _ = _dataset(tmp_path)
    result = evaluate(root, tmp_path / "out", CONFIG)
    text = result.markdown_path.read_text(encoding="utf-8")

    assert SYNTHETIC_NOTE + "." in text
    assert "alarm windows by modality" in text
    assert "first alarm modality" in text
    assert "first alarm = expected" in text
    assert MISS_NOTE in text
    for model in MODEL_NAMES:
        assert f"| {model} | wheel_vibration | imu |" in text
    assert "features_pca" not in text
    assert "features_robust_z" not in text


def test_fault_without_expected_modality_is_reported_as_unknown(tmp_path: Path) -> None:
    root, _ = _dataset(tmp_path)
    payload = json.loads((root / "faults.json").read_text(encoding="utf-8"))
    for entry in payload["sessions"].values():
        del entry["expected_modality"]
    (root / "faults.json").write_text(json.dumps(payload), encoding="utf-8")

    result = evaluate(root, tmp_path / "out", CONFIG)
    faults = result.data["models"]["physics"]["test"]["faults"]
    assert all(entry["expected_modality"] is None for entry in faults.values())
    assert all(entry["first_alarm_matches_expected"] is None for entry in faults.values())
    assert "unknown" in result.markdown_path.read_text(encoding="utf-8")


def test_unknown_expected_modality_is_rejected(tmp_path: Path) -> None:
    root, _ = _dataset(tmp_path)
    payload = json.loads((root / "faults.json").read_text(encoding="utf-8"))
    next(iter(payload["sessions"].values()))["expected_modality"] = "lidar"
    (root / "faults.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="expected_modality"):
        evaluate(root, tmp_path / "out", CONFIG)


def test_seeds_write_per_seed_entries_and_an_aggregate(tmp_path: Path) -> None:
    root, _ = _dataset(tmp_path)
    out = tmp_path / "multi"
    result = evaluate_seeds(root, out, CONFIG, [0, 1])

    data = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert data["seeds"] == [0, 1]
    assert set(data["per_seed"]) == {"0", "1"}
    assert data["per_seed"]["0"]["seed"] == 0
    assert data["per_seed"]["1"]["seed"] == 1
    # A seed changes only the split, never the data: both runs see the same sessions.
    sessions = [sum(entry["split"].values(), []) for entry in data["per_seed"].values()]
    assert sorted(sessions[0]) == sorted(sessions[1])
    assert data["per_seed"]["0"]["split"] != data["per_seed"]["1"]["split"]
    for seed in (0, 1):
        assert (out / f"seed-{seed}" / "evaluation.json").exists()
        assert (out / f"seed-{seed}" / "evaluation.md").exists()

    aggregate = data["aggregate"]
    assert "physics.false_alarms_per_hour" in aggregate
    for stats in aggregate.values():
        assert stats["min"] <= stats["mean"] <= stats["max"]
        assert 1 <= stats["n_seeds"] <= 2
    text = result.markdown_path.read_text(encoding="utf-8")
    assert SYNTHETIC_NOTE + "." in text
    assert "| metric | seeds | mean | min | max |" in text

    # The single-seed form is the same evaluation as the matching per-seed entry.
    single = evaluate(root, tmp_path / "single", CONFIG, seed=1)
    assert single.data == data["per_seed"]["1"]


def test_aggregate_metrics_uses_only_seeds_that_have_the_fault_kind() -> None:
    def entry(per_hour: float, faults: dict[str, dict[str, object]]) -> dict[str, object]:
        return {
            "models": {
                "physics": {"test": {"false_alarms": {"per_hour": per_hour}, "faults": faults}}
            }
        }

    fault = {
        "sessions": 2,
        "detected": 1,
        "median_delay_ms": 5000.0,
        "first_alarm_matches_expected": 1,
    }
    aggregate = aggregate_metrics({0: entry(1.0, {"plateau_sag": fault}), 1: entry(3.0, {})})
    assert aggregate["physics.false_alarms_per_hour"] == {
        "n_seeds": 2,
        "mean": 2.0,
        "min": 1.0,
        "max": 3.0,
    }
    assert aggregate["physics.plateau_sag.detection_rate"]["n_seeds"] == 1
    assert aggregate["physics.plateau_sag.detection_rate"]["mean"] == 0.5


def test_cli_seeds_end_to_end(tmp_path: Path) -> None:
    root, _ = _dataset(tmp_path)
    out = tmp_path / "cli-seeds"
    assert (
        main(
            ["evaluate", "--data-dir", str(root), "--out", str(out), "--config", str(CONFIG)]
            + ["--seeds", "0,1"]
        )
        == 0
    )
    assert (out / "seed-0" / "evaluation.json").exists()
    assert (out / "seed-1" / "evaluation.json").exists()
    assert (out / "evaluation.json").exists()


@pytest.mark.parametrize("bad", ["", "0,,1", "a,b", "0,0", "1.5"])
def test_cli_bad_seed_list_exits_non_zero(tmp_path: Path, bad: str) -> None:
    root, _ = _dataset(tmp_path)
    out = tmp_path / "bad-seeds"
    code = main(
        ["evaluate", "--data-dir", str(root), "--out", str(out), "--config", str(CONFIG)]
        + ["--seeds", bad]
    )
    assert code == 2
    assert not (out / "evaluation.json").exists()


def test_cli_rejects_seed_together_with_seeds(tmp_path: Path) -> None:
    root, _ = _dataset(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        main(
            ["evaluate", "--data-dir", str(root), "--out", str(tmp_path / "o")]
            + ["--config", str(CONFIG), "--seed", "1", "--seeds", "0,1"]
        )
    assert excinfo.value.code != 0
