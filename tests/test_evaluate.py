"""Evaluation harness: end-to-end, split isolation and CLI errors (ML-17)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.physics_synthetic import (
    DEFAULT_PHYSICS_SESSIONS,
    IMU_PERIOD_MS,
    PhysicsDataset,
    make_physics_data_dir,
)
from vehicle_ml.baseline import BaselineModel
from vehicle_ml.cli import main
from vehicle_ml.data.sources import imu_columns
from vehicle_ml.data.store import SessionStore
from vehicle_ml.dataset import build_dataset, write_dataset
from vehicle_ml.evaluate import (
    MISS_NOTE,
    MODEL_NAMES,
    NOT_CALIBRATED,
    SYNTHETIC_NOTE,
    UNATTRIBUTED,
    _false_alarms,
    _fault_results,
    _imu_bands,
    aggregate_attribution,
    aggregate_metrics,
    evaluate,
    evaluate_seeds,
    pca_alarm_modality,
    pca_training_means,
    z_alarm_modality,
)
from vehicle_ml.normalize import Standardizer
from vehicle_ml.physics.config import PHYSICS_GROUPS

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "physics_cl250.toml"


def _dataset(tmp_path: Path) -> tuple[Path, PhysicsDataset]:
    root = tmp_path / "data"
    dataset = make_physics_data_dir(root)
    dataset.write_faults(root / "faults.json")
    return root, dataset


def _mixed_imu_dataset(tmp_path: Path) -> tuple[Path, PhysicsDataset]:
    missing = {"phys-00", "phys-05", "phys-10", "phys-27"}
    specs = tuple(
        replace(spec, has_imu=False) if spec.session_id in missing else spec
        for spec in DEFAULT_PHYSICS_SESSIONS
    )
    root = tmp_path / "mixed"
    dataset = make_physics_data_dir(root, specs)
    dataset.write_faults(root / "faults.json")
    return root, dataset


def _without_imu(tmp_path: Path, part: str, name: str) -> tuple[Path, list[str]]:
    """A dataset where every healthy session of ``part`` (seed 0 split) has no IMU."""
    probe_root, probe = _dataset(tmp_path / "probe")
    split = build_dataset(probe_root, tmp_path / "probe-out", seed=0).split
    truths = probe.truths()
    chosen = [sid for sid in split[part] if truths[sid].kind is None]
    assert chosen
    specs = tuple(
        replace(spec, has_imu=False) if spec.session_id in chosen else spec
        for spec in DEFAULT_PHYSICS_SESSIONS
    )
    root = tmp_path / name
    make_physics_data_dir(root, specs).write_faults(root / "faults.json")
    return root, chosen


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
    assert set(data["pca_modality_training_mean_sq"]) == {"can", "imu"}
    assert all(
        mean is None or mean > 0.0 for mean in data["pca_modality_training_mean_sq"].values()
    )

    expected = _expected_fault_kinds(dataset, data["split"])
    for model in MODEL_NAMES:
        test = data["models"][model]["test"]
        assert {"episodes", "hours", "per_hour"} <= set(test["false_alarms"])
        assert set(test["faults"]) == expected
        for entry in test["faults"].values():
            assert {
                "sessions",
                "detected",
                "missed",
                "not_evaluated",
                "median_delay_ms",
                "modalities",
            } <= set(entry)
            assert entry["detected"] + entry["missed"] + entry["not_evaluated"] == entry["sessions"]
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
    means = {"can": 1.0, "imu": 1.0}
    # The IMU features hold the largest normalised share (5.1 against 0.6 on CAN).
    assert pca_alarm_modality(np.array([0.1, 0.2, 5.0, 0.3, 0.1]), modalities, means) == "imu"
    # Per-modality sums, not the single largest entry: three CAN features of 2.0 beat one 5.0.
    assert pca_alarm_modality(np.array([2.0, 2.0, 5.0, 2.0, 0.0]), modalities, means) == "can"
    assert pca_alarm_modality(np.array([0.1, np.nan, 5.0, 0.3, 0.1]), modalities, means) == ""
    with pytest.raises(ValueError, match="one modality per feature"):
        pca_alarm_modality(np.array([1.0, 2.0]), modalities, means)


def test_pca_ratio_removes_the_feature_count_advantage() -> None:
    # Ten IMU features with small residuals sum to 30; two CAN features sum to 20.
    # The raw sum would choose IMU, but the training means make CAN the larger ratio.
    modalities = ["can", "can"] + ["imu"] * 10
    residuals = np.array([20.0, 0.0] + [3.0] * 10)
    means = {"can": 1.0, "imu": 100.0}
    assert pca_alarm_modality(residuals, modalities, means) == "can"


def test_pca_attribution_skips_zero_and_non_finite_training_means() -> None:
    modalities = ["can", "can", "imu"]
    residuals = np.array([1.0, 1.0, 10.0])
    assert pca_alarm_modality(residuals, modalities, {"can": 0.0, "imu": np.nan}) == ""
    assert pca_alarm_modality(residuals, modalities, {"can": 2.0, "imu": 0.0}) == "can"


def test_pca_training_means_read_only_the_training_sessions() -> None:
    train = pd.DataFrame(
        {
            "session_id": ["train"] * 4,
            "start_ms": [0, 1000, 2000, 3000],
            "end_ms": [1000, 2000, 3000, 4000],
            "a": [0.0, 1.0, 2.0, 3.0],
            "b": [1.0, 2.0, 3.0, 4.0],
        }
    )
    features = pd.concat(
        [
            train,
            pd.DataFrame(
                {
                    "session_id": ["val", "test"],
                    "start_ms": [0, 0],
                    "end_ms": [1000, 1000],
                    "a": [1000.0, 2000.0],
                    "b": [1000.0, 2000.0],
                }
            ),
        ],
        ignore_index=True,
    )
    scaler = Standardizer.fit(train, {"train": ["train"]})
    baseline = BaselineModel.fit(train, {"train": ["train"]}, scaler)
    modalities = ["can", "can"]

    means = pca_training_means(baseline, features, modalities, ["train"])
    poisoned = features.copy()
    poisoned.loc[poisoned["session_id"] != "train", ["a", "b"]] = 1e9
    assert pca_training_means(baseline, poisoned, modalities, ["train"]) == means
    assert means["can"] is not None and means["can"] > 0.0


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


def test_sessions_without_imu_are_unscored_but_physics_scores(tmp_path: Path) -> None:
    root, _ = _mixed_imu_dataset(tmp_path)
    result = evaluate(root, tmp_path / "out", CONFIG)
    data = result.data

    unscored = data["baseline_unscored"]
    assert [entry["session_id"] for entry in unscored["train"]] == ["phys-00"]
    assert {entry["session_id"] for entry in unscored["test"]} == {"phys-10", "phys-27"}
    assert {entry["session_id"] for entry in unscored["val"]} == {"phys-05"}
    for part in ("train", "val", "test"):
        for entry in unscored[part]:
            assert entry["missing_modalities"] == ["imu"]
            assert entry["windows"] > 0

    physics_fault = data["models"]["physics"]["test"]["faults"]["wheel_vibration"]
    assert physics_fault["sessions"] == 1
    assert physics_fault["not_evaluated"] == 0
    assert physics_fault["detected"] + physics_fault["missed"] == 1

    for model in ("baseline_pca", "baseline_z"):
        entry = data["models"][model]["test"]["faults"]["wheel_vibration"]
        assert entry["not_evaluated"] == 1
        assert entry["detected"] == 0
        assert entry["missed"] == 0
        assert entry["detected"] + entry["missed"] + entry["not_evaluated"] == entry["sessions"]

    text = result.markdown_path.read_text(encoding="utf-8")
    assert "not scored (missing modality)" in text
    assert "not evaluated" in text
    assert "phys-27" in text

    card_out = tmp_path / "card"
    dataset = build_dataset(root, card_out, kinds=("telemetry", "derived", "imu"), seed=0)
    write_dataset(dataset, card_out, SessionStore(root).sessions(), ("telemetry", "derived", "imu"))
    card = (card_out / "dataset.md").read_text(encoding="utf-8")
    assert "| phys-00 | False |" in card
    assert "| phys-05 | False |" in card
    assert "| phys-27 | False |" in card
    assert "| phys-01 | True |" in card


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

    attribution = data["attribution"]
    assert attribution
    for kinds in attribution.values():
        for entry in kinds.values():
            assert len(entry["first_alarm_modalities"]) == len(entry["seeds"])
            assert entry["matches_expected"] == sum(
                modality == entry["expected_modality"]
                for modality in entry["first_alarm_modalities"]
            )
            if entry["expected_modality"] is None:
                assert entry["fraction_matching_expected"] is None
            else:
                assert entry["fraction_matching_expected"] == pytest.approx(
                    entry["matches_expected"] / len(entry["seeds"])
                )
            assert entry["matches_expected"] + entry["no_alarm_seeds"] + entry[
                "not_evaluated_seeds"
            ] + entry["mixed_seeds"] <= len(entry["seeds"])
    assert "wheel_vibration" in attribution["physics"]
    assert set(attribution["physics"]) <= {
        "thermostat_stuck_open",
        "fan_failure",
        "plateau_sag",
        "clutch_slip",
        "wheel_vibration",
    }
    text = result.markdown_path.read_text(encoding="utf-8")
    assert SYNTHETIC_NOTE + "." in text
    assert "| metric | seeds | mean | min | max |" in text
    assert "## Attribution stability" in text
    assert "first alarm modality per seed" in text

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


def test_aggregate_attribution_treats_no_alarm_as_not_a_match() -> None:
    def entry(
        *,
        detected: int,
        missed: int,
        first_alarm_modalities: dict[str, int],
        not_evaluated: int = 0,
    ) -> dict[str, object]:
        return {
            "models": {
                "physics": {
                    "test": {
                        "faults": {
                            "wheel_vibration": {
                                "sessions": 1,
                                "detected": detected,
                                "missed": missed,
                                "not_evaluated": not_evaluated,
                                "expected_modality": "imu",
                                "first_alarm_modalities": first_alarm_modalities,
                            }
                        }
                    }
                }
            }
        }

    per_seed = {
        0: entry(detected=1, missed=0, first_alarm_modalities={"imu": 1}),
        1: entry(detected=0, missed=1, first_alarm_modalities={}),
    }
    result = aggregate_attribution(per_seed)
    entry_out = result["physics"]["wheel_vibration"]
    assert entry_out["seeds"] == [0, 1]
    assert entry_out["first_alarm_modalities"] == ["imu", "no alarm"]
    assert entry_out["matches_expected"] == 1
    assert entry_out["fraction_matching_expected"] == 0.5
    assert entry_out["no_alarm_seeds"] == 1


def test_aggregate_attribution_reports_all_unscored_seeds() -> None:
    per_seed = {
        0: {
            "models": {
                "baseline_pca": {
                    "test": {
                        "faults": {
                            "wheel_vibration": {
                                "sessions": 1,
                                "detected": 0,
                                "missed": 0,
                                "not_evaluated": 1,
                                "expected_modality": "imu",
                                "first_alarm_modalities": {},
                            }
                        }
                    }
                }
            }
        }
    }
    entry = aggregate_attribution(per_seed)["baseline_pca"]["wheel_vibration"]
    assert entry["first_alarm_modalities"] == ["not evaluated"]
    assert entry["not_evaluated_seeds"] == 1
    assert entry["fraction_matching_expected"] == 0.0


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


def test_all_unscored_seed_is_omitted_not_counted_as_a_miss() -> None:
    def entry(not_evaluated: int) -> dict[str, object]:
        fault = {
            "sessions": 1,
            "detected": 0,
            "missed": 1 - not_evaluated,
            "not_evaluated": not_evaluated,
            "median_delay_ms": None,
            "first_alarm_matches_expected": 0,
        }
        return {
            "models": {
                "baseline_z": {
                    "test": {
                        "false_alarms": {"per_hour": 0.0, "hours": 1.0},
                        "faults": {"wheel_vibration": fault},
                    }
                }
            }
        }

    aggregate = aggregate_metrics({0: entry(0), 1: entry(1)})
    detection = aggregate["baseline_z.wheel_vibration.detection_rate"]
    assert detection == {"n_seeds": 1, "mean": 0.0, "min": 0.0, "max": 0.0}
    assert aggregate["baseline_z.wheel_vibration.first_alarm_expected_rate"]["n_seeds"] == 1
    assert aggregate["baseline_z.wheel_vibration.not_evaluated_rate"]["n_seeds"] == 2
    assert aggregate["baseline_z.wheel_vibration.not_evaluated_rate"]["mean"] == 0.5


def test_baseline_without_validation_scores_is_not_calibrated(tmp_path: Path) -> None:
    root, chosen = _without_imu(tmp_path, "val", "val-no-imu")
    result = evaluate(root, tmp_path / "out", CONFIG)
    data = result.data
    assert set(chosen) <= set(data["split"]["val"])

    assert data["thresholds"]["physics"] is not None
    assert set(data["not_calibrated"]) == {"baseline_pca", "baseline_z"}
    for model in ("baseline_pca", "baseline_z"):
        entry = data["models"][model]
        assert entry["threshold"] is None
        assert entry["test"]["false_alarms"]["episodes"] == 0.0
        assert entry["test"]["false_alarms"]["hours"] == 0.0
        for fault in entry["test"]["faults"].values():
            assert fault["not_evaluated"] == fault["sessions"]
            assert fault["detected"] == 0
            assert fault["missed"] == 0
    assert data["models"]["physics"]["test"]["false_alarms"]["hours"] > 0.0

    saved = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert saved["thresholds"]["baseline_pca"] is None
    text = result.markdown_path.read_text(encoding="utf-8")
    assert f"| baseline_pca | {NOT_CALIBRATED} |" in text


def test_features_without_a_training_value_are_reported_as_excluded(tmp_path: Path) -> None:
    root, _ = _dataset(tmp_path)
    plain = evaluate(root, tmp_path / "plain", CONFIG).data
    assert plain["baseline_excluded_features"] == []
    assert plain["baseline_excluded_modalities"] == []

    no_imu_root, _ = _without_imu(tmp_path, "train", "train-no-imu")
    result = evaluate(no_imu_root, tmp_path / "out", CONFIG)
    data = result.data
    assert data["baseline_excluded_modalities"] == ["imu"]
    assert data["baseline_excluded_features"]
    text = result.markdown_path.read_text(encoding="utf-8")
    assert "excluded from the baselines (modalities: imu)" in text
    assert data["baseline_excluded_features"][0] in text


def _alarm_frame(values: list[float]) -> pd.DataFrame:
    starts = np.arange(len(values)) * 1000
    return pd.DataFrame(
        {"session_id": "s", "start_ms": starts, "end_ms": starts + 1000, "m": values}
    )


def test_false_alarm_episodes_do_not_merge_across_an_unscored_gap() -> None:
    frame = _alarm_frame([5.0, 5.0, np.nan, np.nan, 5.0, 5.0, 0.0])
    result = _false_alarms(frame, "m", 1.0, ["s"])
    assert result["episodes"] == 2.0
    # run 1 covers 0-2000 ms, run 2 covers 4000-7000 ms; the 2000 ms gap is excluded
    assert result["hours"] == pytest.approx(5000 / 3_600_000.0)


def test_false_alarm_hours_skip_unscored_edges() -> None:
    frame = _alarm_frame([np.nan, 0.0, 0.0, np.nan])
    result = _false_alarms(frame, "m", 1.0, ["s"])
    assert result["episodes"] == 0.0
    assert result["hours"] == pytest.approx(2000 / 3_600_000.0)


def test_alarm_without_attributable_modality_is_unattributed() -> None:
    scores = pd.DataFrame(
        {
            "session_id": ["f"] * 2,
            "start_ms": [0, 1000],
            "end_ms": [1000, 2000],
            "baseline_z": [5.0, 5.0],
            "baseline_z_modality": ["", "imu"],
        }
    )
    faults = {"f": {"fault": "k", "onset_ms": 0, "expected_modality": "imu"}}
    result = _fault_results(scores, "baseline_z", 1.0, ["f"], faults)["k"]
    assert result["modalities"] == {UNATTRIBUTED: 1, "imu": 1}
    assert result["first_alarm_modalities"] == {UNATTRIBUTED: 1}
    assert result["first_alarm_matches_expected"] == 0
    per_seed = {0: {"models": {"baseline_z": {"test": {"faults": {"k": result}}}}}}
    assert aggregate_attribution(per_seed)["baseline_z"]["k"]["first_alarm_modalities"] == [
        UNATTRIBUTED
    ]
