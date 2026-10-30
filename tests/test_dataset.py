"""Dataset builder and CLI: split isolation, cards and argument handling."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tests.synthetic import make_data_dir
from vehicle_ml.cli import main
from vehicle_ml.data.store import SessionStore
from vehicle_ml.dataset import _is_metadata, build_dataset, write_dataset
from vehicle_ml.split import assert_no_leakage


@pytest.fixture
def data_dir(tmp_path) -> Path:
    return make_data_dir(tmp_path / "data")


def test_build_dataset_keeps_sessions_whole_and_drops_position(data_dir: Path, tmp_path) -> None:
    result = build_dataset(data_dir, tmp_path / "out", kinds=("telemetry", "imu"))

    assert not result.features.empty
    assert {"session_id", "split", "start_ms", "end_ms"} <= set(result.features.columns)
    parts = {
        part: result.features.loc[result.features["split"] == part, "session_id"].unique().tolist()
        for part in ("train", "val", "test")
    }
    assert_no_leakage(*parts.values())
    assert not any(column.startswith("gps_latitude") for column in result.features.columns)
    assert "ax_g" in result.channels["imu"]
    assert "ax_g" not in result.channels["telemetry"]
    assert result.defs_versions["s1"] == "0.20.0"
    assert result.defs_versions["s2"] is None


def test_build_dataset_refuses_output_inside_data_dir(data_dir: Path) -> None:
    with pytest.raises(ValueError, match="inside --data-dir"):
        build_dataset(data_dir, data_dir / "out")


def test_build_dataset_rejects_unknown_kind(data_dir: Path, tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown kinds"):
        build_dataset(data_dir, tmp_path / "out", kinds=("nope",))


def test_cli_writes_all_outputs_end_to_end(data_dir: Path, tmp_path) -> None:
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "vehicle_ml",
            "dataset",
            "--data-dir",
            str(data_dir),
            "--out",
            str(out),
            "--kinds",
            "telemetry,imu",
            "--channels",
            "all",
            "--window-ms",
            "2000",
            "--stride-ms",
            "1000",
            "--max-gap-ms",
            "500",
            "--seed",
            "7",
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert "wrote" in proc.stdout
    for name in ("features.parquet", "split.json", "scaler.json", "dataset.md"):
        assert (out / name).exists()

    split = json.loads((out / "split.json").read_text())
    assert split["seed"] == 7
    assert split["fractions"] == [0.7, 0.15, 0.15]
    assert set(split["sessions"]) == {"s1", "s2", "s3"}
    assert set(split["sessions"].values()) <= {"train", "val", "test"}

    features = pd.read_parquet(out / "features.parquet")
    assert not features.empty
    assert not any(column.startswith("gps_latitude") for column in features.columns)
    assert features.groupby("session_id")["split"].nunique().max() == 1

    card = (out / "dataset.md").read_text()
    assert "# Dataset card" in card
    assert "## Modality availability" in card
    assert "0.20.0" in card
    assert "unknown" in card


def test_cli_accepts_band_spec(data_dir: Path, tmp_path) -> None:
    out = tmp_path / "bands"
    code = main(
        [
            "dataset",
            "--data-dir",
            str(data_dir),
            "--out",
            str(out),
            "--kinds",
            "imu",
            "--channels",
            "ax_g",
            "--bands",
            "ax_g:0-5,5-20",
        ]
    )

    assert code == 0
    features = pd.read_parquet(out / "features.parquet")
    assert "ax_g.band_0_5hz" in features.columns
    assert "ax_g.band_5_20hz" in features.columns


def test_cli_rejects_bad_arguments(data_dir: Path, tmp_path) -> None:
    out = tmp_path / "out"
    base = ["dataset", "--data-dir", str(data_dir), "--out", str(out)]
    assert main(base) == 0
    assert main([*base, "--kinds", "nope"]) == 2
    assert main([*base, "--bands", "ax_g:bad"]) == 2
    assert main(["dataset", "--data-dir", str(data_dir), "--out", str(data_dir / "out")]) == 2


def test_channels_all_excludes_metadata_by_rule(data_dir: Path, tmp_path) -> None:
    result = build_dataset(data_dir, tmp_path / "out", kinds=("telemetry",), channels=None)

    names = set(result.channels["telemetry"])
    assert "engine_speed_rpm" in names
    assert not {n for n in names if n.endswith(("_age_ms", "_valid")) or n.startswith("can_")}
    assert not {n for n in names if n.startswith("rtt_") or n.startswith("step_gap_")}
    assert "unanswered_did_count" not in names
    assert {"engine_speed_age_ms", "can_flags", "rtt_min_ms", "seq"} <= result.excluded

    explicit = build_dataset(
        data_dir, tmp_path / "out", kinds=("telemetry",), channels=["rtt_min_ms", "can_flags"]
    )
    assert explicit.channels["telemetry"] == ["can_flags", "rtt_min_ms"]
    assert explicit.excluded == set()


def test_is_metadata_rules() -> None:
    for name in (
        "seq",
        "x_age_ms",
        "x_valid",
        "can_bus_state",
        "unanswered_did_count",
        "rtt_sum_ms",
    ):
        assert _is_metadata(name)
    assert not _is_metadata("engine_speed_rpm")


def test_time_base_is_the_largest_kind_and_is_recorded(data_dir: Path, tmp_path) -> None:
    out = tmp_path / "out"
    result = build_dataset(data_dir, out, kinds=("telemetry", "imu"))
    assert result.base_kinds == {"imu": 2, "telemetry": 1}  # s2 has no imu
    write_dataset(result, out, SessionStore(data_dir).sessions(), ("telemetry", "imu"))
    card = (out / "dataset.md").read_text()
    assert "the kind with the most rows" in card
    assert "Auto-excluded" in card


def test_session_without_usable_frames_is_dropped_as_no_data(data_dir: Path, tmp_path) -> None:
    result = build_dataset(
        data_dir, tmp_path / "out", kinds=("telemetry", "imu"), channels=["ax_g"]
    )
    assert result.drops["no_data"] == 1  # s2 has no imu
    assert "missing" not in result.drops


def test_uneven_imu_rate_drops_windows_instead_of_aborting(data_dir: Path, tmp_path) -> None:
    path = data_dir / "sessions" / "s1" / "parquet" / "imu.parquet"
    imu = pd.read_parquet(path)
    imu[imu["device_time_ms"] != 5_000].to_parquet(path, index=False)

    result = build_dataset(
        data_dir, tmp_path / "out", kinds=("imu",), channels=["ax_g"], bands={"ax_g": [(0.0, 5.0)]}
    )
    assert result.drops["rate"] >= 1
    assert not result.features.empty
