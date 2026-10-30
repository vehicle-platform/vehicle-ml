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
from vehicle_ml.dataset import build_dataset
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
