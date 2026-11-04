"""Operating-point proxies: medians, missing data and role-config validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.physics_synthetic import PhysicsSessionSpec, make_physics_data_dir
from vehicle_ml.data.store import SessionStore
from vehicle_ml.operating_point import session_operating_points
from vehicle_ml.physics.config import load_roles

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "physics_cl250.toml"


def test_config_loads_the_two_operating_point_proxies() -> None:
    roles = load_roles(CONFIG)
    assert roles.operating_point == {
        "temperature": "coolant_temp",
        "load": "engine_speed",
    }


def test_operating_point_table_is_optional(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8").split("[operating_point]")[0]
    path = tmp_path / "without-operating-point.toml"
    path.write_text(text, encoding="utf-8")
    roles = load_roles(path)
    assert roles.operating_point == {}


def test_unknown_axis_or_role_is_a_clear_error(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    bad_axis = tmp_path / "bad-axis.toml"
    bad_axis.write_text(
        text.replace('temperature = "coolant_temp"', 'humidity = "coolant_temp"'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown operating-point axis"):
        load_roles(bad_axis)

    bad_role = tmp_path / "bad-role.toml"
    bad_role.write_text(
        text.replace('load = "engine_speed"', 'load = "unknown_role"'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown role"):
        load_roles(bad_role)


def test_medians_follow_the_synthetic_ambient_and_load(tmp_path: Path) -> None:
    root = tmp_path / "data"
    specs = (
        PhysicsSessionSpec("op-cold", seed=1, ambient_c=5.0, load=0.4, duration_ms=20_000),
        PhysicsSessionSpec("op-mid", seed=2, ambient_c=20.0, load=0.8, duration_ms=20_000),
        PhysicsSessionSpec("op-hot", seed=3, ambient_c=35.0, load=1.2, duration_ms=20_000),
    )
    make_physics_data_dir(root, specs)
    roles = load_roles(CONFIG)
    points = session_operating_points(
        SessionStore(root), [spec.session_id for spec in specs], roles
    )

    temperatures = [points[spec.session_id].value("temperature") for spec in specs]
    loads = [points[spec.session_id].value("load") for spec in specs]
    assert all(value is not None for value in temperatures)
    assert temperatures == sorted(temperatures)
    assert loads == sorted(loads)
    for spec in specs:
        point = points[spec.session_id]
        assert point.samples["temperature"] > 0
        assert point.samples["load"] > 0


def test_missing_role_column_gives_none_not_a_guess(tmp_path: Path) -> None:
    root = tmp_path / "data"
    spec = PhysicsSessionSpec("op-missing", seed=4, ambient_c=20.0, load=1.0)
    make_physics_data_dir(root, (spec,))
    parquet = root / "sessions" / spec.session_id / "parquet" / "telemetry.parquet"
    telemetry = pd.read_parquet(parquet)
    telemetry.drop(columns=["coolant_temp_degc"]).to_parquet(parquet, index=False)

    point = session_operating_points(SessionStore(root), [spec.session_id], load_roles(CONFIG))[
        spec.session_id
    ]
    assert point.value("temperature") is None
    assert point.samples["temperature"] == 0
    assert point.value("load") is not None
    assert point.samples["load"] > 0
