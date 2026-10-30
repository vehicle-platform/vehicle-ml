"""Defs bridge, server naming rule and role config validation (ML-10)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vehicle_ml.data.naming import physical_column, signal_key, unit_suffix
from vehicle_ml.defs import defs_version, vehicle_cl250, vehicle_derived
from vehicle_ml.physics.config import REQUIRED_ROLES, load_roles

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "physics_cl250.toml"


def test_bridge_imports_the_pinned_defs_version() -> None:
    assert defs_version() == "v0.20.0"
    assert "ENGINE_SPEED" in vehicle_cl250.DIDS
    assert "gear_estimate" in vehicle_derived.SIGNALS


@pytest.mark.parametrize(
    ("defs_signal", "expected"),
    [
        ("ENGINE_SPEED", "engine_speed_rpm"),
        ("VEHICLE_SPEED", "vehicle_speed_kmh"),
        ("COOLANT_TEMP", "coolant_temp_degc"),
        ("BATTERY_VOLTAGE", "battery_voltage_v"),
    ],
)
def test_naming_matches_the_server_examples(defs_signal: str, expected: str) -> None:
    assert physical_column(defs_signal) == expected


def test_signal_key_and_unit_suffix_fallbacks() -> None:
    assert signal_key("ENGINE_SPEED") == "engine_speed"
    assert unit_suffix("rpm") == "rpm"
    assert unit_suffix("m/s^2") == "ms2"


def test_valid_role_config_loads() -> None:
    roles = load_roles(CONFIG)
    assert tuple(roles.roles) == REQUIRED_ROLES
    assert roles.column("engine_speed") == "engine_speed_rpm"
    assert roles.column("coolant_temp") == "coolant_temp_degc"
    assert roles.column("gear_estimate") == "gear_estimate"
    assert roles.kind("gear_estimate") == "derived"


def test_unknown_did_fails_with_the_name(tmp_path: Path) -> None:
    path = tmp_path / "roles.toml"
    text = CONFIG.read_text().replace('"did.ENGINE_SPEED"', '"did.FOO"')
    path.write_text(text)
    with pytest.raises(ValueError, match="did.FOO"):
        load_roles(path)


def test_unknown_derived_fails_with_the_name(tmp_path: Path) -> None:
    path = tmp_path / "roles.toml"
    text = CONFIG.read_text().replace('"derived.gear_estimate"', '"derived.nope"')
    path.write_text(text)
    with pytest.raises(ValueError, match="derived.nope"):
        load_roles(path)


def test_missing_role_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "roles.toml"
    text = CONFIG.read_text().replace('engine_speed = "did.ENGINE_SPEED"\n', "")
    path.write_text(text)
    with pytest.raises(ValueError, match="engine_speed"):
        load_roles(path)


def test_bad_prefix_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "roles.toml"
    text = CONFIG.read_text().replace('"did.ENGINE_SPEED"', '"sensor.ENGINE_SPEED"')
    path.write_text(text)
    with pytest.raises(ValueError, match="prefix"):
        load_roles(path)
