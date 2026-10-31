"""Per-vehicle role map for the physics estimators (D-078, invariant 2).

Library code names roles (``coolant_temp``, ``engine_speed``, ...), never defs signal
names. A TOML file maps each role to a pinned vehicle-defs signal: ``did.<NAME>`` for
``vehicle_defs.vehicle_cl250.DIDS`` or ``derived.<name>`` for
``vehicle_defs.vehicle_derived.SIGNALS``. Every name is validated on load, so a typo or
a signal absent from the pinned defs fails before any data is read.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from vehicle_ml.data.naming import physical_column
from vehicle_ml.defs import vehicle_cl250, vehicle_derived

REQUIRED_ROLES = (
    "engine_speed",
    "vehicle_speed",
    "coolant_temp",
    "battery_voltage",
    "gear_estimate",
    "gearbox_ratio_measured",
)


def _validate_signal(role: str, spec: str) -> tuple[str, str]:
    prefix, separator, name = spec.partition(".")
    if not separator or not name:
        raise ValueError(f"role {role!r} must name did.<NAME> or derived.<name>, got {spec!r}")
    if prefix == "did":
        if name not in vehicle_cl250.DIDS:
            raise ValueError(f"role {role!r} names unknown defs DID {name!r} ({spec!r})")
    elif prefix == "derived":
        if name not in vehicle_derived.SIGNALS:
            raise ValueError(f"role {role!r} names unknown derived signal {name!r} ({spec!r})")
    else:
        raise ValueError(f"role {role!r} has unsupported signal prefix {prefix!r}")
    return prefix, name


@dataclass(frozen=True)
class RoleMap:
    """Validated role -> vehicle-defs signal mapping."""

    roles: Mapping[str, str]
    source: Path | None = None

    def signal(self, role: str) -> str:
        try:
            return self.roles[role]
        except KeyError as exc:
            raise ValueError(f"role {role!r} is not in the role map") from exc

    def kind(self, role: str) -> str:
        prefix, _ = _validate_signal(role, self.signal(role))
        return prefix

    def name(self, role: str) -> str:
        _, name = _validate_signal(role, self.signal(role))
        return name

    def column(self, role: str) -> str:
        """The Parquet column for a role: a wide telemetry column or a derived signal name."""
        if self.kind(role) == "did":
            return physical_column(self.name(role))
        return self.name(role)

    def require(self, *roles: str) -> None:
        missing = [role for role in roles if role not in self.roles]
        if missing:
            raise ValueError(f"role map is missing roles: {', '.join(missing)}")


def load_roles(path: str | Path, required: tuple[str, ...] = REQUIRED_ROLES) -> RoleMap:
    """Load and validate a role config; fail on an unknown defs name or a missing role."""
    config_path = Path(path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    raw = data.get("roles")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{config_path}: expected a non-empty [roles] table")

    roles: dict[str, str] = {}
    for role, spec in raw.items():
        if not isinstance(role, str) or not role:
            raise ValueError(f"{config_path}: role names must be non-empty strings")
        if not isinstance(spec, str) or not spec:
            raise ValueError(f"{config_path}: role {role!r} must map to a signal string")
        _validate_signal(role, spec)
        roles[role] = spec

    missing = [role for role in required if role not in roles]
    if missing:
        raise ValueError(f"{config_path}: missing required roles: {', '.join(missing)}")

    return RoleMap(roles=roles, source=config_path)


__all__ = ["REQUIRED_ROLES", "RoleMap", "load_roles"]
