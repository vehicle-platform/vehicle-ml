"""Signal-name to telemetry-column mapping, copied from vehicle-server.

Source: ``vehicle-server/src/vehicle_server/signals.py`` (``signal_key``,
``unit_suffix``, ``physical_column``). The server writes decoded telemetry columns as
``<lowercased defs name>_<normalised unit>``; vehicle-ml must read the same names.
The unit suffix table is intentionally identical so the two repositories cannot drift
silently.
"""

from __future__ import annotations

from vehicle_ml.defs import vehicle_cl250

_DID_UNIT = 5

_UNIT_SUFFIX = {
    "rpm": "rpm",
    "km/h": "kmh",
    "%": "pct",
    "degC": "degc",
    "V": "v",
}


def signal_key(defs_signal: str) -> str:
    """The lowercase key used for a defs signal name, e.g. ENGINE_SPEED -> engine_speed."""
    return defs_signal.lower()


def unit_suffix(unit: str) -> str:
    if unit in _UNIT_SUFFIX:
        return _UNIT_SUFFIX[unit]
    return "".join(ch for ch in unit.lower() if ch.isalnum()) or "unit"


def physical_unit(defs_signal: str) -> str:
    """The physical unit of a defs DID (from the generated DIDS table)."""
    return vehicle_cl250.DIDS[defs_signal][_DID_UNIT]


def physical_column(defs_signal: str) -> str:
    """The telemetry Parquet column for a defs DID, e.g. COOLANT_TEMP -> coolant_temp_degc."""
    return f"{signal_key(defs_signal)}_{unit_suffix(physical_unit(defs_signal))}"


__all__ = ["physical_column", "physical_unit", "signal_key", "unit_suffix"]
