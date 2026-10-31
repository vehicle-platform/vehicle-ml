"""Feature-column provenance from the pinned vehicle-defs tables (ML-19).

A feature column is ``<channel>.<stat>`` or ``<channel>.band_<low>_<high>hz``.
The channel is a telemetry column for a defs DID, a derived signal name, or a
server-side sensor column. The source and modality are derived from the defs tables,
never from a hand-written channel list:

- ``vehicle_cl250.DIDS`` -> ``did.<NAME>`` (modality ``can``);
- ``vehicle_derived.SIGNALS`` -> ``derived.<name>`` (modality ``can``);
- ``vehicle_derived.SENSORS`` -> ``sensor.<name>`` (modality ``imu`` for
  ``sensor.imu_*``, otherwise ``other``).

The server column names for sensors follow the BLE block field and unit (the IMU
scaled columns are ``ax_g`` / ``gx_dps``, the GPS scaled columns are
``ground_speed_mps`` and so on), so they are derived from the generated BLE schema
and the sensor unit rather than copied into vehicle-ml.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from vehicle_ml.data.naming import physical_column, unit_suffix
from vehicle_ml.defs import ble, vehicle_cl250, vehicle_derived

IDENTITY_COLUMNS = ("session_id", "start_ms", "end_ms")


@dataclass(frozen=True)
class FeatureMetadata:
    """Provenance of one feature column."""

    source: str
    modality: str
    group: str | None = None


def _snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


# BLE block units use the server's scaled-column suffixes, which are not the DID
# unit-suffix table (the server writes ``gx_dps`` and ``ground_speed_mps``).
_BLE_UNIT_SUFFIX = {"g": "g", "deg/s": "dps", "m/s": "mps", "m/s2": "ms2"}


def _ble_unit_suffix(unit: str) -> str:
    return _BLE_UNIT_SUFFIX.get(unit, unit_suffix(unit))


def _sensor_columns() -> dict[str, str]:
    """Server column -> defs sensor name, derived from the generated tables."""
    columns: dict[str, str] = {}
    for sensor, spec in vehicle_derived.SENSORS.items():
        phase0 = spec.get("phase0")
        if not isinstance(phase0, str) or phase0.count(".") != 2:
            continue
        _, block, field = phase0.split(".")
        block_spec = ble.SCHEMA.get(block, {})
        if block == "imuBlock":
            scale = block_spec.get("scale", {})
            scale_name = "accel" if field.startswith("a") else "gyro"
            unit = str(scale.get(scale_name, {}).get("unit", spec.get("unit", "unit")))
            columns[field] = sensor
            columns[f"{field}_{_ble_unit_suffix(unit)}"] = sensor
        else:
            field_name = _snake_case(field)
            columns[field_name] = sensor
            columns[f"{field_name}_{_ble_unit_suffix(str(spec.get('unit', 'unit')))}"] = sensor
    return columns


_DID_COLUMNS = {physical_column(name): name for name in vehicle_cl250.DIDS}
_DERIVED_COLUMNS = {name: name for name in vehicle_derived.SIGNALS}
_SENSOR_COLUMNS = _sensor_columns()


def _channel(column: str) -> str:
    return column.partition(".")[0]


def feature_source(column: str) -> str:
    """The series behind a feature column, e.g. ``did.ENGINE_SPEED``.

    Raises ``ValueError`` naming the column when its channel is not in the pinned
    defs tables.
    """
    channel = _channel(column)
    if channel in _DID_COLUMNS:
        return f"did.{_DID_COLUMNS[channel]}"
    if channel in _DERIVED_COLUMNS:
        return f"derived.{_DERIVED_COLUMNS[channel]}"
    if channel in _SENSOR_COLUMNS:
        return f"sensor.{_SENSOR_COLUMNS[channel]}"
    raise ValueError(f"unknown feature channel {channel!r} in column {column!r}")


def feature_modality(source: str) -> str:
    """``can`` for DID/derived sources, ``imu`` for IMU sensors, else ``other``."""
    if source.startswith(("did.", "derived.")):
        return "can"
    if source.startswith("sensor.imu_"):
        return "imu"
    if source.startswith(("sensor.", "metadata.")):
        return "other"
    raise ValueError(f"unknown feature source {source!r}")


def feature_metadata(
    columns: Iterable[str], *, allow_unknown: bool = False
) -> dict[str, FeatureMetadata]:
    """Metadata for every feature column, keyed by the full column name.

    ``allow_unknown`` is for storage metadata that a caller explicitly selected; it
    labels those columns ``metadata.<channel>`` instead of failing. Feature frames use
    the strict default so a typo cannot silently become an unsourced feature.
    """
    result: dict[str, FeatureMetadata] = {}
    for column in columns:
        if column in IDENTITY_COLUMNS:
            continue
        try:
            source = feature_source(column)
        except ValueError:
            if not allow_unknown:
                raise
            source = f"metadata.{_channel(column)}"
        result[column] = FeatureMetadata(source=source, modality=feature_modality(source))
    return result


def feature_sources(columns: Iterable[str]) -> dict[str, str]:
    """``column -> source`` for every feature column."""
    return {column: metadata.source for column, metadata in feature_metadata(columns).items()}


def feature_modalities(columns: Iterable[str]) -> dict[str, str]:
    """``column -> modality`` for every feature column."""
    return {column: metadata.modality for column, metadata in feature_metadata(columns).items()}


__all__ = [
    "FeatureMetadata",
    "feature_metadata",
    "feature_modality",
    "feature_modalities",
    "feature_source",
    "feature_sources",
]
