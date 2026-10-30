"""Position-column filter, copied from vehicle-mcp/src/vehicle_mcp/tools.py.

Invariant 7: position (lat/lon/height) must never reach a model. The rule is the same
one vehicle-mcp applies to its series: a horizontal component anywhere in the name, a
vertical one only as the last token, so a relative quantity such as
``derived.altitude_change`` stays.
"""

from __future__ import annotations

import re

_HORIZONTAL_TOKENS = frozenset({"lat", "lon", "lng", "latitude", "longitude"})
_VERTICAL_TOKENS = frozenset({"height", "alt", "altitude"})
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def is_position_name(name: str) -> bool:
    """True when a column name looks like a GPS position component."""
    normalized = _CAMEL_BOUNDARY.sub("_", name).lower()
    tokens = [token for token in _TOKEN_SPLIT.split(normalized) if token]
    if any(token in _HORIZONTAL_TOKENS for token in tokens):
        return True
    return bool(tokens) and tokens[-1] in _VERTICAL_TOKENS


def drop_position_columns(columns: list[str]) -> list[str]:
    """Return the columns that are safe to hand to a model."""
    return [column for column in columns if not is_position_name(column)]
