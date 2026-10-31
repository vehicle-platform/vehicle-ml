"""Bridge to the pinned vehicle-defs submodule (external/vehicle-defs).

This is the vehicle-ml counterpart of ``vehicle-server/src/vehicle_server/defs.py``:
the submodule's ``gen/python`` directory is added to ``sys.path`` and ``vehicle_defs``
is imported from there. No generated code is copied into this repository
(invariant 2). ``defs_pin.txt`` is the fallback version when git metadata is absent,
for example in a Docker image built without the submodule's ``.git`` directory.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFS_ROOT = _REPO_ROOT / "external" / "vehicle-defs"
_DEFS_PYTHON = _DEFS_ROOT / "gen" / "python"
_FALLBACK_VERSION_FILE = Path(__file__).resolve().parent / "defs_pin.txt"

if _DEFS_PYTHON.is_dir() and str(_DEFS_PYTHON) not in sys.path:
    sys.path.insert(0, str(_DEFS_PYTHON))

try:
    import vehicle_defs  # noqa: E402
    from vehicle_defs import (  # noqa: E402
        ble,
        vehicle_cl250,
        vehicle_constants,
        vehicle_derived,
    )
except ImportError as exc:  # pragma: no cover - fails fast at startup
    raise ImportError(
        "vehicle_defs could not be imported from "
        f"{_DEFS_PYTHON}. Did you run "
        "`git submodule update --init --recursive`?"
    ) from exc


def defs_version() -> str:
    """Version of the pinned vehicle-defs submodule (tag when git metadata exists)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(_DEFS_ROOT), "describe", "--tags", "--always"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        version = result.stdout.strip()
        if version:
            return version
    except (OSError, subprocess.CalledProcessError):
        pass

    try:
        return _FALLBACK_VERSION_FILE.read_text().strip() or "unknown"
    except OSError:
        return "unknown"


__all__ = [
    "ble",
    "vehicle_defs",
    "vehicle_cl250",
    "vehicle_constants",
    "vehicle_derived",
    "defs_version",
]
