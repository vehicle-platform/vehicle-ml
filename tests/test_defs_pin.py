"""The fallback defs pin must not drift from the submodule tag (ML-22)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import vehicle_ml.defs as defs_module

PIN_FILE = Path(defs_module.__file__).resolve().parent / "defs_pin.txt"
DEFS_ROOT = Path(__file__).resolve().parents[1] / "external" / "vehicle-defs"


def test_defs_pin_equals_the_submodule_tag() -> None:
    try:
        result = subprocess.run(
            ["git", "-C", str(DEFS_ROOT), "describe", "--tags", "--exact-match"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"no git tag metadata for external/vehicle-defs: {exc}")
    assert PIN_FILE.read_text(encoding="utf-8").strip() == result.stdout.strip()
