"""Scaffold smoke test: the package imports and exposes its version."""

from vehicle_ml import __version__


def test_version_is_set() -> None:
    assert __version__ == "0.1.0"
