"""Shared errors for the stage-1 physics estimators."""

from __future__ import annotations


class NotIdentifiableError(ValueError):
    """The data does not excite the model enough to identify its parameters."""
