"""Per-gear ratio table and clutch-slip residual.

The table is the median ``gearbox_ratio_measured`` per ``gear_estimate`` learned from
train sessions only; gear 0 (no gear) is ignored. The residual is
``measured - table[gear]`` and is NaN for gear 0 or a gear with too few training
samples.

``gear_estimate`` itself is matched to the vehicle-defs constants by the derived
signal, so this table is a learned check on that match, not a second source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vehicle_ml.physics.errors import NotIdentifiableError


@dataclass(frozen=True)
class GearTable:
    """Median measured ratio per engaged gear."""

    ratios: dict[int, float]

    @classmethod
    def fit(
        cls,
        gearbox_ratio_measured: np.ndarray,
        gear_estimate: np.ndarray,
        *,
        min_samples: int = 5,
    ) -> GearTable:
        ratio = np.asarray(gearbox_ratio_measured, dtype=float)
        gear = np.asarray(gear_estimate, dtype=float)
        if ratio.ndim != 1 or ratio.size != gear.size:
            raise ValueError("ratio and gear must be 1-D arrays of equal length")
        usable = np.isfinite(ratio) & np.isfinite(gear) & (gear > 0)
        if not usable.any():
            raise NotIdentifiableError("gear table is not identifiable: no engaged-gear samples")

        ratios: dict[int, float] = {}
        for number in sorted({int(value) for value in gear[usable]}):
            values = ratio[usable & (gear == number)]
            if values.size >= min_samples:
                ratios[number] = float(np.median(values))
        if not ratios:
            raise NotIdentifiableError(
                f"gear table is not identifiable: no gear has {min_samples} samples"
            )
        return cls(ratios=ratios)

    def residual(self, gearbox_ratio_measured: np.ndarray, gear_estimate: np.ndarray) -> np.ndarray:
        """Measured minus table ratio; NaN for gear 0 or a gear outside the table."""
        ratio = np.asarray(gearbox_ratio_measured, dtype=float)
        gear = np.asarray(gear_estimate, dtype=float)
        if ratio.ndim != 1 or ratio.size != gear.size:
            raise ValueError("ratio and gear must be 1-D arrays of equal length")
        result = np.full(ratio.shape, np.nan, dtype=float)
        for number, value in self.ratios.items():
            mask = (gear == number) & np.isfinite(ratio)
            result[mask] = ratio[mask] - value
        return result


__all__ = ["GearTable", "NotIdentifiableError"]
