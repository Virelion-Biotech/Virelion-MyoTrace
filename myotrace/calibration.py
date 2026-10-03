from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForceCalibration:
    """Explicit instrument-specific motion-to-force calibration.

    This object must only be fit when paired motion and force ground truth are available.
    """

    slope: float
    intercept: float
    r2: float
    rmse: float
    n: int
    units: str
    method: str = "ordinary_least_squares"

    def predict(self, motion: np.ndarray | float) -> np.ndarray | float:
        return self.slope * np.asarray(motion) + self.intercept


def fit_force_calibration(motion: np.ndarray, force: np.ndarray, *, units: str = "arbitrary_force_units") -> ForceCalibration:
    """Fit a simple auditable linear calibration from motion index to measured force."""
    x = np.asarray(motion, dtype=float).reshape(-1)
    y = np.asarray(force, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 3:
        raise ValueError("motion and force must have the same length and contain at least 3 paired observations")
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        raise ValueError("At least 3 finite paired observations are required")
    x, y = x[keep], y[keep]
    if np.ptp(x) == 0:
        raise ValueError("motion values must vary")
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    residual = y - pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(residual**2)))
    return ForceCalibration(float(slope), float(intercept), float(r2), rmse, int(x.size), units)
