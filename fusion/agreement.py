from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlandAltman:
    mean_bias: float
    sd_difference: float
    loa_lower: float
    loa_upper: float
    n: int


def bland_altman(method_a: np.ndarray, method_b: np.ndarray) -> BlandAltman:
    """Agreement summary for two measurements of the same quantity."""
    a = np.asarray(method_a, dtype=float).reshape(-1)
    b = np.asarray(method_b, dtype=float).reshape(-1)
    if a.size != b.size or a.size < 2:
        raise ValueError("paired methods require the same number of observations and n>=2")
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    if d.size < 2:
        raise ValueError("at least 2 finite paired observations are required")
    bias = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    return BlandAltman(bias, sd, bias - 1.96 * sd, bias + 1.96 * sd, int(d.size))


def coefficient_of_variation(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    mean = float(np.mean(x))
    return float(np.std(x, ddof=1) / mean) if mean != 0 else float("nan")
