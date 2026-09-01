from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BootstrapSummary:
    estimate: float
    lower: float
    upper: float
    n_boot: int
    seed: int


def bootstrap_mean(values: np.ndarray, *, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> BootstrapSummary:
    """Non-parametric bootstrap CI for a beat-level summary."""
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return BootstrapSummary(float(np.nanmean(x)) if x.size else np.nan, np.nan, np.nan, 0, seed)
    if n_boot < 100:
        raise ValueError("n_boot must be >= 100")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    estimates = np.mean(x[idx], axis=1)
    return BootstrapSummary(float(np.mean(x)), float(np.quantile(estimates, alpha / 2)), float(np.quantile(estimates, 1 - alpha / 2)), n_boot, seed)


def bootstrap_statistic(values: np.ndarray, statistic: Callable[[np.ndarray], float], *, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> BootstrapSummary:
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return BootstrapSummary(np.nan, np.nan, np.nan, 0, seed)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        estimates[i] = statistic(x[rng.integers(0, x.size, x.size)])
    return BootstrapSummary(float(statistic(x)), float(np.quantile(estimates, alpha / 2)), float(np.quantile(estimates, 1 - alpha / 2)), n_boot, seed)
