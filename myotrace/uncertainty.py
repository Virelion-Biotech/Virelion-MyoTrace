from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from ._validation import bootstrap_parameters


@dataclass(frozen=True)
class BootstrapSummary:
    estimate: float
    lower: float
    upper: float
    n_boot: int
    seed: int


def bootstrap_mean(values: np.ndarray, *, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> BootstrapSummary:
    """Non-parametric bootstrap CI for a beat-level summary."""
    bootstrap_parameters(n_boot, alpha)
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return BootstrapSummary(float(np.nanmean(x)) if x.size else np.nan, np.nan, np.nan, 0, seed)
    if n_boot < 100:
        raise ValueError("n_boot must be >= 100")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    estimates = np.mean(x[idx], axis=1)
    return BootstrapSummary(
        float(np.mean(x)),
        float(np.quantile(estimates, alpha / 2)),
        float(np.quantile(estimates, 1 - alpha / 2)),
        n_boot,
        seed,
    )


def bootstrap_statistic(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> BootstrapSummary:
    bootstrap_parameters(n_boot, alpha)
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return BootstrapSummary(np.nan, np.nan, np.nan, 0, seed)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        estimates[i] = statistic(x[rng.integers(0, x.size, x.size)])
    return BootstrapSummary(
        float(statistic(x)),
        float(np.quantile(estimates, alpha / 2)),
        float(np.quantile(estimates, 1 - alpha / 2)),
        n_boot,
        seed,
    )


def independent_unit_mean(values, unit_ids, *, n_boot=2000, seed=42, alpha=0.05):
    """Equal-weight biological-unit mean and CI; nested beats/frames do not increase n."""
    bootstrap_parameters(n_boot, alpha)
    x = np.asarray(values, dtype=float)
    units = np.asarray(unit_ids, dtype=object)
    if x.ndim != 1 or units.shape != x.shape or not x.size or not np.isfinite(x).all():
        raise ValueError("Finite values and one biological unit ID per observation are required")
    if any(not isinstance(u, str) or not u.strip() for u in units):
        raise ValueError("Biological unit IDs must be explicit nonblank strings")
    ids = sorted(set(units))
    if len(ids) < 2:
        raise ValueError("At least two independent biological units are required for a CI")
    means = np.array([np.mean(x[units == u]) for u in ids])
    ci = bootstrap_mean(means, n_boot=n_boot, seed=seed, alpha=alpha)
    return {
        "estimate": ci.estimate,
        "lower": ci.lower,
        "upper": ci.upper,
        "n_independent": len(ids),
        "n_observations": len(x),
        "resampling_unit": "biological_unit",
        "estimand": "equal-weight mean of biological-unit means",
        "unit_means": dict(zip(ids, means.tolist())),
    }
