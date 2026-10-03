from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ReplicatePlan:
    biological_replicates: int
    technical_replicates_per_biological: int
    independent_days: int
    independent_operators: int
    blinded_analysis: bool
    held_out_external_set: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.biological_replicates < 3:
            out.append("fewer_than_three_biological_replicates")
        if self.independent_days < 2:
            out.append("single_or_insufficient_experimental_day")
        if self.independent_operators < 2:
            out.append("single_operator")
        if not self.blinded_analysis:
            out.append("analysis_not_blinded")
        if not self.held_out_external_set:
            out.append("no_external_holdout")
        return out


def bootstrap_ci(values: Iterable[float], statistic=np.mean, *, n_boot: int = 5000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Deterministic percentile bootstrap CI; resampling is at the supplied unit of replication."""
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(x, size=(n_boot, x.size), replace=True)
    estimates = np.asarray([statistic(row) for row in samples], dtype=float)
    return float(statistic(x)), float(np.quantile(estimates, alpha / 2)), float(np.quantile(estimates, 1 - alpha / 2))


def sensitivity_to_weight(weights: dict[str, float], scores: dict[str, float], *, perturbation: float = 0.1) -> dict[str, float]:
    """One-at-a-time normalized perturbation; useful for auditing a composite score."""
    if set(weights) != set(scores):
        raise ValueError("weights and scores must contain identical modalities")
    base = float(sum(weights[k] * scores[k] for k in weights) / max(sum(weights.values()), np.finfo(float).eps))
    result: dict[str, float] = {"baseline": base}
    for key in weights:
        w = dict(weights)
        w[key] *= 1.0 + perturbation
        value = float(sum(w[k] * scores[k] for k in w) / max(sum(w.values()), np.finfo(float).eps))
        result[f"delta_{key}"] = value - base
    return result
