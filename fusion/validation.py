from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr


@dataclass(frozen=True)
class GroupComparison:
    group_a: str
    group_b: str
    n_a: int
    n_b: int
    median_a: float
    median_b: float
    delta: float
    rank_biserial: float
    p_value: float


def compare_groups(scores: Mapping[str, float], labels: Mapping[str, str], group_a: str, group_b: str) -> GroupComparison:
    a = np.asarray([scores[k] for k, v in labels.items() if v == group_a and k in scores and np.isfinite(scores[k])], dtype=float)
    b = np.asarray([scores[k] for k, v in labels.items() if v == group_b and k in scores and np.isfinite(scores[k])], dtype=float)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Both groups require at least one finite score")
    stat = mannwhitneyu(a, b, alternative="two-sided", method="auto")
    u = float(stat.statistic)
    rank_biserial = 2.0 * u / (len(a) * len(b)) - 1.0
    return GroupComparison(group_a, group_b, len(a), len(b), float(np.median(a)), float(np.median(b)), float(np.median(b) - np.median(a)), rank_biserial, float(stat.pvalue))


def rank_correlation(scores: Sequence[float], reference: Sequence[float]) -> dict[str, float]:
    x, y = np.asarray(scores, dtype=float), np.asarray(reference, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return {"rho": np.nan, "p_value": np.nan, "n": float(keep.sum())}
    r = spearmanr(x[keep], y[keep])
    return {"rho": float(r.statistic), "p_value": float(r.pvalue), "n": float(keep.sum())}


def leave_one_modality_out(values: Mapping[str, Mapping[str, float]], config) -> dict[str, dict[str, float]]:
    """Sensitivity analysis: recompute each sample after removing one modality."""
    from .model import calculate_index
    modalities = ("mechanical", "electrical", "molecular")
    out: dict[str, dict[str, float]] = {}
    for sample_id, row in values.items():
        per_sample: dict[str, float] = {}
        for removed in modalities:
            filtered = {k: v for k, v in row.items() if not k.lower().startswith(removed + ":")}
            result = calculate_index(sample_id, filtered, config)
            per_sample[removed] = float(result.composite_score)
        out[sample_id] = per_sample
    return out
