from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .model import FeatureReference, FusionConfig, FusionResult, calculate_index


@dataclass(frozen=True)
class ReferenceSet:
    name: str
    references: Mapping[str, FeatureReference]
    metadata: Mapping[str, str] | None = None


def build_config(reference_set: ReferenceSet, modality_weights: Mapping[str, float] | None = None) -> FusionConfig:
    return FusionConfig(references=reference_set.references, modality_weights=modality_weights or {"mechanical": 1 / 3, "electrical": 1 / 3, "molecular": 1 / 3})


def score_samples(samples: Mapping[str, Mapping[str, float]], config: FusionConfig) -> list[FusionResult]:
    return [calculate_index(sample_id, values, config) for sample_id, values in samples.items()]


def separation_summary(results: list[FusionResult], labels: Mapping[str, str]) -> dict[str, float]:
    """Descriptive adult-vs-fetal benchmark with rank-based discrimination statistics."""
    groups: dict[str, list[float]] = {}
    for r in results:
        label = labels.get(r.sample_id)
        if label is not None and np.isfinite(r.composite_score):
            groups.setdefault(label, []).append(r.composite_score)
    summary: dict[str, float] = {"n_scored": float(sum(map(len, groups.values())))}
    for label, vals in groups.items():
        summary[f"{label}_n"] = float(len(vals))
        summary[f"{label}_mean"] = float(np.mean(vals))
        summary[f"{label}_median"] = float(np.median(vals))
        summary[f"{label}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    if "adult" in groups and "fetal" in groups:
        adult, fetal = np.asarray(groups["adult"]), np.asarray(groups["fetal"])
        delta = float(np.median(adult) - np.median(fetal))
        # AUC is exactly the normalized Mann-Whitney U statistic for this binary ranking task.
        pairwise = sum(float(a > f) + 0.5 * float(a == f) for a in adult for f in fetal)
        auc = pairwise / max(1, len(adult) * len(fetal))
        summary.update({"adult_minus_fetal_median": delta, "adult_vs_fetal_auc": float(auc)})
    return summary
