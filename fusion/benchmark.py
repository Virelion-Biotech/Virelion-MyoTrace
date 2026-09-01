from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .model import FeatureReference, FusionConfig, FusionResult, calculate_index


@dataclass(frozen=True)
class ReferenceSet:
    name: str
    references: Mapping[str, FeatureReference]


def build_config(reference_set: ReferenceSet, modality_weights: Mapping[str, float] | None = None) -> FusionConfig:
    return FusionConfig(
        references=reference_set.references,
        modality_weights=modality_weights or {"mechanical": 1 / 3, "electrical": 1 / 3, "molecular": 1 / 3},
    )


def score_samples(samples: Mapping[str, Mapping[str, float]], config: FusionConfig) -> list[FusionResult]:
    return [calculate_index(sample_id, values, config) for sample_id, values in samples.items()]


def separation_summary(results: list[FusionResult], labels: Mapping[str, str]) -> dict[str, float]:
    """Descriptive benchmark; this is not a model-validation statistic."""
    groups: dict[str, list[float]] = {}
    for r in results:
        label = labels.get(r.sample_id)
        if label is not None and np.isfinite(r.composite_score):
            groups.setdefault(label, []).append(r.composite_score)
    summary: dict[str, float] = {"n_scored": float(sum(map(len, groups.values())))}
    for label, vals in groups.items():
        summary[f"{label}_n"] = float(len(vals))
        summary[f"{label}_mean"] = float(np.mean(vals))
        summary[f"{label}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    if "adult" in groups and "fetal" in groups:
        summary["adult_minus_fetal"] = float(np.mean(groups["adult"]) - np.mean(groups["fetal"]))
    return summary
