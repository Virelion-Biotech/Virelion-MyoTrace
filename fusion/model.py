from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping


@dataclass(frozen=True)
class FeatureReference:
    fetal: float
    adult: float
    higher_is_mature: bool = True
    weight: float = 1.0

    def score(self, value: float) -> float:
        value = float(value)
        if not isfinite(value):
            return float("nan")
        lo, hi = sorted((self.fetal, self.adult))
        if hi == lo:
            return 0.5
        raw = (value - lo) / (hi - lo)
        score = raw if self.adult >= self.fetal else 1.0 - raw
        if not self.higher_is_mature:
            score = 1.0 - score
        return max(0.0, min(1.0, score))


@dataclass(frozen=True)
class FusionConfig:
    references: Mapping[str, FeatureReference] = field(default_factory=dict)
    modality_weights: Mapping[str, float] = field(
        default_factory=lambda: {"mechanical": 1 / 3, "electrical": 1 / 3, "molecular": 1 / 3}
    )

    def normalized_modality_weights(self) -> dict[str, float]:
        clean = {k: max(0.0, float(v)) for k, v in self.modality_weights.items()}
        total = sum(clean.values())
        if total <= 0:
            raise ValueError("At least one modality weight must be positive")
        return {k: v / total for k, v in clean.items()}


@dataclass(frozen=True)
class FusionResult:
    sample_id: str
    composite_score: float
    modality_scores: dict[str, float]
    feature_scores: dict[str, float]
    coverage: float
    status: str


def _feature_modality(name: str) -> str:
    prefix = name.split(":", 1)[0].lower()
    if prefix in {"mechanical", "electrical", "molecular"}:
        return prefix
    return "molecular"


def calculate_index(sample_id: str, values: Mapping[str, float], config: FusionConfig) -> FusionResult:
    """Calculate a 0-100 multimodal maturity score from user-supplied adult/fetal references.

    No biological cutoffs are silently hard-coded. The caller supplies reference endpoints,
    allowing calibration to an explicit cell model, assay, species, and laboratory protocol.
    """
    weights = config.normalized_modality_weights()
    feature_scores: dict[str, float] = {}
    by_modality: dict[str, list[tuple[float, float]]] = {}
    for feature, value in values.items():
        ref = config.references.get(feature)
        if ref is None:
            continue
        score = ref.score(float(value))
        if score != score:
            continue
        feature_scores[feature] = score
        modality = _feature_modality(feature)
        by_modality.setdefault(modality, []).append((score, max(0.0, ref.weight)))
    modality_scores = {
        modality: sum(s * w for s, w in rows) / sum(w for _, w in rows)
        for modality, rows in by_modality.items()
        if rows and sum(w for _, w in rows) > 0
    }
    contributing = [(modality_scores[m], weights.get(m, 0.0)) for m in modality_scores if weights.get(m, 0.0) > 0]
    if not contributing:
        return FusionResult(sample_id, float("nan"), {}, feature_scores, 0.0, "insufficient_features")
    numerator = sum(score * weight for score, weight in contributing)
    denominator = sum(weight for _, weight in contributing)
    coverage = denominator
    composite = 100.0 * numerator / denominator
    status = "complete" if coverage >= 0.999 else "partial"
    return FusionResult(sample_id, composite, modality_scores, feature_scores, coverage, status)
