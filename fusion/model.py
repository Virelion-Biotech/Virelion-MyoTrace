from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class FeatureReference:
    fetal: float
    adult: float
    higher_is_mature: bool = True
    weight: float = 1.0
    transform: str = "linear"

    def _transform(self, value: float) -> tuple[float, float, float]:
        f, a, v = float(self.fetal), float(self.adult), float(value)
        if self.transform == "log":
            if min(f, a, v) <= 0:
                raise ValueError("log reference requires positive values")
            return np.log(f), np.log(a), np.log(v)
        return f, a, v

    def score(self, value: float) -> float:
        value = float(value)
        if not isfinite(value):
            return float("nan")
        f, a, v = self._transform(value)
        if a == f:
            return 0.5
        raw = (v - f) / (a - f)
        score = raw if self.higher_is_mature else 1.0 - raw
        return float(np.clip(score, 0.0, 1.0))


@dataclass(frozen=True)
class FusionConfig:
    references: Mapping[str, FeatureReference] = field(default_factory=dict)
    modality_weights: Mapping[str, float] = field(default_factory=lambda: {"mechanical": 1 / 3, "electrical": 1 / 3, "molecular": 1 / 3})
    minimum_modality_coverage: float = 0.50
    contradiction_threshold: float = 0.45

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
    coherence_score: float
    confidence: float
    uncertainty_width: float

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "composite_score": self.composite_score,
            "coverage": self.coverage,
            "status": self.status,
            "coherence_score": self.coherence_score,
            "confidence": self.confidence,
            "uncertainty_width": self.uncertainty_width,
            "modality_scores": self.modality_scores,
            "feature_scores": self.feature_scores,
        }


def _feature_modality(name: str) -> str:
    prefix = name.split(":", 1)[0].lower()
    return prefix if prefix in {"mechanical", "electrical", "molecular"} else "molecular"


def calculate_index(sample_id: str, values: Mapping[str, float], config: FusionConfig) -> FusionResult:
    weights = config.normalized_modality_weights()
    feature_scores: dict[str, float] = {}
    by_modality: dict[str, list[tuple[float, float]]] = {}
    for feature, value in values.items():
        ref = config.references.get(feature)
        if ref is None:
            continue
        try:
            score = ref.score(float(value))
        except (TypeError, ValueError, FloatingPointError):
            continue
        if not isfinite(score):
            continue
        feature_scores[feature] = score
        modality = _feature_modality(feature)
        by_modality.setdefault(modality, []).append((score, max(0.0, ref.weight)))

    modality_scores = {
        modality: float(sum(s * w for s, w in rows) / sum(w for _, w in rows))
        for modality, rows in by_modality.items()
        if rows and sum(w for _, w in rows) > 0
    }
    contributing = [(modality_scores[m], weights[m]) for m in modality_scores if weights.get(m, 0.0) > 0]
    if not contributing:
        return FusionResult(sample_id, np.nan, {}, feature_scores, 0.0, "insufficient_features", np.nan, 0.0, np.nan)

    numerator = sum(score * weight for score, weight in contributing)
    denominator = sum(weight for _, weight in contributing)
    composite = 100.0 * numerator / denominator
    coverage = float(denominator)

    # Agreement is high when modalities are pointing to a similar maturity state.
    if len(modality_scores) == 1:
        coherence = 1.0
    else:
        vals = np.array(list(modality_scores.values()), dtype=float)
        coherence = float(np.clip(1.0 - np.std(vals, ddof=0) / 0.5, 0.0, 1.0))
    uncertainty_width = float(100.0 * (1.0 - coherence) + 30.0 * (1.0 - coverage))
    confidence = float(np.clip(coverage * (0.5 + 0.5 * coherence), 0.0, 1.0))

    status = "complete" if coverage >= 0.999 else "partial"
    if coverage < config.minimum_modality_coverage:
        status = "low_coverage"
    if len(modality_scores) >= 2 and coherence < (1.0 - config.contradiction_threshold):
        status = "discordant_modalities"
    return FusionResult(sample_id, float(np.clip(composite, 0.0, 100.0)), modality_scores, feature_scores, coverage, status, coherence, confidence, uncertainty_width)
