import math

from fusion.model import FeatureReference, FusionConfig, calculate_index


def test_fusion_returns_100_for_adult_endpoints() -> None:
    refs = {
        "mechanical:mean_bpm": FeatureReference(60, 120),
        "electrical:fpd_ms": FeatureReference(150, 300),
        "molecular:MYH7_MYH6_ratio": FeatureReference(0.2, 2.0),
    }
    values = {k: refs[k].adult for k in refs}
    result = calculate_index("adult-1", values, FusionConfig(refs))
    assert math.isclose(result.composite_score, 100.0)
    assert result.status == "complete"


def test_partial_modality_is_explicit() -> None:
    refs = {"mechanical:mean_bpm": FeatureReference(60, 120)}
    result = calculate_index("x", {"mechanical:mean_bpm": 90}, FusionConfig(refs))
    assert 0 < result.composite_score < 100
    assert result.status == "partial"
    assert result.coverage == 1 / 3
