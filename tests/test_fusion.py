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
    assert result.status == "low_coverage"
    assert result.coverage == 1 / 3
    assert result.coverage < FusionConfig(refs).minimum_modality_coverage


def test_lower_adult_reference_preserves_fetal_and_adult_endpoints():
    from fusion.model import FeatureReference
    for transform in ("linear", "log"):
        reference = FeatureReference(fetal=150, adult=100, higher_is_mature=False, transform=transform)
        assert reference.score(150) == 0.0
        assert reference.score(100) == 1.0
        assert 0.0 < reference.score(125) < 1.0
