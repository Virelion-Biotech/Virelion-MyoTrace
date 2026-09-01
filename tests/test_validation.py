import numpy as np

from fusion.model import FeatureReference, FusionConfig, calculate_index
from fusion.validation import compare_groups, leave_one_modality_out


def test_fusion_reports_confidence_and_coherence() -> None:
    cfg = FusionConfig(references={
        "mechanical:beat_rate_bpm": FeatureReference(60, 120),
        "electrical:fpd_ms": FeatureReference(100, 200),
        "molecular:MYH7_MYH6_ratio": FeatureReference(0.2, 2.0),
    })
    r = calculate_index("x", {"mechanical:beat_rate_bpm": 90, "electrical:fpd_ms": 150, "molecular:MYH7_MYH6_ratio": 1.1}, cfg)
    assert 0 <= r.composite_score <= 100
    assert 0 <= r.coherence_score <= 1
    assert 0 <= r.confidence <= 1
    assert r.uncertainty_width >= 0


def test_group_comparison_has_finite_statistics() -> None:
    scores = {f"a{i}": float(i) for i in range(5)} | {f"b{i}": float(i + 10) for i in range(5)}
    labels = {f"a{i}": "fetal" for i in range(5)} | {f"b{i}": "adult" for i in range(5)}
    r = compare_groups(scores, labels, "fetal", "adult")
    assert r.p_value <= 1
    assert r.n_a == r.n_b == 5
    assert r.delta > 0


def test_leave_one_modality_out_is_defined() -> None:
    cfg = FusionConfig(references={
        "mechanical:x": FeatureReference(0, 1),
        "electrical:x": FeatureReference(0, 1),
        "molecular:x": FeatureReference(0, 1),
    })
    out = leave_one_modality_out({"s": {"mechanical:x": .5, "electrical:x": .5, "molecular:x": .5}}, cfg)
    assert set(out["s"]) == {"mechanical", "electrical", "molecular"}
