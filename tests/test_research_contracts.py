import json

import numpy as np
import pandas as pd
import pytest

from fusion.model import FeatureReference, FusionConfig, calculate_index
from fusion.integrate import assemble_fusion_features, row_to_feature_mapping
from fusion.report import write_json, results_table
from fusion.benchmark import ReferenceSet, build_config, score_samples, separation_summary
from fusion.validation import rank_correlation
from fusion.markers import marker_panel_dict, MarkerDefinition
from fusion.agreement import bland_altman
from myotrace.uncertainty import bootstrap_mean, bootstrap_statistic
from myotrace.validation_protocol import bootstrap_ci
from myotrace.robust import assess_signal_quality, robust_preprocess, spectral_features
from myotrace.advanced import trace_qc, cycle_average, cross_correlation_lag
from myotrace.morphology import beat_templates
from myotrace.calibration import fit_force_calibration
from myotrace.provenance import sha256_file, build_provenance, write_json as write_provenance


@pytest.mark.parametrize(
    "args",
    [
        {"fetal": 0, "adult": 0},
        {"fetal": np.nan, "adult": 1},
        {"fetal": 0, "adult": 1, "weight": -1},
        {"fetal": 0, "adult": 1, "transform": "log"},
        {"fetal": 0, "adult": 1, "transform": "sqrt"},
    ],
)
def test_reference_validation(args):
    with pytest.raises(ValueError):
        FeatureReference(**args)


@pytest.mark.parametrize("weights", [{"mechanical": -1}, {"unknown": 1}, {"mechanical": np.nan}, {"mechanical": 0}])
def test_weight_validation(weights):
    with pytest.raises(ValueError):
        FusionConfig(modality_weights=weights)


def test_missing_features_and_zero_weight_modality():
    config = FusionConfig(
        references={
            "mechanical:a": FeatureReference(0, 1),
            "mechanical:b": FeatureReference(0, 1),
            "electrical:c": FeatureReference(0, 1),
        },
        modality_weights={"mechanical": 1, "electrical": 0},
    )
    result = calculate_index("a", {"mechanical:a": 1, "electrical:c": 0}, config)
    assert result.composite_score == 100
    assert result.coverage == 0.5
    assert result.coherence_score == 1
    assert result.status == "partial"
    assert result.to_dict()["uncertainty_kind"] == "heuristic_not_confidence_interval"


def test_fusion_join_benchmark_and_exports(tmp_path):
    mechanical = pd.DataFrame({"sample_id": ["a", "a", "b"], "amplitude": [0, 0, 1]})
    electrical = pd.DataFrame({"sample_id": ["b", "c"], "fpd_ms": [200, 100]})
    molecular = pd.DataFrame({"sample_id": ["c"], "TNNI3_TNNI1_ratio": [2]})
    table = assemble_fusion_features(mechanical, electrical=electrical, molecular=molecular)
    assert len(table) == 3
    assert row_to_feature_mapping({"sample_id": "a", "x": "bad", "y": None, "z": "2"}) == {"z": 2}
    refs = ReferenceSet("technical", {"mechanical:amplitude": FeatureReference(0, 1)})
    config = build_config(refs, {"mechanical": 1})
    results = score_samples({"a": {"mechanical:amplitude": 0}, "b": {"mechanical:amplitude": 1}, "empty": {}}, config)
    assert separation_summary(results, {"a": "fetal", "b": "adult"})["adult_vs_fetal_auc"] == 1
    path = tmp_path / "fusion.json"
    write_json(results, path)
    assert json.loads(path.read_text(), parse_constant=lambda v: pytest.fail(v))[-1]["composite_score"] is None
    assert len(results_table(results)) == 3
    assert marker_panel_dict({"a": 1}) == {"a": 1}
    with pytest.raises(ValueError):
        MarkerDefinition("x", "bad")
    with pytest.raises(ValueError):
        rank_correlation([1, 2, 3], [1])
    assert rank_correlation([1, 2, 3], [1, 2, 3])["rho"] == 1
    assert np.isnan(rank_correlation([1], [1])["rho"])


@pytest.mark.parametrize("function", [bootstrap_mean, bootstrap_statistic, bootstrap_ci])
@pytest.mark.parametrize("args", [{"n_boot": 0}, {"n_boot": 100.5}, {"alpha": 0}, {"alpha": np.nan}, {"alpha": 1}])
def test_bootstrap_parameters(function, args):
    positional = ([1, 2, 3], np.mean) if function == bootstrap_statistic else ([1, 2, 3],)
    with pytest.raises(ValueError):
        function(*positional, **args)


def test_bootstrap_and_calibration(tmp_path):
    a = bootstrap_mean(np.arange(20), n_boot=100, seed=9)
    b = bootstrap_mean(np.arange(20), n_boot=100, seed=9)
    assert a == b and a.lower < a.estimate < a.upper
    assert bootstrap_statistic([1, 2, 3], np.mean, n_boot=100).n_boot == 100
    assert bootstrap_mean([]).n_boot == 0
    assert bootstrap_statistic([], np.mean).n_boot == 0
    x = np.arange(10.0)
    cal = fit_force_calibration(x, 3 * x + 2, units="uN")
    np.testing.assert_allclose(cal.predict(np.array([10, 11])), [32, 35])
    for args in [([1, 1, 1], [1, 2, 3]), ([1], [1]), ([np.nan] * 3, [1, 2, 3])]:
        with pytest.raises(ValueError):
            fit_force_calibration(*args)
    with pytest.raises(ValueError):
        bland_altman([1], [1])
    with pytest.raises(ValueError):
        bland_altman([np.nan, 1], [1, 1])
    path = tmp_path / "input"
    path.write_bytes(b"data")
    p = build_provenance(path, version="test", parameters={"fps": 100})
    assert p.source_sha256 == sha256_file(path)
    write_provenance(p, tmp_path / "provenance.json")


def test_signal_quality_is_not_lag_one_smoothness():
    fps = 100
    t = np.arange(1200) / fps
    clean = assess_signal_quality(np.sin(2 * np.pi * 1.5 * t), fps)
    assert clean.quality_score > 0.5
    assert clean.periodicity > 0.8
    noise = assess_signal_quality(np.random.default_rng(8).normal(size=t.size), fps)
    assert noise.periodicity < 0.15
    assert noise.quality_score < clean.quality_score
    assert np.isnan(spectral_features(np.ones(100), fps)["dominant_frequency_hz"])
    mixed = np.sin(2 * np.pi * t)
    mixed[0:3] = [np.nan, np.inf, -np.inf]
    assert assess_signal_quality(mixed, fps).missing_fraction == 3 / 1200
    assert np.isfinite(robust_preprocess(mixed, fps)).all()
    assert not trace_qc(np.full(100, np.nan), fps).usable
    for x in [[], [np.nan] * 10, [np.inf] * 10]:
        with pytest.raises(ValueError):
            robust_preprocess(x, fps)
    with pytest.raises(ValueError):
        spectral_features(np.ones(100), np.nan)
    with pytest.raises(ValueError):
        cross_correlation_lag([1, np.nan, 3], [1, 2, 3])
    assert cross_correlation_lag(np.ones(10), np.ones(10)) == (0, 0)


def test_templates_and_cycle_bounds():
    fps = 100
    t = np.arange(1000) / fps
    x = np.sin(2 * np.pi * t)
    template, stability, dispersion = beat_templates(x, np.arange(0.25, 9.5, 1), fps)
    assert len(template) == 100 and stability > 0.99 and dispersion < 0.01
    assert beat_templates(np.ones(1000), np.arange(1, 9), fps)[0].size == 0
    assert beat_templates(x, np.array([1]), fps)[0].size == 0
    with pytest.raises(ValueError):
        cycle_average(x, [-1, 100])


def test_public_utilities_invalid_and_empty_contracts():
    from myotrace.benchmark import benchmark_synthetic
    from myotrace.kinetics import prepare_signal
    from myotrace.qc import assess_frames
    from myotrace.schema import SampleRecord
    from myotrace.serialization import json_safe

    assert benchmark_synthetic().passed
    assert SampleRecord("a", {"x": 1}).to_frame().sample_id.iloc[0] == "a"
    assert json_safe(np.float64(np.inf)) is None
    with pytest.raises(ValueError):
        prepare_signal(np.ones(100), np.nan)
    with pytest.raises(ValueError):
        prepare_signal(np.ones(5), 100)
    with pytest.raises(ValueError):
        prepare_signal(np.full(100, np.nan), 30)
    assert np.isfinite(prepare_signal(np.r_[np.nan, np.arange(100.0)], 30)).all()
    with pytest.raises(ValueError):
        assess_frames(np.ones((3, 32, 32)), 0)
    with pytest.raises(ValueError):
        robust_preprocess(np.ones(20), 30, median_kernel_s=-1)
    with pytest.raises(ValueError):
        spectral_features(np.ones(20), 30, min_hz=2, max_hz=1)
    assert spectral_features(np.ones(2), 1, min_hz=0.4, max_hz=0.5)["band_power"] == 0
    with pytest.raises(ValueError):
        assemble_fusion_features(pd.DataFrame({"sample_id": ["a"]}))
    with pytest.raises(ValueError):
        assemble_fusion_features(pd.DataFrame({"amplitude": [1]}))
