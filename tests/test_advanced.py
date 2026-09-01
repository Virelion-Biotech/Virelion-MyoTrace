import numpy as np

from myotrace.advanced import cross_correlation_lag, cycle_average, morphology_similarity, trace_qc
from myotrace.validation_protocol import ReplicatePlan, bootstrap_ci, sensitivity_to_weight


def test_trace_qc_detects_flat_signal():
    q = trace_qc(np.ones(600), 100)
    assert "flat_signal" in q.flags
    assert not q.usable


def test_cross_correlation_alignment():
    x = np.sin(np.linspace(0, 20, 500))
    y = np.roll(x, 7)
    lag, corr = cross_correlation_lag(x, y)
    assert abs(lag) == 7
    assert corr > 0.95


def test_cycle_average_and_similarity():
    t = np.linspace(0, 10, 1000)
    x = np.sin(2 * np.pi * t)
    peaks = np.array([50, 150, 250, 350, 450, 550, 650, 750, 850, 950])
    cycles = cycle_average(x, peaks, n_points=50)
    assert cycles.shape == (9, 50)
    assert morphology_similarity(cycles) > 0.99


def test_bootstrap_is_reproducible():
    a = bootstrap_ci([1, 2, 3, 4], n_boot=1000, seed=11)
    b = bootstrap_ci([1, 2, 3, 4], n_boot=1000, seed=11)
    assert a == b


def test_weight_sensitivity_and_replicate_warnings():
    result = sensitivity_to_weight({"mechanical": 1, "electrical": 1}, {"mechanical": 0.8, "electrical": 0.6})
    assert result["baseline"] == 0.7
    plan = ReplicatePlan(2, 3, 1, 1, False, False)
    assert len(plan.warnings()) == 5
