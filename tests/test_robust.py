import numpy as np

from myotrace.robust import assess_signal_quality, robust_preprocess, spectral_features
from myotrace.synthetic import cardiac_motion


def test_synthetic_signal_has_expected_frequency() -> None:
    s = cardiac_motion(duration_s=20, fps=100, bpm=90, noise_sd=0.01)
    features = spectral_features(s.motion, s.fps)
    assert abs(features["dominant_frequency_hz"] - 1.5) < 0.15


def test_quality_score_is_bounded() -> None:
    s = cardiac_motion(duration_s=12, fps=100, bpm=90, noise_sd=0.02)
    x = robust_preprocess(s.motion, s.fps)
    q = assess_signal_quality(x, s.fps)
    assert 0 <= q.quality_score <= 1
    assert q.missing_fraction == 0
