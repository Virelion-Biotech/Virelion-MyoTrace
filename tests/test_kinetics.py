import numpy as np

from myotrace.kinetics import analyze_trace, summarize_beats


def test_analyze_trace_detects_synthetic_beats() -> None:
    fps = 100.0
    t = np.arange(0, 12, 1 / fps)
    signal = 0.5 + 0.2 * (1 + np.sin(2 * np.pi * 1.5 * t))
    beats = analyze_trace(signal, fps)
    assert len(beats) >= 10
    summary = summarize_beats(beats)
    assert 85 < summary["mean_bpm"] < 95
