from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinetics import analyze_trace
from .synthetic import cardiac_motion


@dataclass(frozen=True)
class DetectionBenchmark:
    true_bpm: float
    estimated_bpm: float
    absolute_error_bpm: float
    beat_count_error: int
    passed: bool


def benchmark_synthetic(*, seed: int = 7, bpm: float = 90.0, fps: float = 100.0) -> DetectionBenchmark:
    truth = cardiac_motion(bpm=bpm, fps=fps, seed=seed)
    beats = analyze_trace(truth.motion, fps, min_bpm=30, max_bpm=240)
    estimate = float(np.nanmean([b.beat_rate_bpm for b in beats])) if beats else np.nan
    expected_count = max(0, len(truth.beat_times_s) - 1)
    error = abs(estimate - bpm) if np.isfinite(estimate) else np.inf
    return DetectionBenchmark(bpm, estimate, error, len(beats) - expected_count, bool(error <= 3.0 and abs(len(beats) - expected_count) <= 1))
