from __future__ import annotations

import numpy as np


def _resample(x: np.ndarray, n: int = 100) -> np.ndarray:
    t = np.linspace(0.0, 1.0, len(x))
    target = np.linspace(0.0, 1.0, n)
    return np.interp(target, t, x)


def beat_templates(signal: np.ndarray, peak_times_s: np.ndarray, fps: float, *, n_points: int = 100) -> tuple[np.ndarray, float, float]:
    """Return mean normalized beat template, shape stability and beat-shape dispersion."""
    x = np.asarray(signal, dtype=float).reshape(-1)
    peaks = np.asarray(peak_times_s, dtype=float)
    if fps <= 0 or peaks.size < 3:
        return np.empty(0), np.nan, np.nan
    indices = np.unique(np.clip(np.rint(peaks * fps).astype(int), 1, x.size - 2))
    beats: list[np.ndarray] = []
    for i in range(1, len(indices)):
        lo, hi = indices[i - 1], indices[i]
        if hi - lo < 4:
            continue
        segment = x[lo:hi]
        centered = segment - np.min(segment)
        scale = np.ptp(centered)
        if scale <= 0:
            continue
        beats.append(_resample(centered / scale, n_points))
    if len(beats) < 2:
        return np.empty(0), np.nan, np.nan
    matrix = np.vstack(beats)
    template = np.mean(matrix, axis=0)
    correlations: list[float] = []
    distances: list[float] = []
    for row in matrix:
        if np.std(row) > 0 and np.std(template) > 0:
            correlations.append(float(np.corrcoef(row, template)[0, 1]))
        distances.append(float(np.sqrt(np.mean((row - template) ** 2))))
    stability = float(np.mean(correlations)) if correlations else np.nan
    dispersion = float(np.mean(distances)) if distances else np.nan
    return template, stability, dispersion
