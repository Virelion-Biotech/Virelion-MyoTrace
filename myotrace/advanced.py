from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.signal import correlate, find_peaks, welch


@dataclass(frozen=True)
class TraceQC:
    n_samples: int
    duration_s: float
    sampling_hz: float
    finite_fraction: float
    dynamic_range: float
    rms: float
    dominant_frequency_hz: float
    periodicity: float
    drift_slope_per_s: float
    saturation_fraction: float
    usable: bool
    flags: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def trace_qc(signal: np.ndarray, fps: float, *, min_duration_s: float = 5.0) -> TraceQC:
    """Quantify signal integrity without changing the underlying measurement."""
    x = np.asarray(signal, dtype=float).reshape(-1)
    if fps <= 0:
        raise ValueError("fps must be positive")
    finite = np.isfinite(x)
    y = x[finite]
    flags: list[str] = []
    if y.size == 0:
        flags.append("no_finite_samples")
        return TraceQC(len(x), len(x) / fps, fps, 0.0, 0.0, 0.0, np.nan, 0.0, np.nan, 0.0, False, tuple(flags))
    median = float(np.median(y))
    filled = np.nan_to_num(x, nan=median, posinf=median, neginf=median)
    dynamic = float(np.percentile(filled, 99) - np.percentile(filled, 1))
    rms = float(np.sqrt(np.mean((filled - np.mean(filled)) ** 2)))
    t = np.arange(len(filled), dtype=float) / fps
    slope = float(np.polyfit(t, filled, 1)[0]) if len(filled) > 1 else np.nan
    if len(filled) >= 8:
        f, p = welch(filled - np.mean(filled), fs=fps, nperseg=min(len(filled), max(8, int(fps * 10))))
        mask = f > 0
        dom = float(f[mask][np.argmax(p[mask])]) if np.any(mask) else np.nan
        ac = correlate(filled - np.mean(filled), filled - np.mean(filled), mode="full")
        ac = ac[len(ac) // 2:]
        periodicity = float(ac[min(len(ac) - 1, max(1, int(fps * 1.0)))] / max(ac[0], np.finfo(float).eps))
    else:
        dom, periodicity = np.nan, 0.0
    sat = float(np.mean((filled <= np.min(filled)) | (filled >= np.max(filled)))) if len(filled) else 0.0
    if len(x) / fps < min_duration_s:
        flags.append("short_recording")
    if np.mean(finite) < 0.995:
        flags.append("missing_samples")
    if dynamic <= np.finfo(float).eps:
        flags.append("flat_signal")
    if np.isfinite(periodicity) and periodicity < 0.05:
        flags.append("weak_periodicity")
    usable = not any(f in flags for f in ("no_finite_samples", "short_recording", "flat_signal"))
    return TraceQC(len(x), len(x) / fps, fps, float(np.mean(finite)), dynamic, rms, dom, periodicity, slope, sat, usable, tuple(flags))


def cross_correlation_lag(reference: np.ndarray, candidate: np.ndarray) -> tuple[int, float]:
    """Return integer-sample lag and normalized correlation for modality alignment."""
    a = np.asarray(reference, dtype=float).reshape(-1)
    b = np.asarray(candidate, dtype=float).reshape(-1)
    n = min(len(a), len(b))
    if n < 3:
        raise ValueError("Signals must contain at least 3 samples")
    a, b = a[:n] - np.mean(a[:n]), b[:n] - np.mean(b[:n])
    c = correlate(a, b, mode="full")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return 0, 0.0
    i = int(np.argmax(c))
    lag = i - (n - 1)
    return lag, float(c[i] / denom)


def cycle_average(signal: np.ndarray, peaks: np.ndarray, *, n_points: int = 200) -> np.ndarray:
    """Resample individual cycles onto phase [0,1] for morphology comparison."""
    x = np.asarray(signal, dtype=float).reshape(-1)
    p = np.asarray(peaks, dtype=int)
    if len(p) < 2 or n_points < 10:
        raise ValueError("At least two peaks and >=10 phase points are required")
    cycles = []
    phase = np.linspace(0.0, 1.0, n_points)
    for a, b in zip(p[:-1], p[1:]):
        if b <= a + 2:
            continue
        t = np.linspace(0.0, 1.0, b - a, endpoint=True)
        cycles.append(np.interp(phase, t, x[a:b]))
    if not cycles:
        raise ValueError("No valid cycles")
    return np.asarray(cycles)


def morphology_similarity(cycles: np.ndarray) -> float:
    """Mean pairwise Pearson similarity of normalized beat waveforms."""
    c = np.asarray(cycles, dtype=float)
    if c.ndim != 2 or c.shape[0] < 2:
        return np.nan
    z = c - np.mean(c, axis=1, keepdims=True)
    norms = np.linalg.norm(z, axis=1)
    valid = norms > 0
    if valid.sum() < 2:
        return np.nan
    z = z[valid] / norms[valid, None]
    corr = z @ z.T
    tri = corr[np.triu_indices_from(corr, k=1)]
    return float(np.mean(tri)) if tri.size else np.nan
