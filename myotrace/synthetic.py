from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticTrace:
    time_s: np.ndarray
    motion: np.ndarray
    beat_times_s: np.ndarray
    fps: float


def cardiac_motion(*, duration_s: float = 20.0, fps: float = 100.0, bpm: float = 90.0, amplitude: float = 1.0, rise_fraction: float = 0.30, noise_sd: float = 0.03, drift: float = 0.0, seed: int = 7) -> SyntheticTrace:
    """Generate a deterministic asymmetric cardiac-like contraction waveform for unit tests."""
    if duration_s <= 1 or fps <= 0 or bpm <= 0:
        raise ValueError("duration_s, fps and bpm must be positive; duration_s must exceed 1 s")
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fps))
    t = np.arange(n) / fps
    period = 60.0 / bpm
    phase = np.mod(t, period) / period
    rise = np.clip(phase / max(rise_fraction, 1e-3), 0, 1)
    fall = np.clip((1 - phase) / max(1 - rise_fraction, 1e-3), 0, 1)
    pulse = (np.minimum(rise, fall) ** 1.5) * amplitude
    signal = pulse + drift * (t - t.mean()) / max(duration_s, 1) + rng.normal(0, noise_sd, n)
    beat_times = np.arange(0, duration_s, period)
    return SyntheticTrace(t, signal, beat_times, fps)
