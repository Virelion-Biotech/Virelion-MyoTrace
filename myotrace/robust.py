from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import detrend, welch, correlate, savgol_filter


try:
    _trapz = np.trapezoid  # NumPy >= 2.0
except AttributeError:  # pragma: no cover - exercised on NumPy < 2.0
    _trapz = np.trapz


@dataclass(frozen=True)
class SignalQuality:
    snr_db: float
    periodicity: float
    dominant_frequency_hz: float
    drift_fraction: float
    clipping_fraction: float
    missing_fraction: float
    quality_score: float
    flags: tuple[str, ...]


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("signal is empty")
    missing = ~np.isfinite(x)
    if not missing.any():
        return x
    med = np.median(x[np.isfinite(x)]) if np.isfinite(x).any() else np.nan
    if not np.isfinite(med):
        raise ValueError("signal contains no finite values")
    return np.nan_to_num(x, nan=med, posinf=med, neginf=med)


def robust_preprocess(signal: Iterable[float], fps: float, *, median_kernel_s: float = 0.15) -> np.ndarray:
    """Prepare a motion trace while preserving beat-scale morphology."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    raw = np.asarray(list(signal), dtype=float)
    x = _finite(raw)
    if x.size < 9:
        raise ValueError("signal is too short")
    if not np.isfinite(median_kernel_s) or median_kernel_s <= 0:
        raise ValueError("median_kernel_s must be positive finite")
    if np.ptp(x) == 0:
        return np.zeros_like(x)
    k = max(3, int(round(median_kernel_s * fps)) | 1)
    if k >= x.size:
        k = x.size - 1 if x.size % 2 == 0 else x.size
    if k >= 3:
        x = median_filter(x, size=k, mode="nearest")
    x = detrend(x, type="linear")
    scale = float(np.nanpercentile(np.abs(x), 95))
    return x / scale if scale > 0 else x


def spectral_features(
    signal: Iterable[float], fps: float, *, min_hz: float = 0.2, max_hz: float = 5.0
) -> dict[str, float]:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    if not np.isfinite(min_hz) or not np.isfinite(max_hz) or not 0 <= min_hz < max_hz:
        raise ValueError("Require 0 <= min_hz < max_hz")
    x = _finite(np.asarray(list(signal), dtype=float))
    nperseg = min(x.size, max(16, int(fps * 8)))
    freqs, power = welch(x, fs=fps, nperseg=nperseg)
    keep = (freqs >= min_hz) & (freqs <= min(max_hz, fps / 2))
    if not np.any(keep):
        return {"dominant_frequency_hz": np.nan, "spectral_entropy": np.nan, "band_power": 0.0}
    p, f = power[keep], freqs[keep]
    if np.sum(p) <= np.finfo(float).eps:
        return {"dominant_frequency_hz": np.nan, "spectral_entropy": np.nan, "band_power": 0.0}
    idx = int(np.argmax(p))
    prob = p / max(float(np.sum(p)), np.finfo(float).eps)
    entropy = float(-np.sum(prob * np.log(prob + 1e-12)) / np.log(max(2, len(prob))))
    band_power = float(_trapz(p, f)) if len(f) > 1 else float(p[0])
    return {"dominant_frequency_hz": float(f[idx]), "spectral_entropy": entropy, "band_power": band_power}


def assess_signal_quality(signal: Iterable[float], fps: float) -> SignalQuality:
    raw = np.asarray(list(signal), dtype=float).reshape(-1)
    missing_fraction = float(np.mean(~np.isfinite(raw))) if raw.size else 1.0
    x = _finite(raw)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    spec = spectral_features(x, fps)
    # Smoothing-residual ratio is a descriptive proxy, not calibrated physical SNR.
    k = min(max(5, int(round(fps * 0.1)) | 1), x.size if x.size % 2 else x.size - 1)
    smooth = savgol_filter(x, k, min(3, k - 1)) if k >= 3 else np.full_like(x, np.mean(x))
    noise = float(np.std(x - smooth)) + 1e-9
    signal_scale = float(np.std(smooth)) + 1e-9
    snr_db = float(20 * np.log10(signal_scale / noise))
    periodicity = beat_periodicity(x, fps)
    slope = float(np.polyfit(np.arange(x.size), x, 1)[0]) if x.size > 1 else 0.0
    drift = abs(slope) * x.size / (np.std(x) + 1e-9)
    span = np.ptp(x)
    clipping = float(np.mean((x <= np.min(x) + span * 1e-6) | (x >= np.max(x) - span * 1e-6))) if span else 1.0
    flags: list[str] = []
    if snr_db < 6:
        flags.append("low_snr")
    if periodicity < 0.05:
        flags.append("weak_periodicity")
    if drift > 0.25:
        flags.append("residual_drift")
    if clipping > 0.05:
        flags.append("possible_clipping")
    if missing_fraction > 0:
        flags.append("missing_or_nonfinite_samples")
    if not np.isfinite(spec["dominant_frequency_hz"]):
        flags.append("no_dominant_frequency")
    q = float(np.clip((snr_db - 3) / 12, 0, 1)) * float(np.clip((periodicity + 0.1) / 0.6, 0, 1))
    q *= float(np.clip(1 - drift, 0, 1)) * float(np.clip(1 - clipping * 5, 0, 1)) * (1 - missing_fraction)
    return SignalQuality(
        snr_db, periodicity, spec["dominant_frequency_hz"], drift, clipping, missing_fraction, q, tuple(flags)
    )


def beat_periodicity(signal: np.ndarray, fps: float) -> float:
    """Maximum autocorrelation at 0.25–2 s lags; excludes lag-one smoothness."""
    x = np.asarray(signal, dtype=float)
    x = x - np.mean(x)
    ac = correlate(x, x, mode="full", method="fft")[x.size - 1 :]
    lo, hi = max(1, int(np.ceil(fps * 0.25))), min(x.size, int(fps * 2) + 1)
    if ac[0] <= np.finfo(float).eps or hi <= lo:
        return 0.0
    return float(np.clip(np.max(ac[lo:hi]) / ac[0], 0, 1))
