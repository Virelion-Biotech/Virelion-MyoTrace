from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.signal import detrend, medfilt, welch


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
    med = np.nanmedian(x)
    if not np.isfinite(med):
        raise ValueError("signal contains no finite values")
    return np.nan_to_num(x, nan=med, posinf=med, neginf=med)


def robust_preprocess(signal: Iterable[float], fps: float, *, median_kernel_s: float = 0.15) -> np.ndarray:
    """Prepare a motion trace while preserving beat-scale morphology."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    raw = np.asarray(list(signal), dtype=float)
    x = _finite(raw)
    if x.size < 9:
        raise ValueError("signal is too short")
    k = max(3, int(round(median_kernel_s * fps)) | 1)
    if k >= x.size:
        k = x.size - 1 if x.size % 2 == 0 else x.size
    if k >= 3:
        x = medfilt(x, kernel_size=k)
    x = detrend(x, type="linear")
    scale = float(np.nanpercentile(np.abs(x), 95))
    return x / scale if scale > 0 else x


def spectral_features(signal: Iterable[float], fps: float, *, min_hz: float = 0.2, max_hz: float = 5.0) -> dict[str, float]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    x = _finite(np.asarray(list(signal), dtype=float))
    nperseg = min(x.size, max(16, int(fps * 8)))
    freqs, power = welch(x, fs=fps, nperseg=nperseg)
    keep = (freqs >= min_hz) & (freqs <= min(max_hz, fps / 2))
    if not np.any(keep):
        return {"dominant_frequency_hz": np.nan, "spectral_entropy": np.nan, "band_power": 0.0}
    p, f = power[keep], freqs[keep]
    idx = int(np.argmax(p))
    prob = p / max(float(np.sum(p)), np.finfo(float).eps)
    entropy = float(-np.sum(prob * np.log(prob + 1e-12)) / np.log(max(2, len(prob))))
    band_power = float(np.trapz(p, f)) if len(f) > 1 else float(p[0])
    return {"dominant_frequency_hz": float(f[idx]), "spectral_entropy": entropy, "band_power": band_power}


def assess_signal_quality(signal: Iterable[float], fps: float) -> SignalQuality:
    raw = np.asarray(list(signal), dtype=float).reshape(-1)
    missing_fraction = float(np.mean(~np.isfinite(raw))) if raw.size else 1.0
    x = _finite(raw)
    if fps <= 0:
        raise ValueError("fps must be positive")
    spec = spectral_features(x, fps)
    noise = float(np.median(np.abs(x - np.median(x)))) * 1.4826 + 1e-9
    signal_scale = float(np.std(x)) + 1e-9
    snr_db = float(20 * np.log10(signal_scale / noise))
    ac = np.correlate(x - np.mean(x), x - np.mean(x), mode="full")[x.size - 1:]
    periodicity = float(ac[1] / ac[0]) if ac.size > 1 and ac[0] > 0 else 0.0
    slope = float(np.polyfit(np.arange(x.size), x, 1)[0])
    drift = abs(slope) * x.size / (np.std(x) + 1e-9)
    span = np.ptp(x)
    clipping = float(np.mean((x <= np.min(x) + span * 1e-6) | (x >= np.max(x) - span * 1e-6))) if span else 1.0
    flags: list[str] = []
    if snr_db < 6: flags.append("low_snr")
    if periodicity < 0.05: flags.append("weak_periodicity")
    if drift > 0.25: flags.append("residual_drift")
    if clipping > 0.05: flags.append("possible_clipping")
    if missing_fraction > 0: flags.append("missing_or_nonfinite_samples")
    if not np.isfinite(spec["dominant_frequency_hz"]): flags.append("no_dominant_frequency")
    q = float(np.clip((snr_db - 3) / 12, 0, 1)) * float(np.clip((periodicity + 0.1) / 0.6, 0, 1))
    q *= float(np.clip(1 - drift, 0, 1)) * float(np.clip(1 - clipping * 5, 0, 1)) * (1 - missing_fraction)
    return SignalQuality(snr_db, periodicity, spec["dominant_frequency_hz"], drift, clipping, missing_fraction, q, tuple(flags))
