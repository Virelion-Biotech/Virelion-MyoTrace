from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


@dataclass(frozen=True)
class BeatMetrics:
    beat_index: int
    peak_time_s: float
    interval_s: float
    beat_rate_bpm: float
    amplitude: float
    rise_time_s: float
    relaxation_time_s: float
    time_to_peak_s: float
    baseline: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _bandpass(signal: np.ndarray, fps: float, low_hz: float, high_hz: float) -> np.ndarray:
    nyq = 0.5 * fps
    high = min(high_hz / nyq, 0.99)
    low = max(low_hz / nyq, 1e-4)
    if low >= high:
        return signal - np.nanmedian(signal)
    b, a = butter(3, [low, high], btype="band")
    return filtfilt(b, a, signal)


def prepare_signal(signal: np.ndarray, fps: float, *, low_hz: float = 0.3, high_hz: float = 8.0) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size < 5:
        raise ValueError("Signal is too short")
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=float(np.nanmedian(x)))
    return _bandpass(x, fps, low_hz, high_hz)


def analyze_trace(
    signal: np.ndarray,
    fps: float,
    *,
    prominence_fraction: float = 0.15,
    min_bpm: float = 30.0,
    max_bpm: float = 240.0,
) -> list[BeatMetrics]:
    """Extract beat-level mechanical kinetics from a motion trace."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    x = prepare_signal(signal, fps)
    span = float(np.percentile(x, 95) - np.percentile(x, 5))
    prominence = max(span * prominence_fraction, np.finfo(float).eps)
    min_distance = max(1, int(fps * 60.0 / max_bpm))
    peaks, props = find_peaks(x, prominence=prominence, distance=min_distance)
    results: list[BeatMetrics] = []
    if peaks.size == 0:
        return results
    for idx, peak in enumerate(peaks):
        prev_peak = peaks[idx - 1] if idx else None
        interval = float((peak - prev_peak) / fps) if prev_peak is not None else np.nan
        bpm = float(60.0 / interval) if np.isfinite(interval) and interval > 0 else np.nan
        if np.isfinite(bpm) and not (min_bpm <= bpm <= max_bpm):
            continue
        left = max(0, peak - int(round(0.8 * fps)))
        right = min(x.size, peak + int(round(0.8 * fps)))
        local = x[left:right]
        baseline = float(np.percentile(local, 10)) if local.size else float(x[peak])
        amplitude = float(x[peak] - baseline)

        # Use the nearest local minimum before the peak as the contraction onset.
        valley_candidates, _ = find_peaks(-x[left : peak + 1])
        onset = left + int(valley_candidates[-1]) if valley_candidates.size else left
        rise = float((peak - onset) / fps)

        # Relaxation is measured to the first crossing near baseline after the peak.
        threshold = baseline + 0.2 * amplitude
        post = np.flatnonzero(x[peak:right] <= threshold)
        relax = float(post[0] / fps) if post.size else np.nan
        results.append(
            BeatMetrics(
                beat_index=len(results),
                peak_time_s=float(peak / fps),
                interval_s=interval,
                beat_rate_bpm=bpm,
                amplitude=amplitude,
                rise_time_s=rise,
                relaxation_time_s=relax,
                time_to_peak_s=rise,
                baseline=baseline,
            )
        )
    return results


def beats_to_frame_table(beats: list[BeatMetrics], sample_id: str) -> pd.DataFrame:
    """Return CardioScore/ElectroTrace-friendly tidy beat records."""
    df = pd.DataFrame([b.to_dict() for b in beats])
    if df.empty:
        return pd.DataFrame(
            columns=["sample_id", "beat_index", "peak_time_s", "interval_s", "beat_rate_bpm", "amplitude", "rise_time_s", "relaxation_time_s", "time_to_peak_s", "baseline"]
        )
    df.insert(0, "sample_id", sample_id)
    df["modality"] = "mechanical"
    return df


def summarize_beats(beats: list[BeatMetrics]) -> dict[str, float]:
    if not beats:
        return {"n_beats": 0.0, "mean_bpm": np.nan, "sd_bpm": np.nan, "mean_amplitude": np.nan, "mean_rise_time_s": np.nan, "mean_relaxation_time_s": np.nan}
    bpm = np.array([b.beat_rate_bpm for b in beats], dtype=float)
    return {
        "n_beats": float(len(beats)),
        "mean_bpm": float(np.nanmean(bpm)),
        "sd_bpm": float(np.nanstd(bpm, ddof=1)) if len(bpm) > 1 else 0.0,
        "mean_amplitude": float(np.nanmean([b.amplitude for b in beats])),
        "mean_rise_time_s": float(np.nanmean([b.rise_time_s for b in beats])),
        "mean_relaxation_time_s": float(np.nanmean([b.relaxation_time_s for b in beats])),
    }
