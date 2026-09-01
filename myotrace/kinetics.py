from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, filtfilt


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
    width_50_s: float
    area_abs: float
    beat_quality: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _bandpass(signal: np.ndarray, fps: float, low_hz: float, high_hz: float) -> np.ndarray:
    nyq = 0.5 * fps
    high = min(high_hz / nyq, 0.98)
    low = max(low_hz / nyq, 1e-4)
    if low >= high or x := np.array([]):
        return signal - np.nanmedian(signal)
    b, a = butter(3, [low, high], btype="band")
    return filtfilt(b, a, signal)


def prepare_signal(signal: np.ndarray, fps: float, *, low_hz: float = 0.25, high_hz: float = 8.0) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size < max(20, int(fps * 1.5)):
        raise ValueError("Signal is too short for reliable beat kinetics")
    if not np.all(np.isfinite(x)):
        med = float(np.nanmedian(x))
        x = np.nan_to_num(x, nan=med, posinf=med, neginf=med)
    b, a = butter(3, [max(low_hz / (0.5 * fps), 1e-4), min(high_hz / (0.5 * fps), 0.98)], btype="band")
    return filtfilt(b, a, x)


def _crossing_time(x: np.ndarray, start: int, stop: int, level: float, fps: float, direction: str) -> float:
    segment = x[start:stop]
    if direction == "up":
        idx = np.flatnonzero(segment >= level)
    else:
        idx = np.flatnonzero(segment <= level)
    return float(idx[0] / fps) if idx.size else np.nan


def analyze_trace(signal: np.ndarray, fps: float, *, prominence_fraction: float = 0.12, min_bpm: float = 30.0, max_bpm: float = 240.0) -> list[BeatMetrics]:
    """Extract beat-level mechanical kinetics with morphology-aware metrics."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    x = prepare_signal(signal, fps)
    span = float(np.percentile(x, 95) - np.percentile(x, 5))
    prominence = max(span * prominence_fraction, np.finfo(float).eps)
    distance = max(1, int(np.floor(fps * 60.0 / max_bpm)))
    peaks, props = find_peaks(x, prominence=prominence, distance=distance)
    out: list[BeatMetrics] = []
    for i, peak in enumerate(peaks):
        interval = float((peak - peaks[i - 1]) / fps) if i else np.nan
        bpm = float(60.0 / interval) if np.isfinite(interval) and interval > 0 else np.nan
        if np.isfinite(bpm) and not (min_bpm <= bpm <= max_bpm):
            continue
        half_window = max(2, int(0.55 * fps))
        left = max(0, peak - half_window)
        right = min(x.size, peak + half_window)
        local = x[left:right]
        baseline = float(np.percentile(local, 15))
        amplitude = float(max(x[peak] - baseline, 0.0))
        if amplitude <= 0:
            continue
        onset_candidates, _ = find_peaks(-x[left:peak + 1], distance=max(1, int(0.08 * fps)))
        onset = left + int(onset_candidates[-1]) if onset_candidates.size else left
        rise = float((peak - onset) / fps)
        threshold = baseline + 0.2 * amplitude
        relaxation = _crossing_time(x, peak, right, threshold, fps, "down")
        width50 = np.nan
        try:
            from scipy.signal import peak_widths
            width50 = float(peak_widths(x, [peak], rel_height=0.5)[0][0] / fps)
        except Exception:
            pass
        lo = onset
        hi = min(right, onset + max(1, int(round(max(interval if np.isfinite(interval) else 1.0, 0.5) * fps))))
        area = float(np.trapezoid(np.abs(x[lo:hi] - baseline), dx=1.0 / fps)) if hi > lo else 0.0
        expected = np.nanmedian(np.diff(peaks) / fps) if len(peaks) > 2 else np.nan
        regularity = float(np.exp(-abs(interval - expected) / expected)) if np.isfinite(expected) and expected > 0 and np.isfinite(interval) else 0.5
        morphology = float(np.clip(1.0 - abs(rise - (relaxation if np.isfinite(relaxation) else rise)) / max(rise + (relaxation if np.isfinite(relaxation) else rise), 1e-6), 0, 1))
        quality = float(np.clip(0.65 * regularity + 0.35 * morphology, 0, 1))
        out.append(BeatMetrics(len(out), peak / fps, interval, bpm, amplitude, rise, relaxation, rise, baseline, width50, area, quality))
    return out


def beats_to_frame_table(beats: list[BeatMetrics], sample_id: str) -> pd.DataFrame:
    columns = list(BeatMetrics.__dataclass_fields__.keys())
    if not beats:
        return pd.DataFrame(columns=["sample_id", *columns, "modality"])
    df = pd.DataFrame([b.to_dict() for b in beats])
    df.insert(0, "sample_id", sample_id)
    df["modality"] = "mechanical"
    return df


def summarize_beats(beats: list[BeatMetrics]) -> dict[str, float]:
    if not beats:
        return {"n_beats": 0.0, "mean_bpm": np.nan, "sd_bpm": np.nan, "cv_bpm": np.nan, "mean_amplitude": np.nan, "mean_rise_time_s": np.nan, "mean_relaxation_time_s": np.nan, "mean_width_50_s": np.nan, "mean_area_abs": np.nan, "mean_beat_quality": np.nan, "regularity_index": np.nan}
    bpm = np.array([b.beat_rate_bpm for b in beats], dtype=float)
    intervals = np.array([b.interval_s for b in beats], dtype=float)
    mean_bpm = float(np.nanmean(bpm))
    return {
        "n_beats": float(len(beats)),
        "mean_bpm": mean_bpm,
        "sd_bpm": float(np.nanstd(bpm, ddof=1)) if len(bpm) > 1 else 0.0,
        "cv_bpm": float(np.nanstd(bpm, ddof=1) / mean_bpm) if len(bpm) > 1 and mean_bpm > 0 else np.nan,
        "mean_amplitude": float(np.nanmean([b.amplitude for b in beats])),
        "mean_rise_time_s": float(np.nanmean([b.rise_time_s for b in beats])),
        "mean_relaxation_time_s": float(np.nanmean([b.relaxation_time_s for b in beats])),
        "mean_width_50_s": float(np.nanmean([b.width_50_s for b in beats])),
        "mean_area_abs": float(np.nanmean([b.area_abs for b in beats])),
        "mean_beat_quality": float(np.nanmean([b.beat_quality for b in beats])),
        "regularity_index": float(1.0 - np.nanstd(intervals, ddof=1) / np.nanmean(intervals)) if np.sum(np.isfinite(intervals)) > 1 and np.nanmean(intervals) > 0 else np.nan,
    }
