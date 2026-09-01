from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, peak_widths


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


def prepare_signal(signal: np.ndarray, fps: float, *, low_hz: float = 0.25, high_hz: float = 8.0) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if fps <= 0:
        raise ValueError("fps must be positive")
    if x.size < max(20, int(fps * 1.5)):
        raise ValueError("Signal is too short for reliable beat kinetics")
    if not np.all(np.isfinite(x)):
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            raise ValueError("Signal contains no finite values")
        x = np.nan_to_num(x, nan=float(np.median(finite)), posinf=float(np.max(finite)), neginf=float(np.min(finite)))
    nyq = 0.5 * fps
    low = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 0.98)
    if low >= high:
        return x - np.median(x)
    b, a = butter(3, [low, high], btype="band")
    return filtfilt(b, a, x)


def _first_crossing_down(x: np.ndarray, start: int, stop: int, level: float, fps: float) -> float:
    if stop <= start:
        return np.nan
    idx = np.flatnonzero(x[start:stop] <= level)
    return float(idx[0] / fps) if idx.size else np.nan


def _last_crossing_up(x: np.ndarray, start: int, stop: int, level: float, fps: float) -> float:
    if stop <= start:
        return np.nan
    idx = np.flatnonzero(x[start:stop] <= level)
    return float((idx[-1]) / fps) if idx.size else np.nan


def analyze_trace(signal: np.ndarray, fps: float, *, prominence_fraction: float = 0.12, min_bpm: float = 30.0, max_bpm: float = 240.0) -> list[BeatMetrics]:
    """Extract beat-level mechanical kinetics with conservative quality scoring."""
    x = prepare_signal(signal, fps)
    span = float(np.percentile(x, 95) - np.percentile(x, 5))
    prominence = max(span * prominence_fraction, np.finfo(float).eps)
    distance = max(1, int(np.floor(fps * 60.0 / max_bpm)))
    peaks, properties = find_peaks(x, prominence=prominence, distance=distance)
    out: list[BeatMetrics] = []
    expected_interval = float(np.median(np.diff(peaks)) / fps) if len(peaks) >= 3 else np.nan
    for peak in peaks:
        previous = peaks[peaks < peak]
        interval = float((peak - previous[-1]) / fps) if previous.size else np.nan
        bpm = float(60.0 / interval) if np.isfinite(interval) and interval > 0 else np.nan
        if np.isfinite(bpm) and not (min_bpm <= bpm <= max_bpm):
            continue
        window = max(2, int(0.55 * fps))
        left = max(0, peak - window)
        right = min(x.size, peak + window)
        local = x[left:right]
        baseline = float(np.percentile(local, 15))
        amplitude = float(max(x[peak] - baseline, 0.0))
        if amplitude <= 0:
            continue
        onset_level = baseline + 0.20 * amplitude
        half_level = baseline + 0.50 * amplitude
        offset_level = baseline + 0.80 * amplitude
        onset_candidates = np.flatnonzero(x[left:peak + 1] <= onset_level)
        onset = left + int(onset_candidates[-1]) if onset_candidates.size else left
        rise = float((peak - onset) / fps)
        relaxation = _first_crossing_down(x, peak, right, onset_level, fps)
        width50 = np.nan
        if peak > 0 and peak < x.size - 1:
            try:
                width50 = float(peak_widths(x, [peak], rel_height=0.5)[0][0] / fps)
            except (ValueError, IndexError):
                width50 = np.nan
        end = min(right, peak + int(round(max(interval if np.isfinite(interval) else 1.0, 0.5) * fps)))
        area = float(np.trapezoid(np.abs(x[onset:end] - baseline), dx=1.0 / fps)) if end > onset else 0.0
        regularity = float(np.exp(-abs(interval - expected_interval) / expected_interval)) if np.isfinite(expected_interval) and expected_interval > 0 and np.isfinite(interval) else 0.5
        morphology = float(np.clip(1.0 - abs(rise - (relaxation if np.isfinite(relaxation) else rise)) / max(rise + (relaxation if np.isfinite(relaxation) else rise), 1e-6), 0, 1))
        prominence_value = float(properties.get("prominences", np.array([0.0]))[list(peaks).index(peak)]) if len(properties.get("prominences", [])) == len(peaks) else span
        prominence_score = float(np.clip(prominence_value / max(span, np.finfo(float).eps), 0, 1))
        quality = float(np.clip(0.50 * regularity + 0.30 * morphology + 0.20 * prominence_score, 0, 1))
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
    keys = ["n_beats", "mean_bpm", "sd_bpm", "cv_bpm", "mean_amplitude", "mean_rise_time_s", "mean_relaxation_time_s", "mean_width_50_s", "mean_area_abs", "mean_beat_quality", "regularity_index"]
    if not beats:
        return {k: np.nan for k in keys}
    bpm = np.asarray([b.beat_rate_bpm for b in beats], dtype=float)
    intervals = np.asarray([b.interval_s for b in beats], dtype=float)
    mean_bpm = float(np.nanmean(bpm))
    mean_interval = float(np.nanmean(intervals))
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
        "regularity_index": float(np.clip(1.0 - np.nanstd(intervals, ddof=1) / mean_interval, 0, 1)) if len(intervals) > 1 and mean_interval > 0 else np.nan,
    }
