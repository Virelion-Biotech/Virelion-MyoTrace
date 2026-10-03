from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QCReport:
    frame_count: int
    fps: float
    duration_s: float
    intensity_std: float
    motion_fraction: float
    dropped_or_flat_fraction: float
    usable: bool
    reasons: tuple[str, ...]


def assess_frames(frames: np.ndarray, fps: float) -> QCReport:
    x = np.asarray(frames, dtype=np.float32)
    if x.ndim != 3 or x.shape[0] < 3:
        raise ValueError("frames must have shape (n_frames, height, width) with n_frames >= 3")
    frame_std = x.reshape(x.shape[0], -1).std(axis=1)
    frame_means = x.reshape(x.shape[0], -1).mean(axis=1)
    intensity_std = float(np.std(frame_means))
    diff = np.mean(np.abs(np.diff(x, axis=0)), axis=(1, 2))
    q = float(np.percentile(diff, 25))
    motion_fraction = float(np.mean(diff > max(q * 1.5, np.finfo(float).eps)))
    flat = float(np.mean(frame_std < np.finfo(float).eps))
    reasons: list[str] = []
    if fps < 10:
        reasons.append("low_fps")
    if intensity_std > max(float(np.mean(frame_means)) * 0.25, 0.02):
        reasons.append("strong_global_intensity_drift")
    if flat > 0.05:
        reasons.append("flat_or_corrupt_frames")
    if motion_fraction < 0.01:
        reasons.append("negligible_detectable_motion")
    usable = len(reasons) == 0
    return QCReport(
        frame_count=int(x.shape[0]),
        fps=float(fps),
        duration_s=float(max(0.0, (x.shape[0] - 1) / fps)),
        intensity_std=intensity_std,
        motion_fraction=motion_fraction,
        dropped_or_flat_fraction=flat,
        usable=usable,
        reasons=tuple(reasons),
    )
