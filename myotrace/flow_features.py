from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FlowFrameFeatures:
    mean_speed: float
    median_speed: float
    p95_speed: float
    motion_area_fraction: float
    directional_coherence: float


def summarize_flow_field(flow: np.ndarray, *, threshold: float | None = None) -> FlowFrameFeatures:
    """Summarize one dense optical-flow field without discarding spatial information."""
    f = np.asarray(flow, dtype=float)
    if f.ndim != 3 or f.shape[-1] != 2:
        raise ValueError("flow must have shape (height, width, 2)")
    vx, vy = f[..., 0], f[..., 1]
    speed = np.hypot(vx, vy)
    mean_speed = float(np.mean(speed))
    median_speed = float(np.median(speed))
    p95 = float(np.percentile(speed, 95))
    cutoff = float(threshold) if threshold is not None else float(np.percentile(speed, 75))
    motion_area = float(np.mean(speed >= cutoff))
    resultant = np.hypot(np.mean(vx), np.mean(vy))
    mean_vector = float(np.mean(speed))
    coherence = float(np.clip(resultant / max(mean_vector, np.finfo(float).eps), 0.0, 1.0))
    return FlowFrameFeatures(mean_speed, median_speed, p95, motion_area, coherence)
