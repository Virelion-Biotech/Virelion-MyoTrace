from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import normalize_frames, validate_frame_stack


@dataclass(frozen=True)
class FlowConfig:
    method: str = "farneback"
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    motion_percentile: float = 75.0
    ensemble_weight_lk: float = 0.5


def _farneback_signal(frames: np.ndarray, config: FlowConfig) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for optical flow; install the 'video' extra.") from exc
    x = normalize_frames(frames)
    out = np.empty(x.shape[0] - 1, dtype=np.float64)
    prev = x[0]
    for i in range(1, x.shape[0]):
        flow = cv2.calcOpticalFlowFarneback(prev, x[i], None, config.pyr_scale, config.levels, config.winsize, config.iterations, config.poly_n, config.poly_sigma, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        out[i - 1] = float(np.percentile(mag, config.motion_percentile))
        prev = x[i]
    return out


def _lk_signal(frames: np.ndarray, config: FlowConfig) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for optical flow; install the 'video' extra.") from exc
    x = (normalize_frames(frames) * 255).astype(np.uint8)
    features = cv2.goodFeaturesToTrack(x[0], maxCorners=300, qualityLevel=0.01, minDistance=5)
    if features is None or len(features) < 3:
        raise ValueError("Lucas-Kanade could not initialize enough trackable features")
    out = np.full(x.shape[0] - 1, np.nan, dtype=np.float64)
    prev = x[0]
    prev_pts = features
    for i in range(1, x.shape[0]):
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev, x[i], prev_pts, None)
        if curr_pts is None or status is None:
            prev, prev_pts = x[i], features
            continue
        good = status.ravel().astype(bool)
        if good.sum() >= 3:
            delta = curr_pts[good] - prev_pts[good]
            out[i - 1] = float(np.median(np.linalg.norm(delta, axis=1)))
            prev_pts = curr_pts[good].reshape(-1, 1, 2)
        else:
            prev_pts = features
        prev = x[i]
    return np.nan_to_num(out, nan=0.0)


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - np.mean(x)) / max(np.std(x), np.finfo(float).eps)


def optical_flow_trace(frames: np.ndarray, config: FlowConfig | None = None) -> np.ndarray:
    """Convert frames into a motion-intensity signal.

    ``ensemble`` combines independently computed Farneback and Lucas–Kanade traces after robust
    scaling. This is intended as a consensus signal, not as a claim of superior accuracy until
    validated against an external ground truth.
    """
    validate_frame_stack(frames)
    cfg = config or FlowConfig()
    method = cfg.method.lower()
    if method == "farneback":
        return _farneback_signal(frames, cfg)
    if method in {"lk", "lucas-kanade", "lucaskanade"}:
        return _lk_signal(frames, cfg)
    if method == "ensemble":
        farneback = _zscore(_farneback_signal(frames, cfg))
        lk = _zscore(_lk_signal(frames, cfg))
        w = float(np.clip(cfg.ensemble_weight_lk, 0.0, 1.0))
        return (1.0 - w) * farneback + w * lk
    raise ValueError(f"Unknown optical-flow method: {cfg.method!r}")


def frame_timestamps(n_frames: int, fps: float) -> np.ndarray:
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("fps must be a positive finite number")
    return np.arange(max(0, n_frames - 1), dtype=np.float64) / fps
