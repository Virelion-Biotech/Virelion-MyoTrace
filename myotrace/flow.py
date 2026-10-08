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

    def __post_init__(self) -> None:
        if self.method.lower() not in {"farneback", "lk", "lucas-kanade", "lucaskanade", "ensemble"}:
            raise ValueError("Unknown optical-flow method")
        if not np.isfinite(self.pyr_scale) or not 0 < self.pyr_scale < 1:
            raise ValueError("pyr_scale must lie strictly between 0 and 1")
        for name in ("levels", "winsize", "iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.poly_n not in {5, 7} or not np.isfinite(self.poly_sigma) or self.poly_sigma <= 0:
            raise ValueError("poly_n must be 5 or 7 and poly_sigma must be positive finite")
        if not np.isfinite(self.motion_percentile) or not 0 <= self.motion_percentile <= 100:
            raise ValueError("motion_percentile must lie in [0, 100]")
        if not np.isfinite(self.ensemble_weight_lk) or not 0 <= self.ensemble_weight_lk <= 1:
            raise ValueError("ensemble_weight_lk must lie in [0, 1]")


def _farneback_signal(frames: np.ndarray, config: FlowConfig, mask: np.ndarray | None = None) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for optical flow; install the 'video' extra.") from exc
    x = np.rint(normalize_frames(frames) * 255).astype(np.uint8)
    out = np.empty(x.shape[0] - 1, dtype=np.float64)
    prev = x[0]
    for i in range(1, x.shape[0]):
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            x[i],
            None,
            config.pyr_scale,
            config.levels,
            config.winsize,
            config.iterations,
            config.poly_n,
            config.poly_sigma,
            0,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        out[i - 1] = float(np.percentile(mag[mask] if mask is not None else mag, config.motion_percentile))
        prev = x[i]
    return out


def _lk_signal(frames: np.ndarray, config: FlowConfig, mask: np.ndarray | None = None) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for optical flow; install the 'video' extra.") from exc
    x = (normalize_frames(frames) * 255).astype(np.uint8)
    features = cv2.goodFeaturesToTrack(
        x[0],
        maxCorners=300,
        qualityLevel=0.01,
        minDistance=5,
        mask=mask.astype(np.uint8) * 255 if mask is not None else None,
    )
    if features is None or len(features) < 3:
        raise ValueError("Lucas-Kanade could not initialize enough trackable features")
    out = np.empty(x.shape[0] - 1, dtype=np.float64)
    prev = x[0]
    prev_pts = features
    for i in range(1, x.shape[0]):
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev, x[i], prev_pts, None)
        if curr_pts is None or status is None or status.sum() < 3:
            # Retry using points detected in the actual previous frame, never frame zero.
            prev_pts = cv2.goodFeaturesToTrack(
                prev,
                maxCorners=300,
                qualityLevel=0.01,
                minDistance=5,
                mask=mask.astype(np.uint8) * 255 if mask is not None else None,
            )
            if prev_pts is None or len(prev_pts) < 3:
                raise ValueError(f"Lucas-Kanade lost tracking at frame {i}")
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev, x[i], prev_pts, None)
        if curr_pts is None or status is None or status.sum() < 3:
            raise ValueError(f"Lucas-Kanade lost tracking at frame {i}")
        good = status.ravel().astype(bool)
        delta = (curr_pts[good] - prev_pts[good]).reshape(-1, 2)
        out[i - 1] = float(np.median(np.linalg.norm(delta, axis=1)))
        prev_pts = curr_pts[good].reshape(-1, 1, 2)
        prev = x[i]
    if not np.all(np.isfinite(out)):
        raise ValueError("Lucas-Kanade produced nonfinite motion")
    return out


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    scale = float(np.std(x))
    # Avoid inflating sub-millipixel tracking jitter into unit-scale events.
    if scale <= 1e-3 * max(1.0, abs(float(np.mean(x)))):
        return np.zeros_like(x)
    return (x - np.mean(x)) / scale


def optical_flow_trace(
    frames: np.ndarray, config: FlowConfig | None = None, *, mask: np.ndarray | None = None
) -> np.ndarray:
    """Convert frames into a motion-intensity signal.

    ``ensemble`` combines independently computed Farneback and Lucas–Kanade traces after robust
    scaling. This is intended as a consensus signal, not as a claim of superior accuracy until
    validated against an external ground truth.
    """
    validate_frame_stack(frames)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != frames.shape[1:] or not mask.any():
            raise ValueError("mask must match cropped frame dimensions and contain selected pixels")
    cfg = config or FlowConfig()
    method = cfg.method.lower()
    if method == "farneback":
        return _farneback_signal(frames, cfg, mask)
    if method in {"lk", "lucas-kanade", "lucaskanade"}:
        return _lk_signal(frames, cfg, mask)
    if method == "ensemble":
        farneback = _zscore(_farneback_signal(frames, cfg, mask))
        lk = _zscore(_lk_signal(frames, cfg, mask))
        w = float(np.clip(cfg.ensemble_weight_lk, 0.0, 1.0))
        return (1.0 - w) * farneback + w * lk
    raise ValueError(f"Unknown optical-flow method: {cfg.method!r}")


def frame_timestamps(n_frames: int, fps: float) -> np.ndarray:
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("fps must be a positive finite number")
    return np.arange(max(0, n_frames - 1), dtype=np.float64) / fps
