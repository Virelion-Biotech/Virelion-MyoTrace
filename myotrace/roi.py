from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    width: int
    height: int

    def validate(self, frame_shape: tuple[int, int]) -> None:
        h, w = frame_shape
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("ROI coordinates and dimensions must be positive")
        if self.x + self.width > w or self.y + self.height > h:
            raise ValueError("ROI extends outside frame bounds")


def crop_frames(frames: np.ndarray, roi: ROI | None = None, mask: np.ndarray | None = None) -> np.ndarray:
    """Crop frames to an explicit ROI and optionally apply a binary mask."""
    x = np.asarray(frames)
    if x.ndim != 3:
        raise ValueError("frames must have shape (n_frames, height, width)")
    if roi is not None:
        roi.validate((x.shape[1], x.shape[2]))
        x = x[:, roi.y:roi.y + roi.height, roi.x:roi.x + roi.width]
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != x.shape[1:]:
            raise ValueError("mask shape must match the spatial frame dimensions")
        x = np.where(m[None, ...], x, np.median(x, axis=(1, 2), keepdims=True))
    return x
