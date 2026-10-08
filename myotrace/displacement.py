"""Signed principal displacement from fixed-reference dense optical flow.

This is a CPU research signal, not a physiological contraction-direction label.
Unlike magnitude, projection retains reversal of a coherent displacement field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from .flow import FlowConfig
from .io import normalize_frames, validate_frame_stack


@dataclass(frozen=True)
class DisplacementTrace:
    signal: np.ndarray
    reference_frame: int
    spatial_samples: int
    explained_variance_fraction: float
    flags: tuple[str, ...]


def signed_displacement_trace(
    frames: np.ndarray,
    config: FlowConfig | None = None,
    *,
    mask: np.ndarray | None = None,
    reference_frame: int = 0,
    grid_size: int = 16,
) -> DisplacementTrace:
    """Project reference-to-frame flow onto its leading spatial mode.

    At most grid_size**2 vector locations enter the PCA, limiting working memory.
    Sign is fixed by the largest absolute loading; it does not identify systole.
    No beat labels, target rates or annotations are used to fit the projection.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for signed displacement; install the 'video' extra.") from exc

    validate_frame_stack(frames)
    cfg = config or FlowConfig()
    if cfg.method.lower() != "farneback":
        raise ValueError("Signed displacement requires method=farneback")
    if (
        isinstance(reference_frame, bool)
        or not isinstance(reference_frame, int)
        or not 0 <= reference_frame < len(frames)
    ):
        raise ValueError("reference_frame must be an in-bounds integer")
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or not 4 <= grid_size <= 32:
        raise ValueError("grid_size must be an integer in [4, 32]")
    ys = np.unique(np.linspace(0, frames.shape[1] - 1, grid_size).astype(int))
    xs = np.unique(np.linspace(0, frames.shape[2] - 1, grid_size).astype(int))
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    yy, xx = yy.ravel(), xx.ravel()
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != frames.shape[1:]:
            raise ValueError("mask must match cropped frame dimensions")
        keep = mask[yy, xx]
        yy, xx = yy[keep], xx[keep]
    if len(yy) < 4:
        raise ValueError("Displacement grid must contain at least four selected locations")
    images = np.rint(normalize_frames(frames) * 255).astype(np.uint8)
    vectors = np.empty((len(images), 2 * len(yy)), dtype=np.float64)
    for i, image in enumerate(images):
        if i == reference_frame or np.array_equal(image, images[reference_frame]):
            vectors[i] = 0
            continue
        field = cv2.calcOpticalFlowFarneback(
            images[reference_frame],
            image,
            None,
            cfg.pyr_scale,
            cfg.levels,
            cfg.winsize,
            cfg.iterations,
            cfg.poly_n,
            cfg.poly_sigma,
            0,
        )
        vectors[i] = field[yy, xx].reshape(-1)
    if not np.all(np.isfinite(vectors)):
        raise ValueError("Reference optical flow produced nonfinite displacement")
    centered = vectors - vectors.mean(axis=0)
    covariance = centered.T @ centered
    total = float(np.trace(covariance))
    if total <= 1e-8 * len(images) * len(yy):
        return DisplacementTrace(np.zeros(len(images)), reference_frame, len(yy), 0.0, ("negligible_displacement",))
    eigenvalue, eigenvector = eigh(covariance, subset_by_index=[covariance.shape[0] - 1] * 2)
    loading = eigenvector[:, 0]
    if loading[np.argmax(np.abs(loading))] < 0:
        loading = -loading
    signal = centered @ loading / np.sqrt(len(yy))
    ratio = float(np.clip(eigenvalue[0] / total, 0, 1))
    flags = ("multiple_displacement_modes",) if ratio < 0.8 else ()
    return DisplacementTrace(signal, reference_frame, len(yy), ratio, flags)
