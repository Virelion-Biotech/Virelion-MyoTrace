"""Acquisition-scale conversion for optical motion proxies, without force/strain claims."""

from __future__ import annotations
import numpy as np


def physical_motion(signal, *, micrometers_per_pixel, fps, signal_mode, method):
    if any(isinstance(x, bool) or not np.isfinite(x) or x <= 0 for x in [micrometers_per_pixel, fps]):
        raise ValueError("Acquisition scale and frame rate must be positive finite numbers")
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Motion must be a nonempty finite vector")
    if method == "ensemble":
        raise ValueError("Standardized ensemble motion has no physical pixel scale")
    if signal_mode == "signed_displacement":
        if len(values) < 2:
            raise ValueError("Signed displacement velocity requires at least two frames")
        return {
            "principal_displacement_um": values * micrometers_per_pixel,
            "principal_velocity_um_s": np.gradient(values * micrometers_per_pixel, 1 / fps),
        }
    if signal_mode == "motion":
        return {"motion_speed_um_s": values * micrometers_per_pixel * fps}
    raise ValueError("Unknown motion signal mode")
