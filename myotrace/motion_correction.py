from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotionCorrectionReport:
    enabled: bool
    corrected_frames: int
    median_translation_px: float
    max_translation_px: float
    failed_fraction: float


def correct_global_translation(frames: np.ndarray, *, max_corners: int = 200) -> tuple[np.ndarray, MotionCorrectionReport]:
    """Compensate rigid camera/sample translation using phase-independent feature tracking.

    The transform is intentionally limited to translation. This prevents the correction model
    from explaining away genuine contractile deformation with an overly flexible registration.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for motion correction; install the 'video' extra.") from exc
    x = np.asarray(frames)
    if x.ndim != 3 or x.shape[0] < 3:
        raise ValueError("frames must have shape (n_frames, y, x)")
    ref = x[0].astype(np.float32)
    out = np.empty_like(x)
    out[0] = x[0]
    shifts: list[float] = []
    failed = 0
    for i in range(1, x.shape[0]):
        prev8 = cv2.normalize(ref, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cur8 = cv2.normalize(x[i].astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        p0 = cv2.goodFeaturesToTrack(prev8, maxCorners=max_corners, qualityLevel=0.01, minDistance=6)
        if p0 is None or len(p0) < 4:
            out[i] = x[i]
            failed += 1
            continue
        p1, status, _ = cv2.calcOpticalFlowPyrLK(prev8, cur8, p0, None)
        if p1 is None or status is None or int(status.sum()) < 4:
            out[i] = x[i]
            failed += 1
            continue
        delta = (p1 - p0).reshape(-1, 2)[status.ravel().astype(bool)]
        dx, dy = np.median(delta, axis=0)
        mag = float(np.hypot(dx, dy))
        shifts.append(mag)
        matrix = np.float32([[1, 0, -dx], [0, 1, -dy]])
        out[i] = cv2.warpAffine(x[i], matrix, (x.shape[2], x.shape[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    shifts_array = np.asarray(shifts, dtype=float)
    report = MotionCorrectionReport(True, int(x.shape[0] - 1 - failed), float(np.median(shifts_array)) if shifts_array.size else 0.0, float(np.max(shifts_array)) if shifts_array.size else 0.0, float(failed / max(1, x.shape[0] - 1)))
    return out, report
