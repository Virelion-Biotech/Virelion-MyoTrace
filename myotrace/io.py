from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VideoFrames:
    frames: np.ndarray
    fps: float | None
    source: str

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.n_frames - 1) / self.fps) if self.fps else 0.0


def load_video(path: str | Path, *, max_frames: int | None = None, gray: bool = True) -> VideoFrames:
    """Load an AVI/MP4/MOV stream through OpenCV.

    The loader deliberately returns frames in memory so the signal stage is deterministic.
    For large acquisitions, pass ``max_frames`` during exploratory QC or implement chunked IO
    upstream before production-scale processing.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for video loading; install the 'video' extra.") from exc

    if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0):
        raise ValueError("max_frames must be a positive integer")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = None
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"Video contained no readable frames: {path}")
    return VideoFrames(np.stack(frames), fps=fps, source=str(path))


def load_tiff_stack(path: str | Path) -> VideoFrames:
    """Load a TIFF image sequence as a frame stack."""
    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("tifffile is required for TIFF stacks; install the 'tiff' extra.") from exc
    with tifffile.TiffFile(path) as tif:
        if len(tif.series) != 1:
            raise ValueError("TIFF contains multiple series; select one time sequence upstream")
        series = tif.series[0]
        metadata = tif.imagej_metadata or {}
        has_time = metadata.get("finterval") is not None or metadata.get("fps") is not None
        if series.axes not in {"TYX", "QYX", "IYX", "YX"} and not (series.axes == "ZYX" and has_time):
            raise ValueError(
                f"Expected a grayscale time sequence; TIFF axes are {series.axes!r}. Select channels/Z upstream."
            )
        arr = np.asarray(series.asarray())
        fps = None
        interval = metadata.get("finterval")
        if interval is not None and metadata.get("tunit", "sec").lower() in {"s", "sec", "second", "seconds"}:
            interval = float(interval)
            if np.isfinite(interval) and interval > 0:
                fps = 1.0 / interval
        if fps is None and metadata.get("fps") is not None:
            candidate = float(metadata["fps"])
            if np.isfinite(candidate) and candidate > 0:
                fps = candidate
        if fps is None and tif.ome_metadata:
            from xml.etree import ElementTree

            pixels = next(
                (e for e in ElementTree.fromstring(tif.ome_metadata).iter() if e.tag.rsplit("}", 1)[-1] == "Pixels"),
                None,
            )
            if pixels is not None and "TimeIncrement" in pixels.attrib:
                factor = {"s": 1.0, "ms": 0.001, "µs": 1e-6}.get(pixels.get("TimeIncrementUnit", "s"))
                interval = float(pixels.attrib["TimeIncrement"])
                if factor is not None and np.isfinite(interval) and interval > 0:
                    fps = 1.0 / (interval * factor)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected a grayscale TIFF stack with shape (frames, y, x), got {arr.shape}")
    return VideoFrames(arr, fps=fps, source=str(path))


def normalize_frames(frames: np.ndarray, *, percentile_low: float = 1.0, percentile_high: float = 99.0) -> np.ndarray:
    """Robustly scale arbitrary image dtype/intensity into float32 [0, 1]."""
    if not 0 <= percentile_low < percentile_high <= 100:
        raise ValueError("normalization percentiles must satisfy 0 <= low < high <= 100")
    x = np.asarray(frames, dtype=np.float32)
    if not x.size or not np.all(np.isfinite(x)):
        raise ValueError("frames must contain finite numeric values")
    lo, hi = np.percentile(x, [percentile_low, percentile_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def validate_frame_stack(frames: np.ndarray) -> None:
    if not isinstance(frames, np.ndarray) or frames.ndim != 3:
        raise ValueError("frames must be a NumPy array of shape (n_frames, height, width)")
    if frames.shape[0] < 3:
        raise ValueError("At least three frames are required")
    if min(frames.shape[1:]) < 8:
        raise ValueError("Frames are too small for optical-flow analysis")

    if not np.issubdtype(frames.dtype, np.number) or not np.all(np.isfinite(frames)):
        raise ValueError("frames must contain finite numeric values")
