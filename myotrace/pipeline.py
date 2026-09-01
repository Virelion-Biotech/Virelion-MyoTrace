from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .flow import FlowConfig, optical_flow_trace
from .io import load_tiff_stack, load_video
from .kinetics import analyze_trace, beats_to_frame_table, summarize_beats
from .qc import QCReport, assess_frames


@dataclass(frozen=True)
class VideoAnalysis:
    sample_id: str
    qc: QCReport
    summary: dict[str, float]
    beats: pd.DataFrame
    trace: pd.DataFrame


def analyze_video(
    path: str | Path,
    *,
    sample_id: str | None = None,
    fps_override: float | None = None,
    flow_config: FlowConfig | None = None,
    reject_failed_qc: bool = False,
) -> VideoAnalysis:
    """Run loading -> QC -> optical flow -> beat kinetics as one reproducible pipeline."""
    path = Path(path)
    source = load_tiff_stack(path) if path.suffix.lower() in {".tif", ".tiff"} else load_video(path)
    fps = float(fps_override or source.fps)
    qc = assess_frames(source.frames, fps)
    if reject_failed_qc and not qc.usable:
        raise ValueError(f"Video failed QC: {', '.join(qc.reasons)}")
    motion = optical_flow_trace(source.frames, flow_config)
    times = __import__("numpy").arange(motion.size, dtype=float) / fps
    beats = analyze_trace(motion, fps)
    sid = sample_id or path.stem
    beat_table = beats_to_frame_table(beats, sid)
    trace = pd.DataFrame({"sample_id": sid, "timestamp_s": times, "motion_index": motion, "modality": "mechanical"})
    summary = summarize_beats(beats)
    summary.update({"qc_usable": float(qc.usable), "qc_motion_fraction": qc.motion_fraction})
    return VideoAnalysis(sid, qc, summary, beat_table, trace)
