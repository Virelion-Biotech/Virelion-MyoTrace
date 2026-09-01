from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .flow import FlowConfig, optical_flow_trace
from .io import load_tiff_stack, load_video
from .kinetics import analyze_trace, beats_to_frame_table, summarize_beats
from .morphology import beat_templates
from .motion_correction import MotionCorrectionReport, correct_global_translation
from .provenance import build_provenance
from .qc import QCReport, assess_frames
from .robust import assess_signal_quality, robust_preprocess, spectral_features
from .roi import ROI, crop_frames


@dataclass(frozen=True)
class VideoAnalysis:
    sample_id: str
    qc: QCReport
    summary: dict[str, float]
    beats: pd.DataFrame
    trace: pd.DataFrame
    provenance: dict[str, object]
    motion_correction: MotionCorrectionReport | None = None


def analyze_video(
    path: str | Path,
    *,
    sample_id: str | None = None,
    fps_override: float | None = None,
    flow_config: FlowConfig | None = None,
    reject_failed_qc: bool = False,
    robust: bool = True,
    roi: ROI | None = None,
    correct_motion: bool = False,
) -> VideoAnalysis:
    """Run loading, ROI selection, optional rigid-motion correction, QC, mechanics and provenance."""
    path = Path(path)
    source = load_tiff_stack(path) if path.suffix.lower() in {".tif", ".tiff"} else load_video(path)
    frames = crop_frames(source.frames, roi=roi)
    correction = None
    if correct_motion:
        frames, correction = correct_global_translation(frames)
    fps = float(fps_override or source.fps)
    qc = assess_frames(frames, fps)
    if reject_failed_qc and not qc.usable:
        raise ValueError(f"Video failed QC: {', '.join(qc.reasons)}")
    motion = optical_flow_trace(frames, flow_config)
    analysis_signal = robust_preprocess(motion, fps) if robust else motion
    signal_qc = assess_signal_quality(analysis_signal, fps)
    times = np.arange(motion.size, dtype=float) / fps
    beats = analyze_trace(analysis_signal, fps)
    sid = sample_id or path.stem
    beat_table = beats_to_frame_table(beats, sid)
    trace = pd.DataFrame({"sample_id": sid, "timestamp_s": times, "motion_index": motion, "analysis_signal": analysis_signal, "modality": "mechanical"})
    summary = summarize_beats(beats)
    summary.update({
        "qc_usable": float(qc.usable), "qc_motion_fraction": qc.motion_fraction,
        "signal_quality": signal_qc.quality_score, "signal_snr_db": signal_qc.snr_db,
        "signal_periodicity": signal_qc.periodicity, "dominant_frequency_hz": signal_qc.dominant_frequency_hz,
        **{f"spectral_{k}": v for k, v in spectral_features(analysis_signal, fps).items()},
    })
    if len(beats) >= 3:
        _, morphology_stability, morphology_dispersion = beat_templates(
            analysis_signal, beat_table["peak_time_s"].to_numpy(), fps
        )
        summary["morphology_stability"] = morphology_stability
        summary["morphology_dispersion"] = morphology_dispersion
    else:
        summary["morphology_stability"] = np.nan
        summary["morphology_dispersion"] = np.nan
    if correction is not None:
        summary.update({
            "motion_correction_failed_fraction": correction.failed_fraction,
            "motion_correction_median_translation_px": correction.median_translation_px,
            "motion_correction_max_translation_px": correction.max_translation_px,
        })
    prov = build_provenance(path, version="0.2.0", parameters={
        "fps": fps, "flow": repr(flow_config), "robust": robust,
        "roi": repr(roi), "correct_motion": correct_motion,
    })
    return VideoAnalysis(sid, qc, summary, beat_table, trace, prov.__dict__, correction)
