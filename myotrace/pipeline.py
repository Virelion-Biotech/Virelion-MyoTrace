from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from ._version import __version__
from .displacement import signed_displacement_trace
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
    summary: dict[str, object]
    beats: pd.DataFrame
    trace: pd.DataFrame
    provenance: dict[str, object]
    motion_correction: MotionCorrectionReport | None = None


def analyze_video(
    path: str | Path,
    *,
    sample_id: str | None = None,
    fps_override: float | None = None,
    micrometers_per_pixel: float | None = None,
    flow_config: FlowConfig | None = None,
    reject_failed_qc: bool = False,
    robust: bool = True,
    roi: ROI | None = None,
    mask: np.ndarray | None = None,
    correct_motion: bool = False,
    signal_mode: str = "motion",
    detector: str = "noise_aware",
    reference_frame: int = 0,
) -> VideoAnalysis:
    """Run loading, ROI selection, optional rigid-motion correction, QC, mechanics and provenance."""
    if signal_mode not in {"motion", "signed_displacement"}:
        raise ValueError("signal_mode must be motion or signed_displacement")
    path = Path(path)
    source = load_tiff_stack(path) if path.suffix.lower() in {".tif", ".tiff"} else load_video(path)
    fps_value = source.fps if fps_override is None else fps_override
    if fps_value is None:
        raise ValueError("Acquisition frame rate is unavailable; supply fps_override (CLI: --fps)")
    fps = float(fps_value)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive finite")
    cfg = flow_config or FlowConfig()
    frames = crop_frames(source.frames, roi=roi)
    correction = None
    if correct_motion:
        frames, correction = correct_global_translation(frames)
    qc = assess_frames(frames, fps)
    if correction is not None and correction.failed_fraction > 0.1:
        qc = replace(qc, usable=False, reasons=(*qc.reasons, "motion_correction_failed"))
    if reject_failed_qc and not qc.usable:
        raise ValueError(f"Video failed QC: {', '.join(qc.reasons)}")
    displacement = None
    if signal_mode == "signed_displacement":
        displacement = signed_displacement_trace(frames, cfg, mask=mask, reference_frame=reference_frame)
        motion = displacement.signal
    else:
        motion = optical_flow_trace(frames, cfg, mask=mask)
    analysis_signal = robust_preprocess(motion, fps) if robust else motion
    signal_qc = assess_signal_quality(analysis_signal, fps)
    offset = 0.0 if displacement is not None else 0.5
    times = (np.arange(motion.size, dtype=float) + offset) / fps
    beats = analyze_trace(analysis_signal, fps, detector=detector)
    sid = sample_id or path.stem
    # Optical flow measures intervals; kinetics use their midpoint timestamps.
    beats = [replace(b, peak_time_s=b.peak_time_s + offset / fps) for b in beats]
    beat_table = beats_to_frame_table(beats, sid)
    trace = pd.DataFrame(
        {
            "sample_id": sid,
            "timestamp_s": times,
            "motion_index": motion,
            "analysis_signal": analysis_signal,
            "modality": "mechanical",
        }
    )
    physical = {}
    if micrometers_per_pixel is not None:
        from .physical_units import physical_motion

        physical = physical_motion(
            motion,
            micrometers_per_pixel=micrometers_per_pixel,
            fps=fps,
            signal_mode=signal_mode,
            method=cfg.method.lower(),
        )
        for name, values in physical.items():
            trace[name] = values
    summary = summarize_beats(beats)
    summary["micrometers_per_pixel"] = micrometers_per_pixel
    summary["physical_outputs"] = list(physical)
    summary["physical_interpretation"] = "scaled optical motion proxy; not tissue strain or force"

    summary["measurement_status"] = "motion_events_not_verified_cardiac_beats"
    summary["signal_mode"] = signal_mode
    summary["detector"] = detector
    summary["motion_units"] = (
        "pixels_principal_displacement"
        if displacement is not None
        else ("standardized_consensus" if cfg.method.lower() == "ensemble" else "pixels_per_frame")
    )
    if displacement is not None:
        summary["measurement_status"] = "displacement_cycles_not_verified_cardiac_beats"
        summary["displacement_explained_variance"] = displacement.explained_variance_fraction
        summary["displacement_flags"] = list(displacement.flags)
        summary["displacement_spatial_samples"] = displacement.spatial_samples
        summary["displacement_polarity"] = "largest_loading_positive_not_verified_systole"
    summary["kinetics_units"] = "normalized_motion_index" if robust else summary["motion_units"]
    summary["signal_flags"] = list(signal_qc.flags)
    summary.update(
        {
            "qc_usable": float(qc.usable),
            "qc_motion_fraction": qc.motion_fraction,
            "signal_quality": signal_qc.quality_score,
            "signal_snr_db": signal_qc.snr_db,
            "signal_periodicity": signal_qc.periodicity,
            "dominant_frequency_hz": signal_qc.dominant_frequency_hz,
            **{f"spectral_{k}": v for k, v in spectral_features(analysis_signal, fps).items()},
        }
    )
    if len(beats) >= 3:
        _, morphology_stability, morphology_dispersion = beat_templates(
            analysis_signal, beat_table["peak_time_s"].to_numpy() - offset / fps, fps
        )
        summary["morphology_stability"] = morphology_stability
        summary["morphology_dispersion"] = morphology_dispersion
    else:
        summary["morphology_stability"] = np.nan
        summary["morphology_dispersion"] = np.nan
    if correction is not None:
        summary.update(
            {
                "motion_correction_failed_fraction": correction.failed_fraction,
                "motion_correction_median_translation_px": correction.median_translation_px,
                "motion_correction_max_translation_px": correction.max_translation_px,
            }
        )
    prov = build_provenance(
        path,
        version=__version__,
        parameters={
            "fps": fps,
            "micrometers_per_pixel": micrometers_per_pixel,
            "scale_source": "caller acquisition metadata" if micrometers_per_pixel is not None else "unknown",
            "fps_source": "override" if fps_override is not None else "acquisition_metadata",
            "flow": asdict(cfg),
            "robust": robust,
            "roi": asdict(roi) if roi else None,
            "correct_motion": correct_motion,
            "mask_sha256": hashlib.sha256(np.asarray(mask, dtype=bool).tobytes()).hexdigest()
            if mask is not None
            else None,
            "timestamp_convention": "frame_timestamp" if displacement is not None else "frame_interval_midpoint",
            "signal_mode": signal_mode,
            "reference_frame": reference_frame if displacement is not None else None,
            "detector": detector,
            "detection_gaussian_sigma_s": 0.04 if detector == "noise_aware" else 0,
            "detection_noise_prominence_multiplier": 5 if detector == "noise_aware" else 0,
            "reject_failed_qc": reject_failed_qc,
        },
    )
    return VideoAnalysis(sid, qc, summary, beat_table, trace, prov.__dict__, correction)
