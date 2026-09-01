"""Virelion-MyoTrace: quantitative cardiac mechanics and multimodal tissue characterization."""

from .calibration import ForceCalibration, fit_force_calibration
from .kinetics import BeatMetrics, analyze_trace, summarize_beats
from .pipeline import VideoAnalysis, analyze_video
from .robust import SignalQuality, assess_signal_quality, robust_preprocess
from .uncertainty import BootstrapSummary, bootstrap_mean

__all__ = [
    "BeatMetrics", "VideoAnalysis", "analyze_trace", "analyze_video", "summarize_beats",
    "SignalQuality", "assess_signal_quality", "robust_preprocess", "BootstrapSummary", "bootstrap_mean",
    "ForceCalibration", "fit_force_calibration",
]
__version__ = "0.2.0"
