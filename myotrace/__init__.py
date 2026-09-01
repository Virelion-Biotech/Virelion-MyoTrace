"""Virelion-MyoTrace: quantitative cardiac mechanics and multimodal tissue characterization."""

from .advanced import TraceQC, cross_correlation_lag, cycle_average, morphology_similarity, trace_qc
from .calibration import ForceCalibration, fit_force_calibration
from .kinetics import BeatMetrics, analyze_trace, summarize_beats
from .pipeline import VideoAnalysis, analyze_video
from .robust import SignalQuality, assess_signal_quality, robust_preprocess
from .uncertainty import BootstrapSummary, bootstrap_mean
from .validation_protocol import ReplicatePlan, bootstrap_ci, sensitivity_to_weight

__all__ = [
    "BeatMetrics", "VideoAnalysis", "analyze_trace", "analyze_video", "summarize_beats",
    "SignalQuality", "assess_signal_quality", "robust_preprocess", "BootstrapSummary", "bootstrap_mean",
    "ForceCalibration", "fit_force_calibration", "TraceQC", "trace_qc", "cross_correlation_lag",
    "cycle_average", "morphology_similarity", "ReplicatePlan", "bootstrap_ci", "sensitivity_to_weight",
]
__version__ = "0.3.0"
