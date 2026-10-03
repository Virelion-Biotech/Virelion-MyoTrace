# Virelion-MyoTrace

MyoTrace is a Python toolkit for extracting quantitative motion and beat-level mechanical features from cardiac-cell or tissue video and TIFF data. It also provides optional multimodal feature fusion.

## What it contains

- Farneback and Lucas–Kanade optical-flow analysis.
- Ensemble motion estimates for sensitivity analysis.
- Frame, ROI, mask, drift, and signal-quality checks.
- Beat-rate and beat-interval analysis.
- Contraction/relaxation timing and motion-index features.
- Spectral and signal-quality descriptors.
- Provenance, parameter manifests, and SHA-256 input hashes.
- Paired force calibration when instrument-specific ground truth is available.
- Optional mechanical/electrical/molecular feature fusion.
- Repeatability, agreement, and sensitivity-analysis utilities.

Motion magnitude is a motion index. It is not force or stress unless calibrated against an appropriate mechanical measurement.

## Installation

```bash
pip install -e '.[all,dev]'
```

## Usage

CLI:

```bash
myotrace recording.mp4 --sample-id EHT_001 --out results/
myotrace recording.mp4 --method ensemble --correct-motion --out results/
```

Python:

```python
from myotrace import analyze_video, fit_force_calibration

result = analyze_video("recording.mp4", sample_id="EHT_001", correct_motion=True)
print(result.summary)
```

Force calibration requires paired instrument measurements:

```python
cal = fit_force_calibration(motion_values, force_values, units="uN")
```

## Inputs and outputs

**Inputs:** cardiac-cell/tissue video or TIFF data, sample identifiers, ROI/mask information, frame-rate metadata, preprocessing parameters, and optional paired force/electrical/molecular features.

**Outputs:** motion traces, beat metrics, contraction/relaxation measurements, spectral/QC features, calibration results, multimodal feature tables, summaries, and provenance records. Typical files include `motion_trace.csv`, `beat_metrics.csv`, `summary.json`, `provenance.json`, and `qc.txt`.

## Validation

The repository includes synthetic timing benchmarks and utilities for group comparison, reference correlation, leave-one-modality-out analysis, Bland–Altman summaries, and repeatability. Biological validation requires independent recordings, external comparison, test/retest data, appropriate mechanical ground truth, locked reference panels, and independent batches or laboratories.

## Limitations

Optical-flow outputs depend on image quality, frame rate, motion, preprocessing, segmentation/ROI choices, and camera stability. Synthetic benchmarks do not establish biological validity. Force claims require instrument-specific calibration. The multimodal maturity index is a computational framework, not a clinically validated maturity scale.

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See `LICENSE`.
