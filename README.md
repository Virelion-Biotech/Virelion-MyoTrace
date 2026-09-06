# Virelion-MyoTrace

MyoTrace is a Python toolkit for extracting quantitative motion and beat-level mechanical features from cardiac-cell or tissue video and TIFF data. It also provides optional multimodal feature fusion.

## Scope

- Farneback and Lucas–Kanade optical-flow analysis;
- ensemble motion estimates for sensitivity analysis;
- frame, ROI, mask, drift, and signal-quality checks;
- beat-rate and beat-interval analysis;
- contraction/relaxation timing and motion-index features;
- spectral and signal-quality descriptors;
- provenance, parameter manifests, and SHA-256 input hashes;
- paired force calibration when instrument-specific ground truth is available;
- optional mechanical/electrical/molecular feature fusion;
- repeatability, agreement, and sensitivity-analysis utilities.

Motion magnitude is a motion index. It is not force or stress unless calibrated against an appropriate mechanical measurement.

## Installation

```bash
pip install -e '.[all,dev]'
```

## CLI

```bash
myotrace recording.mp4 --sample-id EHT_001 --out results/
myotrace recording.mp4 --method ensemble --correct-motion --out results/
```

Typical outputs:

```text
motion_trace.csv
beat_metrics.csv
summary.json
provenance.json
qc.txt
```

## Python API

```python
from myotrace import analyze_video, fit_force_calibration

result = analyze_video("recording.mp4", sample_id="EHT_001", correct_motion=True)
print(result.summary)
```

Force calibration should only be performed with paired instrument measurements:

```python
cal = fit_force_calibration(motion_values, force_values, units="uN")
```

## Multimodal fusion

The fusion module accepts explicitly named features such as `mechanical:mean_bpm`, `electrical:fpd_ms`, and molecular measurements. Missing modalities reduce coverage rather than being silently imputed. Reference ranges and weights are configuration inputs and should be locked before confirmatory analysis.

The resulting maturity index is a computational calibration framework, not a clinically validated maturity scale.

## Validation

The repository includes synthetic timing benchmarks and utilities for group comparison, reference correlation, leave-one-modality-out analysis, Bland–Altman summaries, and repeatability. A stronger biological validation program requires independent recordings, external comparison, test/retest data, force ground truth, locked reference panels, and independent batches/laboratories.

## Scientific limitations

Optical-flow outputs depend on image quality, frame rate, motion, preprocessing, segmentation/ROI choices, and camera stability. Synthetic benchmarks do not establish biological validity. Force claims require instrument-specific calibration.

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See `LICENSE`.

## Citation

Cite the repository release and the datasets, recordings, or experimental methods used for validation.
