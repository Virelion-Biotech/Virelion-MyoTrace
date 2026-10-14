# Virelion-MyoTrace

**0.5.0:** CPU accuracy recovery adds noise-aware detection and an optional signed displacement path. See [paired recovery results and remaining failures](docs/ACCURACY_RECOVERY.md).

MyoTrace is a Python toolkit for extracting quantitative motion and beat-level mechanical features from cardiac-cell or tissue video and TIFF data. It also provides optional multimodal feature fusion.


<!-- VALIDATION_PRODUCT_STATUS:BEGIN -->
## Validation and product completion status

**Reviewed 2026-10-14; implementation base `6772d2e6db11`.** **Status: Analytical accuracy recovery; public recordings lack independent beat/force truth.** This is an evidence-backed scope assessment, not a percentage-complete estimate or a declaration that the product is finished.

| Area | Current evidence / remaining gate |
|---|---|
| Software / operational checks | Automated test sources and CI definitions are present; this documentation audit does not report a new algorithm/data-validation run. |
| Scientific or domain evidence | Noise-aware detection reduced independent generated-trace failures from 56 to 8 of 432; signed displacement passed 48 generated-video cases. Public recordings execute/repeat, but full-field multimode projections and method disagreement remain; biological beats and force are not qualified. |
| Remaining for the declared product scope | Acquire independent beat/timing and instrument-calibrated mechanical references with ROI/acquisition metadata; resolve multimode/drift/unsigned double-event failures; validate across batches/labs and test-retest; add bounded-memory streaming for large acquisitions. |

Evidence and limits: [repository report](docs/ACCURACY_RECOVERY.md). The [organization status index](https://github.com/Virelion-Biotech/Virelion-HeartTwin/blob/main/docs/STACK_VALIDATION_STATUS.md) separates software, numerical, data-readiness and empirical qualification. Code/schema execution or a completed download is not a substitute for endpoint-specific validation.
<!-- VALIDATION_PRODUCT_STATUS:END -->

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

CPU execution is sufficient for every implemented method. No neural-network training, CUDA, or GPU is required. See [the CPU audit](docs/CPU_AUDIT.md) for tested behavior and retained failures.

## Installation

```bash
pip install -e '.[all,dev]'
```

## Usage

CLI:

```bash
myotrace recording.mp4 --sample-id EHT_001 --out results/
myotrace recording.mp4 --method ensemble --correct-motion --out results/
myotrace recording.tif --fps 100 --roi 0 0 128 128 --out results/
myotrace recording.tif --fps 100 --signal-mode signed_displacement --out displacement-results/
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

TIFF frame rates come from ImageJ `finterval`/`fps` or OME `TimeIncrement` metadata. Supply `--fps` when unavailable; MyoTrace never assumes 30 fps. TIFF input must be a single grayscale time sequence. Ambiguous Z/channel stacks must be selected upstream. ImageJ Z-labeled stacks with explicit time metadata are accepted.

`--allow-qc-fail` retains flagged recordings for investigation. Static recordings export zero events and JSON `null` for undefined rates.

## Inputs and outputs

**Inputs:** cardiac-cell/tissue video or TIFF data, sample identifiers, ROI/mask information, frame-rate metadata, preprocessing parameters, and optional paired force/electrical/molecular features.

**Outputs:** motion traces, beat metrics, contraction/relaxation measurements, spectral/QC features, calibration results, multimodal feature tables, summaries, and provenance records. Typical files include `motion_trace.csv`, `beat_metrics.csv`, `summary.json`, `provenance.json`, and `qc.txt`.

Raw Farneback and Lucas–Kanade traces are pixel displacement magnitudes per frame interval; timestamps are interval midpoints. Ensemble traces are standardized consensus values, and robust preprocessing normalizes amplitudes. Signed displacement has N frame-timestamp samples and retains direction; its principal-mode polarity is not a verified physiological contraction direction. The default noise-aware detector uses explicit 40 ms Gaussian smoothing and a residual-noise prominence floor; `--detector legacy` retains the prior detector. These amplitudes are not directly comparable across preprocessing modes or recording frame rates.

The legacy `beat_metrics.csv`, `n_beats`, and `mean_bpm` fields describe **detected motion events**. Unsigned optical flow can have separate contraction and relaxation peaks: a 60-cycle/min generated recording gives approximately 120 motion events/min. Outputs explicitly mark `measurement_status=motion_events_not_verified_cardiac_beats`. Do not interpret these as verified cardiac rate or physiological contraction/relaxation kinetics without a reference for your acquisition.

## Validation

The repository includes synthetic timing benchmarks and utilities for group comparison, reference correlation, leave-one-modality-out analysis, Bland–Altman summaries, and repeatability. Biological validation requires independent recordings, external comparison, test/retest data, appropriate mechanical ground truth, locked reference panels, and independent batches or laboratories.

Reproduce the CPU analytical checks:

```bash
pip install -e '.[all,dev]'
pytest -q
python scripts/validate_cpu.py --out /tmp/myotrace-analytical
OPENBLAS_NUM_THREADS=1 python scripts/validate_recovery_cpu.py
# Downloads three checksum-pinned external hiPSC-CM movies (~307 MB):
OPENBLAS_NUM_THREADS=1 python scripts/validate_public_videos.py
```

Full reports, failed cases, source hashes, and public input identifiers are in `validation/cpu/`; 0.4 reports are historical, while 0.5 recovery evidence is under `validation/cpu/recovery/`. Public data are fetched from their source and retain the original dataset terms. CI checks six Python/NumPy combinations, the analytical grid, public-video execution, and an installed wheel outside the checkout.

## Limitations

Optical-flow outputs depend on image quality, frame rate, motion, preprocessing, segmentation/ROI choices, and camera stability. Synthetic benchmarks do not establish biological validity. Force claims require instrument-specific calibration. The multimodal maturity index is a computational framework, not a clinically validated maturity scale. Coverage accounts for missing locked features as well as missing modalities. `confidence` and `uncertainty_width` are heuristic descriptors, not validated probabilities or confidence intervals. The smoothing-residual `signal_snr_db` is also a descriptive proxy, not a calibrated physical SNR. Global translation correction can remove rigid biological motion as well as camera drift; confirm its appropriateness on your recordings. Full frame stacks are held in memory, so large acquisitions may require preprocessing or a future streaming implementation.

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See `LICENSE`.
