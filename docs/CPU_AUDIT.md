# MyoTrace 0.4.0 CPU audit

All implemented methods run on CPU. No GPU, Colab notebook, model training, or user-returned results were needed. This release repairs software and checks analytical behavior; it does not establish a validated biological assay.

## Reproduced failures and repairs

The starting commit was `857130b` (0.3.0). Its 18 tests passed with about 57% statement coverage, despite not exercising video analysis end to end. [Baseline measurements](../validation/cpu/baseline.json) preserve the original behavior.

- A one-pixel translation returned about 0.00015 pixels in Farneback. Convert normalized frames to uint8 in the 0–255 intensity range before optical flow. [OpenCV's documented convention](https://docs.opencv.org/4.10.0/dc/d6b/group__video__track.html) informed this change; known-displacement tests measure its effect.
- Lucas–Kanade returned about 0.54 pixels because it took a norm over the wrong array dimension. Use Euclidean displacement for each tracked point. Reinitialize from the actual previous image; tracking loss is an explicit error instead of a silently substituted zero.
- Ensemble standardization amplified negligible constant-speed jitter. Traces with standard deviation at or below `0.001 * max(1, abs(mean))` are treated as constant during consensus scaling. This is an explicit numerical floor for sensitivity analysis, not a universal imaging detection limit.
- A static recording produced a false motion event after median-filter zero padding. Use nearest-edge padding and return a zero normalized trace for exactly constant inputs. Center signals before bandpass filtering.
- No-beat summaries contained a NaN beat count and caused the CLI to fail after writing files. Export zero events, leave unestimable metrics undefined, and serialize undefined numbers as JSON null throughout CLI/HeartTwin/fusion exports.
- TIFF loading assumed 30 fps and ignored acquisition metadata. Read supported ImageJ/OME timing; require an explicit override when absent. Reject ambiguous channels, Z stacks without time metadata, and multiple TIFF series. Reject zero/nonfinite overrides.
- Provenance reported 0.2.0 and recorded `None` rather than the actual default flow configuration. Record 0.4.0, effective parameters, dependency versions, acquisition/override timing, input SHA-256, mask SHA-256, and midpoint timestamp convention.
- Frame QC rejected constant-speed motion because it required motion above 1.5 times an internal quartile. Measure the nonzero frame-change fraction; this remains an intensity-difference descriptor, not a calibrated motion assay.
- Signal QC called a clean sinusoid zero quality because the waveform MAD was used as noise. Use a smoothing-residual ratio and autocorrelation at 0.25–2-second lags rather than lag-one smoothness. Both are heuristic descriptors; neither is a physical SNR or calibrated assay pass/fail probability.
- Fusion coverage could report complete coverage after only one feature in a larger locked panel. Account for missing weighted features within each modality. Exclude zero-weight modalities from coherence; reject invalid anchors, transforms, weights and thresholds. Label uncertainty as heuristic.
- Added finite-input/parameter validation, per-sample timestamp validation, ROI CLI selection, spatial mask selection for flow aggregation/tracking, valid bootstrap settings, and packaging/license/citation hygiene.

## Analytical evidence

[The executable protocol](../scripts/validate_cpu.py) records every case, pass and failure, in [the complete report](../validation/cpu/current/results.json). Thresholds are included in that report.

- 72 flow cases: three texture seeds, four displacements (0.25, 0.5, 1, 2 pixels), three intensity dtypes, and two methods. All pass the maximum absolute error threshold of 0.1 pixel.
- 432 waveform cases: three seeds, four rates (45–150 events/min), three sampling rates (25/50/100 Hz), three noise amplitudes, two drift levels, and raw/robust preprocessing. All noise-free and 0.03-noise cases pass. 64 cases fail at noise SD 0.15: 42 raw, 22 robust. Failures remain in the report. Passing requires absolute rate error ≤3, precision and recall ≥0.9, with event matching tolerance 0.05 s + one sample. These parameters do not establish performance on arbitrary biological recordings.
- A generated sinusoidal displacement completes 60 physical cycles/min but produces approximately 120 motion events/min for all three methods. This counterexample is retained in the report and traces. Legacy beat-named outputs are explicitly marked as unverified motion events; physiological beat/kinetic interpretation requires a reference or a future separately validated directional/displacement method.

## Public biological input execution

[The source manifest](../validation/cpu/public_sources.json) pins three SarcAsM high-speed ACTN2-citrine hiPSC-CM TIFFs (10/20/30 kPa) to a Git commit and SHA-256. The dataset is attributed to Haertter and colleagues, [Zenodo 8232838](https://doi.org/10.5281/zenodo.8232838), CC-BY-NC-4.0. Source images are not redistributed in this repository. The reproducible downloader checks full-file hashes before analysis.

[Public-video results](../validation/cpu/public/results.json) retain summaries, frame QC, acquisition rates, source provenance, and exact repeated-run checks for every movie/method combination. These are external-input execution and repeatability checks, **not** accuracy estimates: annotated cardiac events, matched force measurements, and reference maturity labels were not supplied. QC flags and undefined values remain visible. The toolkit has not been fitted to this dataset. All nine movie/method combinations execute successfully and produce identical traces on two runs. Method agreement is not universal: in the 20 kPa movie, Farneback yields about 79.9 motion events/min, Lucas–Kanade 105.5, and ensemble 82.1. The report flags this >10-events/min disagreement; no method is designated ground truth.

## Release verification and remaining scope

81 automated tests pass with approximately 94% statement coverage in the tested CPU environment. The release tests actual AVI/TIFF decoding, all three optical-flow methods, ROI/mask handling, translation correction, CLI and HeartTwin entry points, strict JSON, numerical utilities, fusion, packaging, and invalid-input behavior. CI covers Python 3.10–3.12 with NumPy 1.x/2.x, CPU analytical validation, public-video execution, and installed-wheel operation outside the source checkout. Generated outputs are reproducible scripts, not GPU checkpoints.

No independent force validation, physiological event annotation, multi-laboratory accuracy study, calibrated QC classifier, or maturity-index validation was created by this audit. The existing [validation protocol](VALIDATION.md) describes the evidence required. Bootstrap resampling assumes the supplied observations are appropriate independent units; do not treat correlated beats as independent biological replicates. Video is decoded into memory; variable-frame-rate timing and streaming of large recordings are not implemented.
