# MyoTrace 0.5.0 accuracy recovery

The 0.4.0 software checks passed, but its strict analytical suite still contained 64 failed noisy-trace cases out of 432, and its unsigned video motion signal could produce separate contraction and relaxation events. These are accuracy failures, not GPU shortages. This release improves those failures on CPU and retains unsuccessful cases. The [0.4 audit](CPU_AUDIT.md) and its reports remain unchanged.

## What changed

`analyze_trace` now defaults to `detector="noise_aware"`. After the existing bandpass, it applies a Gaussian with sigma 0.04 seconds, then requires prominence at least the larger of the existing 12% signal-span threshold and five times `1.4826 * MAD(filtered - smoothed)`. The robust residual scale is a heuristic noise estimate, not an exact Gaussian null model. Peak detection and reported widths/amplitudes/timings use the same smoothed waveform, avoiding inconsistent zero widths. The fixed settings were selected on the development grid before the separate confirmation grid was evaluated. Smoothing changes measured morphology; it is recorded in provenance and must not be interpreted as physiological kinetics validation.

`detector="legacy"` reproduces the 0.4 detection algorithm. This explicit option is available in Python, CLI, and HeartTwin. The default video signal remains unsigned motion for compatibility; changing peak detection cannot restore direction discarded by magnitude.

A new `signal_mode="signed_displacement"` computes Farneback flow between a selected reference image and every frame, then projects the spatial displacement field onto its first principal component. The PCA uses at most 256 sampled vector locations by default (512 components), keeping the projection small enough for CPU. The vector series is centered across frames before PCA. Dividing scores by the square root of the number of locations gives a signed spatial displacement coordinate in pixels; the centered trace is not absolute distance from a relaxed state. The vector-field sign is fixed by its largest absolute loading, not by physiological annotation. Reversing motion thus reverses the signal instead of generating a second positive magnitude peak.

This principal-component extension is inspired by reference-image displacement analysis; it is not presented as an exact reproduction of a published algorithm. [Optical-flow based non-invasive analysis of cardiomyocyte contractility](https://pmc.ncbi.nlm.nih.gov/articles/PMC5583397/) distinguishes consecutive-frame velocity doublets from reference-image displacement. [SciPy's peak-detection documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html) discusses smoothing noisy peak signals. Neither source validates this particular implementation.

The projection reports explained variance and flags a leading-mode fraction below 0.8. Sign does not identify systole, and a dominant mode could be camera drift, illumination artifact, or a noncardiac movement. The method must be assessed with appropriate ROI/mask, acquisition conditions and reference annotations. Masked grids require at least four selected sampled locations. Farneback is the supported engine; unsupported combinations fail explicitly.

Identical reference and target images yield exact zero flow. This avoids numerical optical-flow artifacts creating a first-frame impulse in static data. Frame-based displacement traces have N samples at frame timestamps, while interval-based motion traces have N-1 samples at interval midpoints. CLI output now prints the measurement qualification status as well as the legacy beat-named fields.

## Paired analytical evidence

[Protocol and full results](../validation/cpu/recovery/results.json) preserve every pass/failure and the exact source hashes. [The script](../scripts/validate_recovery_cpu.py) regenerates the checks with no downloads or GPU. Existing thresholds were not relaxed: rate error ≤3/min, precision/recall ≥0.9, timing match within 0.05 seconds plus one sample.

| Protocol | Legacy failures | Noise-aware failures | Cases per detector |
|---|---:|---:|---:|
| Original development grid | 64 | 27 | 432 |
| Independent confirmation grid | 56 | 8 | 432 |

Development uses the existing asymmetric triangular generator, seeds 7/29/41, 45/60/90/150 events/min, 25/50/100 Hz, two drift levels and raw/robust preprocessing. Confirmation uses an independent asymmetric Gaussian pulse generator, seeds 101/211/307, rates 55/75/115/175, 30/60/120 Hz, separate drift/noise conditions and the same paired preprocessing choices. Confirmation was not used to retune the detector. All noise-free and lower-noise cases pass. Remaining failures are at noise SD 0.15; some retain good overall rates but miss the strict event-timing criterion. They are not relabeled as successes.

48 independent generated-video cases vary texture seeds, 55/95/145 cycles/min, 30/60 Hz, starting phase and rigid translation versus radial deformation. All signed-displacement cases pass rate error ≤3/min and absolute waveform correlation ≥0.95. Phase/polarity are assessed separately from contraction identity. Unsigned estimates are recorded alongside signed estimates, including rate-doubling/aliasing behavior. This proves behavior on the defined generated motions, not arbitrary cardiac motion.

## Public recordings and release checks

[Public results](../validation/cpu/recovery/public/results.json) cover the three checksum-pinned SarcAsM hiPSC-CM recordings with all three unsigned methods plus signed displacement, repeated twice. Full traces, events, QC, projection diagnostics, acquisition metadata and provenance are retained. Method-disagreement statistics compare only unsigned methods; signed cycle rates and unsigned event rates are different quantities and are not interchangeable. These recordings lack independent beat/force annotation, so execution/repeatability is evaluated without claiming biological accuracy. All 12 configurations execute and repeat exactly. All three full-field signed projections flag multiple modes: explained fractions are about 0.363, 0.573 and 0.507 (10/20/30 kPa). Thus none is qualified as a single-mode biological beat assay. Candidate signed cycle rates are about 41.6, 71.8 and 84.8/min; they are descriptive results, not ground-truth heart rates. The 20 kPa unsigned methods also retain a >10/min disagreement. These biological qualifications remain unresolved; this report does not count execution success as accuracy success.

91 tests exercise the installed APIs and CLI, projection behavior, static inputs, masks/ROI, timing and invalid settings. Both NumPy 1.26/OpenCV 4.10 and NumPy 2.5/OpenCV 5 environments pass. CI also repeats the analytical recovery, public-file downloads and an installed-wheel smoke test. No GPU notebook was needed.

## Use

```bash
pip install -e '.[all,dev]'
myotrace recording.tif --fps 100 --signal-mode signed_displacement --out results/
myotrace recording.tif --fps 100 --detector legacy --out legacy-results/
OPENBLAS_NUM_THREADS=1 python scripts/validate_recovery_cpu.py
OPENBLAS_NUM_THREADS=1 python scripts/validate_public_videos.py
```

```python
from myotrace import analyze_video
result = analyze_video("recording.tif", fps_override=100,
                       signal_mode="signed_displacement", reference_frame=0)
```

The work was completed on CPU. A T4 would not provide missing physiological annotations or guarantee recovery of timing lost in noisy data. Force calibration, contraction/relaxation polarity, multiple independent beating foci, across-site accuracy and maturity-index biological validation remain separate validation tasks. This release is not described as a perfect cardiac assay.
