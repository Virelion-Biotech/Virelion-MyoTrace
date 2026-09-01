# Virelion-MyoTrace

**Research-grade video cardiac mechanics + auditable multimodal tissue characterization.**

MyoTrace is the mechanical arm of the Virelion cardiac characterization stack. It turns brightfield/phase-contrast video or TIFF stacks into quantitative beat-level mechanics, signal-quality metrics, spectral descriptors and reproducibility metadata. Its fusion layer combines mechanical, electrical and molecular endpoints without hiding reference assumptions behind a black-box score.

> **Positioning:** contractility-only analysis is an established field. MyoTrace is designed around interoperability, reproducibility, uncertainty-aware reporting and multimodal benchmarking. Motion magnitude is a *motion index*, not force. Force/stress requires instrument-specific calibration against a transducer, pillar deflection or other mechanical ground truth.

## Architecture

```text
Video / TIFF
    |
    +--> frame QC --> ROI / mask --> optional camera-motion correction
                                      |
                                      +--> Farneback / Lucas-Kanade / ensemble optical flow
                                                |
                                                +--> robust preprocessing
                                                +--> spectral + signal QC
                                                +--> beat detection / morphology
                                                +--> uncertainty + provenance

CardioScore / ElectroTrace ----+
                               +--> aligned feature table --> Fusion Index
Transcriptomic / qPCR ---------+                         |
                                                         +--> modality scores
                                                         +--> coherence
                                                         +--> confidence
                                                         +--> coverage
                                                         +--> sensitivity analysis
```

## Core capabilities

### Mechanical engine
- Dense Farneback optical flow.
- Sparse Lucas–Kanade tracking.
- Cross-method **ensemble** signal for robustness/sensitivity analysis.
- Robust frame intensity normalization.
- Explicit frame-level QC for low FPS, flat frames, intensity drift and negligible motion.
- Explicit rectangular ROI and binary-mask controls.
- Conservative rigid-translational camera-motion correction; no flexible registration that could absorb biological deformation.
- Robust median/detrend preprocessing with raw-vs-processed sensitivity control.
- Beat rate and interval distributions.
- Motion amplitude, rise/time-to-peak, relaxation, 50% width and area-under-motion.
- Beat-level morphology/quality scoring and sample-level regularity/CV descriptors.
- Spectral dominant frequency, entropy and band power.
- Deterministic synthetic cardiac-like traces for regression benchmarks.

### Scientific integrity layer
Every analysis can carry:

- source SHA-256;
- software/Python/platform information;
- complete analysis parameters;
- signal SNR, periodicity, drift, clipping and missingness;
- QC flags rather than silent failure;
- bootstrap summaries for uncertainty estimation.

### Mechanical ground-truth calibration
A dedicated calibration module fits an explicit motion-index-to-force relationship only when paired measurements are supplied. It reports slope, intercept, RMSE and R² so force claims remain tied to an instrument-specific calibration experiment rather than being inferred from pixel motion alone.

### Multimodal fusion
The Composite Cardiac Tissue Maturity Index is intentionally transparent:

1. Each endpoint has explicit fetal/adult reference values.
2. Directionality can be inverted for markers where lower values indicate maturity.
3. Log-transformed endpoints are supported for multiplicative measurements.
4. Features are grouped into mechanical/electrical/molecular modalities.
5. Modality weights are normalized and exposed.
6. Missing modalities reduce coverage rather than being silently imputed.
7. Cross-modality disagreement produces a coherence score and can set status to `discordant_modalities`.
8. The result exposes confidence and an uncertainty-width heuristic so a high score is not confused with a high-confidence score.

The default score is a **calibration framework**, not a clinically validated maturity scale. Reference endpoints must be locked prospectively and externally validated before biological claims are made.

## Validation framework

`myotrace.benchmark` provides a deterministic synthetic timing benchmark. `fusion.validation` and `fusion.agreement` provide:

- Mann–Whitney group comparisons with rank-biserial effect size;
- Spearman reference correlation;
- leave-one-modality-out sensitivity analysis;
- Bland–Altman agreement summaries;
- coefficient-of-variation repeatability summaries.

The intended publication-grade validation program is:

1. synthetic ground truth for timing robustness;
2. public-video external validation against established analysis software;
3. inter-method agreement and test/retest repeatability on identical recordings;
4. paired force-ground-truth calibration before any force language;
5. locked reference-panel calibration using training material only;
6. fully held-out validation across independent differentiation batches and, ideally, independent laboratories;
7. perturbation testing for focus, illumination, motion blur, frame rate, spatial resolution and compression;
8. blinded analysis where feasible.

A detailed protocol is provided in `docs/VALIDATION.md`.

No automated score substitutes for biological ground truth.

## Interoperability contract

The stable biological join key is `sample_id`. Time-resolved mechanics use `timestamp_s`; modality summaries are exposed with explicit prefixes such as `mechanical:mean_bpm`, `electrical:fpd_ms`, and `molecular:MYH7_MYH6_ratio`.

This enables downstream composition with Virelion-CardioScore and Virelion-ElectroTrace without copying their implementations into MyoTrace.

## Install

```bash
pip install -e '.[all,dev]'
```

## CLI

```bash
myotrace recording.mp4 --sample-id EHT_001 --out results/
```

For robustness analysis:

```bash
myotrace recording.mp4 --method ensemble --correct-motion --out ensemble-results/
```

Outputs:

```text
results/
├── motion_trace.csv
├── beat_metrics.csv
├── summary.json
├── provenance.json
└── qc.txt
```

Use `--raw-signal` to bypass robust preprocessing for sensitivity analysis, or `--allow-qc-fail` to inspect data that failed automated QC.

## Python API

```python
from myotrace import analyze_video, fit_force_calibration

result = analyze_video("recording.mp4", sample_id="EHT_001", correct_motion=True)
print(result.summary)
print(result.provenance)

# Only when paired mechanical ground truth exists:
cal = fit_force_calibration(motion_values, force_values, units="uN")
print(cal.r2, cal.rmse)
```

## Fusion API

```python
from fusion.model import FeatureReference, FusionConfig, calculate_index

config = FusionConfig(references={
    "mechanical:mean_bpm": FeatureReference(60, 120),
    "electrical:fpd_ms": FeatureReference(100, 200),
    "molecular:MYH7_MYH6_ratio": FeatureReference(0.2, 2.0, transform="log"),
})

result = calculate_index("EHT_001", {
    "mechanical:mean_bpm": 105,
    "electrical:fpd_ms": 165,
    "molecular:MYH7_MYH6_ratio": 1.1,
}, config)

print(result.to_dict())
```

## Research status

The repository is engineered for serious research use, but **Nature-level publication is not a software feature**. The missing ingredient is empirical evidence: independent recordings, matched multimodal constructs, blinded comparison, ground-truth force calibration, test/retest data, external validation and a preregistered statistical analysis plan.

That distinction is intentional. MyoTrace is being built so those experiments can be executed and audited cleanly rather than encoded after the fact.

## License

GNU Affero General Public License v3.0 (AGPL-3.0). See `LICENSE`.
