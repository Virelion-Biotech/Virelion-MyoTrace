# Virelion-MyoTrace

**Video-based cardiac contractility analysis + multimodal tissue-maturity fusion.**

Virelion-MyoTrace is the mechanical arm of the Virelion cardiac characterization stack. It converts brightfield/phase-contrast video into reproducible beat-level mechanical kinetics, then exposes a transparent fusion layer for combining mechanical, electrical, and molecular measurements at the `sample_id` level.

> **Scientific positioning:** MyoTrace is not presented as a first-in-field contractility tracker. The contribution is interoperability and an explicit, auditable multimodal maturity index. The motion output is a calibrated-independent *motion index*, not force. Force/stress claims require an instrument-specific calibration dataset.

## What is implemented

### MyoTrace
- AVI/MP4/MOV loading through OpenCV.
- TIFF-stack loading through `tifffile`.
- Robust intensity normalization.
- Dense Farnebäck optical-flow motion trace.
- Sparse Lucas-Kanade alternative.
- Beat detection with configurable physiological bounds.
- Beat-level amplitude, interval, beat rate, rise/time-to-peak, and relaxation kinetics.
- Explicit video QC for low frame rate, global intensity drift, flat/corrupt frames, and negligible motion.
- Tidy outputs designed for joining with other Virelion modality tables.

### Fusion layer
- Marker definitions for common maturation-panel features such as `MYH7_MYH6_ratio`, `TNNI3_TNNI1_ratio`, SERCA2A/ATP2A2 and GJA1/Cx43.
- User-supplied fetal/adult reference endpoints rather than hidden biological thresholds.
- Per-feature 0-1 maturation scores and a 0-100 composite score.
- Equal-by-default modality weights across mechanical, electrical, and molecular data; all weights are configurable.
- Partial-data handling with explicit coverage and status fields.
- Benchmark helpers for adult-vs-fetal reference separation.
- A modality assembler that joins MyoTrace, CardioScore-style, and molecular tables by `sample_id`.

This first release intentionally does **not** claim a validated clinical or biological maturity scale. The index becomes publication-grade only after calibration and external validation on matched multimodal tissue constructs.

## Installation

```bash
pip install -e .[all,dev]
```

For video only:

```bash
pip install -e .[video]
```

## Command line

```bash
myotrace recording.mp4 --sample-id EHT_001 --out results/
```

The command writes:

```text
results/
├── motion_trace.csv
├── beat_metrics.csv
└── qc.txt
```

The trace schema contains `sample_id`, `timestamp_s`, `motion_index`, and `modality`. Beat output contains one row per detected beat.

## Python example

```python
from myotrace import analyze_video

result = analyze_video("recording.mp4", sample_id="EHT_001")
print(result.summary)
print(result.beats.head())
```

## Fusion example

```python
from fusion.model import FeatureReference, FusionConfig, calculate_index

references = {
    "mechanical:beat_rate_bpm": FeatureReference(fetal=60, adult=120),
    "electrical:fpd_ms": FeatureReference(fetal=150, adult=300),
    "molecular:MYH7_MYH6_ratio": FeatureReference(fetal=0.2, adult=2.0),
}

config = FusionConfig(references=references)
result = calculate_index(
    "EHT_001",
    {
        "mechanical:beat_rate_bpm": 100,
        "electrical:fpd_ms": 250,
        "molecular:MYH7_MYH6_ratio": 1.5,
    },
    config,
)

print(result.composite_score)  # 0-100
print(result.modality_scores)
print(result.coverage)
```

## Interoperability contract

The stable join key is `sample_id`. Time-resolved mechanical data use `timestamp_s`; modality-level summaries can be aggregated and prefixed as `mechanical:<feature>`, `electrical:<feature>`, and `molecular:<feature>`.

This makes MyoTrace compatible with a downstream CardioScore/ElectroTrace integration without coupling the packages or duplicating their implementations.

## Validation philosophy

MyoTrace is deliberately conservative about claims:

1. Validate motion extraction against public EHT/organoid videos and synthetic motion traces.
2. Compare beat timing and kinetics with an established analysis method on the same recordings.
3. For force, acquire paired deflection/transducer data and fit a separate calibration model.
4. For the fusion score, lock a reference panel and reference endpoints before testing held-out data.
5. Avoid training a high-capacity ML fusion model until enough matched multimodal samples exist; the transparent weighted index is the baseline.
6. Report missing modalities and QC failures rather than silently imputing them.

## Research references

The algorithmic design should be compared against established contractility-analysis approaches such as MUSCLEMOTION, OpenHeartWare, PIV-based EHT monitoring, and EHT Analysis. This repository does not copy their source code.

## Development

```bash
pip install -e .[all,dev]
pytest
ruff check .
```

The project is licensed under **GNU Affero General Public License v3.0 (AGPL-3.0)**. See `LICENSE`.
