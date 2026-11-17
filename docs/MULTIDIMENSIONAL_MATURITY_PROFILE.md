# Multidimensional cardiomyocyte and engineered-tissue maturity profile v1

MyoTrace now defines eight independent dimensions: sarcomeric organization, contractility/beat mechanics, electrophysiological maturity, calcium-transient maturity, metabolic maturation, cell-cycle/proliferation status, electrical coupling/synchrony, and tissue architecture/alignment.

Each dimension carries its own status, value/unit, modality, method, calibration, sample/timepoint, QC and source artifact SHA-256. Missing values are represented as `null` with `not_measured`, `not_estimable` or `not_applicable`; they are never converted to zero. Modality and unit compatibility are checked. Physical force/velocity units require explicit calibration metadata.

## Video-only behavior

`video_only_profile(...)` fills only the contractility/beat-mechanics dimension from existing video summary data. That value remains a proxy because MyoTrace's motion events are not verified cardiac beats. Static recordings produce `not_estimable` with null value. Electrophysiology, calcium, metabolic, cell-cycle, sarcomeric and architecture dimensions remain explicitly unmeasured.

## Multimodal use

```python
from fusion.maturity_profile import build_maturity_profile

profile = build_maturity_profile(
    sample_id="EHT-001",
    timepoint="day-21",
    measurements={
        "sarcomeric_organization": {
            "status": "measured", "value": 0.72, "unit": "fraction",
            "modality": "imaging", "method": "alpha-actinin image analysis",
            "calibration": {"pixel_size_um": 0.2},
            "qc": {"status": "pass"},
            "source_artifact_sha256": "<sha256>",
        }
    },
)
```

The supplied measurement must also identify sample/timepoint. The profile does not calculate a composite maturity score. The legacy fusion index remains a separate explicitly configured exploratory tool and is not automatically invoked by this profile.

## Interpretation

Early cardiac marker expression does not establish mature functional cardiomyocytes. Contractility, electrical maturity, calcium handling, metabolism, proliferation and tissue architecture are distinct endpoints. PEG-hydrogel reprogramming studies should record early colony formation separately from later mechanical/electrical functional endpoints. A profile is an organized evidence record, not a validated universal maturity scale.
