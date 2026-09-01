# Virelion-MyoTrace validation protocol

This document defines the evidence required before presenting MyoTrace or the Composite Cardiac Tissue Maturity Index as a validated research assay.

## 1. Analytical validity

### Synthetic ground truth
Use deterministic synthetic traces to quantify:

- beat detection sensitivity and precision;
- heart-rate absolute error;
- missed/extra beat count;
- robustness to additive noise;
- robustness to baseline drift;
- robustness to dropped/non-finite samples;
- robustness to frame-rate reduction;
- parameter sensitivity.

Report the full parameter grid and random seeds.

### Video perturbation suite
Start from a fixed reference recording and generate blinded perturbations covering:

- global illumination changes;
- slow intensity drift;
- local occlusion;
- camera translation;
- blur;
- compression;
- reduced frame rate;
- reduced spatial resolution.

The primary endpoint should be degradation in beat timing and kinetic estimates, not merely whether the software produces an output.

## 2. Method comparison

For recordings that have an established analysis result, compare MyoTrace against the reference method on identical videos.

Recommended reporting:

- Bland–Altman bias and limits of agreement;
- Spearman correlation;
- absolute and relative error;
- beat-count agreement;
- repeatability across repeated analyses;
- sensitivity to analyst-defined ROI where applicable.

Agreement should be evaluated on independent samples rather than only on repeated measurements from the same sample.

## 3. Mechanical ground truth

Optical-flow amplitude is not force.

To claim force or stress:

1. collect paired video and calibrated force/transducer or pillar-deflection measurements;
2. split calibration and validation at the biological batch level;
3. fit calibration on training batches only;
4. report slope, intercept, RMSE and R²;
5. evaluate calibration on unseen batches;
6. test calibration stability across imaging sessions and tissue geometries.

Never use the same paired observations for both calibration and final performance claims.

## 4. Multimodal maturity index

Lock the following before testing the held-out set:

- marker panel;
- endpoint transforms;
- fetal/adult or other reference anchors;
- feature weights;
- modality weights;
- missing-data policy;
- QC exclusion criteria;
- statistical analysis plan.

The held-out set should contain matched mechanical, electrical and molecular measurements from the same biological construct or prespecified matched experimental unit.

### Required analyses

- adult-like versus immature discrimination;
- rank correlation with an independently defined maturation reference;
- leave-one-modality-out sensitivity;
- modality discordance analysis;
- bootstrap confidence intervals;
- batch/site robustness;
- external validation using an independently generated cohort.

Do not train a high-capacity ML fusion model and then evaluate it on the same biological cohort used to tune features, references or weights.

## 5. Experimental design

The biological replicate is the experimental unit. Do not treat individual video frames or beats as independent biological replicates.

Record at minimum:

- donor/line;
- differentiation batch;
- tissue/model type;
- construct geometry;
- media/conditioning protocol;
- imaging frame rate and exposure;
- temperature and acquisition conditions;
- perturbation/drug conditions;
- video file checksum;
- analysis software version and parameters.

Where feasible, randomize acquisition order and blind the analysis to condition labels.

## 6. Publication-ready reporting

A strong paper should report both successful and failed analyses, pre-specified exclusions, parameter sensitivity and uncertainty.

Software release should include:

- exact version;
- environment specification;
- immutable input identifiers/checksums;
- example data that can legally be redistributed;
- machine-readable outputs;
- complete analysis configuration;
- tests and continuous integration;
- citation metadata.

**Important:** a publication-grade repository is not equivalent to a validated biological assay. The latter requires independent experimental evidence.
