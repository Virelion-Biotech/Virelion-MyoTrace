# Publication-grade validation roadmap

MyoTrace can be engineered to publication-grade software standards, but no software implementation alone establishes biological validity. A high-impact paper requires prospective experimental evidence.

## 1. Analytical validity

- Lock an analysis specification before testing.
- Evaluate sensitivity to frame rate, compression, focus, illumination, ROI size, and camera drift.
- Establish repeatability within recording and reproducibility across recordings, days, operators, and instruments.
- Report intraclass correlation coefficients (ICC), coefficient of variation, Bland–Altman agreement, and confidence intervals.
- Include negative controls and deliberately degraded videos.

## 2. Orthogonal mechanical validation

For EHTs or engineered tissues with an accepted force measurement, acquire synchronized video and force traces from the same biological construct.

- Fit calibration only on a prespecified training set.
- Evaluate the locked calibration on independent constructs.
- Report slope, intercept, R², MAE/RMSE and agreement plots.
- Never label uncalibrated optical displacement as absolute force.

## 3. Biological validity

Use biologically independent constructs rather than treating beats or frames as independent biological replicates. Test known perturbations expected to alter contractility and kinetics. Prespecify primary endpoints and avoid selecting metrics after seeing group separation.

## 4. Multimodal validity

For the Fusion Index, acquire matched mechanical, electrophysiological, and molecular measurements from the same construct batch.

- Freeze reference definitions before model evaluation.
- Keep an untouched external validation cohort.
- Compare transparent weighted scoring against simpler baselines.
- Report ablations: mechanical-only, electrical-only, molecular-only, and all-modalities.
- Report missing-modality performance and discordance rather than imputing a falsely complete score.

## 5. Generalization

Validate across at least two independent laboratories or acquisition systems when feasible. Include multiple cell lines/donors and model formats. Split at the biological-unit level so videos from the same construct never leak across train and test sets.

## 6. Reproducibility

Every published result should be reconstructable from:

1. immutable input hashes;
2. exact software version/commit;
3. environment lockfile;
4. analysis configuration;
5. reference definitions;
6. random seeds;
7. machine-readable outputs;
8. prespecified statistical analysis.

## 7. What would justify a strong methods claim

The strongest eventual claim is not that MyoTrace is universally superior to every existing contractility tool. It is that a validated, interoperable mechanical representation can be reproduced across acquisition conditions and integrated with electrical and molecular phenotypes to improve characterization of cardiac tissue state.

Until those experiments are completed, the repository should use language such as **"research software," "candidate metric," and "validation framework"**, not clinical or universal diagnostic claims.
