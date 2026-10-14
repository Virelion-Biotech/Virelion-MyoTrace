# SarcAsM Colab run and method disagreement

The supplied `07_MyoTrace_SarcAsM_v1_results.zip` completed its declared scope
in 1544.82 seconds (25 minutes 45 seconds). The worker returned zero and
`RUN_STATUS.json` reports `COMPLETE_SCOPED`, with scientific validation explicitly
`NOT_ESTABLISHED_BY_EXECUTION`. The code was pinned to
`6772d2e6db11a2ef46f2bef16a91e5c5e3d3d7a4`.

## Retained evidence

All 54 unique archive members are retained in
[`validation/colab_sarcasm_2026_10_14`](../validation/colab_sarcasm_2026_10_14/).
All 53 entries covered by the original artifact manifest match their SHA-256
hashes. The manifest correctly excludes itself. Supplied launchers are archived
as `.py.txt` for provenance, never executed in this review; `ARCHIVE_PATHS.json`
maps their original names. All other original files retain their original bytes.
The input movie and structural archive receipts record matching producer
checksums, but those raw inputs are not included in this results ZIP and were
not independently downloaded in this review. Raw source data are not redistributed.

| Endpoint | Recorded result | Interpretation |
|---|---|---|
| Three public movies, four configurations each | 12/12 execute and repeat exactly | Input interoperability/repeatability |
| Independent generated traces, noise-aware detector | 8 failures of 432, versus 56 for legacy | Analytical improvement with retained failures |
| Development generated traces, noise-aware detector | 27 failures of 432, versus 64 for legacy | Development evidence |
| Generated signed-displacement videos | 48/48 pass | Recovery in the tested synthetic conditions |
| Structural annotation inventory | Labels present; `READINESS_ONLY` | No frozen mask/length prediction adapter evaluated |

## 20 kPa disagreement diagnosis

The local review replays detection on each of the 12 exported analysis signals.
Every exported event time and mean instantaneous event rate is reproduced within
numerical tolerance. This checks detector/export consistency, not an independent
rerun of optical flow on raw movies.

| Method | Events | Mean instantaneous motion event rate/min |
|---|---:|---:|
| Farneback | 7 | 68.52795 |
| Lucas–Kanade | 9 | 80.24858 |
| Ensemble | 9 | 80.69576 |

All seven Farneback events match events from both alternatives within
`0.05 + 1/fps` seconds (0.06641 s). Lucas–Kanade additionally selects peaks at
1.66521 and 5.66829 seconds; the ensemble selects peaks at 1.64881 and 5.63547
seconds. This accounts for the different interval distributions and the
12.16781/min reported rate span. The reported mean is the arithmetic mean of
instantaneous inter-event rates, rather than event count divided by duration.

Farneback measures a dense-field 75th-percentile displacement magnitude, while
Lucas–Kanade measures the median magnitude of tracked feature displacements.
Their waveforms need not have identical smaller peaks. The same detector uses
40 ms smoothing and the larger of 12% signal span or five residual-noise scales
as its prominence threshold. These differing waveforms therefore produce
different selections without demonstrating a software defect.

For the ensemble, the two extra peak prominences are about 0.156 and 0.177,
against a default threshold of 0.15358. Raising the prominence fraction from
0.12 to 0.15 removes those peaks, producing seven events and 68.645/min.
However, Lucas–Kanade retains nine events throughout the exploratory fraction
sweep 0.06–0.18. Lowering the Farneback fraction to 0.09 produces eight events
and 78.802/min. These changes can reduce a rate difference while selecting
different events: agreement alone is not a valid tuning objective.

**Decision: preserve the detector and its defaults, retain the disagreement
flag, and do not select a winning method.** No independent beat annotations
establish whether the two additional events are real biological events,
contraction/relaxation components, or tracking artifacts. The ensemble is derived
from the other two methods, so it is not an independent validating reference.

The signed-displacement alternative also cannot resolve this benchmark: all
three full-field recordings carry `multiple_displacement_modes` flags, with
explained fractions 0.36333, 0.57304 and 0.50658. Its principal-mode polarity is
not verified systole. Unsigned motion events must not be relabeled as cardiac
beats or force measurements.

## Remaining work and reproduction

A defensible resolution needs independently labeled events on these recordings
or a suitable new benchmark, acquisition-linked ROIs/masks, and inspection of
the additional peaks against video. Regional motion, feature tracking quality,
and drift correction are candidates for evaluation on a development set followed
by a locked independent holdout. None is shown to improve biological accuracy
by this export. Independent force, strain and maturation references remain
unavailable in this selected archive; structural accuracy also needs a frozen
sarcomere prediction adapter. The eight independent synthetic failures remain
documented in the retained recovery report.

To reproduce the checksum audit, detector replay, candidate peak diagnostics,
and exploratory sweep from the repository root:

```bash
pip install -e '.[all,dev]'
python scripts/diagnose_sarcasm_export.py --out /tmp/sarcasm-diagnostic.json
```

The committed derived report is `DISAGREEMENT_DIAGNOSTIC.json`. It does not
replace the original results. This import changes evidence and documentation;
it makes no detector or optical-flow implementation changes.
