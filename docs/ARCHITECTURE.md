# MyoTrace architecture

```text
raw video / TIFF stack
        |
        v
[provenance + metadata]
        |
        v
[frame QC] ---> [ROI / mask] ---> [motion correction]
                                      |
                                      v
                         [optical-flow engine]
                           /              \
                    Farneback          Lucas-Kanade
                           \              /
                            \            /
                             v          v
                          [motion field]
                               |
                     +---------+---------+
                     |                   |
                     v                   v
              [global trace]      [spatial features]
                     |
                     v
             [robust preprocessing]
                     |
                     v
                [beat detection]
                     |
          +----------+-----------+
          |          |           |
          v          v           v
       kinetics   morphology     QC
          |          |           |
          +----------+-----------+
                     |
                     v
             [tidy mechanical data]
                     |
         +-----------+------------+
         |                        |
         v                        v
 [force calibration]       [CardioScore /
                            molecular data]
         |                        |
         +-----------+------------+
                     v
              [Fusion Index]
                     |
       +-------------+-------------+
       |             |             |
       v             v             v
     score       uncertainty    discordance
       |             |             |
       +-------------+-------------+
                     v
          reproducible report bundle
```

## Design principles

1. **Measurement before interpretation.** MyoTrace reports optical motion measurements; calibration is a separate layer.
2. **QC is data.** A metric without its acquisition and signal-quality context is incomplete.
3. **Biological replication is explicit.** Beats are technical observations, not independent biological replicates.
4. **Modalities remain separable.** The Fusion Index never erases the underlying mechanical, electrical, or molecular evidence.
5. **Everything is auditable.** Input hashes, configuration, reference endpoints, and software versions are retained.
6. **Baselines matter.** Existing validated tools should remain comparison baselines; MyoTrace is not designed to win by hiding competing methods.
