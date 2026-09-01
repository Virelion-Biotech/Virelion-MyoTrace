import numpy as np
import pandas as pd

from myotrace.qc import assess_frames
from myotrace.schema import merge_modalities, validate_trace_table


def test_qc_flags_no_motion() -> None:
    frames = np.ones((20, 32, 32), dtype=np.uint8)
    report = assess_frames(frames, 30)
    assert not report.usable
    assert "negligible_detectable_motion" in report.reasons


def test_schema_and_merge() -> None:
    a = pd.DataFrame({"sample_id": ["s1"], "timestamp_s": [0.0], "motion_index": [0.1], "modality": ["mechanical"]})
    validate_trace_table(a)
    b = pd.DataFrame({"sample_id": ["s1"], "fpd_ms": [250.0], "modality": ["electrical"]})
    merged = merge_modalities(a, b)
    assert len(merged) == 2
