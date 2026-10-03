from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


REQUIRED_TRACE_COLUMNS = ("sample_id", "timestamp_s", "motion_index", "modality")


def validate_trace_table(table: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_TRACE_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"Trace table missing columns: {missing}")
    if table.empty:
        raise ValueError("Trace table is empty")
    if (table["timestamp_s"].diff().dropna() < 0).any():
        raise ValueError("timestamp_s must be non-decreasing")


def merge_modalities(
    mechanical: pd.DataFrame,
    electrical: pd.DataFrame | None = None,
    molecular: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long-format interoperability helper using sample_id as the stable join key."""
    frames = [mechanical]
    for frame in (electrical, molecular):
        if frame is not None:
            frames.append(frame)
    for frame in frames:
        if "sample_id" not in frame.columns:
            raise ValueError("Every modality table must contain sample_id")
    return pd.concat(frames, ignore_index=True, sort=False)


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    values: Mapping[str, Any]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"sample_id": self.sample_id, **dict(self.values)}])
