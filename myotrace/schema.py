from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd
import numpy as np


REQUIRED_TRACE_COLUMNS = ("sample_id", "timestamp_s", "motion_index", "modality")


def validate_trace_table(table: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_TRACE_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"Trace table missing columns: {missing}")
    if table.empty:
        raise ValueError("Trace table is empty")
    if (
        table["sample_id"].isna().any()
        or not np.isfinite(table[["timestamp_s", "motion_index"]].to_numpy(dtype=float)).all()
    ):
        raise ValueError("Trace identifiers and measurements must be present and finite")
    for _, group in table.groupby("sample_id", sort=False):
        if (group["timestamp_s"].diff().dropna() <= 0).any():
            raise ValueError("timestamp_s must be strictly increasing within each sample")


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
        return pd.DataFrame([{**dict(self.values), "sample_id": self.sample_id}])
