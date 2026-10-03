from __future__ import annotations

from typing import Mapping

import pandas as pd


def summarize_by_sample(table: pd.DataFrame, *, modality: str, metrics: tuple[str, ...]) -> pd.DataFrame:
    """Collapse a modality table to one row per sample and prefix metric names for fusion."""
    if "sample_id" not in table.columns:
        raise ValueError("Modality table must contain sample_id")
    present = [c for c in metrics if c in table.columns]
    if not present:
        raise ValueError(f"No requested metrics found for {modality}")
    out = table.groupby("sample_id", as_index=False)[present].mean(numeric_only=True)
    return out.rename(columns={c: f"{modality}:{c}" for c in present})


def assemble_fusion_features(
    mechanical: pd.DataFrame,
    *,
    mechanical_metrics: tuple[str, ...] = ("beat_rate_bpm", "amplitude", "rise_time_s", "relaxation_time_s"),
    electrical: pd.DataFrame | None = None,
    electrical_metrics: tuple[str, ...] = ("beat_rate_bpm", "fpd_ms"),
    molecular: pd.DataFrame | None = None,
    molecular_metrics: tuple[str, ...] = ("MYH7_MYH6_ratio", "TNNI3_TNNI1_ratio", "ATP2A2_SERCA2A", "GJA1_CX43"),
) -> pd.DataFrame:
    """Join mechanical/electrical/molecular measurements by sample_id for index calculation."""
    pieces = [summarize_by_sample(mechanical, modality="mechanical", metrics=mechanical_metrics)]
    if electrical is not None:
        pieces.append(summarize_by_sample(electrical, modality="electrical", metrics=electrical_metrics))
    if molecular is not None:
        pieces.append(summarize_by_sample(molecular, modality="molecular", metrics=molecular_metrics))
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on="sample_id", how="outer")
    return out


def row_to_feature_mapping(row: Mapping[str, object]) -> dict[str, float]:
    features: dict[str, float] = {}
    for key, value in row.items():
        if key == "sample_id" or value is None:
            continue
        try:
            features[key] = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return features
