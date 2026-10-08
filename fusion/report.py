from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .model import FusionResult
from myotrace.serialization import json_safe


def results_table(results: Iterable[FusionResult]):
    import pandas as pd

    rows = []
    for r in results:
        row = {
            "sample_id": r.sample_id,
            "composite_score": r.composite_score,
            "coverage": r.coverage,
            "status": r.status,
            "coherence_score": r.coherence_score,
            "confidence": r.confidence,
            "uncertainty_width": r.uncertainty_width,
        }
        row.update({f"modality:{k}": v for k, v in r.modality_scores.items()})
        row.update({f"feature:{k}": v for k, v in r.feature_scores.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def write_json(results: Iterable[FusionResult], path: str | Path) -> None:
    payload = [r.to_dict() for r in results]
    Path(path).write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
