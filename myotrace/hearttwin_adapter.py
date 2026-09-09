"""HeartTwin local-command adapter for the mechanical.analyze capability.

Reads a HeartTwin payload from HEARTTWIN_PAYLOAD, locates a mechanical
observation's input_path, runs MyoTrace analysis, and writes a compact,
JSON-serializable result to stdout.
"""
from __future__ import annotations

import json
import os
import sys

from .flow import FlowConfig
from .pipeline import analyze_video


def _find_input_path(payload: dict) -> tuple[str, dict]:
    """Locate the mechanical observation carrying the analysis input."""
    for obs in payload.get("observations", []):
        if obs.get("modality") == "mechanical" and "input_path" in obs.get("values", {}):
            return obs["values"]["input_path"], obs["values"]
    raise ValueError(
        "No 'mechanical' observation with an 'input_path' value was provided; "
        "mechanical.analyze requires modality='mechanical' and values.input_path."
    )


def main() -> int:
    raw = os.environ.get("HEARTTWIN_PAYLOAD")
    if not raw:
        print("HEARTTWIN_PAYLOAD environment variable not set", file=sys.stderr)
        return 1
    try:
        payload = json.loads(raw)
        video_path, params = _find_input_path(payload)
        cfg = FlowConfig(
            method=params.get("method", "farneback"),
            motion_percentile=params.get("motion_percentile", 75.0),
        )
        result = analyze_video(
            video_path,
            sample_id=payload.get("entity_id"),
            fps_override=params.get("fps"),
            flow_config=cfg,
            reject_failed_qc=not params.get("allow_qc_fail", False),
            robust=not params.get("raw_signal", False),
            correct_motion=params.get("correct_motion", False),
        )
        output = {
            "sample_id": result.sample_id,
            "summary": result.summary,
            "qc": {
                "usable": result.qc.usable,
                "reasons": result.qc.reasons,
                "fps": result.qc.fps,
                "frame_count": result.qc.frame_count,
            },
            "provenance": result.provenance,
        }
        print(json.dumps(output, allow_nan=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - convert to HeartTwin error contract
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
