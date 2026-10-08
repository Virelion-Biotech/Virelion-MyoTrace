"""CPU execution check on three public SarcAsM hiPSC-CM movies.

No annotated beat/force ground truth accompanies these movies; this is an
external-input interoperability and repeatability check, not assay validation.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
import urllib.request

import numpy as np
import cv2

from myotrace.pipeline import analyze_video
from myotrace.flow import FlowConfig
from myotrace.io import load_tiff_stack
from myotrace.serialization import json_safe


def run(manifest_path, data, out):
    cv2.setNumThreads(1)
    manifest = json.loads(manifest_path.read_text())
    data.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in manifest["movies"]:
        path = data / source["name"]
        if not path.exists():
            with urllib.request.urlopen(source["url"], timeout=120) as response, path.open("wb") as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != source["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")
        loaded = load_tiff_stack(path)
        for method in ("farneback", "lk", "ensemble"):
            try:
                result = analyze_video(path, flow_config=FlowConfig(method=method))
                second = analyze_video(path, flow_config=FlowConfig(method=method))
                np.testing.assert_array_equal(result.trace.motion_index, second.trace.motion_index)
                stem = path.stem + "_" + method
                result.trace.to_csv(out / (stem + "_trace.csv"), index=False)
                result.beats.to_csv(out / (stem + "_events.csv"), index=False)
                rows.append(
                    dict(
                        movie=source["name"],
                        method=method,
                        shape=list(loaded.frames.shape),
                        fps=loaded.fps,
                        executed=True,
                        repeatable=True,
                        qc=result.qc.__dict__,
                        summary=result.summary,
                        provenance=result.provenance,
                    )
                )
            except Exception as exc:
                rows.append(dict(movie=source["name"], method=method, executed=False, error=str(exc)))
    agreement = []
    for source in manifest["movies"]:
        selected = [r for r in rows if r["movie"] == source["name"] and r["executed"]]
        rates = {r["method"]: r["summary"]["mean_bpm"] for r in selected}
        finite = [v for v in rates.values() if np.isfinite(v)]
        span = max(finite) - min(finite) if finite else None
        agreement.append(
            dict(
                movie=source["name"],
                motion_event_rates=rates,
                span_per_min=span,
                flagged=span is not None and span > 10,
                interpretation="Descriptive method disagreement; no method is designated ground truth",
            )
        )
    report = dict(
        source=manifest,
        validation_scope="External input execution and exact repeatability only; no annotated beats/force/maturity ground truth",
        results=rows,
        method_agreement=agreement,
        disagreement_flag_threshold_per_min=10,
        executed=sum(r["executed"] for r in rows),
        total=len(rows),
    )
    (out / "results.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"executed": report["executed"], "total": len(rows)}))
    if report["executed"] != len(rows):
        raise SystemExit("External video execution failed; all errors retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("validation/cpu/public_sources.json"))
    parser.add_argument("--data", type=Path, default=Path("/tmp/myotrace-public-videos"))
    parser.add_argument("--out", type=Path, default=Path("validation/cpu/public"))
    args = parser.parse_args()
    run(args.manifest, args.data, args.out)
