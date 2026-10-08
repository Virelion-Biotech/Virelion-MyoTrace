"""Deterministic analytical tests, including reported failures; no GPU or network."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import tifffile

from myotrace.flow import FlowConfig, optical_flow_trace
from myotrace.kinetics import analyze_trace, summarize_beats
from myotrace.pipeline import analyze_video
from myotrace.serialization import json_safe
from myotrace.synthetic import cardiac_motion
from myotrace.robust import robust_preprocess


def texture(seed, size=96):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(20, 235, (size, size), dtype=np.uint8), (3, 3), 0)


def shifted(a, dx, dy=0):
    return cv2.warpAffine(
        a, np.float32([[1, 0, dx], [0, 1, dy]]), (a.shape[1], a.shape[0]), borderMode=cv2.BORDER_REFLECT
    )


def validate(out: Path):
    cv2.setNumThreads(1)
    out.mkdir(parents=True, exist_ok=True)
    flow_rows = []
    for seed, dx, dtype, method in product(
        (19, 29, 41), (0.25, 0.5, 1.0, 2.0), ("uint8", "uint16", "float32"), ("farneback", "lk")
    ):
        a = texture(seed)
        frames = np.stack([shifted(a, i * dx) for i in range(8)])
        if dtype == "uint16":
            frames = frames.astype(np.uint16) * 257
        elif dtype == "float32":
            frames = frames.astype(np.float32) / 255
        estimate = optical_flow_trace(frames, FlowConfig(method=method))
        error = float(np.max(np.abs(estimate - dx)))
        flow_rows.append(
            dict(seed=seed, displacement_px=dx, dtype=dtype, method=method, max_error_px=error, passed=error <= 0.1)
        )
    signal_rows = []
    for seed, bpm, fps, noise, drift, robust in product(
        (7, 29, 41), (45, 60, 90, 150), (25, 50, 100), (0, 0.03, 0.15), (0, 0.3), (False, True)
    ):
        truth = cardiac_motion(duration_s=12, fps=fps, bpm=bpm, noise_sd=noise, drift=drift, seed=seed)
        beats = analyze_trace(robust_preprocess(truth.motion, fps) if robust else truth.motion, fps)
        summary = summarize_beats(beats)
        # Generator's maximum is at rise_fraction * period, not period onset.
        expected = truth.beat_times_s + 0.3 * 60 / bpm
        expected = expected[(expected >= 0.3) & (expected < 11.7)]
        detected = np.asarray([b.peak_time_s for b in beats])
        detected = detected[(detected >= 0.3) & (detected < 11.7)]
        distances = np.abs(expected[:, None] - detected[None, :])
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(distances)
        matched = int(np.sum(distances[rows, cols] <= 0.05 + 1 / fps))
        precision = matched / len(detected) if len(detected) else 0
        recall = matched / len(expected) if len(expected) else 0
        error = abs(summary["mean_bpm"] - bpm)
        signal_rows.append(
            dict(
                seed=seed,
                bpm=bpm,
                fps=fps,
                noise_sd=noise,
                drift=drift,
                robust=robust,
                detected=len(detected),
                expected=len(expected),
                precision=precision,
                recall=recall,
                absolute_error_bpm=error,
                passed=bool(error <= 3 and precision >= 0.9 and recall >= 0.9),
            )
        )
    # A sinusoidal displacement returns to its initial position once per second,
    # but unsigned speed has TWO peaks each second. Preserve this counterexample.
    fps = 40
    positions = 3 * np.sin(2 * np.pi * np.arange(320) / fps)
    frames = np.stack([shifted(texture(19), dx) for dx in positions])
    path = out / "sinusoidal_displacement.tif"
    tifffile.imwrite(path, frames, photometric="minisblack", metadata={"axes": "TYX"})
    video_rows = []
    for method in ("farneback", "lk", "ensemble"):
        result = analyze_video(path, fps_override=fps, flow_config=FlowConfig(method=method))
        result.trace.to_csv(out / f"{method}_trace.csv", index=False)
        result.beats.to_csv(out / f"{method}_events.csv", index=False)
        video_rows.append(
            dict(
                method=method,
                physical_cycle_bpm=60,
                observed_motion_event_bpm=result.summary["mean_bpm"],
                observed_motion_events=result.summary["n_beats"],
                summary=result.summary,
                provenance=result.provenance,
            )
        )
    report = dict(
        protocol={
            "flow_max_error_px": 0.1,
            "signal_max_error_bpm": 3,
            "signal_precision_recall_min": 0.9,
            "match_tolerance_s": ".05 + 1/fps",
            "limits": "Analytical synthetic tests; not biological/force/maturity validation",
        },
        environment={
            k: importlib.metadata.version(k)
            for k in ("numpy", "scipy", "pandas", "opencv-python-headless", "tifffile", "virelion-myotrace")
        },
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for root in ("myotrace", "fusion", "scripts")
            for p in sorted(Path(root).glob("*.py"))
        },
        flow_cases=flow_rows,
        trace_cases=signal_rows,
        video_cases=video_rows,
        totals={
            "flow_cases": len(flow_rows),
            "flow_failed": sum(not r["passed"] for r in flow_rows),
            "trace_cases": len(signal_rows),
            "trace_failed": sum(not r["passed"] for r in signal_rows),
        },
    )
    (out / "results.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    # TIFF is deterministic and reproducible; avoid committing bulky generated images.
    path.unlink()
    print(json.dumps(report["totals"]))
    if report["totals"]["flow_failed"]:
        raise SystemExit("Known-displacement optical flow failed")
    clean = [r for r in signal_rows if r["noise_sd"] == 0]
    if any(not r["passed"] for r in clean):
        raise SystemExit("Noise-free signal benchmark failed; report retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/myotrace-analytical-validation"))
    validate(parser.parse_args().out)
