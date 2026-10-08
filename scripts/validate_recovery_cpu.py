"""Paired legacy/new CPU accuracy checks with separate confirmation cases.

Development seeds/rates reproduce the published 0.4 failures. Confirmation
seeds/rates and pulse generator are excluded from algorithm selection.
"""

from __future__ import annotations

import argparse
from itertools import product
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from myotrace.synthetic import cardiac_motion
from myotrace.robust import robust_preprocess
from myotrace.kinetics import analyze_trace, summarize_beats
from myotrace.flow import FlowConfig, optical_flow_trace
from myotrace.displacement import signed_displacement_trace
from myotrace.serialization import json_safe


def trace_cases(split):
    if split == "development":
        params = product((7, 29, 41), (45, 60, 90, 150), (25, 50, 100), (0, 0.03, 0.15), (0, 0.3), (False, True))
    else:
        params = product((101, 211, 307), (55, 75, 115, 175), (30, 60, 120), (0, 0.05, 0.15), (0, 0.2), (False, True))
    rows = []
    for seed, bpm, fps, noise, drift, robust in params:
        period = 60 / bpm
        if split == "development":
            truth = cardiac_motion(duration_s=12, fps=fps, bpm=bpm, noise_sd=noise, drift=drift, seed=seed)
            signal = truth.motion
            expected = truth.beat_times_s + 0.3 * period
        else:
            # Independent asymmetric Gaussian pulse generator, not cardiac_motion.
            t = np.arange(int(12 * fps)) / fps
            expected = np.arange(0.4, 12, period)
            signal = np.zeros_like(t)
            for peak in expected:
                width = np.where(t < peak, 0.11 * period, 0.21 * period)
                signal += np.exp(-0.5 * ((t - peak) / width) ** 2)
            signal += np.random.default_rng(seed).normal(0, noise, t.size) + drift * (t - 6) / 12
        signal = robust_preprocess(signal, fps) if robust else signal
        expected = expected[(expected >= 0.3) & (expected < 11.7)]
        for detector in ("legacy", "noise_aware"):
            beats = analyze_trace(signal, fps, detector=detector)
            summary = summarize_beats(beats)
            detected = np.array([b.peak_time_s for b in beats])
            detected = detected[(detected >= 0.3) & (detected < 11.7)]
            delta = np.abs(expected[:, None] - detected[None, :])
            a, b = linear_sum_assignment(delta)
            matched = int(np.sum(delta[a, b] <= 0.05 + 1 / fps))
            precision = matched / max(1, len(detected))
            recall = matched / max(1, len(expected))
            error = abs(summary["mean_bpm"] - bpm)
            rows.append(
                dict(
                    split=split,
                    seed=seed,
                    bpm=bpm,
                    fps=fps,
                    noise_sd=noise,
                    drift=drift,
                    robust=robust,
                    detector=detector,
                    precision=precision,
                    recall=recall,
                    error_bpm=error,
                    expected=len(expected),
                    detected=len(detected),
                    passed=bool(error <= 3 and precision >= 0.9 and recall >= 0.9),
                )
            )
    return rows


def video_cases():
    rows = []
    for seed, bpm, fps, phase, kind in product(
        (101, 211), (55, 95, 145), (30, 60), (0, 0.37), ("translation", "radial_deformation")
    ):
        rng = np.random.default_rng(seed)
        a = cv2.GaussianBlur(rng.integers(20, 235, (80, 80), dtype=np.uint8), (3, 3), 0)
        t = np.arange(int(8 * fps)) / fps
        displacement = np.sin(2 * np.pi * bpm / 60 * t + phase)
        images = []
        for value in displacement:
            if kind == "translation":
                matrix = np.float32([[1, 0, 2 * value], [0, 1, value]])
            else:
                scale = 1 + 0.025 * value
                matrix = cv2.getRotationMatrix2D((40, 40), 0, scale)
            images.append(cv2.warpAffine(a, matrix, (80, 80), borderMode=cv2.BORDER_REFLECT))
        frames = np.stack(images)
        signed = signed_displacement_trace(frames)
        summary = summarize_beats(analyze_trace(signed.signal, fps))
        unsigned = optical_flow_trace(frames, FlowConfig())
        unsigned_summary = summarize_beats(analyze_trace(unsigned, fps))
        correlation = abs(float(np.corrcoef(signed.signal, displacement)[0, 1]))
        error = abs(summary["mean_bpm"] - bpm)
        rows.append(
            dict(
                seed=seed,
                bpm=bpm,
                fps=fps,
                phase=phase,
                kind=kind,
                signed_cycle_rate=summary["mean_bpm"],
                unsigned_motion_event_rate=unsigned_summary["mean_bpm"],
                absolute_waveform_correlation=correlation,
                explained_variance=signed.explained_variance_fraction,
                error_bpm=error,
                passed=bool(error <= 3 and correlation >= 0.95),
            )
        )
    return rows


def validate(out):
    cv2.setNumThreads(1)
    out.mkdir(parents=True, exist_ok=True)
    traces = trace_cases("development") + trace_cases("confirmation")
    videos = video_cases()
    totals = {}
    for split, detector in product(("development", "confirmation"), ("legacy", "noise_aware")):
        selected = [r for r in traces if r["split"] == split and r["detector"] == detector]
        totals[split + "_" + detector] = {"cases": len(selected), "failed": sum(not r["passed"] for r in selected)}
    totals["signed_video"] = {"cases": len(videos), "failed": sum(not r["passed"] for r in videos)}
    report = dict(
        protocol={
            "sigma_s": 0.04,
            "noise_multiplier": 5,
            "noise_estimator": "1.4826*MAD(filtered - smoothed)",
            "rate_error_max": 3,
            "precision_recall_min": 0.9,
            "match_tolerance_s": ".05+1/fps",
            "video_waveform_abs_correlation_min": 0.95,
            "confirmation_selection": "Frozen after development; independent seeds/rates/fps/Gaussian pulse generator",
            "scope": "Analytical correctness only; unverified biological contraction polarity, beat labels, force or maturity",
        },
        totals=totals,
        traces=traces,
        videos=videos,
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("myotrace", "fusion", "scripts")
            for p in sorted(Path(folder).glob("*.py"))
        },
    )
    (out / "results.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps(totals))
    if totals["signed_video"]["failed"]:
        raise SystemExit("Signed-video analytical validation failed")
    for split in ("development", "confirmation"):
        if totals[split + "_noise_aware"]["failed"] >= totals[split + "_legacy"]["failed"]:
            raise SystemExit("Noise detector did not improve the paired benchmark")
    if any(not r["passed"] for r in traces if r["noise_sd"] == 0 and r["detector"] == "noise_aware"):
        raise SystemExit("Noise-free detector regression")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("validation/cpu/recovery"))
    validate(parser.parse_args().out)
