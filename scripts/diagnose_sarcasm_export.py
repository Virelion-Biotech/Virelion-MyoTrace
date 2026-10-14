"""Replay exported traces without downloading movies or running supplied launchers.

The prominence sweep is exploratory sensitivity analysis, not parameter selection
or accuracy validation. No optical-flow method is treated as ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from myotrace.kinetics import analyze_trace, prepare_signal, summarize_beats


def run(root: Path) -> dict:
    mapping = json.loads((root / "ARCHIVE_PATHS.json").read_text())
    hashes = json.loads((root / "ARTIFACT_HASHES.json").read_text())
    for original, digest in hashes.items():
        actual = hashlib.sha256((root / mapping[original]).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Payload checksum mismatch: {original}")
    movies = []
    for name in ("10kPa", "20kPa", "30kPa"):
        directory = root / "results" / name
        original = json.loads((directory / "results.json").read_text())
        rows = []
        for result in original["results"]:
            method = result["method"]
            fps = result["fps"]
            trace = pd.read_csv(directory / f"{name}_{method}_trace.csv")
            exported = pd.read_csv(directory / f"{name}_{method}_events.csv")
            signal = trace.analysis_signal.to_numpy()
            offset = 0.0 if method == "signed_displacement" else 0.5
            beats = analyze_trace(signal, fps)
            times = np.array([b.peak_time_s + offset / fps for b in beats])
            np.testing.assert_allclose(times, exported.peak_time_s.to_numpy(), rtol=0, atol=1e-10)
            summary = summarize_beats(beats)
            np.testing.assert_allclose(summary["mean_bpm"], result["summary"]["mean_bpm"], rtol=0, atol=1e-8)
            filtered = prepare_signal(signal, fps)
            smoothed = gaussian_filter1d(filtered, sigma=0.04 * fps, mode="nearest")
            residual = filtered - smoothed
            noise_floor = float(5 * 1.4826 * np.median(np.abs(residual - np.median(residual))))
            span = float(np.percentile(smoothed, 95) - np.percentile(smoothed, 5))
            threshold = max(0.12 * span, noise_floor, np.finfo(float).eps)
            peaks, properties = find_peaks(smoothed, prominence=0, distance=max(1, int(np.floor(fps * 60 / 240))))
            candidates = [
                {"time_s": float((peak + offset) / fps), "prominence": float(prominence),
                 "passes_default_prominence": bool(prominence >= threshold)}
                for peak, prominence in zip(peaks, properties["prominences"])
            ]
            sensitivity = []
            for fraction in (0.06, 0.09, 0.12, 0.15, 0.18):
                selected = analyze_trace(signal, fps, prominence_fraction=fraction)
                sensitivity.append({"prominence_fraction": fraction, "events": len(selected),
                                    "mean_motion_event_rate_per_min": summarize_beats(selected)["mean_bpm"]})
            rows.append({"method": method, "export_replayed": True, "events": len(beats),
                         "mean_motion_event_rate_per_min": summary["mean_bpm"],
                         "event_times_s": times.tolist(), "default_prominence_threshold": threshold,
                         "span_floor": 0.12 * span, "residual_noise_floor": noise_floor,
                         "candidate_peaks": candidates, "prominence_sensitivity": sensitivity})
        # Peak spacing exceeds twice this tolerance, so nearest matches are unique.
        by_method = {row["method"]: row for row in rows}
        tolerance = 0.05 + 1 / original["results"][0]["fps"]
        comparisons = []
        base = np.array(by_method["farneback"]["event_times_s"])
        for method in ("lk", "ensemble"):
            other = np.array(by_method[method]["event_times_s"])
            distances = np.abs(base[:, None] - other[None, :])
            matched = distances <= tolerance
            assert np.all(matched.sum(axis=0) <= 1) and np.all(matched.sum(axis=1) <= 1)
            comparisons.append({"methods": ["farneback", method], "tolerance_s": tolerance,
                                "matched_events": int(matched.sum()),
                                "farneback_only_s": base[~matched.any(axis=1)].tolist(),
                                "other_only_s": other[~matched.any(axis=0)].tolist()})
        movies.append({"movie": name, "methods": rows, "event_matching": comparisons,
                       "original_method_agreement": original["method_agreement"]})
    return {"scope": "Export integrity, detector replay and exploratory prominence sensitivity only",
            "payload_hashes_verified": len(hashes), "traces_replayed": 12,
            "raw_movie_optical_flow_rerun": False, "ground_truth_available": False,
            "parameter_selection_performed": False, "movies": movies}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("validation/colab_sarcasm_2026_10_14"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"payload_hashes_verified": report["payload_hashes_verified"],
                      "traces_replayed": report["traces_replayed"]}))
