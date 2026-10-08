"""Smoke-test the installed wheel from outside the checkout."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tifffile

import myotrace
from fusion.model import FeatureReference, FusionConfig, calculate_index

assert myotrace.__version__ == "0.5.0"
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    a = np.random.default_rng(17).integers(20, 235, (64, 64), dtype=np.uint8)
    frames = np.stack(
        [
            cv2.warpAffine(
                a,
                np.float32([[1, 0, 2 * np.sin(2 * np.pi * i / 40)], [0, 1, 0]]),
                (64, 64),
                borderMode=cv2.BORDER_REFLECT,
            )
            for i in range(240)
        ]
    )
    path = root / "input.tif"
    tifffile.imwrite(path, frames, photometric="minisblack", metadata={"axes": "TYX"})
    result = myotrace.analyze_video(path, fps_override=40)
    assert len(result.trace) == 239
    signed = myotrace.analyze_video(path, fps_override=40, signal_mode="signed_displacement")
    assert len(signed.trace) == 240
    assert abs(signed.summary["mean_bpm"] - 60) < 3
    assert signed.summary["displacement_explained_variance"] > 0.95
    subprocess.run(
        [sys.executable, "-m", "myotrace.cli", str(path), "--fps", "40", "--out", str(root / "output")], check=True
    )
    summary = json.loads(
        (root / "output/summary.json").read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v))
    )
    assert summary["n_beats"] > 0
    assert (
        calculate_index(
            "a", {"mechanical:x": 1}, FusionConfig({"mechanical:x": FeatureReference(0, 1)}, {"mechanical": 1})
        ).composite_score
        == 100
    )
print("Installed wheel CLI, TIFF pipeline, strict JSON and fusion passed")
