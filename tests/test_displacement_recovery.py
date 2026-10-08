import json
import sys

import cv2
import numpy as np
import pytest
import tifffile

from myotrace import signed_displacement_trace, analyze_video
from myotrace.flow import FlowConfig
from myotrace.kinetics import analyze_trace, summarize_beats
from myotrace.cli import main


def frames(phase=0, deform=False):
    a = cv2.GaussianBlur(np.random.default_rng(91).integers(20, 235, (64, 64), dtype=np.uint8), (3, 3), 0)
    x = np.sin(2 * np.pi * np.arange(240) / 40 + phase)
    out = []
    for d in x:
        matrix = (
            cv2.getRotationMatrix2D((32, 32), 0, 1 + 0.04 * d) if deform else np.float32([[1, 0, 2 * d], [0, 1, d]])
        )
        out.append(cv2.warpAffine(a, matrix, (64, 64), borderMode=cv2.BORDER_REFLECT))
    return np.stack(out), x


@pytest.mark.parametrize("phase", [0, 0.5, 1.5])
@pytest.mark.parametrize("deform", [False, True])
def test_signed_displacement_preserves_one_cycle(phase, deform):
    images, truth = frames(phase, deform)
    result = signed_displacement_trace(images)
    assert abs(np.corrcoef(result.signal, truth)[0, 1]) > 0.98
    assert result.explained_variance_fraction > 0.95
    assert summarize_beats(analyze_trace(result.signal, 40))["mean_bpm"] == pytest.approx(60, abs=2)
    np.testing.assert_allclose(signed_displacement_trace(images, reference_frame=80).signal, result.signal, atol=0.03)


def test_signed_static_and_parameter_validation():
    images, _ = frames()
    static = np.stack([images[0]] * 30)
    np.testing.assert_array_equal(signed_displacement_trace(static).signal, np.zeros(30))
    for kwargs in [
        {"grid_size": 3},
        {"reference_frame": -1},
        {"reference_frame": 240},
        {"config": FlowConfig(method="lk")},
        {"mask": np.zeros((64, 64), dtype=bool)},
    ]:
        with pytest.raises(ValueError):
            signed_displacement_trace(images, **kwargs)


def test_signed_pipeline_cli_and_timestamps(tmp_path, monkeypatch, capsys):
    images, _ = frames(0.5)
    path = tmp_path / "cycles.tif"
    tifffile.imwrite(path, images, photometric="minisblack", metadata={"axes": "TYX"})
    result = analyze_video(path, fps_override=40, signal_mode="signed_displacement")
    assert len(result.trace) == 240
    assert result.trace.timestamp_s.iloc[0] == 0
    assert result.summary["mean_bpm"] == pytest.approx(60, abs=2)
    assert result.summary["measurement_status"] == "displacement_cycles_not_verified_cardiac_beats"
    assert result.provenance["parameters"]["timestamp_convention"] == "frame_timestamp"
    monkeypatch.setattr(
        sys,
        "argv",
        ["myotrace", str(path), "--fps", "40", "--signal-mode", "signed_displacement", "--out", str(tmp_path / "out")],
    )
    assert main() == 0
    assert "displacement_cycles_not_verified" in capsys.readouterr().out
    assert json.loads((tmp_path / "out/summary.json").read_text())["detector"] == "noise_aware"
    with pytest.raises(ValueError):
        analyze_video(path, fps_override=40, signal_mode="bad")
    with pytest.raises(ValueError):
        analyze_trace(np.ones(100), 40, detector="bad")


def test_noise_detector_rejects_reproduced_spurious_peaks():
    from myotrace.synthetic import cardiac_motion

    truth = cardiac_motion(duration_s=12, fps=25, bpm=45, noise_sd=0.15, seed=7)
    legacy = analyze_trace(truth.motion, 25, detector="legacy")
    repaired = analyze_trace(truth.motion, 25)
    assert len(repaired) < len(legacy)
    assert abs(summarize_beats(repaired)["mean_bpm"] - 45) < abs(summarize_beats(legacy)["mean_bpm"] - 45)


def test_multiple_independent_motion_modes_are_flagged_and_mask_can_isolate_one():
    rng = np.random.default_rng(48)
    a = cv2.GaussianBlur(rng.integers(20, 235, (96, 96), dtype=np.uint8), (3, 3), 0)
    t = np.arange(240) / 40
    images = []
    for time in t:
        left = cv2.warpAffine(
            a[:, :48],
            np.float32([[1, 0, 2 * np.sin(2 * np.pi * time)], [0, 1, 0]]),
            (48, 96),
            borderMode=cv2.BORDER_REFLECT,
        )
        right = cv2.warpAffine(
            a[:, 48:],
            np.float32([[1, 0, 0], [0, 1, 2 * np.sin(2 * np.pi * 1.414 * time)]]),
            (48, 96),
            borderMode=cv2.BORDER_REFLECT,
        )
        images.append(np.hstack([left, right]))
    images = np.stack(images)
    assert "multiple_displacement_modes" in signed_displacement_trace(images).flags
    mask = np.zeros((96, 96), dtype=bool)
    mask[:, :40] = True
    isolated = signed_displacement_trace(images, mask=mask)
    assert isolated.explained_variance_fraction > 0.95
    assert summarize_beats(analyze_trace(isolated.signal, 40))["mean_bpm"] == pytest.approx(60, abs=2)
