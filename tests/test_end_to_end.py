import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
import pytest
import tifffile

from myotrace import __version__, analyze_video
from myotrace.flow import FlowConfig, optical_flow_trace, frame_timestamps
from myotrace.io import load_video, load_tiff_stack, normalize_frames, validate_frame_stack
from myotrace.kinetics import analyze_trace, summarize_beats, BeatMetrics
from myotrace.roi import ROI, crop_frames
from myotrace.motion_correction import correct_global_translation
from myotrace.flow_features import summarize_flow_field
from myotrace.schema import validate_trace_table


def texture(seed=19):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(20, 235, (96, 96), dtype=np.uint8), (3, 3), 0)


def translate(a, dx):
    return cv2.warpAffine(
        a, np.float32([[1, 0, dx], [0, 1, 0]]), (a.shape[1], a.shape[0]), borderMode=cv2.BORDER_REFLECT
    )


@pytest.mark.parametrize("method", ["farneback", "lk"])
@pytest.mark.parametrize("dx", [0.5, 1.0, 2.0])
def test_known_translation(method, dx):
    a = texture()
    frames = np.stack([translate(a, i * dx) for i in range(6)])
    trace = optical_flow_trace(frames, FlowConfig(method=method))
    np.testing.assert_allclose(trace, dx, atol=0.08)


def test_ensemble_does_not_amplify_constant_speed_jitter():
    frames = np.stack([translate(texture(), i) for i in range(6)])
    np.testing.assert_array_equal(optical_flow_trace(frames, FlowConfig(method="ensemble")), np.zeros(5))


@pytest.mark.parametrize(
    "args",
    [
        {"motion_percentile": -1},
        {"motion_percentile": np.nan},
        {"ensemble_weight_lk": 2},
        {"levels": 0},
        {"iterations": 1.5},
        {"poly_n": 3},
        {"poly_sigma": np.inf},
        {"pyr_scale": 1},
        {"method": "bad"},
    ],
)
def test_invalid_flow_config(args):
    with pytest.raises(ValueError):
        FlowConfig(**args)


def recording(tmp_path, static=False):
    fps = 40
    t = np.arange(320) / fps
    dx = np.zeros_like(t) if static else 3 * np.sin(2 * np.pi * t)
    frames = np.stack([translate(texture(), value) for value in dx])
    path = tmp_path / "motion.tif"
    tifffile.imwrite(path, frames, photometric="minisblack", metadata={"axes": "TYX"})
    return path, frames, fps


@pytest.mark.parametrize("method", ["farneback", "lk", "ensemble"])
@pytest.mark.parametrize("robust", [True, False])
def test_tiff_pipeline_and_provenance(tmp_path, method, robust):
    path, frames, fps = recording(tmp_path)
    result = analyze_video(
        path, fps_override=fps, flow_config=FlowConfig(method=method), robust=robust, roi=ROI(4, 4, 80, 80)
    )
    assert len(result.trace) == len(frames) - 1
    assert result.trace.timestamp_s.iloc[0] == 0.5 / fps
    assert result.provenance["version"] == __version__
    assert result.provenance["parameters"]["flow"]["method"] == method
    assert result.provenance["parameters"]["fps_source"] == "override"
    assert result.summary["measurement_status"] == "motion_events_not_verified_cardiac_beats"
    validate_trace_table(result.trace)


def test_missing_fps_and_invalid_override(tmp_path):
    path, _, _ = recording(tmp_path)
    assert load_tiff_stack(path).fps is None
    with pytest.raises(ValueError, match="frame rate"):
        analyze_video(path)
    for value in [0, -1, np.nan, np.inf]:
        with pytest.raises(ValueError, match="fps"):
            analyze_video(path, fps_override=value)


def test_imagej_fps_and_invalid_axes(tmp_path):
    path = tmp_path / "imagej.tif"
    frames = np.stack([texture()] * 4)
    tifffile.imwrite(path, frames, imagej=True, metadata={"axes": "TYX", "finterval": 0.02, "tunit": "sec"})
    assert load_tiff_stack(path).fps == 50
    tifffile.imwrite(path, frames, metadata={"axes": "ZYX"}, photometric="minisblack")
    with pytest.raises(ValueError, match="axes"):
        load_tiff_stack(path)


def test_video_loader_and_pipeline(tmp_path):
    _, frames, fps = recording(tmp_path)
    path = tmp_path / "motion.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (96, 96))
    assert writer.isOpened()
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    loaded = load_video(path, max_frames=12)
    assert loaded.frames.shape == (12, 96, 96)
    assert loaded.fps == fps
    assert load_video(path, max_frames=3, gray=False).frames.shape[-1] == 3
    assert len(analyze_video(path).trace) == 319
    with pytest.raises(ValueError):
        load_video(path, max_frames=0)
    with pytest.raises(FileNotFoundError):
        load_video(tmp_path / "missing.avi")
    bad = tmp_path / "bad.avi"
    bad.write_bytes(b"bad")
    with pytest.raises(ValueError):
        load_video(bad)


def test_static_cli_and_hearttwin_strict_json(tmp_path):
    path, _, fps = recording(tmp_path, static=True)
    out = tmp_path / "output"
    run = subprocess.run(
        [sys.executable, "-m", "myotrace.cli", str(path), "--fps", str(fps), "--allow-qc-fail", "--out", str(out)],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    summary = json.loads((out / "summary.json").read_text(), parse_constant=lambda v: pytest.fail(v))
    assert summary["n_beats"] == 0
    assert summary["mean_bpm"] is None
    payload = {
        "entity_id": "cell1",
        "observations": [
            {"modality": "mechanical", "values": {"input_path": str(path), "fps": fps, "allow_qc_fail": True}}
        ],
    }
    env = dict(os.environ, HEARTTWIN_PAYLOAD_STDIN="1")
    run = subprocess.run(
        [sys.executable, "-m", "myotrace.hearttwin_adapter"],
        input=json.dumps(payload),
        env=env,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["sample_id"] == "cell1"
    bad = subprocess.run(
        [sys.executable, "-m", "myotrace.hearttwin_adapter"], input="{}", env=env, capture_output=True, text=True
    )
    assert bad.returncode == 1
    with pytest.raises(ValueError, match="QC"):
        analyze_video(path, fps_override=fps, reject_failed_qc=True)


def test_translation_correction(tmp_path):
    frames = np.stack([translate(texture(), 0.5 * i) for i in range(15)])
    corrected, report = correct_global_translation(frames)
    assert report.failed_fraction == 0
    assert report.max_translation_px == pytest.approx(7, abs=0.1)
    assert np.mean(optical_flow_trace(corrected)) < 0.08
    _, failed = correct_global_translation(np.zeros((5, 32, 32), dtype=np.uint8))
    assert failed.failed_fraction == 1
    path, _, fps = recording(tmp_path)
    r = analyze_video(path, fps_override=fps, correct_motion=True)
    assert r.motion_correction is not None


def test_frame_roi_mask_validation():
    frames = np.stack([texture()] * 4)
    mask = np.ones((96, 96), dtype=bool)
    mask[:10] = False
    assert crop_frames(frames, mask=mask).shape == frames.shape
    for roi in [ROI(-1, 0, 10, 10), ROI(90, 0, 10, 10), ROI(0.1, 0, 10, 10)]:
        with pytest.raises(ValueError):
            crop_frames(frames, roi=roi)
    with pytest.raises(ValueError):
        crop_frames(frames, mask=np.ones((4, 4)))
    for x in [np.zeros((2, 32, 32)), np.zeros((3, 2, 2)), np.full((3, 32, 32), np.nan)]:
        with pytest.raises(ValueError):
            validate_frame_stack(x)
    with pytest.raises(ValueError):
        normalize_frames(frames, percentile_low=99, percentile_high=1)
    with pytest.raises(ValueError):
        optical_flow_trace(np.zeros_like(frames), FlowConfig(method="lk"))
    with pytest.raises(ValueError):
        frame_timestamps(3, np.nan)


def test_no_beats_and_single_beat_summary():
    assert summarize_beats([])["n_beats"] == 0
    beat = BeatMetrics(0, 1, np.nan, np.nan, 1, 0.2, 0.3, 0.2, 0, 0.5, 0.3, 0.5)
    summary = summarize_beats([beat])
    assert np.isnan(summary["mean_bpm"])
    assert np.isnan(summary["sd_bpm"])
    with pytest.raises(ValueError):
        analyze_trace(np.ones(100), 30, min_bpm=250, max_bpm=240)


def test_spatial_features_and_multisample_schema():
    field = np.zeros((10, 10, 2))
    assert summarize_flow_field(field).motion_area_fraction == 0
    field[..., 0] = 2
    feature = summarize_flow_field(field, threshold=1)
    assert feature.mean_speed == 2
    assert feature.directional_coherence == 1
    with pytest.raises(ValueError):
        summarize_flow_field(field * np.nan)
    table = pd.DataFrame(
        {
            "sample_id": ["a", "a", "b", "b"],
            "timestamp_s": [0, 1, 0, 1],
            "motion_index": [1] * 4,
            "modality": ["mechanical"] * 4,
        }
    )
    validate_trace_table(table)
    table.loc[1, "timestamp_s"] = 0
    with pytest.raises(ValueError):
        validate_trace_table(table)


def test_cli_and_adapter_entrypoints_in_process(tmp_path, monkeypatch, capsys):
    from myotrace.cli import main
    from myotrace.hearttwin_adapter import main as adapter_main

    path, _, fps = recording(tmp_path, static=True)
    out = tmp_path / "entrypoint"
    monkeypatch.setattr(
        sys,
        "argv",
        ["myotrace", str(path), "--fps", str(fps), "--allow-qc-fail", "--roi", "4", "4", "80", "80", "--out", str(out)],
    )
    assert main() == 0
    assert "beats=0" in capsys.readouterr().out
    monkeypatch.delenv("HEARTTWIN_PAYLOAD", raising=False)
    monkeypatch.delenv("HEARTTWIN_PAYLOAD_STDIN", raising=False)
    assert adapter_main() == 1
    monkeypatch.setenv("HEARTTWIN_PAYLOAD", "{}")
    assert adapter_main() == 1
    payload = {
        "entity_id": "a",
        "observations": [
            {"modality": "mechanical", "values": {"input_path": str(path), "fps": fps, "allow_qc_fail": True}}
        ],
    }
    monkeypatch.setenv("HEARTTWIN_PAYLOAD", json.dumps(payload))
    assert adapter_main() == 0
    assert json.loads(capsys.readouterr().out)["summary"]["n_beats"] == 0


def test_ome_and_imagej_slice_time_metadata(tmp_path):
    path = tmp_path / "ome.ome.tif"
    frames = np.stack([texture()] * 4)
    tifffile.imwrite(
        path,
        frames,
        ome=True,
        metadata={"axes": "TYX", "TimeIncrement": 20, "TimeIncrementUnit": "ms"},
        photometric="minisblack",
    )
    assert load_tiff_stack(path).fps == 50
    path = tmp_path / "ij.tif"
    tifffile.imwrite(path, frames, imagej=True, metadata={"axes": "ZYX", "finterval": 0.02})
    assert load_tiff_stack(path).fps == 50


def test_mask_selects_moving_region_without_background_percentile_bias():
    a = texture()
    frames = np.zeros((6, 160, 160), dtype=np.uint8)
    for i in range(6):
        frames[i, 32:128, 32:128] = translate(a, i)
    mask = np.zeros((160, 160), dtype=bool)
    mask[40:120, 40:120] = True
    for method in ("farneback", "lk"):
        measured = optical_flow_trace(frames, FlowConfig(method=method), mask=mask)
        np.testing.assert_allclose(measured, 1, atol=0.08)
    with pytest.raises(ValueError):
        optical_flow_trace(frames, mask=np.zeros((160, 160), dtype=bool))
