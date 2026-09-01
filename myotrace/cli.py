from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import analyze_video


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="myotrace", description="Video-based cardiac contractility analysis")
    p.add_argument("video", type=Path, help="AVI/MP4/MOV or TIFF stack")
    p.add_argument("--sample-id", default=None)
    p.add_argument("--fps", type=float, default=None, help="Override acquisition frame rate")
    p.add_argument("--method", choices=["farneback", "lk"], default="farneback")
    p.add_argument("--out", type=Path, default=Path("myotrace-output"))
    p.add_argument("--allow-qc-fail", action="store_true", help="Process data even when QC flags it")
    return p


def main() -> int:
    args = build_parser().parse_args()
    result = analyze_video(
        args.video,
        sample_id=args.sample_id,
        fps_override=args.fps,
        reject_failed_qc=not args.allow_qc_fail,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    result.trace.to_csv(args.out / "motion_trace.csv", index=False)
    result.beats.to_csv(args.out / "beat_metrics.csv", index=False)
    (args.out / "qc.txt").write_text(
        f"usable={result.qc.usable}\nreasons={','.join(result.qc.reasons)}\n"
        f"fps={result.qc.fps}\nframes={result.qc.frame_count}\n",
        encoding="utf-8",
    )
    print(f"sample={result.sample_id}")
    print(f"beats={int(result.summary['n_beats'])}")
    print(f"mean_bpm={result.summary['mean_bpm']:.3f}")
    print(f"qc={'PASS' if result.qc.usable else 'FLAG'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
