"""Run detection+tracking on a video file with LibreYOLO's ByteTrack pipeline.

Usage:
    python scripts/track_yolo_video.py \\
        --weights LibreYOLO9t.pt \\
        --video path/to/input.mp4 \\
        --output runs/track/out.mp4 \\
        --csv runs/track/tracks.csv \\
        --conf 0.25 --iou 0.45

Writes two artifacts:
  - An annotated video with boxes + track IDs drawn per frame.
  - A CSV of per-frame tracks:
        frame, track_id, class, conf, x1, y1, x2, y2
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True,
                   help="Path or HF filename of detection weights (e.g. LibreYOLO9t.pt)")
    p.add_argument("--video", required=True, type=Path, help="Input video path")
    p.add_argument("--output", type=Path, default=None,
                   help="Annotated output video. Default: runs/track/<stem>.mp4")
    p.add_argument("--csv", type=Path, default=None,
                   help="Per-frame track CSV. Default: runs/track/<stem>.csv")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Tracker's first-association threshold")
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--classes", type=int, nargs="*", default=None)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--vid-stride", type=int, default=1)
    p.add_argument("--device", default="", help="'', 'cpu', 'cuda', 'mps'")
    # Tracker config knobs (forwarded to TrackConfig)
    p.add_argument("--track-buffer", type=int, default=30)
    p.add_argument("--frame-rate", type=int, default=30)
    p.add_argument("--match-thresh", type=float, default=0.8)
    p.add_argument("--min-consecutive-frames", type=int, default=1)
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    from libreyolo import LibreYOLO

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    out_video = args.output or Path("runs/track") / f"{args.video.stem}.mp4"
    out_csv = args.csv or Path("runs/track") / f"{args.video.stem}.csv"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"[track] Loading weights: {args.weights}", flush=True)
    model = LibreYOLO(args.weights, device=args.device or "auto")

    print(f"[track] Tracking {args.video}", flush=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "class", "conf", "x1", "y1", "x2", "y2"])

        frame_idx = 0
        total_detections = 0
        for result in model.track(
            source=str(args.video),
            track_conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            classes=args.classes,
            max_det=args.max_det,
            vid_stride=args.vid_stride,
            save=True,
            output_path=str(out_video),
            track_buffer=args.track_buffer,
            frame_rate=args.frame_rate,
            match_thresh=args.match_thresh,
            minimum_consecutive_frames=args.min_consecutive_frames,
        ):
            frame_idx += 1
            if result.track_id is None or len(result.boxes) == 0:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()
            tid = result.track_id.cpu().numpy()
            for (x1, y1, x2, y2), c, s, t in zip(xyxy, cls, conf, tid):
                writer.writerow([frame_idx, int(t), int(c), f"{float(s):.4f}",
                                 f"{float(x1):.2f}", f"{float(y1):.2f}",
                                 f"{float(x2):.2f}", f"{float(y2):.2f}"])
                total_detections += 1

    print(f"[track] Wrote {frame_idx} frames, {total_detections} tracked detections", flush=True)
    print(f"[track] Video: {out_video}", flush=True)
    print(f"[track] CSV:   {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
