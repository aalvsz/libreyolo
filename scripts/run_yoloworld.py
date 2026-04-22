"""Run LibreYOLOWorld open-vocabulary detection on an image.

Usage:
    python scripts/run_yoloworld.py \\
        --image photo.jpg \\
        --prompts "person" "dog" "traffic cone"

WARNING (MVP scope): weights are random-init — outputs are currently noise.
See docs/agentic-features/blog/yolo-world-integration.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--prompts", required=True, nargs="+",
                   help="List of class-name strings")
    p.add_argument("--weights", default=None,
                   help="Optional path to a LibreYOLOWorld checkpoint. Default: random-init.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="", help="'', 'cpu', 'cuda', 'mps'")
    p.add_argument("--conf", type=float, default=0.001,
                   help="Confidence threshold (MVP weights are random, keep this low)")
    p.add_argument("--max-det", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.image.exists():
        print(f"[run_yoloworld] image not found: {args.image}", file=sys.stderr)
        return 2

    from libreyolo.models.yoloworld import LibreYOLOWorld

    print(f"[run_yoloworld] prompts: {args.prompts}", flush=True)
    model = LibreYOLOWorld(
        model_path=args.weights,
        prompts=args.prompts,
        imgsz=args.imgsz,
        device=args.device or "auto",
    )

    result = model(str(args.image), conf=args.conf, max_det=args.max_det)
    res = result[0] if isinstance(result, list) else result
    print(f"[run_yoloworld] detections: {len(res.boxes)}", flush=True)
    if len(res.boxes) > 0:
        cls_ids = res.boxes.cls.long().tolist()
        conf = res.boxes.conf.tolist()
        xyxy = res.boxes.xyxy.tolist()
        for i, (c, s, box) in enumerate(zip(cls_ids, conf, xyxy)):
            label = args.prompts[c] if 0 <= c < len(args.prompts) else f"cls_{c}"
            print(f"  {i:>2}. {label!r:<30}  conf={s:.3f}  xyxy=[{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")

    if args.weights is None:
        print("[run_yoloworld] NOTE: no weights provided → outputs are random-init noise. "
              "See docs/agentic-features/blog/yolo-world-integration.md for scope.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
