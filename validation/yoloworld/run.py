"""Run open-vocabulary detection on diverse images, save annotated outputs.

Each image gets a curated prompt list; we record:
  - per-image detection counts and confidences
  - annotated JPGs side-by-side
  - a JSON summary with all detections for reproducibility

Run: python validation/yoloworld/run.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from libreyolo.models.yoloworld import LibreYOLOWorld
from libreyolo.models.yoloworld.weight_porting import port_state_dict


# Per-image curated prompts — what we'd reasonably expect to find in each.
IMAGE_PROMPTS = {
    "street_cats.jpg":  ["cat", "remote control", "couch", "blanket"],
    "kitchen.jpg":      ["chair", "dining table", "bottle", "cup", "wine glass", "bowl"],
    "snowboard.jpg":    ["person", "snowboard", "skis", "tree"],
    "surfer.jpg":       ["person", "surfboard", "wave", "ocean"],
    "skateboard.jpg":   ["person", "skateboard", "ground", "shadow"],
    "dining.jpg":       ["person", "wine glass", "dining table", "bowl", "fork", "knife", "cup"],
    "traffic.jpg":      ["car", "truck", "bus", "person", "traffic light", "stop sign"],
}

CHECKPOINT = "/Users/ander.alvarez/.cache/huggingface/hub/models--wondervictor--YOLO-World-V2.1/snapshots/c620164ee3979bf49b895c8a8e0f49aeaca89209/s_stage2-4466ab94.pth"


def annotate(img_path: Path, detections, prompts, out_path: Path) -> None:
    """Draw boxes + labels onto a copy of the image."""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    palette = [
        (255, 80, 80), (80, 255, 80), (80, 120, 255), (255, 200, 60),
        (200, 80, 255), (60, 220, 220), (240, 140, 60), (140, 60, 240),
    ]
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        cls_id = det["cls"]
        conf = det["conf"]
        color = palette[cls_id % len(palette)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{prompts[cls_id]}  {conf:.2f}"
        # Text background
        try:
            bbox = draw.textbbox((x1, max(0, y1 - 22)), label, font=font)
            draw.rectangle(bbox, fill=color)
        except Exception:
            pass
        draw.text((x1 + 2, max(0, y1 - 22)), label, fill=(0, 0, 0), font=font)
    img.save(out_path, quality=90)


def main() -> int:
    in_dir = Path("validation/yoloworld/inputs")
    out_dir = Path("validation/yoloworld/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building LibreYOLOWorld(size='s') ...")
    model = LibreYOLOWorld(size="s", imgsz=640, prompts=["object"], device="cpu")
    print("Porting V2.1-S weights ...")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    res = port_state_dict(ckpt, model.model, strict=False)
    print(f"  loaded {res['loaded']}, mismatches {res['shape_mismatches']}, "
          f"missing-after-port {res['missing_after_port']}")
    if res['shape_mismatches']:
        print("  shape mismatches:")
        for sm in res['sample_shape_mismatches']:
            print(f"    {sm}")
        return 1

    summary = {
        "model": "wondervictor/YOLO-World-V2.1 / s_stage2",
        "weight_load": {"loaded_keys": res["loaded"], "shape_mismatches": res["shape_mismatches"]},
        "results": [],
    }

    for img_name, prompts in IMAGE_PROMPTS.items():
        img_path = in_dir / img_name
        if not img_path.exists():
            print(f"SKIP {img_name} (missing)")
            continue

        model.set_prompts(prompts)
        with torch.no_grad():
            result = model(str(img_path), conf=0.05, iou=0.5, max_det=30)
        r = result[0] if isinstance(result, list) else result

        detections = []
        for cls, conf, xyxy in zip(
            r.boxes.cls.long().tolist(),
            r.boxes.conf.tolist(),
            r.boxes.xyxy.tolist(),
        ):
            detections.append({
                "cls": int(cls),
                "label": prompts[cls] if 0 <= cls < len(prompts) else f"cls_{cls}",
                "conf": round(float(conf), 4),
                "xyxy": [round(float(v), 1) for v in xyxy],
            })

        out_img = out_dir / f"{img_path.stem}.detected.jpg"
        annotate(img_path, detections, prompts, out_img)

        print(f"\n{img_name}  prompts={prompts}")
        print(f"  -> {len(detections)} detections")
        for d in detections[:8]:
            print(f"    {d['label']:<18}  conf={d['conf']}  xyxy={d['xyxy']}")
        if len(detections) > 8:
            print(f"    (+{len(detections)-8} more)")
        print(f"  annotated -> {out_img}")

        summary["results"].append({
            "image": img_name,
            "prompts": prompts,
            "num_detections": len(detections),
            "detections": detections,
            "annotated": str(out_img),
        })

    out_json = out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
