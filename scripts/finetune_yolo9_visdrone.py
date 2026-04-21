"""Fine-tune a YOLOv9 detector on VisDrone aerial imagery.

End-to-end: downloads VisDrone (optional), converts its annotations to YOLO
format, fine-tunes a YOLOv9 model pretrained on COCO, and optionally pushes
the result to the Hugging Face Hub.

VisDrone has 10 useful classes (pedestrian, people, bicycle, car, van, truck,
tricycle, awning-tricycle, bus, motor) across ~6.5k train / 550 val images.
Aerial imagery is a genuinely different distribution from COCO, so fine-tuning
recovers a lot of accuracy vs zero-shot COCO detection.

Example (assuming VisDrone already downloaded locally):

    python scripts/finetune_yolo9_visdrone.py \\
        --visdrone-root /data/VisDrone \\
        --size s --epochs 50 --batch 16 --imgsz 640 \\
        --push --hf-repo ander2221/libreyolo-yolo9s-visdrone

Or with HF Hub download:

    python scripts/finetune_yolo9_visdrone.py \\
        --hf-dataset Voxel51/visdrone2019-det \\
        --size s --epochs 50 --batch 16 --imgsz 640 \\
        --push --hf-repo ander2221/libreyolo-yolo9s-visdrone

VisDrone annotation format (one .txt per image, comma-separated):
    bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncation,occlusion

We discard category 0 (ignored-regions) and 11 (others); remap 1-10 → 0-9.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Tuple


# VisDrone category ids 1..10 — skip 0 (ignored) and 11 (others)
VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]
VISDRONE_ID_TO_YOLO: dict[int, int] = {cid: cid - 1 for cid in range(1, 11)}


# ---------------------------------------------------------------------------
# VisDrone → YOLO annotation conversion
# ---------------------------------------------------------------------------


def visdrone_line_to_yolo(
    line: str, img_w: int, img_h: int
) -> str | None:
    """Convert one VisDrone annotation line to YOLO format.

    Returns a YOLO-format line `"cls cx cy w h"` (normalized, 6 decimals), or
    `None` if the detection should be skipped (ignored category, zero-size,
    or out-of-range class).
    """
    parts = [p.strip() for p in line.strip().split(",") if p.strip()]
    if len(parts) < 6:
        return None
    try:
        x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        cat = int(parts[5])
    except ValueError:
        return None
    if cat not in VISDRONE_ID_TO_YOLO:
        return None
    if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
        return None
    yolo_cls = VISDRONE_ID_TO_YOLO[cat]
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    bw = w / img_w
    bh = h / img_h
    # Clamp to [0, 1] (VisDrone bboxes occasionally poke over image edges)
    cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
    bw, bh = max(0.0, min(1.0, bw)), max(0.0, min(1.0, bh))
    if bw <= 0 or bh <= 0:
        return None
    return f"{yolo_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _image_size(path: Path) -> Tuple[int, int]:
    """Return (width, height) without loading the full image into memory."""
    from PIL import Image
    with Image.open(path) as im:
        return im.size  # (w, h)


def convert_visdrone_split(
    images_dir: Path, annotations_dir: Path, out_labels_dir: Path
) -> dict:
    """Convert a VisDrone split's annotations to YOLO format.

    Reads `{annotations_dir}/{stem}.txt` for each image, writes the same
    stem under `out_labels_dir` in YOLO format. Returns a summary dict.
    """
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in images_dir.iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    summary = {"images": 0, "labels_written": 0, "skipped_images": 0, "skipped_lines": 0}

    for img in images:
        ann = annotations_dir / (img.stem + ".txt")
        if not ann.exists():
            summary["skipped_images"] += 1
            continue
        w, h = _image_size(img)
        out_lines = []
        for line in ann.read_text().splitlines():
            y = visdrone_line_to_yolo(line, w, h)
            if y is None:
                summary["skipped_lines"] += 1
            else:
                out_lines.append(y)
        (out_labels_dir / (img.stem + ".txt")).write_text("\n".join(out_lines))
        summary["images"] += 1
        summary["labels_written"] += len(out_lines)

    return summary


def build_yolo_dataset(visdrone_root: Path, out_root: Path) -> Path:
    """Link VisDrone images + converted labels into a standard YOLO layout.

    Returns the path to the written `data.yaml`. `out_root` is populated with:

        out_root/
            images/train/
            images/val/
            labels/train/
            labels/val/
            data.yaml
    """
    out_root.mkdir(parents=True, exist_ok=True)
    for split_name, src_suffix in [("train", "DET-train"), ("val", "DET-val")]:
        src = visdrone_root / f"VisDrone2019-{src_suffix}"
        if not src.exists():
            raise FileNotFoundError(
                f"VisDrone split missing at {src}. Expected layout:\n"
                f"  {visdrone_root}/VisDrone2019-DET-train/{{images,annotations}}/\n"
                f"  {visdrone_root}/VisDrone2019-DET-val/{{images,annotations}}/"
            )
        images_src = src / "images"
        anns_src = src / "annotations"
        images_dst = out_root / "images" / split_name
        labels_dst = out_root / "labels" / split_name

        # Symlink images (fast, no copy)
        images_dst.parent.mkdir(parents=True, exist_ok=True)
        if images_dst.exists():
            if images_dst.is_symlink() or images_dst.is_file():
                images_dst.unlink()
            else:
                shutil.rmtree(images_dst)
        images_dst.symlink_to(images_src.resolve(), target_is_directory=True)

        summary = convert_visdrone_split(images_src, anns_src, labels_dst)
        print(f"[visdrone] {split_name}: {json.dumps(summary)}", flush=True)

    data = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(VISDRONE_CLASSES),
        "names": VISDRONE_CLASSES,
    }
    import yaml
    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(yaml.dump(data, sort_keys=False))
    return data_yaml


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--visdrone-root", type=Path,
                     help="Already-downloaded VisDrone dir containing "
                          "VisDrone2019-DET-train/ and VisDrone2019-DET-val/")
    src.add_argument("--hf-dataset", default=None,
                     help="HF dataset repo id to download (e.g. 'Voxel51/visdrone2019-det'). "
                          "Downloaded under --cache-dir.")

    p.add_argument("--cache-dir", type=Path, default=Path(".cache/visdrone"),
                   help="Where to extract the dataset + write YOLO-format annotations")

    p.add_argument("--size", default="s", choices=["t", "s", "m", "c"])
    p.add_argument("--init-weights", default=None,
                   help="Path to init weights. Default: auto-download "
                        "`LibreYOLO9{size}.pt` (COCO-pretrained).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr0", type=float, default=0.005)
    p.add_argument("--optimizer", default="SGD", choices=["SGD", "Adam", "AdamW"])
    p.add_argument("--device", default="", help="'', 'cpu', 'cuda', 'mps'")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="yolo9_visdrone")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", action="store_false", dest="amp")
    p.add_argument("--patience", type=int, default=20)

    p.add_argument("--push", action="store_true")
    p.add_argument("--hf-repo", default=None,
                   help="Target HF repo id (e.g. 'ander2221/libreyolo-yolo9s-visdrone')")
    p.add_argument("--hf-private", action="store_true")
    return p.parse_args()


def _download_visdrone_hf(repo_id: str, cache_dir: Path) -> Path:
    """Download VisDrone from an HF dataset repo into cache_dir and return its path."""
    from huggingface_hub import snapshot_download
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(cache_dir / repo_id.replace("/", "__")),
    )
    return Path(path)


def _push_to_hub(results, repo_id, args, data_yaml, private):
    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    best = Path(results["best_checkpoint"])
    api.upload_file(
        path_or_fileobj=str(best),
        path_in_repo=best.name,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload VisDrone-finetuned YOLOv9 checkpoint",
    )
    card = _model_card(repo_id, args, results, data_yaml)
    card_path = Path(results["save_dir"]) / "README.md"
    card_path.write_text(card)
    api.upload_file(
        path_or_fileobj=str(card_path), path_in_repo="README.md",
        repo_id=repo_id, repo_type="model",
        commit_message="Upload model card",
    )
    return f"https://huggingface.co/{repo_id}"


def _model_card(repo_id, args, results, data_yaml) -> str:
    metrics = {k: results.get(k) for k in ("best_mAP50", "best_mAP50_95", "best_epoch", "final_loss")}
    return f"""---
library_name: libreyolo
pipeline_tag: object-detection
tags:
  - libreyolo
  - yolov9
  - visdrone
  - aerial-imagery
  - object-detection
datasets:
  - VisDrone2019-DET
license: mit
---

# {repo_id}

YOLOv9-{args.size} fine-tuned on VisDrone2019-DET aerial imagery using
[LibreYOLO](https://github.com/LibreYOLO/libreyolo).

Classes: {', '.join(VISDRONE_CLASSES)}.

## Training
- size: {args.size}
- init weights: {args.init_weights or f'LibreYOLO9{args.size}.pt (COCO-pretrained)'}
- imgsz: {args.imgsz}, epochs: {args.epochs}, batch: {args.batch}
- optimizer: {args.optimizer} lr0={args.lr0}

## Metrics
```json
{json.dumps({k: v for k, v in metrics.items() if v is not None}, indent=2)}
```

## Usage
```python
from huggingface_hub import hf_hub_download
from libreyolo import LibreYOLO

ckpt = hf_hub_download(repo_id="{repo_id}", filename="best.pt")
model = LibreYOLO(ckpt)
result = model("drone_image.jpg")
for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
    print(box, {{
{chr(10).join(f'        {i}: "{n}",' for i, n in enumerate(VISDRONE_CLASSES))}
    }}[int(cls)], float(conf))
```
"""


def main() -> int:
    args = _parse_args()

    if args.hf_dataset:
        print(f"[finetune_visdrone] Downloading {args.hf_dataset} via HF Hub...", flush=True)
        visdrone_root = _download_visdrone_hf(args.hf_dataset, args.cache_dir)
    else:
        visdrone_root = args.visdrone_root

    print(f"[finetune_visdrone] VisDrone root: {visdrone_root}", flush=True)
    yolo_root = args.cache_dir / "yolo_format"
    data_yaml = build_yolo_dataset(visdrone_root, yolo_root)
    print(f"[finetune_visdrone] YOLO-format dataset ready at {yolo_root}", flush=True)
    print(f"[finetune_visdrone] data.yaml: {data_yaml}", flush=True)

    from libreyolo.models.yolo9.model import LibreYOLO9
    init = args.init_weights or f"LibreYOLO9{args.size}.pt"
    print(f"[finetune_visdrone] Fine-tuning {init} (size={args.size})", flush=True)
    model = LibreYOLO9(model_path=init, size=args.size, device=args.device or "auto")

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        lr0=args.lr0, optimizer=args.optimizer,
        device=args.device, workers=args.workers, seed=args.seed,
        project=args.project, name=args.name, amp=args.amp, patience=args.patience,
    )

    print(f"[finetune_visdrone] Done. Best mAP50-95={results.get('best_mAP50_95')} "
          f"best_epoch={results.get('best_epoch')}", flush=True)
    print(f"[finetune_visdrone] best_checkpoint={results['best_checkpoint']}", flush=True)

    if args.push:
        if not args.hf_repo:
            print("[finetune_visdrone] --push requires --hf-repo", file=sys.stderr)
            return 2
        url = _push_to_hub(results, args.hf_repo, args, data_yaml, args.hf_private)
        print(f"[finetune_visdrone] Pushed to {url}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
