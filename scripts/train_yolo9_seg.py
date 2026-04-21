"""Train a YOLOv9 instance segmentation model on a YOLO-format dataset.

Designed to run both locally (CPU/GPU/MPS) and inside an HF Jobs container:

    # Local
    python scripts/train_yolo9_seg.py \\
        --data path/to/data.yaml \\
        --size t --epochs 50 --batch 16 --imgsz 640

    # HF Jobs (submit from your workstation with hf_hub jobs CLI)
    # See docs/agentic-features/blog/yolo9-instance-segmentation.md

A YOLO-seg data.yaml must include:
    path: <dataset root>
    train: <images-dir relative to path>
    val:   <images-dir relative to path>
    nc:    <num classes>
    names: [cls0, cls1, ...]

Labels: YOLO-polygon format. Each .txt line is either:
    class cx cy w h                               (detection only)
    class x1 y1 x2 y2 ...                         (polygon; >= 3 vertices)
All coordinates normalized to [0, 1].

If --push is set and HF_TOKEN is available (env var or ~/.cache/huggingface/token),
pushes the best checkpoint, a model card, and training config to an HF Hub repo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Path to data.yaml")
    p.add_argument("--size", default="t", choices=["t", "s", "m", "c"], help="YOLOv9 size variant")
    p.add_argument("--init-weights", default=None,
                   help="Init weights path (.pt). Default: auto-download detection weights "
                        "`LibreYOLO9{size}.pt` and initialize seg head from scratch.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640, help="Must be a multiple of 32")
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--optimizer", default="SGD", choices=["SGD", "Adam", "AdamW"])
    p.add_argument("--device", default="", help="'' auto, 'cpu', 'cuda', 'mps'")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="yolo9_seg")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", action="store_false", dest="amp")
    p.add_argument("--patience", type=int, default=50)

    # HF push
    p.add_argument("--push", action="store_true", help="Push artifacts to HF Hub on success")
    p.add_argument("--hf-repo", default=None,
                   help="Target HF repo id (e.g. 'ander2221/libreyolo-yolo9t-seg'). Required with --push.")
    p.add_argument("--hf-private", action="store_true")
    return p.parse_args()


def _maybe_push_to_hub(
    results: dict,
    repo_id: str,
    size: str,
    imgsz: int,
    epochs: int,
    batch: int,
    data_yaml: str,
    private: bool,
) -> str:
    """Push best checkpoint + metadata to an HF Hub model repo. Returns repo URL."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    best = Path(results["best_checkpoint"])
    if not best.exists():
        raise FileNotFoundError(f"Best checkpoint missing: {best}")

    # Push the checkpoint
    api.upload_file(
        path_or_fileobj=str(best),
        path_in_repo=best.name,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload best YOLOv9-seg checkpoint",
    )

    # Write and push a minimal model card
    card = _model_card(
        repo_id=repo_id,
        size=size,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        data_yaml=data_yaml,
        metrics={
            "mAP50": results.get("best_mAP50"),
            "mAP50-95": results.get("best_mAP50_95"),
            "best_epoch": results.get("best_epoch"),
            "final_loss": results.get("final_loss"),
        },
    )
    (Path(results["save_dir"]) / "README.md").write_text(card)
    api.upload_file(
        path_or_fileobj=str(Path(results["save_dir"]) / "README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload model card",
    )

    return f"https://huggingface.co/{repo_id}"


def _model_card(*, repo_id, size, imgsz, epochs, batch, data_yaml, metrics) -> str:
    nice_metrics = json.dumps({k: v for k, v in metrics.items() if v is not None}, indent=2)
    return f"""---
library_name: libreyolo
pipeline_tag: image-segmentation
tags:
  - libreyolo
  - yolov9
  - instance-segmentation
  - object-detection
license: mit
---

# {repo_id}

YOLOv9-{size} instance segmentation checkpoint trained with
[LibreYOLO](https://github.com/LibreYOLO/libreyolo).

## Training
- size: {size}
- imgsz: {imgsz}
- epochs: {epochs}
- batch: {batch}
- data: `{data_yaml}`

## Metrics
```json
{nice_metrics}
```

## Usage
```python
from libreyolo import LibreYOLO9
from huggingface_hub import hf_hub_download

ckpt = hf_hub_download(repo_id="{repo_id}", filename="best.pt")
model = LibreYOLO9(model_path=ckpt, size="{size}", segmentation=True)
result = model("some_image.jpg")
masks = result.masks.data  # (N, H, W) binary masks
```
"""


def main() -> int:
    args = _parse_args()

    # Lazy import so `--help` works without installing torch
    from libreyolo.models.yolo9.model import LibreYOLO9

    init = args.init_weights or f"LibreYOLO9{args.size}.pt"
    print(f"[train_yolo9_seg] Initializing seg model from {init}", flush=True)
    model = LibreYOLO9(
        model_path=init,
        size=args.size,
        segmentation=True,
        device=args.device or "auto",
    )

    print(f"[train_yolo9_seg] Starting training on {args.data}", flush=True)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        optimizer=args.optimizer,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=args.project,
        name=args.name,
        amp=args.amp,
        patience=args.patience,
    )

    print(f"[train_yolo9_seg] Done. Best mAP50-95={results.get('best_mAP50_95')} "
          f"best_epoch={results.get('best_epoch')}", flush=True)
    print(f"[train_yolo9_seg] best_checkpoint={results['best_checkpoint']}", flush=True)

    if args.push:
        if not args.hf_repo:
            print("[train_yolo9_seg] --push set but --hf-repo not provided", file=sys.stderr)
            return 2
        url = _maybe_push_to_hub(
            results=results,
            repo_id=args.hf_repo,
            size=args.size,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            data_yaml=args.data,
            private=args.hf_private,
        )
        print(f"[train_yolo9_seg] Pushed to {url}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
