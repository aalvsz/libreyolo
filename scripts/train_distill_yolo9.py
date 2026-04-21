"""Train a YOLOv9 student with MGD/CWD distillation from a larger YOLOv9 teacher.

Designed to run locally or inside an HF Jobs container. Example:

    # Local: distill YOLOv9-c -> YOLOv9-t on COCO
    python scripts/train_distill_yolo9.py \\
        --data coco/data.yaml \\
        --teacher weights/LibreYOLO9c.pt \\
        --student-size t \\
        --init-student weights/LibreYOLO9t.pt \\
        --epochs 100 --batch 16 --imgsz 640 \\
        --loss-type mgd --loss-weight 0.5

Both teacher and student checkpoints must be loadable by `LibreYOLO(path)`
(the factory auto-detects family + size from the state dict).

See the blog at docs/agentic-features/blog/yolo9-distillation.md for
HF Jobs submission and expected compute cost.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    p.add_argument("--teacher", required=True,
                   help="Path to teacher checkpoint (.pt). Must be loadable by LibreYOLO(...)")
    p.add_argument("--student-size", required=True, choices=["t", "s", "m", "c"],
                   help="YOLOv9 student size")
    p.add_argument("--init-student", default=None,
                   help="Optional student init weights path. Default: auto-download "
                        "`LibreYOLO9{student_size}.pt`.")
    p.add_argument("--loss-type", default="mgd", choices=["mgd", "cwd"])
    p.add_argument("--loss-weight", type=float, default=0.5)
    p.add_argument("--mask-ratio", type=float, default=0.65, help="MGD mask ratio")
    p.add_argument("--tau", type=float, default=1.0, help="CWD temperature")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--optimizer", default="SGD", choices=["SGD", "Adam", "AdamW"])
    p.add_argument("--device", default="", help="'', 'cpu', 'cuda', 'mps'")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="yolo9_distill")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", action="store_false", dest="amp")
    p.add_argument("--patience", type=int, default=50)

    p.add_argument("--push", action="store_true")
    p.add_argument("--hf-repo", default=None,
                   help="Target HF repo id (e.g. 'ander2221/libreyolo-yolo9t-distilled'). Required with --push.")
    p.add_argument("--hf-private", action="store_true")
    return p.parse_args()


def _push_to_hub(results, repo_id, args, private):
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    best = Path(results["best_checkpoint"])
    if not best.exists():
        raise FileNotFoundError(best)

    api.upload_file(
        path_or_fileobj=str(best),
        path_in_repo=best.name,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload distilled YOLOv9 student best checkpoint",
    )

    card = _model_card(repo_id, args, results)
    (Path(results["save_dir"]) / "README.md").write_text(card)
    api.upload_file(
        path_or_fileobj=str(Path(results["save_dir"]) / "README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload model card",
    )
    return f"https://huggingface.co/{repo_id}"


def _model_card(repo_id, args, results) -> str:
    metrics = {
        "best_mAP50": results.get("best_mAP50"),
        "best_mAP50_95": results.get("best_mAP50_95"),
        "best_epoch": results.get("best_epoch"),
        "final_loss": results.get("final_loss"),
    }
    return f"""---
library_name: libreyolo
pipeline_tag: object-detection
tags:
  - libreyolo
  - yolov9
  - knowledge-distillation
  - {args.loss_type}
license: mit
---

# {repo_id}

YOLOv9-{args.student_size} student distilled from `{args.teacher}` using
{args.loss_type.upper()} loss (weight={args.loss_weight}) in
[LibreYOLO](https://github.com/LibreYOLO/libreyolo).

## Training
- student size: {args.student_size}
- teacher: `{Path(args.teacher).name}`
- loss type: {args.loss_type}, weight: {args.loss_weight}
- mask_ratio (MGD): {args.mask_ratio}
- tau (CWD): {args.tau}
- imgsz: {args.imgsz}, epochs: {args.epochs}, batch: {args.batch}
- data: `{args.data}`

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
result = model("image.jpg")
```
"""


def main() -> int:
    args = _parse_args()

    from libreyolo.models.yolo9.model import LibreYOLO9

    init = args.init_student or f"LibreYOLO9{args.student_size}.pt"
    print(f"[train_distill] Initializing student from {init}", flush=True)
    student = LibreYOLO9(
        model_path=init,
        size=args.student_size,
        device=args.device or "auto",
    )

    print(f"[train_distill] Teacher: {args.teacher}", flush=True)
    print(f"[train_distill] Distilling with {args.loss_type.upper()} (weight={args.loss_weight})", flush=True)
    results = student.train(
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
        distill=True,
        distill_teacher=args.teacher,
        distill_loss_type=args.loss_type,
        distill_loss_weight=args.loss_weight,
        distill_mask_ratio=args.mask_ratio,
        distill_tau=args.tau,
    )

    print(f"[train_distill] Done. Best mAP50-95={results.get('best_mAP50_95')} "
          f"best_epoch={results.get('best_epoch')}", flush=True)
    print(f"[train_distill] best_checkpoint={results['best_checkpoint']}", flush=True)

    if args.push:
        if not args.hf_repo:
            print("[train_distill] --push set but --hf-repo missing", file=sys.stderr)
            return 2
        url = _push_to_hub(results, args.hf_repo, args, args.hf_private)
        print(f"[train_distill] Pushed to {url}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
