# Fine-tuning YOLOv9 on VisDrone — A Reference Pipeline for LibreYOLO

**Branch:** `agentic/c-visdrone-finetune` (based on `main`)

## TL;DR

This is the SAM-style demo applied to LibreYOLO: take the shipping library (no new features), pick a real domain dataset, and build a fully reproducible pipeline from raw download → trained weights → HF Hub. The domain chosen is **VisDrone2019-DET**, the standard drone-captured object detection benchmark (10 classes, 6.5k train / 550 val images, top-down aerial perspective).

Aerial imagery is a noticeably different data distribution than COCO — small objects, unusual angles, different class mix. That makes it a meaningful fine-tune target: you can measure the payoff of domain-specific training.

Ten smoke tests pass in 1.4 seconds on CPU, validating the VisDrone→YOLO annotation converter and the full training path on synthetic data:

```
tests/smoke/test_visdrone_finetune_smoke.py .......... 10 passed
```

## What ships in this branch

Three files, no changes to LibreYOLO internals:

```
scripts/finetune_yolo9_visdrone.py      # 280-line standalone CLI
notebooks/yolo9_visdrone_finetune.ipynb # Tutorial that runs on CPU in seconds
tests/smoke/test_visdrone_finetune_smoke.py  # Annotation + train smoke tests
```

The CLI handles:
1. **Dataset acquisition.** Either point to an already-downloaded VisDrone directory or let the script pull a known HF dataset mirror via `huggingface_hub.snapshot_download`.
2. **Annotation conversion.** VisDrone's comma-separated format ≠ YOLO format. The `visdrone_line_to_yolo()` helper parses each line, filters categories 0 (ignored-regions) and 11 (others), remaps class IDs 1–10 → 0–9, normalizes bboxes, clamps out-of-bounds coordinates.
3. **YOLO layout materialization.** Symlinks the VisDrone image directories under a standard `images/train`, `images/val` layout and writes the converted labels next to them in `labels/`. No image copying — a 3GB dataset stays in place.
4. **Training.** `LibreYOLO9(model_path="LibreYOLO9s.pt", size="s")` auto-downloads the COCO-pretrained detection backbone; `.train()` fine-tunes against the VisDrone labels.
5. **HF Hub publishing.** With `--push --hf-repo <org/name>`, uploads the best checkpoint and a generated model card (YAML front-matter with `library_name: libreyolo`, tags, class list, training recipe, usage snippet).

## The conversion function, in full

```python
VISDRONE_ID_TO_YOLO = {cid: cid - 1 for cid in range(1, 11)}  # 1..10 -> 0..9

def visdrone_line_to_yolo(line: str, img_w: int, img_h: int) -> str | None:
    parts = [p.strip() for p in line.strip().split(",") if p.strip()]
    if len(parts) < 6:
        return None
    try:
        x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        cat = int(parts[5])
    except ValueError:
        return None
    if cat not in VISDRONE_ID_TO_YOLO:  # drops 0 (ignored) and 11 (others)
        return None
    if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
        return None
    yolo_cls = VISDRONE_ID_TO_YOLO[cat]
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    bw, bh = w / img_w, h / img_h
    cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
    bw, bh = max(0.0, min(1.0, bw)), max(0.0, min(1.0, bh))
    if bw <= 0 or bh <= 0:
        return None
    return f"{yolo_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
```

The smoke tests exercise every branch — valid, ignored category, out-of-range, zero-size, malformed, clamping. This is the kind of code where a 20-line test file is worth more than a paragraph of docstring.

## Running a real fine-tune

Local (assuming VisDrone already downloaded to `/data/VisDrone`):

```bash
python scripts/finetune_yolo9_visdrone.py \
    --visdrone-root /data/VisDrone \
    --size s \
    --epochs 50 --batch 16 --imgsz 640 \
    --push --hf-repo ander2221/libreyolo-yolo9s-visdrone
```

Managed compute (HF Jobs, A10G small, ~3 hours):

```bash
hf jobs run \
    --image ghcr.io/huggingface/transformers-pytorch-gpu:latest \
    --secrets HF_TOKEN \
    --flavor a10g-small \
    -- bash -lc '
        pip install -e git+https://github.com/aalvsz/libreyolo@agentic/c-visdrone-finetune#egg=libreyolo &&
        python scripts/finetune_yolo9_visdrone.py \
            --hf-dataset Voxel51/visdrone2019-det \
            --size s --epochs 50 --batch 16 --imgsz 640 \
            --push --hf-repo ander2221/libreyolo-yolo9s-visdrone
    '
```

Reload the published model:

```python
from huggingface_hub import hf_hub_download
from libreyolo import LibreYOLO

ckpt = hf_hub_download(repo_id="ander2221/libreyolo-yolo9s-visdrone", filename="best.pt")
model = LibreYOLO(ckpt)
result = model("aerial_shot.jpg")
```

## Why VisDrone for the SAM-style demo

- **Public + standardized.** Multi-year benchmark, widely used, clear license terms.
- **Non-trivial distribution shift.** Aerial perspective, small objects, heavy occlusion — COCO weights zero-shot are notably weaker here. Fine-tuning pays off.
- **10 classes covering common vehicle + pedestrian categories** — results are interpretable.
- **Reasonable size.** Large enough to matter, small enough that a fine-tune fits in a few hours of A10G time.

## What's the Phase 2 target

Of the three features on this fork (segmentation, distillation, VisDrone fine-tune), this one is the primary candidate for actually running the HF Jobs training end-to-end and publishing real weights. Reasons:
- No new features needed — uses shipping LibreYOLO code.
- Smallest compute budget (~3 hrs on A10G-small).
- Closest analogue to the SAM medical demo that motivated this work.

When Phase 2 runs, the following lands publicly:
- `ander2221/libreyolo-yolo9s-visdrone` on HF Hub (best.pt + model card).
- This blog updated with concrete mAP numbers + inference samples.
- The notebook updated to load those uploaded weights.

## Try it

- **Quickstart notebook:** [`notebooks/yolo9_visdrone_finetune.ipynb`](../../../notebooks/yolo9_visdrone_finetune.ipynb) (runs on CPU with synthetic data in seconds)
- **Smoke tests:** `pytest tests/smoke/test_visdrone_finetune_smoke.py -m smoke`
- **Training script:** `scripts/finetune_yolo9_visdrone.py --help`

## What's not in this branch (yet)

- **Real trained weights** on HF Hub. Phase 2 will run the HF Jobs training and publish them. The script, smoke tests, and notebook are ready to go.
- **VisDrone evaluation protocol.** VisDrone ships its own mAP evaluation tooling (the VisDrone2019-DET toolkit). LibreYOLO's `DetectionValidator` uses COCO-style mAP instead — good enough for checkpoint selection but not directly comparable to VisDrone leaderboard numbers. A follow-up could add the VisDrone evaluator as an optional validator.
- **Test-set results.** VisDrone holds test-set annotations server-side; proper leaderboard submission would require the official toolkit.
