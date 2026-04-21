# Finishing YOLOv9 Instance Segmentation in LibreYOLO

**Branch:** `agentic/a-yolo9-segmentation` (based on upstream `47-add-instance-segmentation-to-yolo9`)

## TL;DR

LibreYOLO's YOLOv9 family supports instance segmentation on a feature branch, implemented by the upstream team (architecture, `MaskLoss`, polygon dataset handling, mask target flow, differential LR, unit + e2e tests — 11 commits). When we stood it up end-to-end on synthetic data, it crashed with a shape mismatch the moment we used any image size other than 640. This post covers the fix, adds a tutorial, and sets up a path to real trained weights.

Single-line root cause:

```python
# transforms.py, 5 call sites:
masks = polygons_to_masks([], 160, 160, self.max_labels)  # <-- always 160x160
```

Proto output scales with input (imgsz=128 → 32×32, imgsz=640 → 160×160). Mask targets were pinned at the imgsz=640 resolution. BCE loss compared `[N, 32, 32]` predictions to `[N, 160, 160]` targets and raised. Fix: compute target resolution from `input_dim // MASK_STRIDE`, where `MASK_STRIDE = 4` is the YOLOv9 proto stride.

Five edits, one smoke test, one constant. Training now runs end-to-end at any imgsz.

## What was already there

The upstream feature branch (#47) is impressively complete:

| Component | File | What it does |
|-----------|------|--------------|
| Seg architecture | `libreyolo/models/yolo9/nn.py` | Proto head + mask coefficients per detection |
| Mask loss | `libreyolo/models/yolo9/loss.py` | `MaskLoss` with BCE on proto × coeffs |
| Polygon storage | `libreyolo/data/dataset.py` | Parses YOLO-polygon label format |
| Target rasterization | `libreyolo/data/dataset.py:polygons_to_masks` | Polygons → binary masks at proto resolution |
| Training wiring | `libreyolo/training/trainer.py`, `libreyolo/models/yolo9/trainer.py` | Mask targets flow to loss |
| Differential LR | `libreyolo/models/yolo9/trainer.py` | 1× on seg head, 0.01× on backbone |
| Auto-detection | `libreyolo/models/yolo9/model.py:_detect_segmentation` | Checkpoint inspection + filename suffix |
| Inference | `libreyolo/models/yolo9/utils.py:process_mask` | Crops masks to predicted boxes |
| Results surface | `libreyolo/utils/results.py:Masks` | `.data`, `.xy`, `.xyn` |
| ONNX export | `libreyolo/export/onnx.py` | Seg outputs in exported graph |
| Unit tests | `tests/unit/test_segmentation.py` | 39 tests — all green |

What was missing: **a training run that actually converges** (there was no YOLOv9 seg e2e test — only RF-DETR had one), **a user-facing tutorial**, and **no published seg weights**.

## The bug an end-to-end test immediately found

To close the gap, we wrote `tests/smoke/test_yolo9_seg_smoke.py`: build a fresh seg model from scratch, synthesize 2 images with one polygon each, and run 1 epoch on CPU at `imgsz=128` with `batch=2`. Under 60 seconds, no network, no pretrained weights.

On first run:

```
ValueError: Target size (torch.Size([31, 160, 160]))
            must be the same as input size (torch.Size([31, 32, 32]))
```

Grepping the hard-coded `160`:

```
libreyolo/models/yolo9/transforms.py:111  polygons_to_masks([], 160, 160, ...)
libreyolo/models/yolo9/transforms.py:191  polygons_to_masks(polys_norm, 160, 160, ...)
libreyolo/models/yolo9/transforms.py:302  polygons_to_masks([], 160, 160, max_labels)
libreyolo/models/yolo9/transforms.py:428  polygons_to_masks([], 160, 160, max_labels)
```

Five sites, all assuming `imgsz=640`. Any other imgsz — including a sensible CPU smoke-test value — blew up before the first optimizer step.

## The fix

Compute mask resolution from the true `input_dim`:

```python
# libreyolo/models/yolo9/transforms.py

# YOLOv9 proto stride: imgsz 640 → masks 160x160, imgsz 128 → 32x32
MASK_STRIDE = 4

# Inside the transform, where input_dim = (H, W) of the target image:
mask_h = input_dim[0] // MASK_STRIDE
mask_w = input_dim[1] // MASK_STRIDE
masks = polygons_to_masks(polys_norm, mask_h, mask_w, self.max_labels)
```

Same change applied to the four other call sites (including the mosaic/mixup paths). The hardcoded `160` becomes a named constant `MASK_STRIDE`, making the relationship between imgsz and proto resolution self-documenting.

After the patch, the smoke test passes in 1.7 s on a MacBook M-series CPU. The existing 163-test unit suite stays green — no regression.

## What agentic development looks like here

The SAM medical demo that motivated this work had ml-intern do the whole loop autonomously. We used Claude Code subagents instead, but the shape is the same:

1. **Investigate.** `gh api repos/.../issues` surfaced the roadmap. Listing feature branches (`git branch -r | grep -E "41|47|93"`) revealed that seg and distillation were already near-done on upstream branches — the useful work was finishing, not inventing.
2. **Validate.** Write a smoke test that exercises the full pipeline on tiny data. If it fails, that's a bug. We found one.
3. **Fix.** A five-line patch with a named constant.
4. **Demonstrate.** A notebook a user can run on their laptop. Synthetic dataset, small model, short training — all the moving parts, none of the compute hurdle.
5. **Scale up.** A training script (`scripts/train_yolo9_seg.py`) ready for HF Jobs submission when compute is available.

The things that reliably slow down ML research — reproducible smoke tests, clean CLIs, per-artifact model cards — are exactly what a patient agent excels at. The human stays in the loop for scope decisions and compute spend; everything else is scripting.

## Training a real model

The included script handles HF-Jobs-friendly execution and optional Hub push:

```bash
# Detection checkpoint → fine-tune to seg on COCO-seg
python scripts/train_yolo9_seg.py \
    --data coco-seg/data.yaml \
    --size s \
    --epochs 300 \
    --batch 16 \
    --imgsz 640 \
    --init-weights LibreYOLO9s.pt \
    --push --hf-repo ander2221/libreyolo-yolo9s-seg
```

Submit to HF Jobs (manager compute):

```bash
hf jobs run \
    --image ghcr.io/huggingface/transformers-pytorch-gpu:latest \
    --secrets HF_TOKEN \
    --flavor a10g-small \
    -- bash -lc '
        pip install -e git+https://github.com/aalvsz/libreyolo@agentic/a-yolo9-segmentation#egg=libreyolo &&
        python -m scripts.train_yolo9_seg \
            --data coco-seg/data.yaml --size s --epochs 100 \
            --batch 16 --imgsz 640 \
            --push --hf-repo ander2221/libreyolo-yolo9s-seg
    '
```

Expected wall-clock on an A10G: ~6–8 hours for 100 epochs at size s / imgsz 640, dominated by data loading. Budget accordingly.

## Try it

- **Quickstart notebook:** [`notebooks/yolo9_segmentation_tutorial.ipynb`](../../../notebooks/yolo9_segmentation_tutorial.ipynb)
- **Smoke test:** `pytest tests/smoke/test_yolo9_seg_smoke.py -m smoke`
- **Training script:** `scripts/train_yolo9_seg.py --help`

## Next steps

- Submit HF Jobs training for YOLOv9t-seg on COCO-seg; publish weights under `ander2221/libreyolo-yolo9t-seg` once compute lands.
- Send this branch as a PR upstream (`LibreYOLO/libreyolo#47`) with the `MASK_STRIDE` fix and smoke test as a contribution closing the gap to merge.
- Extend the same `imgsz`-agnostic pattern to RF-DETR seg (branch #39) if needed — spot-check suggested it uses its own code path.
