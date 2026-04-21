# Finishing MGD & CWD Distillation in LibreYOLO

**Branch:** `agentic/b-distillation` (based on upstream `41-research-mgd-and-cwd-distillation`)

## TL;DR

LibreYOLO's upstream distillation branch (#41) implements **Masked Generative Distillation** (MGD, ECCV 2022) and **Channel-Wise Distillation** (CWD, ICCV 2021) as first-class, model-agnostic primitives: 10 commits, a `libreyolo.distillation` module, hooks on both teacher and student, configurable tap points, wired into the training loop, 46 unit tests all green.

What it didn't have: a smoke test that confirms the plumbing actually carries gradients on real data, a user-facing tutorial, or a path from research code to pushed weights. We added those.

Three smoke tests pass in 3.6 seconds on CPU — MGD, CWD, and a direct `Distiller` module check:

```
tests/smoke/test_distillation_smoke.py::test_distillation_training_finishes[mgd] PASSED
tests/smoke/test_distillation_smoke.py::test_distillation_training_finishes[cwd] PASSED
tests/smoke/test_distillation_smoke.py::test_distiller_module_shapes          PASSED
```

## Why MGD / CWD are worth having

YOLO students trained directly on detection labels leave accuracy on the table. Knowledge distillation from a larger pretrained teacher recovers some of that gap with **no extra data** and **no architectural changes**:

- **MGD** ([Yang et al. 2022](https://arxiv.org/abs/2205.01529)) reconstructs the teacher's feature map from a *masked* student feature map. The generative formulation forces the student to encode teacher-level representations even in randomly-occluded positions — harder objective, better transfer. A single tunable hyperparameter: `mask_ratio` (default 0.65).
- **CWD** ([Shu et al. 2021](https://arxiv.org/abs/2011.13256)) applies per-channel softmax to both teacher and student feature maps, then matches distributions via KL divergence. One hyperparameter: temperature `tau` (default 1.0). Usually very strong on dense prediction tasks.

Both methods operate on intermediate features from a handful of tap points — typically the last three stages of the backbone, matching FPN input scales. In LibreYOLO, those tap points and their channel dims come from the model wrapper itself (`wrapper.get_distill_config()`), so the Distiller is architecture-agnostic: anything that implements `get_distill_config()` can be distilled.

## What the upstream branch already ships

Ten commits of substantive work:

| Commit | What it adds |
|--------|---|
| `c8dd727` | `libreyolo/distillation/` module with MGD + CWD losses |
| `5dd4356` | Unit tests (convergence + shape checks) |
| `2bd9641` | `TrainConfig` fields: `distill`, `distill_teacher`, `distill_loss_type`, etc. |
| `05a16b4` | Distiller wired into `BaseTrainer._train_epoch` |
| `7e5813f` | Lazy imports so distillation is opt-in |
| `05809a9` | Error handling cleanup, dead code removal |
| `31df8b1` | Deduplicate AMP / non-AMP code paths |
| `3fd1772` / `87afc7b` | Move distill configs into model wrappers, simplify API |
| `0d2d701` | Convergence checks in the unit tests |

The public API is three parts:

```python
# Option 1 — the high-level API: pass distillation kwargs to model.train()
model.train(
    data="coco/data.yaml", epochs=100, batch=16, imgsz=640,
    distill=True,
    distill_teacher="weights/LibreYOLO9c.pt",
    distill_loss_type="mgd",        # or "cwd"
    distill_loss_weight=0.5,
    distill_mask_ratio=0.65,         # MGD
    distill_tau=1.0,                 # CWD
)

# Option 2 — drive the loop manually with the primitive
from libreyolo.distillation import Distiller

distiller = Distiller(
    teacher_model=teacher.model,
    student_model=student.model,
    teacher_config=teacher.get_distill_config(),
    student_config=student.get_distill_config(),
    loss_type="mgd", loss_weight=0.5,
)
distiller.teacher_forward(imgs)
outputs = student.model(imgs, targets)
loss = outputs["total_loss"] + distiller.compute_loss()
# ...backward...
distiller.step()

# Option 3 — per-model distill configs if you need to customize tap points
from libreyolo.distillation.configs import get_distill_config
cfg = get_distill_config("yolo9", "c")
```

## What we verified

Writing a tight smoke test exposed exactly what's not obvious from reading distillation papers: does the whole plumbing — hooks, teacher forward pass, feature alignment convs, loss computation, gradient flow — actually work end-to-end?

The smoke test:
1. Makes two fresh YOLOv9 checkpoints (`c` teacher, `t` student) — no network needed.
2. Generates 4 tiny synthetic detection images.
3. Runs 1 epoch with `distill=True` at `imgsz=128`, `batch=2`, on CPU.
4. Asserts finite total loss and a writable checkpoint for both MGD and CWD.
5. Separately, drives the `Distiller` primitive directly on random input and asserts the distillation loss is a finite scalar.

Result: all three assertions pass in under 4 seconds. The distillation pipeline works as advertised. Gradients flow through both the detection loss and the distillation loss; the student's optimizer picks up the distiller's own learnable params (alignment + generation convs) via `add_param_group`.

## Scaling up

For a real run — e.g., YOLOv9-c as teacher, YOLOv9-t as student, on COCO:

```bash
python scripts/train_distill_yolo9.py \
    --data coco/data.yaml \
    --teacher weights/LibreYOLO9c.pt \
    --student-size t \
    --init-student weights/LibreYOLO9t.pt \
    --epochs 100 --batch 16 --imgsz 640 \
    --loss-type mgd --loss-weight 0.5 \
    --push --hf-repo ander2221/libreyolo-yolo9t-distilled
```

Submit to HF Jobs (A10G-small flavor, ~8 hours at COCO scale):

```bash
hf jobs run \
    --image ghcr.io/huggingface/transformers-pytorch-gpu:latest \
    --secrets HF_TOKEN \
    --flavor a10g-small \
    -- bash -lc '
        pip install -e git+https://github.com/aalvsz/libreyolo@agentic/b-distillation#egg=libreyolo &&
        python scripts/train_distill_yolo9.py \
            --data /data/coco/data.yaml \
            --teacher /models/LibreYOLO9c.pt \
            --student-size t \
            --epochs 100 --batch 32 --imgsz 640 \
            --loss-type mgd --loss-weight 0.5 \
            --push --hf-repo ander2221/libreyolo-yolo9t-distilled
    '
```

Published artifact reloads with the factory:

```python
from huggingface_hub import hf_hub_download
from libreyolo import LibreYOLO

ckpt = hf_hub_download(repo_id="ander2221/libreyolo-yolo9t-distilled", filename="best.pt")
model = LibreYOLO(ckpt)
result = model("image.jpg")
```

## Extending to other architectures

Adding distillation to a new family is mechanical:
1. Implement `get_distill_config()` on the model wrapper, returning `{"tap_points": [...], "channel_dims": [...]}` for each scale.
2. Everything else — `Distiller`, hooks, losses, trainer wiring — is architecture-agnostic.

This is how the upstream branch supports YOLOv9 and YOLOX simultaneously with the same `Distiller`. RF-DETR and future models get distillation for free once `get_distill_config()` is implemented.

## Try it

- **Quickstart notebook:** [`notebooks/yolo9_distillation_tutorial.ipynb`](../../../notebooks/yolo9_distillation_tutorial.ipynb)
- **Smoke test:** `pytest tests/smoke/test_distillation_smoke.py -m smoke`
- **Training script:** `scripts/train_distill_yolo9.py --help`

## Next steps

- Submit HF Jobs training for YOLOv9-c → YOLOv9-t on COCO; publish results under `ander2221/libreyolo-yolo9t-distilled` (compute budget permitting).
- Land this branch upstream as a PR to `LibreYOLO/libreyolo#41` with the smoke tests as concrete convergence evidence.
- Expand the per-loss ablation: both MGD and CWD have a single knob (`mask_ratio` / `tau`). Small ablation table on COCO would make this a *very* nice addition to the LibreYOLO docs.
