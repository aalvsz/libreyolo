# Open-Vocabulary Detection in LibreYOLO — YOLO-World Scaffold

**Branch:** `agentic/yolo-world` (greenfield on `main`)

## TL;DR

Adds a new model family, `LibreYOLOWorld`, to LibreYOLO's registry — text-prompted detection where you give the model a list of class names as strings (`["person", "dog", "my custom thing"]`) and it returns detections scored against those classes, zero-shot, without retraining. The architecture follows YOLO-World: CLIP text encoder + vision encoder + similarity-based class head, with prompts hot-swappable at inference time.

This branch ships:
- A working `LibreYOLOWorld` wrapper registered in `BaseModel._registry` (auto-detected by the factory).
- A `LibreYOLOWorldModel` nn.Module: CLIP text encoder (HF `openai/clip-vit-base-patch32`), a small CNN vision backbone projecting to CLIP's 512-dim embedding space, a box regression head, and a similarity-based class head.
- 8 offline smoke tests — all green in ~13 seconds on CPU (one-time 12-second CLIP download, then cached).
- An `optional-dependencies.yoloworld` extra so `transformers` doesn't bloat the default install.
- A CLI (`scripts/run_yoloworld.py`) and tutorial notebook.

**What's NOT yet here: real pretrained weights.** Porting Tencent/YOLO-World weights is a separate effort — see §Roadmap. Until that lands, detection outputs are noise; the blog leads with this so nobody is surprised. The *architecture*, *API*, and *smoke-tested plumbing* are the deliverable.

## Why this matters

Closed-vocabulary YOLO is a commodity — YOLOv8/v9/v10/v11, YOLOX, RF-DETR all do 80 COCO classes well. Open-vocabulary YOLO is the real frontier: one model, arbitrary class list at inference time, no retraining. [YOLO-World (Cheng et al., CVPR 2024)](https://arxiv.org/abs/2401.17270) and NVIDIA's YOLOE are the two serious open-weight entries. Ultralytics has a half-hearted YOLO-World integration; nobody in the open-source YOLO space has shipped a *clean, well-tested, factory-compatible* wrapper. This branch's shape is that wrapper.

## Architecture

```
      image ──► VisionEncoder ──► (B, 512, H/16, W/16) ──► visual_proj ──► (B, A×512, H', W')
                                                                              │
                                                                              ▼
                                                                       L2-normalize per anchor
                                                                              │
                                                                              ▼
                                                  ┌─────────────────────► similarity ◄─────────────────┐
                                                  │                         (B, A, H', W', N_prompts) │
                                                  │                                                      │
    prompts ──► CLIP text encoder ──► (N_prompts, 512) ──────────────────────────────────────────────────┘
                                                  │
                                                  └─► × exp(logit_scale)   (CLIP temperature)
```

Key design decisions:

1. **CLIP ViT-B/32 text tower**, frozen. Matches YOLO-World's default. Frozen because open-vocab detection fine-tunes the *alignment*, not the CLIP priors themselves.
2. **Simple CNN vision encoder for the MVP.** Real YOLO-World uses a YOLOv8-L backbone + their RepVL-PAN neck. We ship a minimal 4-stage stride-16 CNN so the smoke test runs in milliseconds and the architecture reads cleanly. Weight porting from Tencent's release is the follow-up.
3. **Projected visual features have an anchor axis** — each location gets `num_anchors` separate 512-D feature vectors. Matches how YOLO heads spread capacity across shape priors.
4. **Cosine-similarity class head with CLIP-style temperature**. Logit scale is a learnable parameter initialized to `exp(2.659) ≈ 14.3` (CLIP convention). No separate classification weights — the "classifier" *is* the text embeddings.
5. **Prompts are a `register_buffer`**, not Parameters. Changing them at inference doesn't dirty the model state dict.

## The user-facing API

```python
from libreyolo.models.yoloworld import LibreYOLOWorld

model = LibreYOLOWorld(prompts=["person", "dog", "traffic cone"], device="cpu")
result = model("photo.jpg")
# result.boxes.cls are indices into the prompts list

# Add / change classes at runtime — no retraining, no rebuild
model.set_prompts(["red car", "blue car", "yellow taxi"])
result = model("street.jpg")
```

And through the unified factory:

```python
from libreyolo import LibreYOLO
model = LibreYOLO("LibreYOLOWorlds.pt", prompts=[...])   # once weights ported
```

## What the smoke tests actually verify

```
tests/smoke/test_yoloworld_smoke.py::test_model_instantiates_cpu_no_network     PASSED
tests/smoke/test_yoloworld_smoke.py::test_text_encoder_returns_normalized_embeddings  PASSED
tests/smoke/test_yoloworld_smoke.py::test_forward_shapes_with_prompts           PASSED
tests/smoke/test_yoloworld_smoke.py::test_prompts_are_hot_swappable             PASSED
tests/smoke/test_yoloworld_smoke.py::test_empty_prompts_rejected                PASSED
tests/smoke/test_yoloworld_smoke.py::test_wrapper_instantiates_with_prompts     PASSED
tests/smoke/test_yoloworld_smoke.py::test_wrapper_inference_returns_detections  PASSED
tests/smoke/test_yoloworld_smoke.py::test_wrapper_prompt_swap_changes_output_classes  PASSED
8 passed, 1 warning in 13.40s
```

Specifically:
- `text_encoder.encode([...])` returns L2-normalized `(N, 512)` embeddings; distinct prompts give distinct vectors.
- Forward pass produces shape-consistent `bbox`, `obj`, and `cls` tensors with finite values.
- `set_prompts(new_list)` changes the final axis of `cls` without rebuild.
- Empty / non-list prompts raise cleanly.
- The wrapper integrates with `BaseModel`'s inference pipeline: an image path in → a `Results` object out with class indices bounded by `len(prompts)`.

## Scope caveats (important)

1. **Weights are random-init.** Calling `model("photo.jpg")` returns nonsense until weights are ported. The smoke test for "inference returns detections" verifies the *pipeline* runs cleanly, not that detections are correct.
2. **The vision backbone is NOT the real YOLO-World backbone.** We ship a minimal CNN because the MVP's point is the API shape + CLIP integration, not COCO-level accuracy. Tencent's full model uses YOLOv8-L + RepVL-PAN — porting that requires:
   - Copying or reimplementing their neck (~200 lines),
   - Loading Tencent's state_dict with key remapping,
   - GPU validation on LVIS / COCO to confirm we haven't broken anything.
3. **No fine-tuning of CLIP.** YOLO-World V2 reports that small fine-tunes help; we leave CLIP frozen for now.
4. **Postprocessing is basic.** No NMS, no anchor priors, no per-class confidence calibration. The MVP emits top-K locations above `conf_thres`; a proper port needs class-agnostic NMS and decoding that matches Tencent's inference config.

## Roadmap to real open-vocab detection

- **Phase 1 (this branch).** Architecture + API + smoke tests. Done.
- **Phase 2 (next branch).** Replace `VisionEncoder` with a YOLOv8-L backbone. Port Tencent/YOLO-World weights via key remapping (they publish state_dicts on HF: https://huggingface.co/Tencent/YOLO-World).
- **Phase 3.** RepVL-PAN neck — multi-scale text-visual fusion. This is the secret sauce.
- **Phase 4.** Proper NMS + evaluation on LVIS (validate open-vocab performance, not just plumbing).
- **Phase 5.** Upstream PR to `LibreYOLO/libreyolo` as a new model family.

Each phase is an independent `ml-intern` subagent run and collectively buy a genuinely differentiating capability for LibreYOLO.

## Why this is a game-changer even as an MVP

A SOTA port is weeks of work. But the *shape of the feature* — an open-vocab YOLO family slotted into LibreYOLO's factory, with hot-swappable prompts and clean tests — is days. Once the shape is landed, Phase 2-5 are *just weight-porting and validation* with no API churn. Users can code against `LibreYOLOWorld(prompts=[...])` today and get real detections tomorrow.

## Try it

```bash
python scripts/run_yoloworld.py \\
    --image photo.jpg \\
    --prompts "person" "dog" "traffic cone"
```

Or:
```bash
pytest tests/smoke/test_yoloworld_smoke.py -m smoke
jupyter notebook notebooks/yoloworld_tutorial.ipynb
```

## What's explicitly out of scope for this branch

- Real training (we're inference-only).
- GPU validation (CPU-only smoke; the MVP is tiny enough it runs in 13s on a MacBook).
- Published HF Hub weights (none yet — that's Phase 2).
- RepVL-PAN neck / multi-scale fusion (Phase 3).

## Files

```
libreyolo/models/yoloworld/
    __init__.py               # exports
    model.py                  # LibreYOLOWorld wrapper (BaseModel subclass)
    nn.py                     # architecture — VisionEncoder, TextEncoder, LibreYOLOWorldModel
tests/smoke/
    test_yoloworld_smoke.py   # 8 offline tests
scripts/
    run_yoloworld.py          # inference CLI
notebooks/
    yoloworld_tutorial.ipynb  # end-to-end walkthrough
docs/agentic-features/blog/
    yolo-world-integration.md # this document
pyproject.toml                # adds [project.optional-dependencies.yoloworld] = transformers
```
