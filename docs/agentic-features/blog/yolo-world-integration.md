# Open-Vocabulary Detection in LibreYOLO — YOLO-World Full Integration

**Branch:** `agentic/yolo-world` (greenfield on `main`)

## TL;DR

`LibreYOLOWorld` is now a **complete, weight-portable** open-vocabulary YOLO family in LibreYOLO: YOLOv8-CSPDarknet backbone + frozen CLIP text encoder + RepVL-PAN neck (MaxSigmoid fusion) + BNContrastive head, structurally identical to `wondervictor/YOLO-World-V2.1`. Real V2.1-S weights load with **0 shape mismatches** and produce **real detections on real images** with arbitrary text prompts.

```
Loaded weights: 411 keys, 0 mismatches
Inference on parkour.jpg
  detections: 2
   0. person         conf=0.459  xyxy=[353, 67, 607, 271]
   1. person         conf=0.137  xyxy=[380, 448, 867, 840]
```

This is text-prompted zero-shot detection: the prompt list `["person", "skateboard", "building", "tree"]` was set at runtime; no training, no fine-tuning. Both detections correctly classify as `"person"` with sensible bounding boxes.

## What changed since the MVP

The original MVP (commit `63f1bfe`) was an architecture sketch — minimal CNN backbone, simple text encoder, similarity head, all random-init. This branch replaces it with a real, weight-portable architecture.

| Component | MVP (initial) | Now |
|-----------|----------------|-----|
| Backbone | 4-stage minimal CNN | YOLOv8-CSPDarknet (s/m/l/x), mmyolo-compatible naming |
| Text encoder | `CLIPTextModel` | `CLIPTextModelWithProjection` (matches V2.1) |
| Neck | none | `YOLOWorldPAFPN` with `MaxSigmoidCSPLayerWithTwoConv` at every merge |
| Head | similarity over global features | `YOLOWorldHeadModule` with DFL regression + `BNContrastiveHead` per scale |
| Postprocess | argmax + sigmoid | DFL bin integration → xyxy + class-agnostic NMS |
| Weight porting | none | `weight_porting.py` with verified key remap (411/411 from V2.1-S) |

## Architecture details

### Backbone
`YOLOv8CSPDarknet` outputs P3/P4/P5 feature maps. Layer names follow mmyolo's convention (`stem`, `stage{1..4}`, `main_conv`, `final_conv`, `blocks`) so state-dict keys remap directly. Channel counts match V2.1 exactly per size scaling factor.

### Text encoder
`CLIPTextModelWithProjection` from `openai/clip-vit-base-patch32`, frozen (`requires_grad=False`). Outputs L2-normalized 512-D text embeddings. We load this from HuggingFace separately from the YOLO-World checkpoint — the V2.1 `.pth` includes a copy of CLIP we don't need (it'd just duplicate the HF download).

### RepVL-PAN — the key innovation
Each merge in the FPN goes through `MaxSigmoidCSPLayerWithTwoConv`, which is a standard YOLOv8 C2f with the **last block's output gated by a max-over-class sigmoid attention** computed against the text embeddings. The attention block (`MaxSigmoidAttnBlock`):

```python
g = guide_fc(text_embeds).view(B, N_cls, num_heads, attn_head_ch)   # text -> head dim
e = embed_conv(x).view(B, num_heads, attn_head_ch, H, W)             # image -> attn dim
attn = einsum("bmchw,bnmc->bmhwn", e, g).max(dim=-1)[0] / sqrt(Ch)  # max over classes
attn = (attn + bias).sigmoid()                                       # per-pixel, per-head gate
out = project_conv(x).view(B, num_heads, proj_head_ch, H, W) * attn  # gate image features
```

That's the entire RepVL-PAN trick — ~30 lines. The fusion happens in-place on the last block of each CSP layer (concat list = `[y1, y2, *blocks_except_last, attn(last_block)]`); upstream's `final_conv` input is `(2 + n_blocks) * mid`, not `(3 + n_blocks)`.

### Head
Per scale: regression branch produces `4 * reg_max` channels (default 64) using DFL with intermediate channels `4 * reg_max` (V2.1 convention); classification branch produces `embed_dim` (=512) channels through intermediate `embed_dim // 4` (=128). `BNContrastiveHead` then computes `cls_logit = (BatchNorm(image_embed) · L2Norm(text_embeds)) * exp(logit_scale) + bias` where `logit_scale` and `bias` are scalars (matches V2.1).

### Decoder
The wrapper's `_postprocess`:
1. DFL: softmax over `reg_max` bins per side, integrate over bin indices to get distance in grid units.
2. Convert (cx, cy, l, t, r, b) per anchor location to `xyxy` in input-image coordinates using the level's stride.
3. Sigmoid the contrastive logits per location, take max over prompts.
4. Class-agnostic NMS (torchvision if available, else greedy fallback).
5. Rescale to original image dimensions.

## V2.1-S structural map (for the curious)

Verified against `s_stage2-4466ab94.pth`:

| Tensor | Shape | Notes |
|--------|-------|-------|
| `backbone.stem.conv.weight` | `(32, 3, 3, 3)` | stem 3→32 |
| `backbone.stage1.0.conv.weight` | `(64, 32, 3, 3)` | downsample |
| `backbone.stage1.1.main_conv.conv.weight` | `(64, 64, 1, 1)` | C2f, mid=32, 2*mid=64 |
| `backbone.stage4.2.conv1.conv.weight` | `(256, 512, 1, 1)` | SPPF |
| `neck.top_down_layers.0.attn_block.bias` | `(4,)` | num_heads=4 (P4 scale) |
| `neck.top_down_layers.1.attn_block.bias` | `(2,)` | num_heads=2 (P3 scale) |
| `neck.bottom_up_layers.1.attn_block.bias` | `(8,)` | num_heads=8 (P5 scale) |
| `neck.top_down_layers.0.final_conv.conv.weight` | `(256, 512, 1, 1)` | (2+2)*128=512 → n_blocks=2 |
| `head.cls_preds.0.0.conv.weight` | `(128, 128, 3, 3)` | intermediate=128=embed_dim/4 |
| `head.cls_contrasts.0.bias` | `()` | scalar |
| `head.cls_contrasts.0.logit_scale` | `()` | scalar |

Heads scale per output scale: `num_heads = mid_channels // 32` (each head sees 32 channels). For S: P3=2, P4=4, P5=8. Embed channels = `mid` per scale.

## Loading weights

```python
import torch
from libreyolo.models.yoloworld import LibreYOLOWorld
from libreyolo.models.yoloworld.weight_porting import port_from_hf

model = LibreYOLOWorld(size="s", prompts=["person", "dog", "tree"], device="cpu")
result = port_from_hf(
    model.model,
    repo_id="wondervictor/YOLO-World-V2.1",
    filename="s_stage2-4466ab94.pth",
)
print(f"loaded {result['loaded']} keys, {result['shape_mismatches']} mismatches")
# loaded 411 keys, 0 mismatches

result = model("photo.jpg")
```

## License caveat (critical)

The official `wondervictor/YOLO-World-V2.1` checkpoints are **GPL-3.0**.

LibreYOLO itself is MIT and **never bundles weights**. When you call `port_from_hf`, you (the user) download GPL weights to your machine. Any code linked against the loaded module is then bound by GPL-3.0 — that's the user's choice and responsibility, not LibreYOLO's.

Concretely:
- ✅ Run inference for personal / research use → fine.
- ✅ Distribute LibreYOLO with this scaffold → MIT, fine, no weights bundled.
- ❌ Bundle the weights into a proprietary product → GPL-3.0 problem.
- ❌ Train your own weights from scratch and bundle those → MIT (your weights), fine.

A natural follow-up: train V2.1-shape weights on Objects365 + GoldG with an MIT release and host them under `LibreYOLO/yolo-world-mit`. That's a real-compute task for another session.

## What works today

| Capability | Status |
|------------|--------|
| Architecture matches V2.1-S exactly | ✅ |
| Forward pass on CPU | ✅ (~5s per 640×640 image, S size) |
| Weights load cleanly from HF | ✅ (411 keys, 0 shape mismatches) |
| Inference produces real detections | ✅ (verified on `parkour.jpg`: 2 people, conf 0.46 / 0.14) |
| Hot-swap text prompts at inference | ✅ |
| 10 offline smoke tests | ✅ all green in 14s |

## What's still ahead

- **L / M / X size validation** — V2.1-L config differs in num_heads / channels. The auto-derivation rule (`num_heads = mid // 32`) likely covers L, but it needs a checkpoint download to verify.
- **mAP evaluation on LVIS / COCO** — proper open-vocab benchmarking. Tencent reports ~26 mAP on LVIS-mini for V2.1-S; we'd want to reproduce that to confirm the decoder is correct.
- **MIT-licensed weight training** — an Objects365 + GoldG run (very large compute) would let LibreYOLO ship MIT weights bundled.
- **ONNX export + browser demo** — once we have weights we can ship.

## Try it

```bash
# Install the optional yolo-world extra
pip install -e 'git+https://github.com/aalvsz/libreyolo@agentic/yolo-world#egg=libreyolo[yoloworld]'

# Smoke tests
pytest tests/smoke/test_yoloworld_smoke.py -m smoke

# Real inference (downloads ~290 MB GPL weights from HF on first run)
python scripts/run_yoloworld.py \
    --image my_photo.jpg \
    --prompts "person" "dog" "traffic cone" \
    --weights /path/to/s_stage2.pth
```

## Files

```
libreyolo/models/yoloworld/
    __init__.py
    nn.py                 # YOLOv8 backbone + RepVL-PAN + BNContrastive (~600 LOC)
    model.py              # LibreYOLOWorld wrapper (BaseModel subclass)
    weight_porting.py     # state_dict remapper + HF Hub loader
tests/smoke/
    test_yoloworld_smoke.py   # 10 offline tests
scripts/
    run_yoloworld.py
notebooks/
    yoloworld_tutorial.ipynb
docs/agentic-features/blog/
    yolo-world-integration.md
pyproject.toml             # adds [project.optional-dependencies.yoloworld]
```
