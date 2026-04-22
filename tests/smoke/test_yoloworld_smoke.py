"""Offline smoke test for the open-vocabulary YOLO-World scaffold.

Validates the architecture + API surface end-to-end on CPU:
  - instantiate the wrapper with a list of prompts,
  - text encoder produces correctly-shaped normalized embeddings,
  - forward pass on a synthetic image returns well-shaped outputs,
  - swapping prompts at runtime updates the classification head width,
  - full inference call returns a populated Results-like dict.

Does NOT validate accuracy — weights are random-init in the MVP.
See docs/agentic-features/blog/yolo-world-integration.md for scope.

Run: pytest tests/smoke/test_yoloworld_smoke.py -v -m smoke
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.models.yoloworld import LibreYOLOWorld, LibreYOLOWorldModel, EMBED_DIM


pytestmark = pytest.mark.smoke


def _synthetic_image(size: int = 256) -> Image.Image:
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Core module tests — no CLIP/HF download
# ---------------------------------------------------------------------------


def test_model_instantiates_cpu_no_network():
    """The raw nn.Module builds on CPU. This DOES download CLIP weights once
    (cached in ~/.cache/huggingface). That's intentional — the text encoder
    is essential."""
    model = LibreYOLOWorldModel(imgsz=256)
    assert model.embed_dim == EMBED_DIM
    assert model.num_anchors > 0
    # No prompts set yet — forward should fail loudly
    with pytest.raises(RuntimeError, match="No text prompts set"):
        model(torch.randn(1, 3, 256, 256))


def test_text_encoder_returns_normalized_embeddings():
    model = LibreYOLOWorldModel(imgsz=256)
    embeds = model.text_encoder.encode(["a photo of a cat", "a photo of a dog", "chair"])
    assert embeds.shape == (3, EMBED_DIM), f"expected (3, {EMBED_DIM}), got {tuple(embeds.shape)}"
    # L2 normalized
    norms = embeds.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), f"norms: {norms}"
    # Distinct prompts → distinct embeddings
    sim = embeds @ embeds.T
    # diagonal ~1, off-diagonal < 1
    assert torch.allclose(sim.diag(), torch.ones(3), atol=1e-4)
    off_diag = sim - torch.eye(3)
    assert off_diag.abs().max() < 0.99, "distinct prompts should not produce identical embeddings"


def test_forward_shapes_with_prompts():
    model = LibreYOLOWorldModel(imgsz=256)
    prompts = ["person", "car", "bicycle", "dog"]
    model.set_prompts(prompts)

    x = torch.randn(2, 3, 256, 256)
    out = model(x)

    B, A = 2, model.num_anchors
    # stride is imgsz / Hp. Our backbone has 4 stride-2 stages → stride=16 → Hp=16
    assert out["stride"] == 16
    Hp = Wp = 256 // 16

    assert out["bbox"].shape == (B, A, Hp, Wp, 4)
    assert out["obj"].shape == (B, A, Hp, Wp)
    assert out["cls"].shape == (B, A, Hp, Wp, len(prompts))

    # Logits are finite
    for key in ("bbox", "obj", "cls"):
        assert torch.isfinite(out[key]).all(), f"non-finite values in {key}"


def test_prompts_are_hot_swappable():
    """Changing the prompt list at runtime changes the class-head width,
    without rebuilding the model."""
    model = LibreYOLOWorldModel(imgsz=256)

    model.set_prompts(["cat", "dog"])
    out_a = model(torch.randn(1, 3, 256, 256))
    assert out_a["cls"].shape[-1] == 2

    model.set_prompts(["red apple", "green apple", "banana", "orange"])
    out_b = model(torch.randn(1, 3, 256, 256))
    assert out_b["cls"].shape[-1] == 4

    assert model.num_prompts == 4
    assert model.prompts == ["red apple", "green apple", "banana", "orange"]


def test_empty_prompts_rejected():
    model = LibreYOLOWorldModel(imgsz=256)
    with pytest.raises(ValueError):
        model.set_prompts([])
    with pytest.raises(ValueError):
        model.set_prompts("not a list")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Wrapper tests — user-facing API
# ---------------------------------------------------------------------------


def test_wrapper_instantiates_with_prompts():
    model = LibreYOLOWorld(prompts=["person", "dog", "traffic cone"], imgsz=256, device="cpu")
    assert model.prompts == ["person", "dog", "traffic cone"]
    assert model.nb_classes == 3


def test_wrapper_inference_returns_detections(tmp_path):
    model = LibreYOLOWorld(prompts=["person", "dog"], imgsz=256, device="cpu")

    img = _synthetic_image(256)
    img_path = tmp_path / "sample.jpg"
    img.save(img_path)

    # Call through the standard BaseModel inference path with a permissive threshold.
    result = model(str(img_path), conf=0.0, iou=0.5, max_det=20)
    res = result[0] if isinstance(result, list) else result
    # Random-init weights → detections may be 0, but the pipeline must return
    # a Results-like object without raising.
    assert res is not None
    assert hasattr(res, "boxes")
    if len(res.boxes) > 0:
        cls_ids = res.boxes.cls.long().tolist()
        # All returned class IDs must be indices into the prompt list.
        assert all(0 <= c < len(model.prompts) for c in cls_ids), f"cls out of range: {cls_ids}"


def test_wrapper_prompt_swap_changes_output_classes(tmp_path):
    model = LibreYOLOWorld(prompts=["cat"], imgsz=256, device="cpu")
    img_path = tmp_path / "sample.jpg"
    _synthetic_image(256).save(img_path)

    _ = model(str(img_path), conf=0.0, iou=0.5, max_det=5)
    model.set_prompts(["dog", "cat", "bird", "fish", "snake"])
    assert model.nb_classes == 5

    result = model(str(img_path), conf=0.0, iou=0.5, max_det=5)
    res = result[0] if isinstance(result, list) else result
    if len(res.boxes) > 0:
        cls_ids = res.boxes.cls.long().tolist()
        assert all(0 <= c < 5 for c in cls_ids)
