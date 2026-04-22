"""LibreYOLOWorld — open-vocabulary YOLO architecture.

Text-prompted detection: given an image and a list of class names as text,
detect those classes zero-shot. Architecture shape:

    image ──► vision encoder (YOLOv9 backbone) ──► spatial features
                                                       │
                                                       ▼
    texts ──► CLIP text encoder ──► text_embeds ──► similarity head ──► boxes + class logits

This is a **scaffold** with a working forward pass and smoke-tested API.
Weight porting from Tencent/YOLO-World (https://github.com/AILab-CVC/YOLO-World)
is future work — the architecture below uses the existing LibreYOLO9 backbone
plus a lightweight CLIP-driven classification head. Real open-vocab accuracy
requires the RepVL-PAN neck and proper weight porting.

See `docs/agentic-features/blog/yolo-world-integration.md` for scope.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Embedding dimension used internally to align vision and text features.
# Projections on both sides map to this space.
EMBED_DIM = 512


class TextEncoder(nn.Module):
    """Lightweight text encoder wrapping HuggingFace CLIP text model.

    We use CLIP because its text embeddings are already aligned to open-vocab
    visual features (that's literally what CLIP is trained for). YOLO-World's
    paper uses a CLIP ViT-B/32 text tower by default — we do the same.
    """

    _HF_MODEL = "openai/clip-vit-base-patch32"

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        # Import lazily so the model is importable without transformers installed.
        from transformers import CLIPTextModel, CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(self._HF_MODEL)
        self.text_model = CLIPTextModel.from_pretrained(self._HF_MODEL)

        # Freeze text encoder (standard in open-vocab detection — we don't
        # fine-tune CLIP during YOLO-World training either).
        for p in self.text_model.parameters():
            p.requires_grad_(False)
        self.text_model.eval()

        clip_dim = self.text_model.config.hidden_size  # 512 for ViT-B/32
        self.proj = (
            nn.Identity() if clip_dim == embed_dim else nn.Linear(clip_dim, embed_dim, bias=False)
        )

    @torch.no_grad()
    def encode(self, prompts: List[str], device: torch.device | None = None) -> torch.Tensor:
        """Return (N_prompts, EMBED_DIM) L2-normalized text embeddings."""
        if device is None:
            device = next(self.text_model.parameters()).device
        tokens = self.tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        out = self.text_model(**tokens)
        # pooler_output is CLIP's [EOS] embedding — standard for text classification.
        x = out.pooler_output
        x = self.proj(x)
        return F.normalize(x, dim=-1)


class VisionEncoder(nn.Module):
    """Minimal CNN backbone producing spatial features.

    For the MVP we use a small custom backbone rather than reusing LibreYOLO9
    directly — it keeps the smoke test decoupled from yolo9 internals and
    avoids having to instantiate a full detection model just to grab features.
    Real weight porting from Tencent/YOLO-World would replace this with their
    YOLOv8-L backbone + RepVL-PAN neck.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, width: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
        )
        self.stage1 = self._block(width, width * 2, stride=2)
        self.stage2 = self._block(width * 2, width * 4, stride=2)
        self.stage3 = self._block(width * 4, width * 8, stride=2)
        self.neck = nn.Conv2d(width * 8, embed_dim, 1)

    @staticmethod
    def _block(cin: int, cout: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, EMBED_DIM, H/16, W/16) feature map."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.neck(x)
        return x


class LibreYOLOWorldModel(nn.Module):
    """Open-vocabulary detection core.

    Combines:
      - A vision encoder producing spatial features at the EMBED_DIM space.
      - A CLIP text encoder producing normalized text embeddings.
      - A bbox head (4 regression outputs per anchor) + objectness.

    At inference, similarity between spatial feature vectors and text
    embeddings yields per-location, per-prompt class logits.
    """

    def __init__(
        self,
        *,
        imgsz: int = 640,
        embed_dim: int = EMBED_DIM,
        width: int = 64,
        num_anchors: int = 3,
    ):
        super().__init__()
        self.imgsz = imgsz
        self.embed_dim = embed_dim
        self.num_anchors = num_anchors

        self.vision_encoder = VisionEncoder(embed_dim=embed_dim, width=width)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)

        # Bbox regression: (B, 4 * num_anchors, H, W) per stride.
        self.bbox_head = nn.Conv2d(embed_dim, 4 * num_anchors, kernel_size=1)
        # Objectness: (B, num_anchors, H, W) — does a box exist here?
        self.obj_head = nn.Conv2d(embed_dim, num_anchors, kernel_size=1)
        # Visual projection into text-aligned space: (B, embed_dim * num_anchors, H, W)
        self.visual_proj = nn.Conv2d(embed_dim, embed_dim * num_anchors, kernel_size=1)

        # Temperature for similarity logits (CLIP convention).
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6593)  # exp ~ 14.3

        # Cached text embeddings — set by `set_prompts()`.
        self.register_buffer("_text_embeds", torch.zeros(0, embed_dim), persistent=False)
        self._current_prompts: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_prompts(self, prompts: List[str]) -> None:
        """Encode and cache text embeddings for the given class names.

        After this call, `forward(images)` returns class logits of shape
        (B, num_anchors, H, W, len(prompts)) scored against these prompts.
        """
        if not isinstance(prompts, (list, tuple)) or not all(isinstance(p, str) for p in prompts):
            raise ValueError("prompts must be a list of strings")
        if len(prompts) == 0:
            raise ValueError("prompts list must be non-empty")
        device = next(self.parameters()).device
        embeds = self.text_encoder.encode(list(prompts), device=device)
        self._text_embeds = embeds
        self._current_prompts = list(prompts)

    @property
    def prompts(self) -> List[str]:
        return list(self._current_prompts)

    @property
    def num_prompts(self) -> int:
        return len(self._current_prompts)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor) -> dict:
        """Forward pass.

        Args:
            images: (B, 3, H, W) float tensor in [0, 1].

        Returns:
            dict with:
              - 'bbox': (B, num_anchors, H', W', 4) bbox regression (raw logits).
              - 'obj':  (B, num_anchors, H', W') objectness logits.
              - 'cls':  (B, num_anchors, H', W', num_prompts) per-prompt class logits.
              - 'stride': scalar effective stride (imgsz / spatial size).
        """
        if self._text_embeds.shape[0] == 0:
            raise RuntimeError(
                "No text prompts set. Call model.set_prompts([...]) before forward()."
            )

        feat = self.vision_encoder(images)  # (B, C, H', W')
        B, _, Hp, Wp = feat.shape
        A = self.num_anchors
        C = self.embed_dim
        T = self._text_embeds.shape[0]

        bbox = self.bbox_head(feat)              # (B, 4*A, H', W')
        obj = self.obj_head(feat)                # (B,   A, H', W')
        vis = self.visual_proj(feat)             # (B, C*A, H', W')

        bbox = bbox.view(B, A, 4, Hp, Wp).permute(0, 1, 3, 4, 2).contiguous()  # (B,A,H,W,4)
        obj = obj.view(B, A, Hp, Wp)

        # Visual features per anchor: (B, A, C, H, W) → (B, A, H, W, C)
        vis = vis.view(B, A, C, Hp, Wp).permute(0, 1, 3, 4, 2).contiguous()
        vis_n = F.normalize(vis, dim=-1)  # unit vectors on the C axis

        # Similarity: (B, A, H, W, C) @ (T, C)^T → (B, A, H, W, T)
        cls = torch.einsum("bahwc,tc->bahwt", vis_n, self._text_embeds)
        cls = cls * self.logit_scale.exp()

        stride = self.imgsz / Hp
        return {"bbox": bbox, "obj": obj, "cls": cls, "stride": stride}
