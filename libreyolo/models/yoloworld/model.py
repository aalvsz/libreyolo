"""LibreYOLOWorld — open-vocabulary YOLO user-facing wrapper.

Thin wrapper around `LibreYOLOWorldModel` that hooks into LibreYOLO's
`BaseModel` registry + provides the user-level API:

    model = LibreYOLOWorld(prompts=["person", "dog", "my custom thing"])
    result = model("photo.jpg")
    # result.boxes has the N_prompts classes indexed 0..N-1 matching `prompts`.

For the MVP, `model(image)` returns a standard `Results` object with `boxes`
(class logits argmax'd per location → class index in the prompts list).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from .nn import LibreYOLOWorldModel


class LibreYOLOWorld(BaseModel):
    """Open-vocabulary YOLO. Text-prompted detection.

    Args:
        model_path: Path to weights. Use `None` for random init (MVP scaffold).
        prompts: Initial list of class-name strings.
        imgsz: Input image size (default 640).
        device: Inference device.

    Example::

        >>> model = LibreYOLOWorld(prompts=["person", "dog", "traffic cone"])
        >>> result = model("photo.jpg")
        >>> # result.boxes.cls are indices into the prompts list
    """

    FAMILY = "yoloworld"
    FILENAME_PREFIX = "LibreYOLOWorld"
    INPUT_SIZES = {"s": 640, "m": 640, "l": 640}
    WEIGHT_EXT = ".pt"

    # -----------------------------------------------------------------
    # Registry classmethods
    # -----------------------------------------------------------------

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return any(
            k.startswith("text_encoder.text_model")
            or k.startswith("visual_proj.")
            or k.startswith("vision_encoder.stem")
            for k in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        # Fresh MVP: size is implicit in architecture; return a stable default.
        return "s"

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        # Open-vocab: classes are driven by runtime prompts, not weights.
        return None

    # -----------------------------------------------------------------
    # Init
    # -----------------------------------------------------------------

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        prompts: Optional[List[str]] = None,
        imgsz: int = 640,
        device: str = "auto",
        **kwargs,
    ):
        self._imgsz = imgsz
        super().__init__(
            model_path=model_path,
            size="s",
            nb_classes=len(prompts) if prompts else 1,  # overridden by set_prompts
            device=device,
            **kwargs,
        )
        if model_path is not None and isinstance(model_path, str) and Path(model_path).exists():
            self._load_weights(model_path)
        if prompts:
            self.set_prompts(prompts)
        else:
            # Ship with a small default prompt list so `model(image)` works post-init.
            self.set_prompts(["object"])

    # -----------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------

    def _init_model(self) -> nn.Module:
        return LibreYOLOWorldModel(imgsz=self._imgsz)

    def _get_available_layers(self):  # base class hook
        return {
            "vision_encoder": self.model.vision_encoder,
            "text_encoder": self.model.text_encoder,
            "bbox_head": self.model.bbox_head,
            "obj_head": self.model.obj_head,
            "visual_proj": self.model.visual_proj,
        }

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def set_prompts(self, prompts: List[str]) -> None:
        """Change the open-vocab class list. Embeddings are recomputed once."""
        self.model.set_prompts(prompts)
        self.nb_classes = len(prompts)

    @property
    def prompts(self) -> List[str]:
        return self.model.prompts

    # -----------------------------------------------------------------
    # Inference plumbing (overrides abstract methods from BaseModel)
    # -----------------------------------------------------------------

    @staticmethod
    def _get_preprocess_numpy():
        """Return a numpy-based preprocess function (for video / array inputs)."""
        import numpy as np

        def _preprocess_numpy(arr: "np.ndarray", input_size: int = 640):
            """Letterbox-free resize → normalize → CHW for a single image."""
            import cv2
            img = cv2.resize(arr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
            img = img[:, :, ::-1]  # BGR → RGB if coming from OpenCV
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            return img, 1.0

        return _preprocess_numpy

    def _preprocess(self, image, color_format: str = "auto", input_size: Optional[int] = None):
        from PIL import Image
        import numpy as np

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        assert isinstance(image, Image.Image)
        size = self._imgsz if input_size is None else input_size
        img = image.resize((size, size))
        arr = np.asarray(img).astype("float32") / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor, image, image.size, 1.0

    def _forward(self, input_tensor: torch.Tensor):
        return self.model(input_tensor)

    def _postprocess(
        self,
        output,
        conf_thres: float,
        iou_thres: float,
        original_size,
        max_det: int = 300,
        **kwargs,
    ):
        """Minimal postprocess: per-location argmax over prompts.

        Returns a dict with keys expected by BaseModel._runner (boxes, conf, cls)
        in the same shape as other LibreYOLO families.
        """
        cls_logits = output["cls"]  # (B, A, H, W, T)
        obj_logits = output["obj"]  # (B, A, H, W)
        bbox = output["bbox"]       # (B, A, H, W, 4)
        stride = output["stride"]

        B, A, H, W, T = cls_logits.shape
        assert B == 1, "MVP postprocess assumes batch=1"

        # Score = sigmoid(objectness) * softmax over class logits
        obj = torch.sigmoid(obj_logits)[0]                 # (A, H, W)
        cls_prob = F.softmax(cls_logits[0], dim=-1)        # (A, H, W, T)
        scores_per_loc, cls_idx = cls_prob.max(dim=-1)     # (A, H, W)
        conf = obj * scores_per_loc                         # (A, H, W)

        mask = conf > conf_thres
        if not mask.any():
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "classes": torch.zeros((0,), dtype=torch.float32),
                "num_detections": 0,
            }

        # Convert grid coordinates + bbox regression → xyxy in original image space.
        yv, xv = torch.meshgrid(
            torch.arange(H, device=bbox.device),
            torch.arange(W, device=bbox.device),
            indexing="ij",
        )
        cx = (xv + 0.5) * stride
        cy = (yv + 0.5) * stride
        cx = cx.unsqueeze(0).expand(A, H, W)  # (A, H, W)
        cy = cy.unsqueeze(0).expand(A, H, W)

        # Bbox regression outputs are raw offsets in pixel space (ltrb from center).
        b = bbox[0]  # (A, H, W, 4)
        # Interpret as (dx, dy, dw, dh) in pixel space (MVP — untrained anyway).
        x1 = cx + b[..., 0] - b[..., 2].abs() / 2
        y1 = cy + b[..., 1] - b[..., 3].abs() / 2
        x2 = cx + b[..., 0] + b[..., 2].abs() / 2
        y2 = cy + b[..., 1] + b[..., 3].abs() / 2
        xyxy = torch.stack([x1, y1, x2, y2], dim=-1)  # (A, H, W, 4)

        sel_conf = conf[mask]
        sel_cls = cls_idx[mask].float()
        sel_box = xyxy[mask]

        # Top-K
        if sel_conf.numel() > max_det:
            topk = sel_conf.topk(max_det)
            sel_conf = topk.values
            sel_cls = sel_cls[topk.indices]
            sel_box = sel_box[topk.indices]

        # Rescale from imgsz coords to original image size
        ow, oh = original_size
        scale_x = ow / self._imgsz
        scale_y = oh / self._imgsz
        sel_box = sel_box.clone()
        sel_box[:, [0, 2]] *= scale_x
        sel_box[:, [1, 3]] *= scale_y

        return {
            "boxes": sel_box.detach().cpu().float(),
            "scores": sel_conf.detach().cpu().float(),
            "classes": sel_cls.detach().cpu().float(),
            "num_detections": int(sel_conf.numel()),
        }
