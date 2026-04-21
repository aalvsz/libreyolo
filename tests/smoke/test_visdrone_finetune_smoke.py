"""Offline smoke test for the VisDrone fine-tune pipeline.

The full fine-tune requires downloading VisDrone (~3GB). This smoke test
validates the **unique parts** of the pipeline — the VisDrone-to-YOLO
annotation converter — and a minimal fine-tune on synthetic data to confirm
that `LibreYOLO9.train()` runs end-to-end on CPU.

Run: pytest tests/smoke/test_visdrone_finetune_smoke.py -v -m smoke
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

# Make the script importable without installing scripts/ as a package
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from finetune_yolo9_visdrone import (  # noqa: E402
    VISDRONE_CLASSES,
    visdrone_line_to_yolo,
    convert_visdrone_split,
    build_yolo_dataset,
)

from libreyolo.models.yolo9.model import LibreYOLO9
from libreyolo.models.yolo9.nn import LibreYOLO9Model


pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# Converter tests — independent of LibreYOLO, very fast
# ---------------------------------------------------------------------------


def test_class_list_has_ten_entries():
    assert len(VISDRONE_CLASSES) == 10
    assert "pedestrian" in VISDRONE_CLASSES and "motor" in VISDRONE_CLASSES


@pytest.mark.parametrize(
    "line,expect_none",
    [
        # VisDrone category 0 = ignored-regions, should be skipped
        ("10,10,20,20,1,0,0,0", True),
        # VisDrone category 11 = others, should be skipped
        ("10,10,20,20,1,11,0,0", True),
        # Zero width — skipped
        ("10,10,0,20,1,3,0,0", True),
        # Valid pedestrian (category 1) -> yolo class 0
        ("100,200,50,80,1,1,0,0", False),
    ],
)
def test_visdrone_line_to_yolo_skips_and_parses(line, expect_none):
    out = visdrone_line_to_yolo(line, img_w=1920, img_h=1080)
    if expect_none:
        assert out is None
    else:
        assert out is not None
        parts = out.split()
        assert len(parts) == 5
        cls, cx, cy, w, h = parts
        # For pedestrian (VisDrone cat=1), YOLO class should be 0
        assert cls == "0"
        # Coords must normalize to (0, 1)
        for v in (cx, cy, w, h):
            f = float(v)
            assert 0.0 < f < 1.0


def test_visdrone_line_to_yolo_clamps_edge_bboxes():
    # Bbox extends beyond image bounds — should clamp to (0, 1) and still return a line
    out = visdrone_line_to_yolo("1910,1070,100,100,1,2,0,0", img_w=1920, img_h=1080)
    assert out is not None
    parts = out.split()
    cx, cy, w, h = [float(v) for v in parts[1:]]
    assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
    assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0


def test_visdrone_line_to_yolo_malformed():
    # Missing fields — returns None
    assert visdrone_line_to_yolo("10,20", 1920, 1080) is None
    assert visdrone_line_to_yolo("", 1920, 1080) is None
    # Non-numeric — returns None
    assert visdrone_line_to_yolo("a,b,c,d,e,f,g,h", 1920, 1080) is None


# ---------------------------------------------------------------------------
# Split-conversion test: write synthetic VisDrone-style files, assert we
# produce well-formed YOLO labels.
# ---------------------------------------------------------------------------


def test_convert_visdrone_split_writes_labels(tmp_path):
    imgs_dir = tmp_path / "images"
    anns_dir = tmp_path / "annotations"
    out_dir = tmp_path / "labels"
    imgs_dir.mkdir(); anns_dir.mkdir()

    # Build 2 synthetic images at 1920x1080 (VisDrone-ish)
    for idx, cat in enumerate([1, 5]):
        Image.fromarray(np.zeros((1080, 1920, 3), dtype=np.uint8)).save(imgs_dir / f"{idx}.jpg")
        (anns_dir / f"{idx}.txt").write_text(
            # One valid line and one ignored-region line
            f"100,200,50,80,1,{cat},0,0\n"
            f"0,0,10,10,1,0,0,0\n"  # ignored — should be dropped
        )

    summary = convert_visdrone_split(imgs_dir, anns_dir, out_dir)
    assert summary["images"] == 2
    assert summary["labels_written"] == 2  # one per image, ignored line dropped
    assert summary["skipped_lines"] == 2   # both ignored lines

    # Check the written YOLO label files
    for idx in (0, 1):
        lines = (out_dir / f"{idx}.txt").read_text().strip().splitlines()
        assert len(lines) == 1
        cls, cx, cy, w, h = lines[0].split()
        # VisDrone cat 1 -> yolo 0, cat 5 -> yolo 4
        expected = {0: "0", 1: "4"}[idx]
        assert cls == expected


def test_build_yolo_dataset_layout(tmp_path):
    """End-to-end converter test: fake VisDrone dir → data.yaml + label files."""
    vd = tmp_path / "VisDrone"
    for split_suffix in ("DET-train", "DET-val"):
        split_dir = vd / f"VisDrone2019-{split_suffix}"
        (split_dir / "images").mkdir(parents=True)
        (split_dir / "annotations").mkdir(parents=True)
        Image.fromarray(np.zeros((720, 1280, 3), dtype=np.uint8)).save(
            split_dir / "images" / "a.jpg"
        )
        (split_dir / "annotations" / "a.txt").write_text(
            "50,50,200,300,1,4,0,0\n"  # car → yolo class 3
        )

    out = tmp_path / "yolo_fmt"
    data_yaml = build_yolo_dataset(vd, out)
    assert data_yaml.exists()

    data = yaml.safe_load(data_yaml.read_text())
    assert data["nc"] == 10
    assert data["names"][3] == "car"

    # Both splits have the label file
    for split in ("train", "val"):
        label = (out / "labels" / split / "a.txt").read_text().strip()
        cls = label.split()[0]
        assert cls == "3"  # car


# ---------------------------------------------------------------------------
# Minimal fine-tune smoke: uses the same trainer path as VisDrone, but on
# 2 synthetic images so it runs in under 3 seconds on CPU.
# ---------------------------------------------------------------------------


def _tiny_dataset(root: Path) -> Path:
    (root / "images/train").mkdir(parents=True, exist_ok=True)
    (root / "images/val").mkdir(parents=True, exist_ok=True)
    (root / "labels/train").mkdir(parents=True, exist_ok=True)
    (root / "labels/val").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for split in ("train", "val"):
        for i in range(2):
            img = rng.integers(60, 110, size=(128, 128, 3), dtype=np.uint8)
            img[24:64, 24:64] = 220
            Image.fromarray(img).save(root / "images" / split / f"{i}.jpg", quality=80)
            (root / "labels" / split / f"{i}.txt").write_text(
                f"0 0.34375 0.34375 0.3125 0.3125\n"
            )
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.dump({
        "path": str(root), "train": "images/train", "val": "images/val",
        "nc": 1, "names": ["obj"],
    }))
    return data_yaml


def test_finetune_training_runs_on_synthetic_data(tmp_path):
    """Smoke: 1-epoch fine-tune on synthetic data, assert convergence plumbing works."""
    data_yaml = _tiny_dataset(tmp_path / "data")

    # Save a fresh detection checkpoint and load via the wrapper
    ckpt = tmp_path / "yolo9t-init.pt"
    net = LibreYOLO9Model(config="t", nb_classes=1)
    torch.save({"model": net.state_dict()}, ckpt)

    model = LibreYOLO9(model_path=str(ckpt), size="t", nb_classes=1, device="cpu")
    results = model.train(
        data=str(data_yaml),
        epochs=1, batch=2, imgsz=128, lr0=0.01, optimizer="SGD",
        device="cpu", workers=0, amp=False, patience=1,
        project=str(tmp_path / "runs"), name="visdrone_smoke",
    )
    assert "final_loss" in results
    assert math.isfinite(results["final_loss"])
    assert Path(results["last_checkpoint"]).exists()
