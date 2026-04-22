# Tracking Pipeline — Finish-The-Last-Mile Design

**Branch:** `agentic/tracking-pipeline` (based on upstream `45-add-support-for-tracking-algorithms`)
**Date:** 2026-04-22
**Author:** ml-intern (agentic)

## Background

LibreYOLO issue #45 shipped ~80% of a ByteTrack-style multi-object tracker on
the upstream branch: `KalmanFilterXYAH` (xyah + constant velocity), `STrack`
lifecycle states, an `iou_distance`/`linear_assignment`/`fuse_score` matching
module, the main `ByteTracker.update()` three-stage association loop, and a
`model.track()` video generator wired through `run_video_inference`. Unit
tests (`tests/unit/test_tracking.py`, 214 passing) and a GPU-only e2e harness
(`tests/e2e/test_tracking.py`, guarded by `requires_cuda`) are in place.

What's missing is the ml-intern artifact layer: CPU-only smoke tests (so the
pipeline can be validated without weights or GPU), a CLI inference script, a
laptop-runnable tutorial notebook, an HF-Jobs preflight script, and a blog
narrative.

## Deliverables (this branch only)

1. **`tests/smoke/test_tracking_smoke.py`** — CPU-only, <60 s, no weight
   downloads. Covers:
   - Single-object ID persistence across synthetic frames.
   - ID switch handling under full occlusion across > `track_buffer` frames.
   - `TrackConfig` round-trip serialization (dataclasses → dict → dataclass).
   - Tracker reset and multi-instance independence.
   - Low-confidence recovery path (BYTE's second association stage).
   - Gated on upstream variants: a BoT-SORT test is included only if a
     `BoTSORTTracker` class is exposed. Upstream ships ByteTrack only, so the
     BoT-SORT slot is filled by an **algorithm-variant parametrisation** over
     config knobs (`fuse_score`, `match_thresh`) that emulate BoT-SORT's
     stricter matching regime. Blog post documents the decision.

2. **`scripts/track_yolo_video.py`** — CLI that takes `--source <video>` (or
   `--synthesize` to skip the weight download), runs a LibreYOLO model +
   ByteTracker, writes an annotated MP4 and a per-frame CSV with
   `frame_idx, track_id, x1, y1, x2, y2, conf, cls, class_name`. Includes
   `--help`.

3. **`notebooks/tracking_tutorial.ipynb`** — laptop CPU notebook.  Builds a
   synthetic "two blobs moving across frames" corpus, runs the tracker via
   the raw API (no detector needed), and plots ID trajectories. Validates
   with `json.load` before commit.

4. **`docs/agentic-features/blog/tracking-pipeline.md`** — narrative:
   what upstream shipped, what the smoke tests cover, any bugs found and
   fixed, CLI demo, recipe for real fine-tuning on MOT17-style data.

5. **`scripts/hf_jobs_tracking_smoke.py`** — mirror of
   `agentic/c-visdrone-finetune`'s UV-script-style HF Jobs preflight.

## Phases

- **Phase 1 (this session, Max-funded):** all five artifacts + smoke tests
  passing locally. Full unit-test suite stays at 214 passing.
- **Phase 2 (gated on HF Jobs credit):** submit `hf_jobs_tracking_smoke.py`
  on `cpu-basic` to prove the clone+install+smoke-test path works on HF
  compute. Skipped by default — blog documents the command.

## Success criteria

- `pytest tests/smoke/test_tracking_smoke.py -v -m smoke` — all green, < 60 s.
- `pytest tests/unit -q` — 214 passing, 8 pre-existing failures that are
  unrelated (RFDETR optional dep missing in CI).
- `python scripts/track_yolo_video.py --help` — CLI parses.
- `python -c "import json; json.load(open('notebooks/tracking_tutorial.ipynb'))"` — valid.

## Out of scope

- Real training of a detection backbone.
- Publishing weights to HF Hub (tracker is deterministic — no weights to
  publish beyond the existing detector checkpoints).
- Re-implementing ByteTrack from scratch — upstream is solid.
- Rewriting the `model.track()` video loop.
