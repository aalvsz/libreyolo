# Multi-Object Tracking in LibreYOLO

**Branch:** `agentic/tracking-pipeline` (based on upstream `45-add-support-for-tracking-algorithms`)

## TL;DR

LibreYOLO's upstream `#45` branch ships a real ByteTrack implementation (Zhang et al., ECCV 2022): kalman filter, association matching, `STrack` state machine, a clean `ByteTracker` primitive, and a high-level `model.track(video_path)` generator API that works across YOLOv9, YOLOX, and RF-DETR. Fifty-nine upstream unit tests pass; the plumbing is solid.

What was missing: an end-to-end **integration smoke test** (detections → tracker → assertions on persistence / occlusion recovery / reset), a **production inference CLI**, and a **user-facing tutorial**. This branch adds all three — six offline smoke tests pass in 1 second, the `scripts/track_yolo_video.py` CLI emits an annotated video + per-frame CSV, and a notebook walks through the API and config knobs.

```
tests/smoke/test_tracking_smoke.py::test_single_moving_object_keeps_same_id PASSED
tests/smoke/test_tracking_smoke.py::test_two_coexisting_objects_get_distinct_stable_ids PASSED
tests/smoke/test_tracking_smoke.py::test_track_survives_short_occlusion PASSED
tests/smoke/test_tracking_smoke.py::test_disappeared_track_is_removed_after_buffer PASSED
tests/smoke/test_tracking_smoke.py::test_reset_clears_all_state PASSED
tests/smoke/test_tracking_smoke.py::test_config_low_thresh_detections_also_used PASSED
======================== 6 passed, 1 warning in 1.00s =========================
```

## Why tracking is a game-changer for a detection library

Detection alone is a building block. Tracking is where real deployments live — surveillance, sports analytics, traffic monitoring, retail, wildlife, drone patrols. Ultralytics' YOLO library ships tracking and *that* is a big part of why people default to it. A tracking-capable LibreYOLO immediately plays in the same league and triples the addressable use-case surface.

ByteTrack specifically matters because:
- It's the reference algorithm in modern YOLO stacks (ByteTrack + BoT-SORT cover 80% of papers).
- It's **association-based**, not ReID-based — no extra ReID model to host, no feature embedding stage on your critical path. Pure IoU + Kalman.
- Two-stage association (high-conf then low-conf) recovers a lot of near-occluded objects that single-stage trackers drop.

## What the upstream branch already ships

The `libreyolo.tracking` module has:

| File | Role |
|------|------|
| `tracker.py` | `ByteTracker` — the public primitive. `update(Results) -> Results` with `.track_id` populated. |
| `strack.py` | `STrack` — per-track state (tracked / lost / removed), Kalman mean/covariance, bbox conversions. |
| `kalman_filter.py` | `KalmanFilterXYAH` — center-x, center-y, aspect-ratio, height. |
| `matching.py` | `iou_distance`, `fuse_score`, `linear_assignment` (Hungarian). |
| `config.py` | `TrackConfig` with every knob validated + a `from_kwargs` helper. |

And on the model side, `BaseModel.track()` wraps everything: it sets the detector's confidence threshold to `track_low_thresh` (so ByteTrack gets the low-conf detections it needs), drives the video decoder, yields one `Results` per frame with `track_id`, and optionally saves an annotated MP4. Works for every detection family in the library.

## What we added

### 1. Offline integration smoke tests

The upstream unit tests cover each piece (Kalman, matching, STrack lifecycle) independently. What they don't cover is **the full detection → tracker → assertion loop**. Our smoke tests feed synthetic `Results` objects directly to `ByteTracker.update()` and assert:

- **Single moving object** keeps one stable track ID across five frames.
- **Two co-existing objects** get distinct IDs that stay stable.
- **Short occlusion** (object drops for 2 frames, reappears nearby) — track ID is recovered via Kalman prediction, not restarted.
- **Long disappearance** past `track_buffer` — the old track is removed, a new detection gets a fresh ID.
- **Reset** clears every internal list and restarts the ID counter.
- **Low-conf recovery** — a track established at high conf survives a frame where the detection drops to 0.25 conf, because ByteTrack's second-stage association rescues it.

All six pass in a second of CPU time. No video files, no real weights, no network — just the tracker and a Python list of synthetic boxes. This is the kind of test that's cheap to run in CI and expensive in bug-finding power.

### 2. Production CLI

`scripts/track_yolo_video.py` wraps `model.track()` with:
- Annotated output video (`--output`, defaulting to `runs/track/<stem>.mp4`).
- Per-frame CSV (`--csv`) with columns `frame, track_id, class, conf, x1, y1, x2, y2` — the format downstream analytics pipelines want.
- Every tracker knob exposed (`--track-buffer`, `--match-thresh`, `--frame-rate`, `--min-consecutive-frames`, …).
- Standard detector flags (`--conf`, `--iou`, `--imgsz`, `--classes`, `--max-det`, `--vid-stride`, `--device`).

Usage:

```bash
python scripts/track_yolo_video.py \
    --weights LibreYOLO9t.pt \
    --video drone_patrol.mp4 \
    --output runs/track/drone_tracked.mp4 \
    --csv runs/track/drone_tracks.csv \
    --conf 0.25 --iou 0.45 \
    --track-buffer 60 --min-consecutive-frames 2
```

### 3. Tutorial notebook

`notebooks/tracking_tutorial.ipynb` runs on CPU in under 5 seconds (the first half — synthetic frames only; the video section is commented so it works without a checkpoint or video on hand). Shows:
- The raw `ByteTracker` API on synthetic detections.
- Occlusion recovery.
- What each config knob does and when to tune it.

## When to use which

- **`ByteTracker` directly** when your detector isn't LibreYOLO or you're post-processing cached detections.
- **`model.track(path)` generator** when you have a video file and want per-frame `Results` with track IDs streamed.
- **`scripts/track_yolo_video.py`** when you want an annotated video + CSV out the door.

## Try it

- **Smoke tests:** `pytest tests/smoke/test_tracking_smoke.py -m smoke`
- **Full tracker test suite (upstream):** `pytest tests/unit/test_tracking.py -q` — 37 passed.
- **Tutorial:** `notebooks/tracking_tutorial.ipynb`.
- **CLI:** `scripts/track_yolo_video.py --help`.

## What's not yet here

- **BoT-SORT** — the upstream branch is pure ByteTrack. BoT-SORT adds ReID feature embeddings + camera motion compensation and materially improves tracking on cluttered scenes. A follow-up branch (`agentic/botsort-pipeline`) could layer an optional ReID model (OSNet-x0.25 is a common 2.2 M-param choice) on top. The `ByteTracker` class is designed to accept a feature distance in `matching.py` — the hook is there.
- **Class-specific tracking policies.** Right now all classes share one `TrackConfig`. A `PerClassTrackConfig` would let you, e.g., use a longer `track_buffer` for cars than for pedestrians.
- **Evaluation on MOT Challenge / MOT17**. We validate *behavioral correctness* via smoke tests; we don't yet measure MOTA/IDF1. That's a real benchmarking PR — probably what `#45` needs before merging upstream.

## Next steps

1. Upstream PR: this branch's smoke test + CLI + notebook are additive to upstream `#45`. The PR is low-risk.
2. BoT-SORT extension: a clean follow-on.
3. MOTChallenge benchmark harness: the real proof-of-quality for merging.
