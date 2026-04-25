"""Analyze tracks.csv and compute track-stability statistics."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> int:
    csv_path = Path("validation/tracking/tracks.csv")
    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "frame": int(r["frame"]),
                "track_id": int(r["track_id"]),
                "cls": int(r["class"]),
                "conf": float(r["conf"]),
                "x1": float(r["x1"]), "y1": float(r["y1"]),
                "x2": float(r["x2"]), "y2": float(r["y2"]),
            })

    n_frames = max((r["frame"] for r in rows), default=0)
    n_total = len(rows)

    # Per-track lifetime
    by_track = defaultdict(list)
    for r in rows:
        by_track[r["track_id"]].append(r)

    n_tracks = len(by_track)
    track_lifetimes = [max(t["frame"] for t in trk) - min(t["frame"] for t in trk) + 1 for trk in by_track.values()]
    mean_lifetime = statistics.mean(track_lifetimes) if track_lifetimes else 0
    median_lifetime = statistics.median(track_lifetimes) if track_lifetimes else 0

    # Tracks that lasted at least 30 frames (1 second at 30fps)
    long_tracks = [t for t in track_lifetimes if t >= 30]

    # Per-frame detection counts
    by_frame = defaultdict(int)
    for r in rows:
        by_frame[r["frame"]] += 1
    per_frame_counts = list(by_frame.values())
    mean_dets_per_frame = statistics.mean(per_frame_counts) if per_frame_counts else 0

    # Confidence distribution
    confs = [r["conf"] for r in rows]
    mean_conf = statistics.mean(confs)
    median_conf = statistics.median(confs)

    # Class distribution (COCO class IDs; 0=person)
    class_counts = defaultdict(int)
    for r in rows:
        class_counts[r["cls"]] += 1

    # Track ID gaps — proxy for fragmentation. For each track, count how many
    # frames within its lifetime have NO detection of this track.
    fragmentations = []
    for tid, trk in by_track.items():
        frames = sorted(t["frame"] for t in trk)
        if len(frames) < 2:
            continue
        present = set(frames)
        span = range(frames[0], frames[-1] + 1)
        gaps = sum(1 for f in span if f not in present)
        fragmentations.append(gaps / len(list(span)))
    mean_frag = statistics.mean(fragmentations) if fragmentations else 0

    # Box-area continuity: track_box_area shouldn't change drastically frame-to-frame
    # for stable tracks. Compute coefficient of variation per track.
    cvs = []
    for tid, trk in by_track.items():
        if len(trk) < 5:
            continue
        areas = [(t["x2"] - t["x1"]) * (t["y2"] - t["y1"]) for t in sorted(trk, key=lambda r: r["frame"])]
        m = statistics.mean(areas)
        s = statistics.stdev(areas) if len(areas) > 1 else 0
        if m > 0:
            cvs.append(s / m)
    mean_cv = statistics.mean(cvs) if cvs else 0

    summary = {
        "frames_processed": n_frames,
        "total_tracked_detections": n_total,
        "unique_tracks": n_tracks,
        "mean_dets_per_frame": round(mean_dets_per_frame, 2),
        "tracks_lasting_>=30_frames": len(long_tracks),
        "track_lifetime_frames": {
            "mean": round(mean_lifetime, 1),
            "median": median_lifetime,
            "max": max(track_lifetimes) if track_lifetimes else 0,
            "p90": round(statistics.quantiles(track_lifetimes, n=10)[8], 1) if len(track_lifetimes) >= 10 else 0,
        },
        "mean_track_fragmentation_ratio": round(mean_frag, 4),
        "mean_box_area_cv": round(mean_cv, 3),
        "mean_detection_confidence": round(mean_conf, 3),
        "median_detection_confidence": round(median_conf, 3),
        "top_5_classes_by_count": sorted(class_counts.items(), key=lambda kv: -kv[1])[:5],
    }

    out_path = Path("validation/tracking/analysis.json")
    out_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))

    # Plain-English interpretation
    print("\n--- Interpretation ---")
    print(f"Processed {n_frames} frames at vid_stride=2 (about {n_frames * 2 // 30}s of video).")
    print(f"Detected and tracked {n_tracks} unique objects, {n_total} total tracked detections.")
    print(f"Average lifetime per track: {mean_lifetime:.1f} frames "
          f"(median {median_lifetime} frames). {len(long_tracks)} tracks survived ≥ 30 frames.")
    print(f"Per-track fragmentation: {mean_frag*100:.1f}% (gaps within track span). "
          f"Lower = more contiguous tracks.")
    print(f"Bounding-box area variation per track: {mean_cv*100:.1f}% CV. "
          f"Below 30% means stable size/distance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
