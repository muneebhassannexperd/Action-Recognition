#!/usr/bin/env python3
"""Export YOLO pose keypoints JSONL from a video (for --input-mode keypoints testing)."""

from __future__ import annotations

import argparse
import json
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

from utils.config import OUTPUTS_DIR, TRACKER_CONFIG, resolve_yolo_model

from pose_detector import PoseDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO pose+track on a video and write per-frame keypoints JSONL.",
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: outputs/<stem>_keypoints.jsonl).",
    )
    parser.add_argument("--pose-model", default=None, help="YOLO pose weights (.pt).")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument(
        "--tracker",
        default=TRACKER_CONFIG,
        choices=("bytetrack.yaml", "botsort.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    import cv2

    args = parse_args()
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    base = os.path.splitext(os.path.basename(args.video))[0]
    output_path = args.output or str(OUTPUTS_DIR / f"{base}_keypoints.jsonl")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = PoseDetector(
        model_path=str(resolve_yolo_model(args.pose_model)),
        device=args.device,
        tracker=args.tracker,
    )
    detector.load_model()
    detector.reset_tracker()

    frame_idx = -1
    written = 0
    with open(output_path, "w", encoding="utf-8") as out:
        while True:
            ret, _frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            timestamp = frame_idx / fps
            tracks = detector.track(_frame)

            record = {
                "frame": frame_idx,
                "timestamp": round(timestamp, 6),
                "fps": fps,
                "width": width,
                "height": height,
                "tracks": [
                    {
                        "track_id": t["track_id"],
                        "bbox": t["bbox"],
                        "confidence": t.get("confidence", 1.0),
                        "keypoints": t.get("keypoints") or [],
                    }
                    for t in tracks
                ],
            }
            out.write(json.dumps(record, separators=(",", ":")) + "\n")
            written += 1

    cap.release()
    print(f"Wrote {written} frames to {output_path}")


if __name__ == "__main__":
    main()
