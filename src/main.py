#!/usr/bin/env python3
"""CTR-GCN violence-action inference CLI (no training)."""

from __future__ import annotations

import argparse
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _SRC)
sys.path.insert(0, _ROOT)

from utils.config import (
    ACTION_MODELS,
    DEFAULT_ACTION_MODEL,
    DEFAULT_WINDOW_SIZE,
    EVENT_MIN_CONFIDENCE,
    INTERACTION_DISTANCE,
    INTERACTION_FRAMES,
    OUTPUTS_DIR,
    POSEC3D_HEATMAP_MODE,
    TRACKER_CONFIG,
    WINDOW_SIZES,
)

from video_processor import VideoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO Pose + action recognition (CTR-GCN or PoseC3D) on CCTV video",
    )
    parser.add_argument("--video", default=None, help="Path to input video.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: outputs/<video_stem>.json).",
    )
    parser.add_argument("--pose-model", default=None, help="YOLO pose weights (.pt).")
    parser.add_argument("--ctrgcn-weights", default=None, help="CTR-GCN joint weights (.pt).")
    parser.add_argument("--posec3d-weights", default=None, help="PoseC3D NTU120 weights (.pth).")
    parser.add_argument(
        "--action-model",
        default=DEFAULT_ACTION_MODEL,
        choices=ACTION_MODELS,
        help="Action recognition backend: ctrgcn (default) or posec3d.",
    )
    parser.add_argument(
        "--posec3d-heatmap",
        default=POSEC3D_HEATMAP_MODE,
        choices=("keypoint", "limb"),
        help="PoseC3D heatmap type: keypoint (COCO-17 joints) or limb (official joint.pth).",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        choices=WINDOW_SIZES,
        help=f"Skeleton buffer length before resampling to 100 frames. One of {WINDOW_SIZES}.",
    )
    parser.add_argument(
        "--inference-stride",
        type=int,
        default=15,
        help="Run action model every N frames once pair buffer is full.",
    )
    parser.add_argument(
        "--interaction-distance",
        type=float,
        default=INTERACTION_DISTANCE,
        help="Max pixel distance between two people to form an interaction pair.",
    )
    parser.add_argument(
        "--interaction-frames",
        type=int,
        default=INTERACTION_FRAMES,
        help="Consecutive frames within distance before pair inference runs.",
    )
    parser.add_argument(
        "--tracker",
        default=TRACKER_CONFIG,
        choices=("bytetrack.yaml", "botsort.yaml"),
        help="Ultralytics tracker config (ByteTrack or BoT-SORT).",
    )
    parser.add_argument(
        "--event-min-confidence",
        type=float,
        default=None,
        help="Min confidence for non-Normal target class in events (default from config).",
    )
    parser.add_argument(
        "--violence-min-confidence",
        type=float,
        default=None,
        help="Alias for --event-min-confidence.",
    )
    parser.add_argument(
        "--show-mapping",
        action="store_true",
        help="Print COCO-17 -> NTU-25 joint mapping rules and exit.",
    )
    parser.add_argument(
        "--no-annotated-video",
        action="store_true",
        help="Skip writing annotated MP4 (JSON is still saved).",
    )
    parser.add_argument(
        "--annotated-output",
        default=None,
        help="Annotated video path (default: outputs/<video_stem>_annotated.mp4).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.show_mapping:
        from joint_mapper import JointMapper
        import json
        print(json.dumps(JointMapper.mapping_documentation(), indent=2))
        return

    if not args.video:
        raise SystemExit("Error: --video is required unless using --show-mapping.")

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    base = os.path.splitext(os.path.basename(args.video))[0]
    output_path = args.output or str(OUTPUTS_DIR / f"{base}.json")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    annotated_path = None
    if not args.no_annotated_video:
        annotated_path = args.annotated_output or str(OUTPUTS_DIR / f"{base}_annotated.mp4")

    event_min_conf = (
        args.event_min_confidence
        if args.event_min_confidence is not None
        else args.violence_min_confidence
        if args.violence_min_confidence is not None
        else EVENT_MIN_CONFIDENCE
    )

    processor = VideoProcessor(
        pose_model_path=args.pose_model,
        action_model=args.action_model,
        ctrgcn_weights_path=args.ctrgcn_weights,
        posec3d_weights_path=args.posec3d_weights,
        posec3d_heatmap_mode=args.posec3d_heatmap,
        device=args.device,
        window_size=args.window_size,
        inference_stride=args.inference_stride,
        event_min_confidence=event_min_conf,
        interaction_distance=args.interaction_distance,
        interaction_frames=args.interaction_frames,
        tracker=args.tracker,
    )

    report = processor.process(
        args.video,
        output_json_path=output_path,
        annotated_output_path=annotated_path,
    )
    print(f"\nSaved JSON: {output_path}")
    if annotated_path:
        print(f"Saved annotated video: {annotated_path}")
    print(f"Target events detected (non-Normal): {len(report['events'])}")
    print(f"Total pair inference segments: {len(report['raw_predictions'])}")


if __name__ == "__main__":
    main()
