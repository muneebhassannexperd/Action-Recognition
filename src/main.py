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
    BEHAVIOR_CUES_KEYPOINT_MODEL,
    BEHAVIOR_CUES_MODULE,
    BEHAVIOR_CUES_MODULE_VERSION,
    DEFAULT_ACTION_MODEL,
    DEFAULT_CAMERA_ID,
    DEFAULT_INPUT_MODE,
    DEFAULT_ORGANIZATION_ID,
    DEFAULT_WINDOW_SIZE,
    EVENT_MIN_CONFIDENCE,
    INPUT_MODES,
    INTERACTION_DISTANCE,
    INTERACTION_FRAMES,
    MOTION_GATE_ENABLED,
    MOTION_THRESHOLD_PAIR,
    MOTION_THRESHOLD_SINGLE,
    OUTPUTS_DIR,
    POSEC3D_HEATMAP_MODE,
    TRACKER_CONFIG,
    WINDOW_SIZES,
    output_format_for_input_mode,
)

from video_processor import VideoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO Pose + action recognition (CTR-GCN or PoseC3D) on CCTV video",
    )
    parser.add_argument("--video", default=None, help="Path to input video (--input-mode video).")
    parser.add_argument(
        "--input-mode",
        default=DEFAULT_INPUT_MODE,
        choices=INPUT_MODES,
        help="video: MP4 in → annotated MP4 + debug JSON out. "
        "keypoints: client JSONL in → behavior_cues JSONL out.",
    )
    parser.add_argument(
        "--keypoints",
        default=None,
        help="Client keypoints input path (required when --input-mode keypoints).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: outputs/<stem>.json for video, <stem>_cues.jsonl for keypoints).",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=DEFAULT_CAMERA_ID,
        help="camera_id for behavior_cues output.",
    )
    parser.add_argument(
        "--organization-id",
        type=int,
        default=DEFAULT_ORGANIZATION_ID,
        help="organization_id for behavior_cues output.",
    )
    parser.add_argument(
        "--keypoint-model",
        default=BEHAVIOR_CUES_KEYPOINT_MODEL,
        help="keypoint_model label in behavior_cues metadata.",
    )
    parser.add_argument(
        "--module-name",
        default=BEHAVIOR_CUES_MODULE,
        help="metadata.module for behavior_cues output.",
    )
    parser.add_argument(
        "--module-version",
        default=BEHAVIOR_CUES_MODULE_VERSION,
        help="metadata.module_version for behavior_cues output.",
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
        help="Run action model every N frames once skeleton buffer is full.",
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
        help="Min confidence for JSON events (default from config).",
    )
    parser.add_argument(
        "--overlay-min-confidence",
        type=float,
        default=None,
        help="Min confidence for annotated video overlays (default: same as --event-min-confidence).",
    )
    parser.add_argument(
        "--motion-gate",
        action=argparse.BooleanOptionalAction,
        default=MOTION_GATE_ENABLED,
        help="Skip PoseC3D when skeleton motion energy is below threshold.",
    )
    parser.add_argument(
        "--motion-threshold-pair",
        type=float,
        default=None,
        help=f"Min normalized pair motion energy (default: {MOTION_THRESHOLD_PAIR}).",
    )
    parser.add_argument(
        "--motion-threshold-single",
        type=float,
        default=None,
        help=f"Min normalized single-track motion energy (default: {MOTION_THRESHOLD_SINGLE}).",
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
        help="Skip annotated MP4 (video mode only; keypoints mode never writes video).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Fallback FPS for keypoints input when not present in the file.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=1920,
        help="Fallback frame width for keypoints input (PoseC3D heatmap scale).",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=1080,
        help="Fallback frame height for keypoints input (PoseC3D heatmap scale).",
    )
    parser.add_argument(
        "--annotated-output",
        default=None,
        help="Annotated video path (default: outputs/<video_stem>_annotated.mp4).",
    )
    return parser.parse_args()


def _resolve_output_path(args: argparse.Namespace, base: str, output_format: str) -> str:
    if args.output:
        return args.output
    if output_format == "behavior_cues":
        return str(OUTPUTS_DIR / f"{base}_cues.jsonl")
    return str(OUTPUTS_DIR / f"{base}.json")


def main() -> None:
    args = parse_args()

    if args.show_mapping:
        from joint_mapper import JointMapper
        import json
        print(json.dumps(JointMapper.mapping_documentation(), indent=2))
        return

    if args.input_mode == "keypoints":
        if not args.keypoints:
            raise SystemExit("Error: --keypoints is required when --input-mode keypoints.")
        if not os.path.exists(args.keypoints):
            raise FileNotFoundError(f"Keypoints file not found: {args.keypoints}")
        if args.video:
            print("Note: --video is ignored in keypoints mode.")
        if args.annotated_output:
            print("Note: --annotated-output is ignored in keypoints mode (no video input).")
        base = os.path.splitext(os.path.basename(args.keypoints))[0]
        if base.endswith("_keypoints"):
            base = base[: -len("_keypoints")]
    else:
        if not args.video:
            raise SystemExit("Error: --video is required when --input-mode video.")
        if not os.path.exists(args.video):
            raise FileNotFoundError(f"Video not found: {args.video}")
        if args.keypoints:
            print("Note: --keypoints is ignored in video mode.")
        base = os.path.splitext(os.path.basename(args.video))[0]

    output_format = output_format_for_input_mode(args.input_mode)
    output_path = _resolve_output_path(args, base, output_format)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    annotated_path = None
    if args.input_mode == "video" and not args.no_annotated_video:
        annotated_path = args.annotated_output or str(OUTPUTS_DIR / f"{base}_annotated.mp4")

    event_min_conf = (
        args.event_min_confidence
        if args.event_min_confidence is not None
        else args.violence_min_confidence
        if args.violence_min_confidence is not None
        else EVENT_MIN_CONFIDENCE
    )
    overlay_min_conf = (
        args.overlay_min_confidence
        if args.overlay_min_confidence is not None
        else event_min_conf
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
        overlay_min_confidence=overlay_min_conf,
        interaction_distance=args.interaction_distance,
        interaction_frames=args.interaction_frames,
        tracker=args.tracker,
        motion_gate_enabled=args.motion_gate,
        motion_threshold_pair=(
            args.motion_threshold_pair
            if args.motion_threshold_pair is not None
            else MOTION_THRESHOLD_PAIR
        ),
        motion_threshold_single=(
            args.motion_threshold_single
            if args.motion_threshold_single is not None
            else MOTION_THRESHOLD_SINGLE
        ),
        input_mode=args.input_mode,
    )

    process_kwargs = dict(
        output_json_path=output_path,
        output_format=output_format,
        camera_id=args.camera_id,
        organization_id=args.organization_id,
        keypoint_model=args.keypoint_model,
        module=args.module_name,
        module_version=args.module_version,
    )

    if args.input_mode == "keypoints":
        report = processor.process_keypoints(
            args.keypoints,
            default_fps=args.fps or 30.0,
            default_width=args.frame_width,
            default_height=args.frame_height,
            **process_kwargs,
        )
    else:
        report = processor.process(
            args.video,
            annotated_output_path=annotated_path,
            **process_kwargs,
        )
    print(f"\nInput mode: {args.input_mode}")
    print(f"Saved output ({output_format}): {output_path}")
    if annotated_path:
        print(f"Saved annotated video: {annotated_path}")
    if output_format == "behavior_cues":
        event_count = len(report.get("behavior_cues", []))
        print(f"Behavior cues emitted: {event_count}")
    else:
        event_count = len(report.get("events", []))
        print(f"Target events detected (non-Normal): {event_count}")
    print(f"Pair inference segments: {report.get('pair_inference_segments', 0)}")
    print(f"Single inference segments: {report.get('single_inference_segments', 0)}")
    motion = report.get("motion_gate", {})
    if motion:
        print(
            f"Motion gate: pair {motion.get('pair_processed', 0)} processed / "
            f"{motion.get('pair_skipped', 0)} skipped, "
            f"single {motion.get('single_processed', 0)} processed / "
            f"{motion.get('single_skipped', 0)} skipped"
        )


if __name__ == "__main__":
    main()
