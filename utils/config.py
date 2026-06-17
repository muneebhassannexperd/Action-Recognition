"""
Pipeline configuration for CTR-GCN inference.

CTR-GCN NTU120 joint model expects skeleton tensors shaped (N, C, T, V, M):
  C=3 (x, y, z), T=100 frames, V=25 NTU RGB+D joints, M=2 persons max.
Values are derived from OffTapWatch/pyskl config
``ctrgcn_pyskl_ntu120_xsub_3dkp/j.py`` and the official CTR-GCN repo.
"""
#Work on Crowd fixation
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Model paths (resolve yolo_models alias and legacy typo folder) ---
_YOLO_CANDIDATES = [
    PROJECT_ROOT / "yolo_models",
    PROJECT_ROOT / "yolo_moddels",
]
YOLO_MODEL_DIR = next((p for p in _YOLO_CANDIDATES if p.is_dir()), _YOLO_CANDIDATES[0])
YOLO_DEFAULT_MODEL = YOLO_MODEL_DIR / "yolo11l-pose.pt"
YOLO_FALLBACK_MODEL = YOLO_MODEL_DIR / "yolo11l-pose.pt"

CTRGCN_JOINT_MODEL_DIR = (
    PROJECT_ROOT / "pretrained_model" / "CTRGCN_NTU120_CSub_joint_84.9"
)
CTRGCN_JOINT_WEIGHTS = CTRGCN_JOINT_MODEL_DIR / "runs-58-57072.pt"

POSEC3D_MODEL_DIR = PROJECT_ROOT / "pretrained_model" / "posec3d"
POSEC3D_JOINT_WEIGHTS = POSEC3D_MODEL_DIR / "joint.pth"
POSEC3D_CHECKPOINT_URL = (
    "http://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/"
    "slowonly_r50_ntu120_xsub/joint.pth"
)

LABELS_FILE = PROJECT_ROOT / "data" / "nturgbd_120.txt"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# --- CTR-GCN input spec (from nturgb+d layout, 3dkp joint config) ---
NUM_JOINTS = 25
NUM_PERSONS = 2
CLIP_LEN = 100
IN_CHANNELS = 3
NUM_CLASSES = 120

# PoseC3D SlowOnly R50 NTU120 XSub (COCO-17 keypoints)
POSEC3D_NUM_JOINTS = 17
POSEC3D_CLIP_LEN = 48
POSEC3D_HEATMAP_SIZE = 64
# ``keypoint``: 17 joint heatmaps (COCO-17). ``limb``: matches official NTU120 joint.pth training.
POSEC3D_HEATMAP_MODE = "limb"

ACTION_MODELS = ("ctrgcn", "posec3d")
DEFAULT_ACTION_MODEL = "posec3d"

# Rolling skeleton windows (frames) before resampling to CLIP_LEN / POSEC3D_CLIP_LEN
WINDOW_SIZES = (30, 48, 60, 90, 100, 120)
DEFAULT_WINDOW_SIZE = 30

# Run CTR-GCN every N frames once pair buffer is full
INFERENCE_STRIDE = 15
#TOP_K = 5
TOP_K = 5

# Two-person interaction gating (pixel distance between bbox centers)
INTERACTION_DISTANCE = 300.0
INTERACTION_FRAMES = 15

# Pair validation before M=2 action inference (see src/pair_validator.py)
PAIR_MIN_DURATION_FRAMES = 30
PAIR_IOU_MIN = 0.05
PAIR_CLOSE_CENTER_DISTANCE = 150.0
PAIR_MIN_MEAN_KEYPOINT_CONF = 0.35
PAIR_WRIST_MAX_DISTANCE = 150.0
PAIR_WRIST_IOU_BYPASS = 0.05
PAIR_SCORE_THRESHOLD = 0.5
PAIR_OVERLAP_SCORE_IOU_REF = 0.3
#abc
# Ultralytics tracker config: "bytetrack.yaml" or "botsort.yaml"
TRACKER_CONFIG = "bytetrack.yaml"

# Min confidence for non-Normal cues in JSON events and annotated video overlays.
EVENT_MIN_CONFIDENCE = 0.12
VIOLENCE_MIN_CONFIDENCE = EVENT_MIN_CONFIDENCE  # backward-compatible alias
# Overlay floor (defaults to EVENT_MIN_CONFIDENCE; override in VideoProcessor if needed).
OVERLAY_MIN_CONFIDENCE = EVENT_MIN_CONFIDENCE

YOLO_KEYPOINT_CONF_THRESHOLD = 0.3
POSE_CONF_THRESHOLD = 0.25

# Skeleton motion-energy gate (see src/motion_energy.py). Calibrated on crowd vs push clips.
MOTION_GATE_ENABLED = True
#MOTION_THRESHOLD_PAIR = 0.020
MOTION_THRESHOLD_PAIR = 0.020
#MOTION_THRESHOLD_SINGLE = 0.012
MOTION_THRESHOLD_SINGLE = 0.012
MOTION_KEYPOINT_CONF = 0.25

# --- CLI I/O modes (video vs client keypoints in; report vs behavior_cues out) ---
INPUT_MODES = ("video", "keypoints")
DEFAULT_INPUT_MODE = "video"

OUTPUT_FORMATS = ("report", "behavior_cues")
# Enforced by --input-mode (see src/main.py); do not mix manually.
VIDEO_OUTPUT_FORMAT = "report"
KEYPOINTS_OUTPUT_FORMAT = "behavior_cues"


def output_format_for_input_mode(input_mode: str) -> str:
    if input_mode == "video":
        return VIDEO_OUTPUT_FORMAT
    if input_mode == "keypoints":
        return KEYPOINTS_OUTPUT_FORMAT
    raise ValueError(f"unknown input_mode: {input_mode!r}")

# behavior_cues delivery envelope (see utils/behavior_cues.py)
BEHAVIOR_CUES_MODULE = "action_recognition_pipeline"
BEHAVIOR_CUES_MODULE_VERSION = "0.1.0"
BEHAVIOR_CUES_KEYPOINT_MODEL = "yolo11n-pose"
DEFAULT_CAMERA_ID = 0
DEFAULT_ORGANIZATION_ID = 0


def load_label_map(path: Path | None = None) -> dict[int, str]:
    label_path = path or LABELS_FILE
    labels: dict[int, str] = {}
    with open(label_path, encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            labels[idx] = line.strip()
    return labels


def resolve_yolo_model(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"YOLO model not found: {p}")
        return p
    if YOLO_DEFAULT_MODEL.exists():
        return YOLO_DEFAULT_MODEL
    if YOLO_FALLBACK_MODEL.exists():
        return YOLO_FALLBACK_MODEL
    raise FileNotFoundError(
        f"No YOLO pose weights under {YOLO_MODEL_DIR}. "
        "Expected best.pt or yolo11l-pose.pt."
    )


def resolve_ctrgcn_weights(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"CTR-GCN weights not found: {p}")
        return p
    if not CTRGCN_JOINT_WEIGHTS.exists():
        raise FileNotFoundError(f"CTR-GCN weights not found: {CTRGCN_JOINT_WEIGHTS}")
    return CTRGCN_JOINT_WEIGHTS


def resolve_posec3d_weights(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"PoseC3D weights not found: {p}")
        return p
    if not POSEC3D_JOINT_WEIGHTS.exists():
        raise FileNotFoundError(
            f"PoseC3D weights not found: {POSEC3D_JOINT_WEIGHTS}. "
            f"Download from {POSEC3D_CHECKPOINT_URL}"
        )
    return POSEC3D_JOINT_WEIGHTS
