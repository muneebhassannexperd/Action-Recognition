"""
Pipeline configuration for CTR-GCN inference.

CTR-GCN NTU120 joint model expects skeleton tensors shaped (N, C, T, V, M):
  C=3 (x, y, z), T=100 frames, V=25 NTU RGB+D joints, M=2 persons max.
Values are derived from OffTapWatch/pyskl config
``ctrgcn_pyskl_ntu120_xsub_3dkp/j.py`` and the official CTR-GCN repo.
"""

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
POSEC3D_HEATMAP_MODE = "keypoint"

ACTION_MODELS = ("ctrgcn", "posec3d")
DEFAULT_ACTION_MODEL = "posec3d"

# Rolling skeleton windows (frames) before resampling to CLIP_LEN / POSEC3D_CLIP_LEN
WINDOW_SIZES = (30, 48, 60, 90, 100, 120)
DEFAULT_WINDOW_SIZE = 30

# Run CTR-GCN every N frames once pair buffer is full
INFERENCE_STRIDE = 25
#TOP_K = 5
TOP_K = 10

# Two-person interaction gating (pixel distance between bbox centers)
INTERACTION_DISTANCE = 250.0
INTERACTION_FRAMES = 30

# Ultralytics tracker config: "bytetrack.yaml" or "botsort.yaml"
TRACKER_CONFIG = "bytetrack.yaml"

# Min confidence for non-Normal target events (see utils/class_mapping.py)

EVENT_MIN_CONFIDENCE = 0.25
VIOLENCE_MIN_CONFIDENCE = EVENT_MIN_CONFIDENCE  # backward-compatible alias

# Single-track fall -> person_falls_after_contact if same track had a pair
# interaction event (push_shove, swing_attempt, grapple_clinch) within this window.
FALL_CONTACT_WINDOW_SECONDS = 2.0

YOLO_KEYPOINT_CONF_THRESHOLD = 0.3
POSE_CONF_THRESHOLD = 0.25


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
