"""
Map YOLO COCO-17 pose keypoints to NTU RGB+D 25-joint skeleton format.

CTR-GCN NTU120 joint weights expect V=25 vertices (layout ``nturgb+d``).
YOLO11 pose outputs 17 COCO keypoints per person.

Reference layouts:
  - NTU: src/models/ntu_graph.py (official CTR-GCN graph/ntu_rgb_d.py)
  - COCO: ultralytics pose, pyskl Graph(layout='coco') in OffTapWatch
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.ntu_graph import NTU_JOINT_NAMES, NUM_NODE

# Ultralytics YOLO pose keypoint order (COCO 17)
YOLO_KEYPOINT_NAMES = [
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
]

COCO = {name: idx for idx, name in enumerate(YOLO_KEYPOINT_NAMES)}


def _kp_xy_conf(keypoints: list[list[float]], idx: int) -> np.ndarray:
    """Return (x, y, conf) for COCO index; zeros if missing."""
    if idx < 0 or idx >= len(keypoints):
        return np.zeros(3, dtype=np.float32)
    pt = keypoints[idx]
    if len(pt) < 2:
        return np.zeros(3, dtype=np.float32)
    conf = float(pt[2]) if len(pt) > 2 else 1.0
    return np.array([float(pt[0]), float(pt[1]), conf], dtype=np.float32)


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = (a + b) / 2.0
    out[2] = min(a[2], b[2])
    return out


def _extrapolate_foot(ankle: np.ndarray, knee: np.ndarray, scale: float = 0.25) -> np.ndarray:
    """Foot joint lies beyond ankle along ankle-knee direction (NTU has no COCO foot)."""
    direction = ankle[:2] - knee[:2]
    norm = np.linalg.norm(direction)
    if norm < 1e-3:
        return ankle.copy()
    foot = ankle.copy()
    foot[:2] = ankle[:2] + (direction / norm) * (norm * scale)
    foot[2] = min(ankle[2], knee[2])
    return foot


def _extrapolate_hand_tip(wrist: np.ndarray, elbow: np.ndarray, scale: float = 0.35) -> np.ndarray:
    """NTU hand-tip joints (#22, #24) extend past the wrist."""
    direction = wrist[:2] - elbow[:2]
    norm = np.linalg.norm(direction)
    if norm < 1e-3:
        return wrist.copy()
    tip = wrist.copy()
    tip[:2] = wrist[:2] + (direction / norm) * (norm * scale)
    tip[2] = min(wrist[2], elbow[2])
    return tip


def _thumb_offset(wrist: np.ndarray, elbow: np.ndarray, side: str) -> np.ndarray:
    """Approximate thumb (#23, #25) with a small perpendicular offset from wrist."""
    direction = wrist[:2] - elbow[:2]
    norm = np.linalg.norm(direction)
    thumb = wrist.copy()
    if norm < 1e-3:
        return thumb
    perp = np.array([-direction[1], direction[0]]) / norm
    sign = 1.0 if side == "left" else -1.0
    thumb[:2] = wrist[:2] + perp * (norm * 0.12 * sign)
    thumb[2] = wrist[2]
    return thumb


# Documented NTU index -> source rule (for maintainers)
NTU_MAPPING_RULES: dict[int, str] = {
    0: "midpoint(left_hip, right_hip) -> spine_base",
    1: "midpoint(spine_base, neck) -> mid_spine",
    2: "midpoint(left_shoulder, right_shoulder) -> neck",
    3: "nose -> head",
    4: "left_shoulder",
    5: "left_elbow",
    6: "left_wrist",
    7: "left_wrist -> left_hand (no palm in COCO)",
    8: "right_shoulder",
    9: "right_elbow",
    10: "right_wrist",
    11: "right_wrist -> right_hand",
    12: "left_hip",
    13: "left_knee",
    14: "left_ankle",
    15: "extrapolate(left_ankle, left_knee) -> left_foot",
    16: "right_hip",
    17: "right_knee",
    18: "right_ankle",
    19: "extrapolate(right_ankle, right_knee) -> right_foot",
    20: "midpoint(spine_base, neck) -> spine",
    21: "extrapolate(left_wrist, left_elbow) -> left_hand_tip",
    22: "thumb_offset(left_wrist, left_elbow) -> left_thumb",
    23: "extrapolate(right_wrist, right_elbow) -> right_hand_tip",
    24: "thumb_offset(right_wrist, right_elbow) -> right_thumb",
}


class JointMapper:
    """
    Convert one frame of YOLO pose keypoints to NTU RGB+D layout.

    Output shape per frame: (25, 3) as (x, y, confidence).
    Channel 3 is kept as keypoint confidence for PreNormalize2D masking;
  z is set to 0 when building the 3D tensor for CTR-GCN (see preprocess).
    """

    num_joints = NUM_NODE
    yolo_joints = len(YOLO_KEYPOINT_NAMES)
    ntu_names = NTU_JOINT_NAMES

    def map_frame(self, keypoints: list[list[float]]) -> np.ndarray:
        """
        Parameters
        ----------
        keypoints : list of [x, y, conf] length 17 from YOLO pose.

        Returns
        -------
        np.ndarray
            Shape (25, 3) NTU skeleton in image coordinates.
        """
        ls = _kp_xy_conf(keypoints, COCO["left_shoulder"])
        rs = _kp_xy_conf(keypoints, COCO["right_shoulder"])
        le = _kp_xy_conf(keypoints, COCO["left_elbow"])
        re = _kp_xy_conf(keypoints, COCO["right_elbow"])
        lw = _kp_xy_conf(keypoints, COCO["left_wrist"])
        rw = _kp_xy_conf(keypoints, COCO["right_wrist"])
        lh = _kp_xy_conf(keypoints, COCO["left_hip"])
        rh = _kp_xy_conf(keypoints, COCO["right_hip"])
        lk = _kp_xy_conf(keypoints, COCO["left_knee"])
        rk = _kp_xy_conf(keypoints, COCO["right_knee"])
        la = _kp_xy_conf(keypoints, COCO["left_ankle"])
        ra = _kp_xy_conf(keypoints, COCO["right_ankle"])
        nose = _kp_xy_conf(keypoints, COCO["nose"])

        spine_base = _midpoint(lh, rh)
        neck = _midpoint(ls, rs)
        mid_spine = _midpoint(spine_base, neck)
        spine = _midpoint(spine_base, neck)

        ntu = np.zeros((NUM_NODE, 3), dtype=np.float32)
        ntu[0] = spine_base
        ntu[1] = mid_spine
        ntu[2] = neck
        ntu[3] = nose
        ntu[4] = ls
        ntu[5] = le
        ntu[6] = lw
        ntu[7] = lw
        ntu[8] = rs
        ntu[9] = re
        ntu[10] = rw
        ntu[11] = rw
        ntu[12] = lh
        ntu[13] = lk
        ntu[14] = la
        ntu[15] = _extrapolate_foot(la, lk)
        ntu[16] = rh
        ntu[17] = rk
        ntu[18] = ra
        ntu[19] = _extrapolate_foot(ra, rk)
        ntu[20] = spine
        ntu[21] = _extrapolate_hand_tip(lw, le)
        ntu[22] = _thumb_offset(lw, le, "left")
        ntu[23] = _extrapolate_hand_tip(rw, re)
        ntu[24] = _thumb_offset(rw, re, "right")
        return ntu

    def map_sequence(self, keypoint_frames: list[list[list[float]]]) -> np.ndarray:
        """Stack frames -> (T, 25, 3)."""
        if not keypoint_frames:
            return np.zeros((0, NUM_NODE, 3), dtype=np.float32)
        return np.stack([self.map_frame(kp) for kp in keypoint_frames], axis=0)

    @staticmethod
    def mapping_documentation() -> dict[str, Any]:
        return {
            "yolo_format": "COCO 17 (ultralytics pose)",
            "ntu_format": "NTU RGB+D 25 (nturgb+d)",
            "ntu_joint_names": NTU_JOINT_NAMES,
            "yolo_joint_names": YOLO_KEYPOINT_NAMES,
            "rules_by_ntu_index": NTU_MAPPING_RULES,
        }


__all__ = ["JointMapper", "NTU_MAPPING_RULES", "YOLO_KEYPOINT_NAMES"]
