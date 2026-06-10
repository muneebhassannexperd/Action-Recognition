"""
NTU RGB+D 25-joint spatial graph for CTR-GCN.

Joint topology matches graph/ntu_rgb_d.py in the official CTR-GCN repo and
pyskl ``Graph(layout='nturgb+d')``.
"""

from __future__ import annotations

import numpy as np

NUM_NODE = 25

# 1-indexed NTU names for documentation / joint_mapper reference
NTU_JOINT_NAMES = [
    "spine_base",           # 0  (NTU #1)
    "mid_spine",            # 1  (NTU #2)
    "neck",                 # 2  (NTU #3)
    "head",                 # 3  (NTU #4)
    "left_shoulder",        # 4  (NTU #5)
    "left_elbow",           # 5  (NTU #6)
    "left_wrist",           # 6  (NTU #7)
    "left_hand",            # 7  (NTU #8)
    "right_shoulder",       # 8  (NTU #9)
    "right_elbow",          # 9  (NTU #10)
    "right_wrist",          # 10 (NTU #11)
    "right_hand",           # 11 (NTU #12)
    "left_hip",             # 12 (NTU #13)
    "left_knee",            # 13 (NTU #14)
    "left_ankle",           # 14 (NTU #15)
    "left_foot",            # 15 (NTU #16)
    "right_hip",            # 16 (NTU #17)
    "right_knee",           # 17 (NTU #18)
    "right_ankle",          # 18 (NTU #19)
    "right_foot",           # 19 (NTU #20)
    "spine",                # 20 (NTU #21)
    "left_hand_tip",        # 21 (NTU #22)
    "left_thumb",           # 22 (NTU #23)
    "right_hand_tip",       # 23 (NTU #24)
    "right_thumb",          # 24 (NTU #25)
]

_SELF_LINK = [(i, i) for i in range(NUM_NODE)]
_INWARD_ORI = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12),
]
INWARD = [(i - 1, j - 1) for (i, j) in _INWARD_ORI]
OUTWARD = [(j, i) for (i, j) in INWARD]


def _edge2mat(link: list[tuple[int, int]], num_node: int) -> np.ndarray:
    a = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in link:
        a[j, i] = 1.0
    return a


def _normalize_digraph(a: np.ndarray) -> np.ndarray:
    dl = np.sum(a, axis=0)
    n = a.shape[0]
    dn = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if dl[i] > 0:
            dn[i, i] = dl[i] ** (-1)
    return np.dot(a, dn)


def get_spatial_graph(num_node: int = NUM_NODE) -> np.ndarray:
    i_mat = _edge2mat(_SELF_LINK, num_node)
    in_mat = _normalize_digraph(_edge2mat(INWARD, num_node))
    out_mat = _normalize_digraph(_edge2mat(OUTWARD, num_node))
    return np.stack((i_mat, in_mat, out_mat))


class Graph:
    """Spatial adjacency used by the official CTR-GCN Model."""

    def __init__(self, labeling_mode: str = "spatial"):
        self.num_node = NUM_NODE
        self.inward = INWARD
        self.outward = OUTWARD
        self.A = get_spatial_graph(NUM_NODE) if labeling_mode == "spatial" else None
        if self.A is None:
            raise ValueError(f"Unsupported labeling_mode: {labeling_mode}")
