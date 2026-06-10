"""PoseC3D preprocessing: raw COCO-17 keypoints -> 17-channel pose heatmaps."""

from __future__ import annotations

import numpy as np
import torch

from utils.config import POSEC3D_CLIP_LEN, POSEC3D_HEATMAP_SIZE, POSEC3D_NUM_JOINTS

EPS = 1e-3

COCO_SKELETONS = (
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7), (7, 9),
    (0, 6), (6, 8), (8, 10), (5, 11), (11, 13), (13, 15),
    (6, 12), (12, 14), (14, 16), (11, 12),
)


def uniform_sample_indices(num_frames: int, clip_len: int = POSEC3D_CLIP_LEN) -> np.ndarray:
    """Deterministic uniform frame indices (inference)."""
    if num_frames <= 0:
        return np.zeros(clip_len, dtype=np.int64)
    if num_frames == clip_len:
        return np.arange(clip_len, dtype=np.int64)
    if num_frames < clip_len:
        return np.arange(clip_len, dtype=np.int64) % num_frames
    return np.linspace(0, num_frames - 1, clip_len).astype(np.int64)


def pose_compact(
    keypoint: np.ndarray,
    img_shape: tuple[int, int],
    padding: float = 0.25,
    threshold: int = 10,
    hw_ratio: float | None = 1.0,
    allow_imgpad: bool = True,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Compact keypoints to a tight bounding box (pyskl PoseCompact).

    Args:
        keypoint: (M, T, V, 2)
        img_shape: (H, W) of the source frame
    """
    h, w = img_shape
    kp = keypoint.astype(np.float32, copy=True)
    kp[np.isnan(kp)] = 0.0

    kp_x = kp[..., 0]
    kp_y = kp[..., 1]
    valid_x = kp_x[kp_x != 0]
    valid_y = kp_y[kp_y != 0]
    if valid_x.size == 0 or valid_y.size == 0:
        return kp, (h, w)

    min_x, max_x = float(valid_x.min()), float(valid_x.max())
    min_y, max_y = float(valid_y.min()), float(valid_y.max())
    if max_x - min_x < threshold or max_y - min_y < threshold:
        return kp, (h, w)

    center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
    half_width = (max_x - min_x) / 2 * (1 + padding)
    half_height = (max_y - min_y) / 2 * (1 + padding)

    if hw_ratio is not None:
        half_height = max(hw_ratio * half_width, half_height)
        half_width = max((1 / hw_ratio) * half_height, half_width)

    min_x, max_x = center[0] - half_width, center[0] + half_width
    min_y, max_y = center[1] - half_height, center[1] + half_height

    if not allow_imgpad:
        min_x, min_y = int(max(0, min_x)), int(max(0, min_y))
        max_x, max_y = int(min(w, max_x)), int(min(h, max_y))
    else:
        min_x, min_y = int(min_x), int(min_y)
        max_x, max_y = int(max_x), int(max_y)

    kp_x = kp[..., 0]
    kp_y = kp[..., 1]
    kp_x[kp_x != 0] -= min_x
    kp_y[kp_y != 0] -= min_y
    new_shape = (max(max_y - min_y, 1), max(max_x - min_x, 1))
    return kp, new_shape


def resize_keypoints(
    keypoint: np.ndarray,
    img_shape: tuple[int, int],
    scale: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize keypoint coordinates to target heatmap size (pyskl Resize, keep_ratio=False)."""
    img_h, img_w = img_shape
    new_w, new_h = scale
    scale_factor = np.array([new_w / img_w, new_h / img_h], dtype=np.float32)
    return keypoint * scale_factor, (new_h, new_w)


def _generate_a_heatmap(
    arr: np.ndarray,
    centers: np.ndarray,
    max_values: np.ndarray,
    sigma: float,
) -> None:
    img_h, img_w = arr.shape
    for center, max_value in zip(centers, max_values):
        if max_value < EPS:
            continue
        mu_x, mu_y = center[0], center[1]
        st_x = max(int(mu_x - 3 * sigma), 0)
        ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
        st_y = max(int(mu_y - 3 * sigma), 0)
        ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
        x = np.arange(st_x, ed_x, 1, dtype=np.float32)
        y = np.arange(st_y, ed_y, 1, dtype=np.float32)
        if not (len(x) and len(y)):
            continue
        y = y[:, None]
        patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma**2)
        patch = patch * max_value
        arr[st_y:ed_y, st_x:ed_x] = np.maximum(arr[st_y:ed_y, st_x:ed_x], patch)


def _generate_a_limb_heatmap(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    start_values: np.ndarray,
    end_values: np.ndarray,
    sigma: float,
) -> None:
    img_h, img_w = arr.shape
    for start, end, start_value, end_value in zip(starts, ends, start_values, end_values):
        value_coeff = min(start_value, end_value)
        if value_coeff < EPS:
            continue
        min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
        min_y, max_y = min(start[1], end[1]), max(start[1], end[1])
        min_x = max(int(min_x - 3 * sigma), 0)
        max_x = min(int(max_x + 3 * sigma) + 1, img_w)
        min_y = max(int(min_y - 3 * sigma), 0)
        max_y = min(int(max_y + 3 * sigma) + 1, img_h)
        x = np.arange(min_x, max_x, 1, dtype=np.float32)
        y = np.arange(min_y, max_y, 1, dtype=np.float32)
        if not (len(x) and len(y)):
            continue
        y = y[:, None]
        d2_start = (x - start[0]) ** 2 + (y - start[1]) ** 2
        d2_end = (x - end[0]) ** 2 + (y - end[1]) ** 2
        d2_ab = (start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2
        if d2_ab < 1:
            _generate_a_heatmap(arr, start[None], np.array([start_value]), sigma)
            continue
        coeff = (d2_start - d2_end + d2_ab) / 2.0 / d2_ab
        a_dominate = coeff <= 0
        b_dominate = coeff >= 1
        seg_dominate = 1 - a_dominate - b_dominate
        position = np.stack([x + np.zeros_like(y), y + np.zeros_like(x)], axis=-1)
        projection = start + np.stack([coeff, coeff], axis=-1) * (end - start)
        d2_line = position - projection
        d2_seg = a_dominate * d2_start + b_dominate * d2_end + seg_dominate * (
            d2_line[:, :, 0] ** 2 + d2_line[:, :, 1] ** 2
        )
        patch = np.exp(-d2_seg / 2.0 / sigma**2) * value_coeff
        arr[min_y:max_y, min_x:max_x] = np.maximum(arr[min_y:max_y, min_x:max_x], patch)


def generate_pose_heatmaps(
    keypoint: np.ndarray,
    keypoint_score: np.ndarray,
    img_shape: tuple[int, int],
    *,
    sigma: float = 0.6,
    with_kp: bool = True,
    with_limb: bool = False,
) -> np.ndarray:
    """
    Generate pseudo heatmaps for all frames.

    Args:
        keypoint: (M, T, V, 2)
        keypoint_score: (M, T, V)
        img_shape: (H, W)

    Returns:
        (T, C, H, W) heatmaps
    """
    assert with_kp ^ with_limb, 'Exactly one of with_kp or with_limb must be True'
    m, num_frame, num_kp, _ = keypoint.shape
    img_h, img_w = img_shape
    num_c = num_kp if with_kp else len(COCO_SKELETONS)
    heatmaps = np.zeros((num_frame, num_c, img_h, img_w), dtype=np.float32)

    for frame_idx in range(num_frame):
        kps = keypoint[:, frame_idx]
        scores = keypoint_score[:, frame_idx]
        frame_maps = heatmaps[frame_idx]
        if with_kp:
            for joint_idx in range(num_kp):
                _generate_a_heatmap(
                    frame_maps[joint_idx],
                    kps[:, joint_idx],
                    scores[:, joint_idx],
                    sigma,
                )
        else:
            for limb_idx, (start_idx, end_idx) in enumerate(COCO_SKELETONS):
                _generate_a_limb_heatmap(
                    frame_maps[limb_idx],
                    kps[:, start_idx],
                    kps[:, end_idx],
                    scores[:, start_idx],
                    scores[:, end_idx],
                    sigma,
                )
    return heatmaps


def preprocess_coco_sequence(
    seq: np.ndarray,
    img_shape: tuple[int, int],
    clip_len: int = POSEC3D_CLIP_LEN,
    heatmap_size: int = POSEC3D_HEATMAP_SIZE,
    heatmap_mode: str = "keypoint",
) -> torch.Tensor:
    """
    Convert raw COCO-17 skeleton sequence to PoseC3D model input.

    Args:
        seq: (M, T, V, 3) or (T, V, 3) — x, y, confidence in pixel coordinates
        img_shape: (H, W) of the video frame
        heatmap_mode: ``keypoint`` (17 joint heatmaps) or ``limb`` (17 limb heatmaps)

    Returns:
        Tensor (1, 1, C, T, H, W) with C=17, T=clip_len, H=W=heatmap_size
    """
    if seq.ndim == 3:
        seq = seq[np.newaxis, ...]
    if seq.shape[2] != POSEC3D_NUM_JOINTS or seq.shape[3] != 3:
        raise ValueError(f"Expected (M, T, {POSEC3D_NUM_JOINTS}, 3), got {seq.shape}")

    m, t, v, _ = seq.shape
    indices = uniform_sample_indices(t, clip_len)
    sampled = seq[:, indices].astype(np.float32)
    keypoint = sampled[..., :2]
    keypoint_score = sampled[..., 2]

    keypoint, compact_shape = pose_compact(keypoint, img_shape)
    keypoint, final_shape = resize_keypoints(
        keypoint,
        compact_shape,
        (heatmap_size, heatmap_size),
    )

    with_kp = heatmap_mode == "keypoint"
    with_limb = heatmap_mode == "limb"
    heatmaps = generate_pose_heatmaps(
        keypoint,
        keypoint_score,
        final_shape,
        with_kp=with_kp,
        with_limb=with_limb,
    )

    # (T, C, H, W) -> (1, C, T, H, W) -> (1, 1, C, T, H, W)
    tensor = torch.from_numpy(heatmaps).float()
    tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0).unsqueeze(0)
    return tensor


__all__ = [
    "preprocess_coco_sequence",
    "uniform_sample_indices",
    "generate_pose_heatmaps",
]
