"""Geometric and pose-quality validation before pair (M=2) action inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# YOLO COCO-17 wrist indices
LEFT_WRIST = 9
RIGHT_WRIST = 10


@dataclass(frozen=True)
class PairValidationConfig:
    min_duration_frames: int = 45
    min_iou: float = 0.05
    close_center_distance_px: float = 120.0
    min_mean_keypoint_conf: float = 0.4
    wrist_max_distance_px: float = 150.0
    wrist_iou_bypass: float = 0.05
    pair_score_threshold: float = 0.5
    interaction_distance_px: float = 250.0
    proximity_weight: float = 0.4
    wrist_weight: float = 0.3
    overlap_weight: float = 0.3
    overlap_score_iou_ref: float = 0.3


@dataclass
class PairValidationResult:
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    pair_score: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def center_distance_px(a: list[float], b: list[float]) -> float:
    ca = bbox_center(a)
    cb = bbox_center(b)
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))


def mean_keypoint_confidence(keypoints: list[list[float]]) -> float:
    confs: list[float] = []
    for pt in keypoints:
        if len(pt) >= 3 and pt[2] > 0.0:
            confs.append(float(pt[2]))
    if not confs:
        return 0.0
    return float(sum(confs) / len(confs))


def _wrist_points(keypoints: list[list[float]]) -> list[tuple[float, float]]:
    wrists: list[tuple[float, float]] = []
    for idx in (LEFT_WRIST, RIGHT_WRIST):
        if idx < len(keypoints):
            pt = keypoints[idx]
            if len(pt) >= 3 and pt[2] > 0.0:
                wrists.append((float(pt[0]), float(pt[1])))
            elif len(pt) >= 2:
                wrists.append((float(pt[0]), float(pt[1])))
    return wrists


def min_wrist_distance_px(kpts_a: list[list[float]], kpts_b: list[list[float]]) -> float:
    wrists_a = _wrist_points(kpts_a)
    wrists_b = _wrist_points(kpts_b)
    if not wrists_a or not wrists_b:
        return float("inf")
    return min(
        float(np.hypot(ax - bx, ay - by))
        for ax, ay in wrists_a
        for bx, by in wrists_b
    )


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def compute_pair_score(
    *,
    center_dist: float,
    wrist_dist: float,
    iou: float,
    config: PairValidationConfig,
) -> tuple[float, float, float, float]:
    proximity_score = _clamp01(1.0 - center_dist / config.interaction_distance_px)
    wrist_proximity_score = _clamp01(1.0 - wrist_dist / config.wrist_max_distance_px)
    overlap_score = _clamp01(iou / config.overlap_score_iou_ref)
    pair_score = (
        config.proximity_weight * proximity_score
        + config.wrist_weight * wrist_proximity_score
        + config.overlap_weight * overlap_score
    )
    return pair_score, proximity_score, wrist_proximity_score, overlap_score


def validate_pair(
    track_a: dict[str, Any],
    track_b: dict[str, Any],
    consecutive_frames: int,
    config: PairValidationConfig,
) -> PairValidationResult:
    """
    Validate a track pair before PoseC3D / CTR-GCN pair inference.

    Returns acceptance flag, human-readable rejection reasons, and metrics.
    """
    reasons: list[str] = []
    bbox_a = track_a["bbox"]
    bbox_b = track_b["bbox"]
    kpts_a = track_a.get("keypoints") or []
    kpts_b = track_b.get("keypoints") or []

    consecutive = consecutive_frames
    iou = bbox_iou(bbox_a, bbox_b)
    center_dist = center_distance_px(bbox_a, bbox_b)
    conf_a = mean_keypoint_confidence(kpts_a)
    conf_b = mean_keypoint_confidence(kpts_b)
    wrist_dist = min_wrist_distance_px(kpts_a, kpts_b)

    pair_score, proximity_score, wrist_score, overlap_score = compute_pair_score(
        center_dist=center_dist,
        wrist_dist=wrist_dist,
        iou=iou,
        config=config,
    )

    metrics = {
        "consecutive_frames": float(consecutive),
        "iou": round(iou, 4),
        "center_distance_px": round(center_dist, 2),
        "mean_keypoint_conf_a": round(conf_a, 4),
        "mean_keypoint_conf_b": round(conf_b, 4),
        "wrist_distance_px": round(wrist_dist, 2) if np.isfinite(wrist_dist) else -1.0,
        "proximity_score": round(proximity_score, 4),
        "wrist_proximity_score": round(wrist_score, 4),
        "overlap_score": round(overlap_score, 4),
        "pair_score": round(pair_score, 4),
    }

    if consecutive < config.min_duration_frames:
        reasons.append("low pair duration")

    overlap_ok = iou >= config.min_iou or center_dist <= config.close_center_distance_px
    if not overlap_ok:
        reasons.append("low overlap")

    if conf_a < config.min_mean_keypoint_conf or conf_b < config.min_mean_keypoint_conf:
        reasons.append("low pose quality")

    wrist_ok = (
        wrist_dist <= config.wrist_max_distance_px
        or iou >= config.wrist_iou_bypass
    )
    if not wrist_ok:
        reasons.append("wrists too far")

    if pair_score < config.pair_score_threshold:
        reasons.append("low pair score")

    return PairValidationResult(
        accepted=not reasons,
        reasons=reasons,
        pair_score=pair_score,
        metrics=metrics,
    )


AVG_METRIC_KEYS = (
    "consecutive_frames",
    "iou",
    "center_distance",
    "mean_keypoint_conf_a",
    "mean_keypoint_conf_b",
    "wrist_distance",
    "pair_score",
)


def validation_record_from_result(
    *,
    frame: int,
    timestamp: float,
    track_a: int,
    track_b: int,
    result: PairValidationResult,
) -> dict[str, Any]:
    """Flatten validation metrics into a JSON-serializable diagnostic row."""
    m = result.metrics
    wrist = m.get("wrist_distance_px", -1.0)
    return {
        "frame": frame,
        "timestamp": round(float(timestamp), 2),
        "track_a": track_a,
        "track_b": track_b,
        "consecutive_frames": int(m.get("consecutive_frames", 0)),
        "iou": m.get("iou"),
        "center_distance": m.get("center_distance_px"),
        "mean_keypoint_conf_a": m.get("mean_keypoint_conf_a"),
        "mean_keypoint_conf_b": m.get("mean_keypoint_conf_b"),
        "wrist_distance": None if wrist is not None and wrist < 0 else wrist,
        "pair_score": round(float(result.pair_score), 4),
        "rejection_reasons": list(result.reasons),
    }


@dataclass
class PairValidationDiagnostics:
    """Collect per-pair validation rows and aggregate stats for threshold tuning."""

    rejected_records: list[dict[str, Any]] = field(default_factory=list)
    accepted_records: list[dict[str, Any]] = field(default_factory=list)
    rejection_reason_counts: dict[str, int] = field(default_factory=dict)

    def record(
        self,
        *,
        frame: int,
        timestamp: float,
        track_a: int,
        track_b: int,
        result: PairValidationResult,
    ) -> None:
        row = validation_record_from_result(
            frame=frame,
            timestamp=timestamp,
            track_a=track_a,
            track_b=track_b,
            result=result,
        )
        if result.accepted:
            accepted_row = {k: v for k, v in row.items() if k != "rejection_reasons"}
            self.accepted_records.append(accepted_row)
            return

        self.rejected_records.append(row)
        for reason in result.reasons:
            self.rejection_reason_counts[reason] = self.rejection_reason_counts.get(reason, 0) + 1

    @staticmethod
    def _average_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
        if not records:
            return {}
        out: dict[str, float] = {}
        for key in AVG_METRIC_KEYS:
            values = [r[key] for r in records if r.get(key) is not None]
            if values:
                out[key] = round(float(sum(values) / len(values)), 4)
        return out

    def to_report(self) -> dict[str, Any]:
        accepted = len(self.accepted_records)
        rejected = len(self.rejected_records)
        return {
            "candidates": accepted + rejected,
            "accepted_pairs": accepted,
            "rejected_pairs": rejected,
            "rejection_reason_counts": dict(sorted(self.rejection_reason_counts.items())),
            "rejected_pair_records": self.rejected_records,
            "average_metrics_accepted": self._average_metrics(self.accepted_records),
            "average_metrics_rejected": self._average_metrics(self.rejected_records),
        }

    def print_summary(self) -> None:
        report = self.to_report()
        if report["candidates"] == 0:
            return
        print(
            f"\nPair validation: {report['accepted_pairs']} accepted, "
            f"{report['rejected_pairs']} rejected "
            f"(of {report['candidates']} candidates)"
        )
        if report["rejection_reason_counts"]:
            print("  Rejection reason counts:")
            for reason, count in report["rejection_reason_counts"].items():
                print(f"    {reason}: {count}")
        if report["average_metrics_accepted"]:
            print("  Average metrics (accepted):")
            for key, val in report["average_metrics_accepted"].items():
                print(f"    {key}: {val}")
        if report["average_metrics_rejected"]:
            print("  Average metrics (rejected):")
            for key, val in report["average_metrics_rejected"].items():
                print(f"    {key}: {val}")


__all__ = [
    "AVG_METRIC_KEYS",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "PairValidationConfig",
    "PairValidationDiagnostics",
    "PairValidationResult",
    "bbox_center",
    "bbox_iou",
    "center_distance_px",
    "compute_pair_score",
    "mean_keypoint_confidence",
    "min_wrist_distance_px",
    "validate_pair",
    "validation_record_from_result",
]
