"""
Map NTU120 class IDs to project target taxonomy.

Unmapped NTU classes resolve to ``Normal`` (id 0).
"""

from __future__ import annotations

from typing import Any

# NTU120 class_id -> target label
CLASS_MAPPING: dict[int, str] = {
    52: "push_shove",
    50: "swing_attempt",
    51: "swing_attempt",
    106: "swing_attempt",
    55: "grapple_clinch",
    108: "stumble_recover",
    43: "stumble_recover",
    42: "stumble_recover",
    54: "aggressive_posture",
    93: "aggressive_posture",
    107: "aggressive_posture",
}

DEFAULT_TARGET_CLASS = "Normal"

TARGET_CLASS_IDS: dict[str, int] = {
    "Normal": 0,
    "push_shove": 1,
    "swing_attempt": 2,
    "grapple_clinch": 3,
    "stumble_recover": 4,
    "aggressive_posture": 5,
}
#
# Pair inference may emit these (including aggressive_posture).
PAIR_TARGET_CLASSES = frozenset({
    "push_shove",
    "swing_attempt",
    "grapple_clinch",
    "aggressive_posture",
})

# NTU classes used to detect fall / instability on the single-person path.
FALL_STUMBLE_NTU_IDS = frozenset({42, 43, 108})

# Single path must never emit these directly from model mapping.
SINGLE_EXCLUDED_TARGETS = frozenset({
    "aggressive_posture",
    "push_shove",
    "swing_attempt",
    "grapple_clinch",
})

# BGR colors for annotated video overlays
TARGET_COLORS: dict[str, tuple[int, int, int]] = {
    "Normal":             (180, 180, 180),  # Light gray
    "push_shove":         (100, 149, 255),  # Bright cornflower blue
    "swing_attempt":      (255, 80,  80),   # Bright red
    "grapple_clinch":     (0, 255,  136),   # Bright amber/yellow
    "stumble_recover":    ( 80, 255, 180),  # Bright mint green
    "aggressive_posture": (220,  80, 255),  # Bright purple/violet
}

TARGET_NTU_IDS = frozenset(CLASS_MAPPING.keys())


def map_ntu_class(ntu_class_id: int) -> tuple[str, int]:
    """Return (target_label, target_class_id). Unmapped NTU ids -> Normal (0)."""
    label = CLASS_MAPPING.get(ntu_class_id, DEFAULT_TARGET_CLASS)
    return label, TARGET_CLASS_IDS[label]


def enrich_ntu_prediction(pred: dict[str, Any]) -> dict[str, Any]:
    """Add target_class and target_class_id to a prediction dict."""
    target, target_id = map_ntu_class(pred["class_id"])
    return {
        **pred,
        "target_class": target,
        "target_class_id": target_id,
        "ntu_class_id": pred["class_id"],
        "ntu_label": pred["label"],
    }


def enrich_top_k(top_k: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [enrich_ntu_prediction(p) for p in top_k]


def best_mapped_from_topk(
    top_k: list[dict[str, Any]],
    min_confidence: float = 0.0,
) -> dict[str, Any] | None:
    """
    Best mapped non-Normal cue for overlay / display (pair inference).

    Returns the highest-confidence mapped production cue at or above
    ``min_confidence``, or None when nothing qualifies (Normal suppression).
    """
    hits = filter_target_events(top_k, min_confidence=min_confidence)
    if not hits:
        return None
    return max(hits, key=lambda x: x["confidence"])


def best_fall_stumble_from_topk(
    top_k: list[dict[str, Any]],
    min_confidence: float = 0.0,
) -> dict[str, Any] | None:
    """Best fall/stumble NTU hit from single-person predictions."""
    hits = [
        enrich_ntu_prediction(p)
        for p in top_k
        if p["class_id"] in FALL_STUMBLE_NTU_IDS and p["confidence"] >= min_confidence
    ]
    if not hits:
        return None
    return max(hits, key=lambda x: x["confidence"])


def filter_target_events(
    top_k: list[dict[str, Any]],
    min_confidence: float = 0.0,
) -> list[dict[str, Any]]:
    """Return enriched pair predictions that map to a non-Normal target above threshold."""
    out: list[dict[str, Any]] = []
    for pred in top_k:
        mapped = enrich_ntu_prediction(pred)
        if mapped["target_class"] != DEFAULT_TARGET_CLASS and mapped["confidence"] >= min_confidence:
            out.append(mapped)
    return out


def filter_single_target_events(
    top_k: list[dict[str, Any]],
    min_confidence: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Single-person events: fall/stumble only (stumble_recover).

    Never emits aggressive_posture or pair-only interaction classes.
    """
    fall_pred = best_fall_stumble_from_topk(top_k, min_confidence=min_confidence)
    if fall_pred is None:
        return []
    return [fall_pred]


__all__ = [
    "CLASS_MAPPING",
    "DEFAULT_TARGET_CLASS",
    "TARGET_CLASS_IDS",
    "TARGET_COLORS",
    "TARGET_NTU_IDS",
    "PAIR_TARGET_CLASSES",
    "FALL_STUMBLE_NTU_IDS",
    "SINGLE_EXCLUDED_TARGETS",
    "best_mapped_from_topk",
    "best_fall_stumble_from_topk",
    "enrich_top_k",
    "enrich_ntu_prediction",
    "filter_target_events",
    "filter_single_target_events",
    "map_ntu_class",
]
