"""
Map NTU120 CTR-GCN class IDs to project target taxonomy.

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
    108: "person_falls_after_contact",
    43: "person_falls_after_contact",
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
    "person_falls_after_contact": 4,
    "stumble_recover": 5,
    "aggressive_posture": 6,
}

# BGR colors for annotated video overlays
TARGET_COLORS: dict[str, tuple[int, int, int]] = {
    "Normal": (160, 160, 160),
    "push_shove": (0, 80, 255),
    "swing_attempt": (0, 165, 255),
    "grapple_clinch": (200, 100, 0),
    "person_falls_after_contact": (255, 0, 200),
    "stumble_recover": (0, 200, 255),
    "aggressive_posture": (0, 0, 255),
}

TARGET_NTU_IDS = frozenset(CLASS_MAPPING.keys())


def map_ntu_class(ntu_class_id: int) -> tuple[str, int]:
    """Return (target_label, target_class_id). Unmapped NTU ids -> Normal (0)."""
    label = CLASS_MAPPING.get(ntu_class_id, DEFAULT_TARGET_CLASS)
    return label, TARGET_CLASS_IDS[label]


def enrich_ntu_prediction(pred: dict[str, Any]) -> dict[str, Any]:
    """Add target_class and target_class_id to a CTR-GCN prediction dict."""
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


def best_mapped_from_topk(top_k: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Pick the best display/event prediction from top-K.

    Prefers the highest-confidence entry among mapped (non-Normal) NTU classes;
    otherwise falls back to top-1 mapped as Normal.
    """
    if not top_k:
        return None

    mapped_hits = [enrich_ntu_prediction(p) for p in top_k if p["class_id"] in CLASS_MAPPING]
    if mapped_hits:
        return max(mapped_hits, key=lambda x: x["confidence"])
    return enrich_ntu_prediction(top_k[0])


def filter_target_events(
    top_k: list[dict[str, Any]],
    min_confidence: float = 0.0,
) -> list[dict[str, Any]]:
    """Return enriched predictions that map to a non-Normal target above threshold."""
    out: list[dict[str, Any]] = []
    for pred in top_k:
        mapped = enrich_ntu_prediction(pred)
        if mapped["target_class"] != DEFAULT_TARGET_CLASS and mapped["confidence"] >= min_confidence:
            out.append(mapped)
    return out


__all__ = [
    "CLASS_MAPPING",
    "DEFAULT_TARGET_CLASS",
    "TARGET_CLASS_IDS",
    "TARGET_COLORS",
    "TARGET_NTU_IDS",
    "best_mapped_from_topk",
    "enrich_top_k",
    "enrich_ntu_prediction",
    "filter_target_events",
    "map_ntu_class",
]
