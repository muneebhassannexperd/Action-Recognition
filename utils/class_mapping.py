"""
Map NTU120 class IDs to project target taxonomy.

Unmapped NTU classes resolve to ``Normal`` (id 0).
"""

from __future__ import annotations

from dataclasses import dataclass
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

# Pair inference may emit these (including aggressive_posture).
PAIR_TARGET_CLASSES = frozenset({
    "push_shove",
    "swing_attempt",
    "grapple_clinch",
    "aggressive_posture",
})

# Pair events that qualify for fall-after-contact correlation.
INTERACTION_CORRELATION_CLASSES = frozenset({
    "push_shove",
    "swing_attempt",
    "grapple_clinch",
})

# NTU classes used to detect fall / instability on the single-person path.
FALL_STUMBLE_NTU_IDS = frozenset({42, 43, 108})

# Single path must never emit these directly from model mapping.
SINGLE_EXCLUDED_TARGETS = frozenset({
    "aggressive_posture",
    "push_shove",
    "swing_attempt",
    "grapple_clinch",
    "person_falls_after_contact",
})

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


@dataclass
class InteractionRecord:
    """Pair interaction event used for fall-after-contact correlation."""

    timestamp: float
    frame: int
    track_a: int
    track_b: int
    target_class: str


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


def best_mapped_from_topk(top_k: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Pick the best display/event prediction from top-K (pair inference).

    Prefers the highest-confidence entry among mapped (non-Normal) NTU classes;
    otherwise falls back to top-1 mapped as Normal.
    """
    if not top_k:
        return None

    mapped_hits = [enrich_ntu_prediction(p) for p in top_k if p["class_id"] in CLASS_MAPPING]
    if mapped_hits:
        return max(mapped_hits, key=lambda x: x["confidence"])
    return enrich_ntu_prediction(top_k[0])


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


def resolve_single_fall_event(
    fall_pred: dict[str, Any],
    track_id: int,
    timestamp: float,
    interaction_history: list[InteractionRecord],
    contact_window_seconds: float,
) -> dict[str, Any]:
    """
    Map a single-path fall/stumble detection to the final target class.

    Recent pair interaction -> person_falls_after_contact, else stumble_recover.
    """
    if had_recent_interaction(
        track_id,
        interaction_history,
        timestamp,
        contact_window_seconds,
    ):
        target_class = "person_falls_after_contact"
    else:
        target_class = "stumble_recover"

    target_class_id = TARGET_CLASS_IDS[target_class]
    return {
        **fall_pred,
        "target_class": target_class,
        "target_class_id": target_class_id,
        "ntu_class_id": fall_pred["ntu_class_id"],
        "ntu_label": fall_pred["ntu_label"],
        "confidence": fall_pred["confidence"],
    }


def had_recent_interaction(
    track_id: int,
    interaction_history: list[InteractionRecord],
    timestamp: float,
    window_seconds: float,
) -> bool:
    """True if track_id had a qualifying pair interaction within the window."""
    for record in interaction_history:
        if timestamp - record.timestamp > window_seconds:
            continue
        if record.target_class not in INTERACTION_CORRELATION_CLASSES:
            continue
        if track_id in (record.track_a, record.track_b):
            return True
    return False


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
    track_id: int,
    timestamp: float,
    interaction_history: list[InteractionRecord],
    contact_window_seconds: float,
    min_confidence: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Single-person events: fall/stumble only, with fall-after-contact correlation.

    Never emits aggressive_posture or pair-only interaction classes.
    """
    fall_pred = best_fall_stumble_from_topk(top_k, min_confidence=min_confidence)
    if fall_pred is None:
        return []
    return [resolve_single_fall_event(
        fall_pred,
        track_id,
        timestamp,
        interaction_history,
        contact_window_seconds,
    )]


__all__ = [
    "CLASS_MAPPING",
    "DEFAULT_TARGET_CLASS",
    "TARGET_CLASS_IDS",
    "TARGET_COLORS",
    "TARGET_NTU_IDS",
    "PAIR_TARGET_CLASSES",
    "INTERACTION_CORRELATION_CLASSES",
    "FALL_STUMBLE_NTU_IDS",
    "SINGLE_EXCLUDED_TARGETS",
    "InteractionRecord",
    "best_mapped_from_topk",
    "best_fall_stumble_from_topk",
    "enrich_top_k",
    "enrich_ntu_prediction",
    "filter_target_events",
    "filter_single_target_events",
    "had_recent_interaction",
    "map_ntu_class",
    "resolve_single_fall_event",
    
]
