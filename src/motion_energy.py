"""Skeleton motion-energy metrics for gating PoseC3D / CTR-GCN inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

MOTION_AVG_KEYS = ("motion_energy", "motion_peak")


@dataclass(frozen=True)
class MotionEnergyResult:
    """Per-track motion stats over a skeleton buffer window."""

    motion_energy: float
    motion_peak: float
    valid_frame_pairs: int
    scale_px: float


def bbox_height(bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(float(y2 - y1), 1.0)


def compute_track_motion_energy(
    sequence: np.ndarray,
    bbox_height: float,
    *,
    keypoint_conf_threshold: float = 0.3,
) -> MotionEnergyResult:
    """
    Mean frame-to-frame joint displacement normalized by person scale.

    Args:
        sequence: (T, V, C) skeleton window; C is x, y, [conf].
        bbox_height: Person scale in pixels for normalization.
    """
    scale = max(float(bbox_height), 1.0)
    if sequence.ndim != 3 or sequence.shape[0] < 2:
        return MotionEnergyResult(0.0, 0.0, 0, scale)

    frame_energies: list[float] = []
    for t in range(1, sequence.shape[0]):
        prev = sequence[t - 1]
        curr = sequence[t]
        displacements: list[float] = []
        for j in range(sequence.shape[1]):
            if prev.shape[-1] >= 3 and curr.shape[-1] >= 3:
                if prev[j, 2] < keypoint_conf_threshold or curr[j, 2] < keypoint_conf_threshold:
                    continue
                if prev[j, 0] == 0 and prev[j, 1] == 0:
                    continue
                if curr[j, 0] == 0 and curr[j, 1] == 0:
                    continue
            dx = float(curr[j, 0] - prev[j, 0])
            dy = float(curr[j, 1] - prev[j, 1])
            displacements.append(float(np.hypot(dx, dy)) / scale)
        if displacements:
            frame_energies.append(float(sum(displacements) / len(displacements)))

    if not frame_energies:
        return MotionEnergyResult(0.0, 0.0, 0, scale)

    return MotionEnergyResult(
        motion_energy=round(float(sum(frame_energies) / len(frame_energies)), 6),
        motion_peak=round(float(max(frame_energies)), 6),
        valid_frame_pairs=len(frame_energies),
        scale_px=round(scale, 2),
    )


def compute_pair_motion_energy(
    energy_a: MotionEnergyResult,
    energy_b: MotionEnergyResult,
) -> dict[str, float]:
    """Combine per-track motion for pair gating (max of each metric)."""
    return {
        "motion_energy": max(energy_a.motion_energy, energy_b.motion_energy),
        "motion_peak": max(energy_a.motion_peak, energy_b.motion_peak),
        "motion_energy_a": energy_a.motion_energy,
        "motion_energy_b": energy_b.motion_energy,
        "motion_peak_a": energy_a.motion_peak,
        "motion_peak_b": energy_b.motion_peak,
    }


def passes_motion_gate(
    motion_energy: float,
    threshold: float | None,
    *,
    enabled: bool,
) -> bool:
    """True when gate is off, threshold unset, or motion meets the floor."""
    if not enabled or threshold is None:
        return True
    return motion_energy >= threshold


@dataclass
class MotionGateDiagnostics:
    """Collect motion-energy rows for calibration and optional gating."""

    gate_enabled: bool = False
    threshold_single: float | None = None
    threshold_pair: float | None = None
    processed_single: list[dict[str, Any]] = field(default_factory=list)
    processed_pair: list[dict[str, Any]] = field(default_factory=list)
    skipped_single: list[dict[str, Any]] = field(default_factory=list)
    skipped_pair: list[dict[str, Any]] = field(default_factory=list)

    def _avg(self, records: list[dict[str, Any]], key: str) -> float | None:
        vals = [r[key] for r in records if r.get(key) is not None]
        if not vals:
            return None
        return round(float(sum(vals) / len(vals)), 6)

    def record_single(
        self,
        *,
        frame: int,
        timestamp: float,
        track_id: int,
        result: MotionEnergyResult,
        threshold: float | None,
        skipped: bool,
    ) -> None:
        row = {
            "frame": frame,
            "timestamp": round(float(timestamp), 2),
            "track_id": track_id,
            "motion_energy": result.motion_energy,
            "motion_peak": result.motion_peak,
            "threshold": threshold,
            "scale_px": result.scale_px,
            "valid_frame_pairs": result.valid_frame_pairs,
            "skipped": skipped,
        }
        if skipped:
            row["rejection_reason"] = "low motion energy"
            self.skipped_single.append(row)
            print(
                f"[motion skip single] t={timestamp:.2f}s track={track_id} "
                f"energy={result.motion_energy:.4f} peak={result.motion_peak:.4f} "
                f"threshold={threshold}"
            )
        else:
            self.processed_single.append(row)
            print(
                f"[motion ok single] t={timestamp:.2f}s track={track_id} "
                f"energy={result.motion_energy:.4f} peak={result.motion_peak:.4f}"
            )

    def record_pair(
        self,
        *,
        frame: int,
        timestamp: float,
        track_a: int,
        track_b: int,
        energy_a: MotionEnergyResult,
        energy_b: MotionEnergyResult,
        combined: dict[str, float],
        threshold: float | None,
        skipped: bool,
    ) -> None:
        row = {
            "frame": frame,
            "timestamp": round(float(timestamp), 2),
            "track_a": track_a,
            "track_b": track_b,
            "motion_energy": combined["motion_energy"],
            "motion_peak": combined["motion_peak"],
            "motion_energy_a": combined["motion_energy_a"],
            "motion_energy_b": combined["motion_energy_b"],
            "threshold": threshold,
            "skipped": skipped,
        }
        if skipped:
            row["rejection_reason"] = "low motion energy"
            self.skipped_pair.append(row)
            print(
                f"[motion skip pair] t={timestamp:.2f}s pair={track_a}-{track_b} "
                f"energy={combined['motion_energy']:.4f} peak={combined['motion_peak']:.4f} "
                f"threshold={threshold}"
            )
        else:
            self.processed_pair.append(row)
            print(
                f"[motion ok pair] t={timestamp:.2f}s pair={track_a}-{track_b} "
                f"energy={combined['motion_energy']:.4f} peak={combined['motion_peak']:.4f}"
            )

    def to_report(self) -> dict[str, Any]:
        all_single = self.processed_single + self.skipped_single
        all_pair = self.processed_pair + self.skipped_pair
        return {
            "gate_enabled": self.gate_enabled,
            "threshold_single": self.threshold_single,
            "threshold_pair": self.threshold_pair,
            "single_processed": len(self.processed_single),
            "single_skipped": len(self.skipped_single),
            "pair_processed": len(self.processed_pair),
            "pair_skipped": len(self.skipped_pair),
            "average_motion_energy_single": self._avg(all_single, "motion_energy"),
            "average_motion_peak_single": self._avg(all_single, "motion_peak"),
            "average_motion_energy_pair": self._avg(all_pair, "motion_energy"),
            "average_motion_peak_pair": self._avg(all_pair, "motion_peak"),
            "average_motion_energy_single_processed": self._avg(self.processed_single, "motion_energy"),
            "average_motion_energy_single_skipped": self._avg(self.skipped_single, "motion_energy"),
            "average_motion_energy_pair_processed": self._avg(self.processed_pair, "motion_energy"),
            "average_motion_energy_pair_skipped": self._avg(self.skipped_pair, "motion_energy"),
            "skipped_single_records": self.skipped_single,
            "skipped_pair_records": self.skipped_pair,
            "processed_single_records": self.processed_single,
            "processed_pair_records": self.processed_pair,
        }

    def print_summary(self) -> None:
        r = self.to_report()
        total_single = r["single_processed"] + r["single_skipped"]
        total_pair = r["pair_processed"] + r["pair_skipped"]
        if total_single == 0 and total_pair == 0:
            return
        print("\n--- Motion gate diagnostics ---")
        print(f"  Gate enabled: {r['gate_enabled']}")
        print(f"  Single: {r['single_processed']} processed, {r['single_skipped']} skipped")
        print(f"  Pair:   {r['pair_processed']} processed, {r['pair_skipped']} skipped")
        if r["average_motion_energy_single"] is not None:
            print(f"  Avg single motion energy: {r['average_motion_energy_single']:.4f}")
        if r["average_motion_energy_pair"] is not None:
            print(f"  Avg pair motion energy:   {r['average_motion_energy_pair']:.4f}")


__all__ = [
    "MOTION_AVG_KEYS",
    "bbox_height",
    "MotionEnergyResult",
    "MotionGateDiagnostics",
    "compute_pair_motion_energy",
    "compute_track_motion_energy",
    "passes_motion_gate",
]
