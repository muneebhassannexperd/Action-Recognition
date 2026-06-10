"""End-to-end pipeline: YOLO track -> interaction pairs -> action recognition -> JSON + video."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import cv2

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _SRC)
sys.path.insert(0, _ROOT)
from utils.class_mapping import (
    best_mapped_from_topk,
    enrich_top_k,
    filter_target_events,
)
from utils.config import (
    DEFAULT_ACTION_MODEL,
    DEFAULT_WINDOW_SIZE,
    EVENT_MIN_CONFIDENCE,
    INFERENCE_STRIDE,
    INTERACTION_DISTANCE,
    INTERACTION_FRAMES,
    TRACKER_CONFIG,
    resolve_yolo_model,
)

from action_recognizer import ActionRecognizer, create_action_recognizer
from interaction_detector import InteractionDetector
from pose_detector import PoseDetector
from skeleton_buffer import create_skeleton_buffer
from video_annotator import TrackActionLabel, annotate_frame, create_video_writer


class VideoProcessor:
    """
    Two-person interaction recognition with a selectable action backend.

    Runs inference only when two tracks stay within ``interaction_distance``
    for ``interaction_frames`` consecutive frames and both skeleton buffers are full.
    """

    def __init__(
        self,
        pose_model_path: str | None = None,
        action_model: str = DEFAULT_ACTION_MODEL,
        ctrgcn_weights_path: str | None = None,
        posec3d_weights_path: str | None = None,
        posec3d_heatmap_mode: str | None = None,
        device: str = "cuda",
        window_size: int = DEFAULT_WINDOW_SIZE,
        inference_stride: int = INFERENCE_STRIDE,
        event_min_confidence: float = EVENT_MIN_CONFIDENCE,
        interaction_distance: float = INTERACTION_DISTANCE,
        interaction_frames: int = INTERACTION_FRAMES,
        tracker: str = TRACKER_CONFIG,
    ) -> None:
        yolo_path = str(resolve_yolo_model(pose_model_path))
        self.action_model = action_model.lower().strip()
        self.pose_detector = PoseDetector(
            model_path=yolo_path,
            device=device,
            tracker=tracker,
        )
        self.interaction_detector = InteractionDetector(
            distance_threshold=interaction_distance,
            min_consecutive_frames=interaction_frames,
        )
        self.buffer = create_skeleton_buffer(self.action_model, window_size=window_size)
        self.recognizer: ActionRecognizer = create_action_recognizer(
            self.action_model,
            device=device,
            ctrgcn_weights_path=ctrgcn_weights_path,
            posec3d_weights_path=posec3d_weights_path,
            heatmap_mode=posec3d_heatmap_mode,
        )
        self.inference_stride = inference_stride
        self.event_min_confidence = event_min_confidence
        self._last_infer_frame: dict[tuple[int, int], int] = {}
        self._track_labels: dict[int, TrackActionLabel] = {}
        self._frame_shape: tuple[int, int] | None = None

    def process(
        self,
        video_path: str,
        output_json_path: str | None = None,
        annotated_output_path: str | None = None,
    ) -> dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_shape = (height, width)

        self.pose_detector.load_model()
        self.pose_detector.reset_tracker()
        self.recognizer.load_model()
        print(f"Action Model: {self.recognizer.get_model_name()}")

        writer: cv2.VideoWriter | None = None
        if annotated_output_path:
            os.makedirs(os.path.dirname(annotated_output_path) or ".", exist_ok=True)
            writer = create_video_writer(annotated_output_path, fps, (width, height))

        events: list[dict[str, Any]] = []
        raw_predictions: list[dict[str, Any]] = []
        seen_event_keys: set[tuple] = set()
        self._last_infer_frame.clear()
        self._track_labels.clear()

        frame_idx = -1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            timestamp = frame_idx / fps

            tracks = self.pose_detector.track(frame)
            self.interaction_detector.update(tracks, frame_idx)
            active_pairs = self.interaction_detector.active_pairs()

            active_ids = {t["track_id"] for t in tracks}
            for tid in list(self.buffer.track_ids()):
                if tid not in active_ids:
                    self.buffer.remove(tid)
                    self._track_labels.pop(tid, None)

            for key in list(self._last_infer_frame):
                if key[0] not in active_ids or key[1] not in active_ids:
                    self._last_infer_frame.pop(key, None)

            for track in tracks:
                tid = track["track_id"]
                kpts = track.get("keypoints") or []
                if kpts:
                    self.buffer.add(tid, kpts)

            for track_a, track_b in active_pairs:
                pair_key = (track_a, track_b)
                if not self.buffer.is_pair_ready(track_a, track_b):
                    continue

                last = self._last_infer_frame.get(pair_key, -self.inference_stride)
                if frame_idx - last < self.inference_stride:
                    continue

                pair_seq = self.buffer.get_pair_sequence(track_a, track_b)
                if pair_seq is None:
                    continue

                predict_kwargs: dict[str, Any] = {}
                if self._frame_shape is not None:
                    predict_kwargs["img_shape"] = self._frame_shape

                top_k = self.recognizer.predict(pair_seq, **predict_kwargs)
                enriched_top_k = enrich_top_k(top_k)
                self._last_infer_frame[pair_key] = frame_idx

                mapped_best = best_mapped_from_topk(top_k)
                if mapped_best is not None:
                    label = TrackActionLabel(
                        target_class=mapped_best["target_class"],
                        target_class_id=mapped_best["target_class_id"],
                        ntu_class_id=mapped_best["ntu_class_id"],
                        ntu_label=mapped_best["ntu_label"],
                        confidence=mapped_best["confidence"],
                        updated_frame=frame_idx,
                    )
                    self._track_labels[track_a] = label
                    self._track_labels[track_b] = label

                segment = {
                    "mode": "pair",
                    "timestamp": round(float(timestamp), 2),
                    "frame": frame_idx,
                    "track_a": track_a,
                    "track_b": track_b,
                    "num_persons": 2,
                    "window_frames": pair_seq.shape[1],
                    "mapped_best": mapped_best,
                    "top_predictions": enriched_top_k,
                }
                raw_predictions.append(segment)

                self._print_pair_predictions(
                    timestamp, track_a, track_b, enriched_top_k, mapped_best,
                )

                for mapped in filter_target_events(top_k, min_confidence=self.event_min_confidence):
                    event_key = (
                        mapped["target_class"],
                        track_a,
                        track_b,
                        round(timestamp, 1),
                    )
                    if event_key in seen_event_keys:
                        continue
                    seen_event_keys.add(event_key)
                    events.append({
                        "timestamp": round(float(timestamp), 2),
                        "frame": frame_idx,
                        "track_a": track_a,
                        "track_b": track_b,
                        "target_class": mapped["target_class"],
                        "target_class_id": mapped["target_class_id"],
                        "ntu_class_id": mapped["ntu_class_id"],
                        "ntu_label": mapped["ntu_label"],
                        "confidence": mapped["confidence"],
                    })

            if writer is not None:
                annotated = annotate_frame(
                    frame,
                    tracks,
                    self._track_labels,
                    frame_idx,
                    active_pairs=active_pairs,
                )
                writer.write(annotated)

        cap.release()
        if writer is not None:
            writer.release()

        report: dict[str, Any] = {
            "video": os.path.basename(video_path),
            "fps": round(float(fps), 2),
            "window_size": self.buffer.window_size,
            "interaction_distance": self.interaction_detector.distance_threshold,
            "interaction_frames": self.interaction_detector.min_consecutive_frames,
            "tracker": self.pose_detector.tracker,
            "events": events,
            "raw_predictions": raw_predictions,
        }
        if annotated_output_path:
            report["annotated_video"] = os.path.basename(annotated_output_path)

        if output_json_path:
            os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

        return report

    @staticmethod
    def _print_pair_predictions(
        timestamp: float,
        track_a: int,
        track_b: int,
        enriched_top_k: list[dict[str, Any]],
        mapped_best: dict[str, Any] | None,
    ) -> None:
        print(
            f"\n[t={timestamp:.2f}s pair={track_a}-{track_b}] "
            f"Top {len(enriched_top_k)} predictions (M=2):"
        )
        for rank, pred in enumerate(enriched_top_k, start=1):
            print(
                f"  {rank}. #{pred['ntu_class_id']:3d} {pred['ntu_label']:<40} "
                f"-> {pred['target_class']} ({pred['confidence']:.4f})"
            )
        if mapped_best:
            print(
                f"  >> overlay: #{mapped_best['target_class_id']} "
                f"{mapped_best['target_class']} ({mapped_best['confidence']:.4f})"
            )


__all__ = ["VideoProcessor"]
