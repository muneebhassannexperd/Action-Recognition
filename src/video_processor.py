"""End-to-end pipeline: video or client keypoints -> action recognition -> JSON + video."""

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
    best_fall_stumble_from_topk,
    best_mapped_from_topk,
    enrich_top_k,
    filter_single_target_events,
    filter_target_events,
)
from utils.behavior_cues import events_to_behavior_cues, write_behavior_cues_jsonl
from utils.config import (
    BEHAVIOR_CUES_KEYPOINT_MODEL,
    BEHAVIOR_CUES_MODULE,
    BEHAVIOR_CUES_MODULE_VERSION,
    DEFAULT_ACTION_MODEL,
    DEFAULT_CAMERA_ID,
    DEFAULT_INPUT_MODE,
    DEFAULT_ORGANIZATION_ID,
    KEYPOINTS_OUTPUT_FORMAT,
    DEFAULT_WINDOW_SIZE,
    EVENT_MIN_CONFIDENCE,
    INFERENCE_STRIDE,
    INTERACTION_DISTANCE,
    INTERACTION_FRAMES,
    PAIR_CLOSE_CENTER_DISTANCE,
    PAIR_IOU_MIN,
    PAIR_MIN_DURATION_FRAMES,
    PAIR_MIN_MEAN_KEYPOINT_CONF,
    PAIR_OVERLAP_SCORE_IOU_REF,
    PAIR_SCORE_THRESHOLD,
    PAIR_WRIST_IOU_BYPASS,
    PAIR_WRIST_MAX_DISTANCE,
    MOTION_GATE_ENABLED,
    MOTION_KEYPOINT_CONF,
    MOTION_THRESHOLD_PAIR,
    MOTION_THRESHOLD_SINGLE,
    TRACKER_CONFIG,
    VIDEO_OUTPUT_FORMAT,
    resolve_yolo_model,
)

from keypoints_input import load_keypoints_stream

from action_recognizer import ActionRecognizer, create_action_recognizer
from interaction_detector import InteractionDetector
from motion_energy import (
    MotionEnergyResult,
    MotionGateDiagnostics,
    bbox_height,
    compute_pair_motion_energy,
    compute_track_motion_energy,
    passes_motion_gate,
)
from pair_validator import (
    PairValidationConfig,
    PairValidationDiagnostics,
    PairValidationResult,
    validate_pair,
)
from pose_detector import PoseDetector
from skeleton_buffer import create_skeleton_buffer
from video_annotator import TrackActionLabel, annotate_frame, create_video_writer


class VideoProcessor:
    """
    Single- and two-person action recognition with a selectable action backend.

    Pair inference runs when two tracks stay within ``interaction_distance`` for
    ``interaction_frames`` consecutive frames. Single inference runs on every
    track with a full buffer, regardless of pair status.
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
        overlay_min_confidence: float | None = None,
        interaction_distance: float = INTERACTION_DISTANCE,
        interaction_frames: int = INTERACTION_FRAMES,
        tracker: str = TRACKER_CONFIG,
        pair_validation: PairValidationConfig | None = None,
        motion_gate_enabled: bool = MOTION_GATE_ENABLED,
        motion_threshold_single: float | None = MOTION_THRESHOLD_SINGLE,
        motion_threshold_pair: float | None = MOTION_THRESHOLD_PAIR,
        motion_keypoint_conf: float = MOTION_KEYPOINT_CONF,
        input_mode: str = DEFAULT_INPUT_MODE,
    ) -> None:
        self.input_mode = input_mode
        self.action_model = action_model.lower().strip()
        self.pose_detector: PoseDetector | None = None
        self.tracker = tracker
        if input_mode == "video":
            yolo_path = str(resolve_yolo_model(pose_model_path))
            self.pose_detector = PoseDetector(
                model_path=yolo_path,
                device=device,
                tracker=tracker,
            )
        elif input_mode != "keypoints":
            raise ValueError(f"input_mode must be 'video' or 'keypoints', got {input_mode!r}")
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
        self.overlay_min_confidence = (
            overlay_min_confidence
            if overlay_min_confidence is not None
            else event_min_confidence
        )
        self.pair_validation = pair_validation or PairValidationConfig(
            min_duration_frames=PAIR_MIN_DURATION_FRAMES,
            min_iou=PAIR_IOU_MIN,
            close_center_distance_px=PAIR_CLOSE_CENTER_DISTANCE,
            min_mean_keypoint_conf=PAIR_MIN_MEAN_KEYPOINT_CONF,
            wrist_max_distance_px=PAIR_WRIST_MAX_DISTANCE,
            wrist_iou_bypass=PAIR_WRIST_IOU_BYPASS,
            pair_score_threshold=PAIR_SCORE_THRESHOLD,
            interaction_distance_px=interaction_distance,
            overlap_score_iou_ref=PAIR_OVERLAP_SCORE_IOU_REF,
        )
        self.motion_gate_enabled = motion_gate_enabled
        self.motion_threshold_single = motion_threshold_single
        self.motion_threshold_pair = motion_threshold_pair
        self.motion_keypoint_conf = motion_keypoint_conf
        self._last_pair_infer_frame: dict[tuple[int, int], int] = {}
        self._last_single_infer_frame: dict[int, int] = {}
        self._pair_track_labels: dict[int, TrackActionLabel] = {}
        self._single_track_labels: dict[int, TrackActionLabel] = {}
        self._frame_shape: tuple[int, int] | None = None

    def _pair_takes_priority(self, track_id: int) -> bool:
        """Pair prediction overrides single when pair has a non-Normal label."""
        label = self._pair_track_labels.get(track_id)
        return label is not None and not label.is_normal

    def _display_labels(self) -> dict[int, TrackActionLabel]:
        """Pair non-Normal labels take priority over single-person labels."""
        labels = dict(self._single_track_labels)
        for tid, pair_label in self._pair_track_labels.items():
            if not pair_label.is_normal or tid not in labels:
                labels[tid] = pair_label
        return labels

    def _validate_pair(
        self,
        track_a: int,
        track_b: int,
        tracks: list[dict[str, Any]],
    ) -> PairValidationResult:
        """Geometric / pose-quality gate before pair action inference."""
        by_id = {t["track_id"]: t for t in tracks}
        ta = by_id.get(track_a)
        tb = by_id.get(track_b)
        if ta is None or tb is None:
            return PairValidationResult(
                accepted=False,
                reasons=["missing track"],
            )

        pair_info = self.interaction_detector.get_pair_candidate(track_a, track_b)
        consecutive = pair_info.consecutive_frames if pair_info is not None else 0
        return validate_pair(ta, tb, consecutive, self.pair_validation)

    def _track_motion_energy(
        self,
        track_id: int,
        tracks_by_id: dict[int, dict[str, Any]],
    ) -> MotionEnergyResult:
        seq = self.buffer.get_sequence(track_id)
        track = tracks_by_id.get(track_id)
        if seq is None or track is None:
            return MotionEnergyResult(0.0, 0.0, 0, 1.0)
        return compute_track_motion_energy(
            seq,
            bbox_height(track["bbox"]),
            keypoint_conf_threshold=self.motion_keypoint_conf,
        )

    def _reset_run_state(self) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        set[tuple],
        PairValidationDiagnostics,
        MotionGateDiagnostics,
    ]:
        self._last_pair_infer_frame.clear()
        self._last_single_infer_frame.clear()
        self._pair_track_labels.clear()
        self._single_track_labels.clear()

        pair_validation_diag = PairValidationDiagnostics()
        motion_gate_diag = MotionGateDiagnostics(
            gate_enabled=self.motion_gate_enabled,
            threshold_single=self.motion_threshold_single,
            threshold_pair=self.motion_threshold_pair,
        )
        return [], [], set(), pair_validation_diag, motion_gate_diag

    def _process_frame(
        self,
        *,
        frame_idx: int,
        timestamp: float,
        tracks: list[dict[str, Any]],
        events: list[dict[str, Any]],
        raw_predictions: list[dict[str, Any]],
        seen_event_keys: set[tuple],
        pair_validation_diag: PairValidationDiagnostics,
        motion_gate_diag: MotionGateDiagnostics,
        predict_kwargs: dict[str, Any],
    ) -> list[tuple[int, int]]:
        self.interaction_detector.update(tracks, frame_idx)
        active_pairs = self.interaction_detector.active_pairs()

        active_ids = {t["track_id"] for t in tracks}
        for tid in list(self.buffer.track_ids()):
            if tid not in active_ids:
                self.buffer.remove(tid)
                self._pair_track_labels.pop(tid, None)
                self._single_track_labels.pop(tid, None)
                self._last_single_infer_frame.pop(tid, None)

        for key in list(self._last_pair_infer_frame):
            if key[0] not in active_ids or key[1] not in active_ids:
                self._last_pair_infer_frame.pop(key, None)

        for track in tracks:
            tid = track["track_id"]
            kpts = track.get("keypoints") or []
            if kpts:
                self.buffer.add(tid, kpts)

        tracks_by_id = {t["track_id"]: t for t in tracks}

        for track_a, track_b in active_pairs:
            pair_key = (track_a, track_b)
            if not self.buffer.is_pair_ready(track_a, track_b):
                continue

            last = self._last_pair_infer_frame.get(pair_key, -self.inference_stride)
            if frame_idx - last < self.inference_stride:
                continue

            validation = self._validate_pair(track_a, track_b, tracks)
            pair_validation_diag.record(
                frame=frame_idx,
                timestamp=timestamp,
                track_a=track_a,
                track_b=track_b,
                result=validation,
            )
            if not validation.accepted:
                reason_str = ", ".join(validation.reasons)
                print(
                    f"[pair reject] t={timestamp:.2f}s pair={track_a}-{track_b}: "
                    f"{reason_str} | score={validation.pair_score:.3f}"
                )
                continue

            energy_a = self._track_motion_energy(track_a, tracks_by_id)
            energy_b = self._track_motion_energy(track_b, tracks_by_id)
            pair_motion = compute_pair_motion_energy(energy_a, energy_b)
            pair_skipped = not passes_motion_gate(
                pair_motion["motion_energy"],
                self.motion_threshold_pair,
                enabled=self.motion_gate_enabled,
            )
            motion_gate_diag.record_pair(
                frame=frame_idx,
                timestamp=timestamp,
                track_a=track_a,
                track_b=track_b,
                energy_a=energy_a,
                energy_b=energy_b,
                combined=pair_motion,
                threshold=self.motion_threshold_pair,
                skipped=pair_skipped,
            )
            if pair_skipped:
                continue

            pair_seq = self.buffer.get_pair_sequence(track_a, track_b)
            if pair_seq is None:
                continue

            top_k = self.recognizer.predict(
                pair_seq,
                inference_mode="pair",
                num_persons=2,
                **predict_kwargs,
            )
            enriched_top_k = enrich_top_k(top_k)
            self._last_pair_infer_frame[pair_key] = frame_idx

            mapped_best = best_mapped_from_topk(
                top_k, min_confidence=self.overlay_min_confidence,
            )
            if mapped_best is not None:
                label = TrackActionLabel(
                    target_class=mapped_best["target_class"],
                    target_class_id=mapped_best["target_class_id"],
                    ntu_class_id=mapped_best["ntu_class_id"],
                    ntu_label=mapped_best["ntu_label"],
                    confidence=mapped_best["confidence"],
                    updated_frame=frame_idx,
                )
                self._pair_track_labels[track_a] = label
                self._pair_track_labels[track_b] = label
            else:
                self._pair_track_labels.pop(track_a, None)
                self._pair_track_labels.pop(track_b, None)

            raw_predictions.append({
                "mode": "pair",
                "timestamp": round(float(timestamp), 2),
                "frame": frame_idx,
                "track_a": track_a,
                "track_b": track_b,
                "num_persons": 2,
                "window_frames": pair_seq.shape[1],
                "mapped_best": mapped_best,
                "top_predictions": enriched_top_k,
            })

            self._print_predictions(
                "pair", timestamp, enriched_top_k, mapped_best,
                track_a=track_a, track_b=track_b,
            )

            for mapped in filter_target_events(top_k, min_confidence=self.event_min_confidence):
                event_key = (
                    "pair",
                    mapped["target_class"],
                    track_a,
                    track_b,
                    round(timestamp, 1),
                )
                if event_key in seen_event_keys:
                    continue
                seen_event_keys.add(event_key)
                events.append({
                    "mode": "pair",
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

        paired_ids = {a for pair in active_pairs for a in pair}

        for tid in sorted(active_ids):
            if tid in paired_ids:
                continue
            if not self.buffer.is_ready(tid):
                continue

            last = self._last_single_infer_frame.get(tid, -self.inference_stride)
            if frame_idx - last < self.inference_stride:
                continue

            single_seq = self.buffer.get_single_sequence(tid)
            if single_seq is None:
                continue

            single_motion = self._track_motion_energy(tid, tracks_by_id)
            single_skipped = not passes_motion_gate(
                single_motion.motion_energy,
                self.motion_threshold_single,
                enabled=self.motion_gate_enabled,
            )
            motion_gate_diag.record_single(
                frame=frame_idx,
                timestamp=timestamp,
                track_id=tid,
                result=single_motion,
                threshold=self.motion_threshold_single,
                skipped=single_skipped,
            )
            if single_skipped:
                continue

            top_k = self.recognizer.predict(
                single_seq,
                inference_mode="single",
                num_persons=1,
                **predict_kwargs,
            )
            enriched_top_k = enrich_top_k(top_k)
            self._last_single_infer_frame[tid] = frame_idx

            mapped_best = best_fall_stumble_from_topk(
                top_k, min_confidence=self.overlay_min_confidence,
            )

            if mapped_best is not None and not self._pair_takes_priority(tid):
                self._single_track_labels[tid] = TrackActionLabel(
                    target_class=mapped_best["target_class"],
                    target_class_id=mapped_best["target_class_id"],
                    ntu_class_id=mapped_best["ntu_class_id"],
                    ntu_label=mapped_best["ntu_label"],
                    confidence=mapped_best["confidence"],
                    updated_frame=frame_idx,
                )
            elif not self._pair_takes_priority(tid):
                self._single_track_labels.pop(tid, None)

            raw_predictions.append({
                "mode": "single",
                "timestamp": round(float(timestamp), 2),
                "frame": frame_idx,
                "track_id": tid,
                "num_persons": 1,
                "window_frames": single_seq.shape[1],
                "mapped_best": mapped_best,
                "top_predictions": enriched_top_k,
            })

            self._print_predictions(
                "single", timestamp, enriched_top_k, mapped_best, track_id=tid,
            )

            for mapped in filter_single_target_events(
                top_k,
                min_confidence=self.event_min_confidence,
            ):
                if self._pair_takes_priority(tid):
                    continue
                event_key = (
                    "single",
                    mapped["target_class"],
                    tid,
                    round(timestamp, 1),
                )
                if event_key in seen_event_keys:
                    continue
                seen_event_keys.add(event_key)
                events.append({
                    "mode": "single",
                    "timestamp": round(float(timestamp), 2),
                    "frame": frame_idx,
                    "track_id": tid,
                    "target_class": mapped["target_class"],
                    "target_class_id": mapped["target_class_id"],
                    "ntu_class_id": mapped["ntu_class_id"],
                    "ntu_label": mapped["ntu_label"],
                    "confidence": mapped["confidence"],
                })

        return active_pairs

    def _build_report(
        self,
        *,
        source_name: str,
        fps: float,
        events: list[dict[str, Any]],
        raw_predictions: list[dict[str, Any]],
        pair_validation_diag: PairValidationDiagnostics,
        motion_gate_diag: MotionGateDiagnostics,
        annotated_output_path: str | None,
        input_mode: str,
        keypoints_source: str | None = None,
    ) -> dict[str, Any]:
        pair_preds = sum(1 for p in raw_predictions if p["mode"] == "pair")
        single_preds = sum(1 for p in raw_predictions if p["mode"] == "single")

        report: dict[str, Any] = {
            "input_mode": input_mode,
            "fps": round(float(fps), 2),
            "window_size": self.buffer.window_size,
            "interaction_distance": self.interaction_detector.distance_threshold,
            "interaction_frames": self.interaction_detector.min_consecutive_frames,
            "event_min_confidence": self.event_min_confidence,
            "overlay_min_confidence": self.overlay_min_confidence,
            "pair_validation": {
                "config": {
                    "min_duration_frames": self.pair_validation.min_duration_frames,
                    "min_iou": self.pair_validation.min_iou,
                    "close_center_distance_px": self.pair_validation.close_center_distance_px,
                    "min_mean_keypoint_conf": self.pair_validation.min_mean_keypoint_conf,
                    "wrist_max_distance_px": self.pair_validation.wrist_max_distance_px,
                    "wrist_iou_bypass": self.pair_validation.wrist_iou_bypass,
                    "pair_score_threshold": self.pair_validation.pair_score_threshold,
                },
                **pair_validation_diag.to_report(),
            },
            "motion_gate": motion_gate_diag.to_report(),
            "events": events,
            "raw_predictions": raw_predictions,
            "pair_inference_segments": pair_preds,
            "single_inference_segments": single_preds,
        }
        if input_mode == "video":
            report["video"] = source_name
            report["tracker"] = self.tracker
        else:
            report["keypoints_source"] = keypoints_source or source_name
        if annotated_output_path:
            report["annotated_video"] = os.path.basename(annotated_output_path)
        return report

    def _write_behavior_cues_jsonl(
        self,
        events: list[dict[str, Any]],
        path: str,
        *,
        fps: float,
        camera_id: int,
        organization_id: int,
        keypoint_model: str,
        module: str,
        module_version: str,
    ) -> list[dict[str, Any]]:
        cue_records = events_to_behavior_cues(
            events,
            camera_id=camera_id,
            organization_id=organization_id,
            fps=fps,
            window_size=self.buffer.window_size,
            action_model=self.action_model,
            keypoint_model=keypoint_model,
            module=module,
            module_version=module_version,
        )
        write_behavior_cues_jsonl(path, cue_records)
        return cue_records

    def _write_output(
        self,
        report: dict[str, Any],
        events: list[dict[str, Any]],
        output_json_path: str | None,
        *,
        output_format: str,
        behavior_cues_jsonl_path: str | None = None,
        fps: float,
        camera_id: int,
        organization_id: int,
        keypoint_model: str,
        module: str,
        module_version: str,
    ) -> dict[str, Any]:
        cues_path = behavior_cues_jsonl_path
        if output_format == "behavior_cues" and output_json_path and not cues_path:
            cues_path = output_json_path

        cue_kwargs = dict(
            fps=fps,
            camera_id=camera_id,
            organization_id=organization_id,
            keypoint_model=keypoint_model,
            module=module,
            module_version=module_version,
        )
        if cues_path:
            os.makedirs(os.path.dirname(cues_path) or ".", exist_ok=True)
            report["behavior_cues"] = self._write_behavior_cues_jsonl(
                events, cues_path, **cue_kwargs,
            )

        if output_json_path and output_format != "behavior_cues":
            os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
            if behavior_cues_jsonl_path:
                report["behavior_cues_jsonl"] = os.path.basename(behavior_cues_jsonl_path)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

        report["output_format"] = output_format
        return report

    def process(
        self,
        video_path: str,
        output_json_path: str | None = None,
        annotated_output_path: str | None = None,
        *,
        behavior_cues_jsonl_path: str | None = None,
        output_format: str = VIDEO_OUTPUT_FORMAT,
        camera_id: int = DEFAULT_CAMERA_ID,
        organization_id: int = DEFAULT_ORGANIZATION_ID,
        keypoint_model: str = BEHAVIOR_CUES_KEYPOINT_MODEL,
        module: str = BEHAVIOR_CUES_MODULE,
        module_version: str = BEHAVIOR_CUES_MODULE_VERSION,
    ) -> dict[str, Any]:
        if self.pose_detector is None:
            raise RuntimeError("Video processing requires input_mode='video'.")

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

        events, raw_predictions, seen_event_keys, pair_validation_diag, motion_gate_diag = (
            self._reset_run_state()
        )

        predict_kwargs: dict[str, Any] = {}
        if self._frame_shape is not None:
            predict_kwargs["img_shape"] = self._frame_shape

        frame_idx = -1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            timestamp = frame_idx / fps

            tracks = self.pose_detector.track(frame)
            active_pairs = self._process_frame(
                frame_idx=frame_idx,
                timestamp=timestamp,
                tracks=tracks,
                events=events,
                raw_predictions=raw_predictions,
                seen_event_keys=seen_event_keys,
                pair_validation_diag=pair_validation_diag,
                motion_gate_diag=motion_gate_diag,
                predict_kwargs=predict_kwargs,
            )

            if writer is not None:
                annotated = annotate_frame(
                    frame,
                    tracks,
                    self._display_labels(),
                    frame_idx,
                    active_pairs=active_pairs,
                )
                writer.write(annotated)

        cap.release()
        if writer is not None:
            writer.release()

        pair_validation_diag.print_summary()
        motion_gate_diag.print_summary()

        report = self._build_report(
            source_name=os.path.basename(video_path),
            fps=fps,
            events=events,
            raw_predictions=raw_predictions,
            pair_validation_diag=pair_validation_diag,
            motion_gate_diag=motion_gate_diag,
            annotated_output_path=annotated_output_path,
            input_mode="video",
        )
        return self._write_output(
            report,
            events,
            output_json_path,
            output_format=output_format,
            behavior_cues_jsonl_path=behavior_cues_jsonl_path,
            fps=fps,
            camera_id=camera_id,
            organization_id=organization_id,
            keypoint_model=keypoint_model,
            module=module,
            module_version=module_version,
        )

    def process_keypoints(
        self,
        keypoints_path: str,
        output_json_path: str | None = None,
        *,
        output_format: str = KEYPOINTS_OUTPUT_FORMAT,
        camera_id: int = DEFAULT_CAMERA_ID,
        organization_id: int = DEFAULT_ORGANIZATION_ID,
        keypoint_model: str = BEHAVIOR_CUES_KEYPOINT_MODEL,
        module: str = BEHAVIOR_CUES_MODULE,
        module_version: str = BEHAVIOR_CUES_MODULE_VERSION,
        default_fps: float = 30.0,
        default_width: int = 1920,
        default_height: int = 1080,
    ) -> dict[str, Any]:
        frame_iter, meta = load_keypoints_stream(
            keypoints_path,
            default_fps=default_fps,
            default_width=default_width,
            default_height=default_height,
        )
        self._frame_shape = (meta.height, meta.width)

        self.recognizer.load_model()
        print(f"Action Model: {self.recognizer.get_model_name()}")
        print(f"Keypoints: {meta.source} ({meta.frame_count} frames @ {meta.fps} fps)")

        events, raw_predictions, seen_event_keys, pair_validation_diag, motion_gate_diag = (
            self._reset_run_state()
        )

        predict_kwargs: dict[str, Any] = {"img_shape": self._frame_shape}

        for frame in frame_iter:
            self._process_frame(
                frame_idx=frame.frame_idx,
                timestamp=frame.timestamp,
                tracks=frame.tracks,
                events=events,
                raw_predictions=raw_predictions,
                seen_event_keys=seen_event_keys,
                pair_validation_diag=pair_validation_diag,
                motion_gate_diag=motion_gate_diag,
                predict_kwargs=predict_kwargs,
            )

        pair_validation_diag.print_summary()
        motion_gate_diag.print_summary()

        report = self._build_report(
            source_name=meta.source,
            fps=meta.fps,
            events=events,
            raw_predictions=raw_predictions,
            pair_validation_diag=pair_validation_diag,
            motion_gate_diag=motion_gate_diag,
            annotated_output_path=None,
            input_mode="keypoints",
            keypoints_source=meta.source,
        )
        return self._write_output(
            report,
            events,
            output_json_path,
            output_format=output_format,
            fps=meta.fps,
            camera_id=camera_id,
            organization_id=organization_id,
            keypoint_model=keypoint_model,
            module=module,
            module_version=module_version,
        )

    @staticmethod
    def _print_predictions(
        mode: str,
        timestamp: float,
        enriched_top_k: list[dict[str, Any]],
        mapped_best: dict[str, Any] | None,
        *,
        track_a: int | None = None,
        track_b: int | None = None,
        track_id: int | None = None,
    ) -> None:
        if mode == "pair":
            header = f"[t={timestamp:.2f}s pair={track_a}-{track_b}]"
        else:
            header = f"[t={timestamp:.2f}s track={track_id}]"
        m_tag = "M=1" if mode == "single" else "M=2"
        print(f"\n{header} Top {len(enriched_top_k)} predictions ({mode}, {m_tag})")
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
