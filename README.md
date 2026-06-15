# CTR-GCN / PoseC3D Action Recognition Pipeline

Inference-only CCTV pipeline for aggressive-interaction cue detection on NTU120 actions, with project-specific class mapping (`push_shove`, `swing_attempt`, `grapple_clinch`, `stumble_recover`, `aggressive_posture`).

## Pipeline overview

**Mode 1 — Video (local dev)**

```
Video → YOLO Pose → ByteTrack → Interaction Detection → Skeleton Buffer
     → Pair validation → Motion gate → PoseC3D / CTR-GCN
     → Debug JSON + Annotated MP4
```

**Mode 2 — Keypoints (Integration)**

```
YOLO keypoints JSONL → Interaction Detection → Skeleton Buffer
     → Pair validation → Motion gate → PoseC3D / CTR-GCN
     → behavior_cues JSONL
```

Runs YOLO + tracking on integration side. This service consumes tracked keypoints and emits format cues.

## Features

- **Default backend:** PoseC3D NTU120 (`joint.pth`) with **limb** heatmaps (configurable in `utils/config.py`)
- **Alternate backend:** CTR-GCN NTU120 (`--action-model ctrgcn`)
- **Pair inference (M=2):** `push_shove`, `swing_attempt`, `grapple_clinch`, `aggressive_posture`
- **Single inference (M=1):** `stumble_recover` only
- **Gates:** pair geometry validation, skeleton motion-energy filter
- **Two CLI modes:** video in → debug outputs; keypoints in →  `behavior_cues` JSONL

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_models.sh
```

Place a custom YOLO pose weights file at `yolo_models/best.pt` if you have one; otherwise `yolo11l-pose.pt` is used (auto-downloaded by Ultralytics on first run if missing).

## Model weights

Weights are **not** committed to git. After clone, run:

```bash
bash scripts/download_models.sh
```

See [pretrained_model/README.md](pretrained_model/README.md) for manual download links.

| Model | Path |
|-------|------|
| PoseC3D NTU120 XSub (default) | `pretrained_model/posec3d/joint.pth` |
| CTR-GCN NTU120 joint | `pretrained_model/CTRGCN_NTU120_CSub_joint_84.9/runs-58-57072.pt` |
| YOLO pose (video mode only) | `yolo_models/yolo11l-pose.pt` or `yolo_models/best.pt` |

### Defaults (`utils/config.py`)

| Setting | Default |
|---------|---------|
| `DEFAULT_ACTION_MODEL` | `posec3d` |
| `POSEC3D_HEATMAP_MODE` | `limb` |
| `DEFAULT_WINDOW_SIZE` | `30` frames |
| `INFERENCE_STRIDE` | `25` (CLI default `--inference-stride` is `15`) |
| `EVENT_MIN_CONFIDENCE` | `0.12` |

Override on the CLI with `--action-model`, `--posec3d-heatmap`, etc.

## Usage

`--input-mode` fixes both input and output — there is no separate `--output-format` flag.

| Mode | Input | Output |
|------|-------|--------|
| `video` (default) | MP4 file | `outputs/<stem>.json` + `outputs/<stem>_annotated.mp4` |
| `keypoints` | keypoints JSONL | `outputs/<stem>_cues.jsonl` (no video) |

### Mode 1 — Video (local dev / testing)

```bash
python3 src/main.py \
  --video Test-Videos/Grapple-Clinch/grapple_high_7.mp4 \
  --device cpu
```

Defaults to **PoseC3D + limb** heatmaps. Writes:

- `outputs/grapple_high_7.json` — full debug report (`events`, `raw_predictions`, `pair_validation`, `motion_gate`, …)
- `outputs/grapple_high_7_annotated.mp4` — overlays with mapped cue labels

Skip the annotated video:

```bash
python3 src/main.py --video path/to/clip.mp4 --no-annotated-video --device cpu
```

### Mode 2 — Keypoints (Integration End)

```bash
python3 src/main.py \
  --input-mode keypoints \
  --keypoints outputs/grapple_high_7_keypoints.jsonl \
  --device cpu
```

Writes `outputs/grapple_high_7_cues.jsonl` — one `behavior_cues` JSON object per line:

- `type`: `"behavior_cues"`
- `family`: `"aggressive_interaction"`
- `cues[].code`: `push_shove`, `swing_attempt`, `grapple_clinch`, `stumble_recover`, or `aggressive_posture`

Optional delivery metadata: `--camera-id`, `--organization-id`, `--keypoint-model`, `--module-name`, `--module-version`.

#### Simulate  keypoints (from a test video)

```bash
python3 scripts/export_keypoints.py \
  --video Test-Videos/Grapple-Clinch/grapple_high_7.mp4 \
  --device cpu
# → outputs/grapple_high_7_keypoints.jsonl

python3 src/main.py \
  --input-mode keypoints \
  --keypoints outputs/grapple_high_7_keypoints.jsonl \
  --device cpu
# → outputs/grapple_high_7_cues.jsonl
```

### Keypoints input schema (JSONL)

One JSON object per line (`.jsonl`). Also accepts a JSON array or `{"frames": [...]}`.

```json
{
  "frame": 265,
  "timestamp": 8.83,
  "fps": 30.0,
  "width": 1920,
  "height": 1080,
  "tracks": [
    {
      "track_id": 1,
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.92,
      "keypoints": [[x, y, conf], ...]
    }
  ]
}
```

**Requirements**

- `track_id` — stable across frames (tracking)
- `bbox` — `[x1, y1, x2, y2]` in pixels
- `keypoints` — **17 COCO joints, YOLO order** (see `src/joint_mapper.py`)
- `fps` / `width` / `height` — on each line or at least the first frame

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--input-mode` | `video` | `video` → debug JSON + annotated MP4; `keypoints` → behavior_cues JSONL |
| `--video` | — | Input video (required for `video` mode) |
| `--keypoints` | — | Input keypoints JSONL (required for `keypoints` mode) |
| `--action-model` | `posec3d` | `posec3d` or `ctrgcn` |
| `--posec3d-heatmap` | `limb` | `limb` (official `joint.pth`) or `keypoint` (COCO-17 joints) |
| `--window-size` | `30` | Skeleton buffer frames: 30, 48, 60, 90, 100, 120 |
| `--inference-stride` | `15` | Run action model every N frames once buffer is full |
| `--interaction-distance` | `250` | Max pixel distance between bbox centers for pair gating |
| `--interaction-frames` | `30` | Consecutive close frames before pair inference |
| `--event-min-confidence` | `0.12` | Min confidence for events / overlays |
| `--motion-gate` / `--no-motion-gate` | on | Skip inference when skeleton motion is below threshold |
| `--motion-threshold-pair` | `0.020` | Pair motion-energy floor |
| `--motion-threshold-single` | `0.012` | Single-track motion-energy floor |
| `--camera-id` | `0` | `camera_id` in behavior_cues output (keypoints mode) |
| `--organization-id` | `0` | `organization_id` in behavior_cues output (keypoints mode) |
| `--tracker` | `bytetrack.yaml` | ByteTrack or BoT-SORT (video mode only) |
| `--no-annotated-video` | off | Skip annotated MP4 (video mode only) |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--output` | auto | Override output path |
| `--show-mapping` | — | Print COCO-17 → NTU-25 joint mapping and exit |

## Project structure

```
src/
  main.py                 CLI entry point
  video_processor.py      Pipeline orchestration (video + keypoints paths)
  keypoints_input.py      keypoints JSONL loader
  pose_detector.py        YOLO pose + tracking (video mode)
  interaction_detector.py Pair proximity gating
  pair_validator.py       Geometric pair validation
  motion_energy.py        Motion-energy gate
  skeleton_buffer.py      Rolling skeleton windows
  joint_mapper.py         COCO-17 → NTU-25 mapping (CTR-GCN)
  recognizers/            PoseC3D and CTR-GCN backends
utils/
  config.py               Thresholds, model paths, I/O mode defaults
  class_mapping.py        NTU120 → project cue codes
  behavior_cues.py        delivery JSONL serializer
scripts/
  download_models.sh      Fetch pretrained weights
  export_keypoints.py     Video → keypoints JSONL ( format simulator)
data/                     NTU120 label file
pretrained_model/         Checkpoints (gitignored)
yolo_models/              YOLO pose weights (gitignored)
outputs/                  JSON, JSONL, annotated videos (gitignored)
```

## Class mapping

NTU120 classes are mapped to project cues in `utils/class_mapping.py`. Unmapped predictions become `Normal` and are suppressed from events / delivery output.

| Cue | Inference path |
|-----|----------------|
| `push_shove` | pair (M=2) |
| `swing_attempt` | pair (M=2) |
| `grapple_clinch` | pair (M=2) |
| `aggressive_posture` | pair (M=2) |
| `stumble_recover` | single (M=1) |

## License

Model weights are subject to their original licenses (OpenMMLab/pyskl, CTR-GCN, Ultralytics). Application code is provided as-is for internal use.
