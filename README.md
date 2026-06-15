# CTR-GCN / PoseC3D Action Recognition Pipeline

Inference-only CCTV pipeline for two-person interaction recognition on NTU120 actions, with project-specific class mapping (`push_shove`, `swing_attempt`, `grapple_clinch`, etc.).

```
Video → YOLO Pose → ByteTrack → Interaction Detection → Skeleton Buffer
     → Action Model (CTR-GCN | PoseC3D) → NTU120 → Mapped JSON + Annotated MP4
```

## Features

- **Dual backends:** `--action-model ctrgcn` (default) or `--action-model posec3d`
- **Pair inference:** runs when two tracks stay within distance for N consecutive frames
- **Outputs:** JSON events + annotated video with mapped target classes

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
| CTR-GCN NTU120 joint | `pretrained_model/CTRGCN_NTU120_CSub_joint_84.9/runs-58-57072.pt` |
| PoseC3D NTU120 XSub | `pretrained_model/posec3d/joint.pth` |
| YOLO pose | `yolo_models/yolo11l-pose.pt` or `yolo_models/best.pt` |

## Usage

```bash
# Default: video in, behavior_cues JSONL out
python3 src/main.py --video path/to/clip.mp4 --device cpu

# Internal debug report (full diagnostics JSON)
python3 src/main.py --video path/to/clip.mp4 --output-format report --device cpu

# PoseC3D backend
python3 src/main.py --video path/to/clip.mp4 --action-model posec3d --device cpu

# Client delivery fields
python3 src/main.py --video path/to/clip.mp4 \
  --output-format behavior_cues \
  --camera-id 0 --organization-id 0 \
  --keypoint-model yolo11n-pose
```

Future: `--input-mode keypoints --keypoints client_tracks.jsonl` (client-supplied YOLO keypoints).

### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--input-mode` | `video` | `video` or `keypoints` (keypoints: coming soon) |
| `--output-format` | `behavior_cues` | `behavior_cues` (JSONL) or `report` (debug JSON) |
| `--action-model` | `posec3d` | `ctrgcn` or `posec3d` |
| `--window-size` | `30` | Skeleton buffer frames (30, 48, 60, 90, 100, 120) |
| `--interaction-distance` | `250` | Max pixel distance for pair gating |
| `--interaction-frames` | `30` | Consecutive close frames before inference |
| `--camera-id` | `0` | `camera_id` in behavior_cues records |
| `--organization-id` | `0` | `organization_id` in behavior_cues records |
| `--tracker` | `bytetrack.yaml` | `bytetrack.yaml` or `botsort.yaml` |
| `--no-annotated-video` | off | Skip annotated MP4 |
| `--device` | `cuda` | `cuda` or `cpu` |

Outputs default to `outputs/<video_stem>_cues.jsonl` (behavior_cues) or `outputs/<video_stem>.json` (report), plus `outputs/<video_stem>_annotated.mp4` unless disabled.

## Project structure

```
src/                  Pipeline source (pose, tracking, recognizers)
utils/                Config, class mapping, device helpers
data/                 NTU120 label file
pretrained_model/     Downloaded checkpoints (gitignored)
yolo_models/          YOLO pose weights (gitignored)
outputs/              JSON + annotated videos (gitignored)
scripts/              Model download helper
```

## Class mapping

NTU120 classes are mapped to project targets in `utils/class_mapping.py`. Unmapped predictions become `Normal`.

## License

Model weights are subject to their original licenses (OpenMMLab/pyskl, CTR-GCN, Ultralytics). Application code is provided as-is for internal use.
