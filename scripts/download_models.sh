#!/usr/bin/env bash
# Download inference weights into pretrained_model/ and yolo_models/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p pretrained_model/posec3d
mkdir -p pretrained_model/CTRGCN_NTU120_CSub_joint_84.9
mkdir -p yolo_models

download() {
  local url="$1"
  local dest="$2"
  if [[ -f "$dest" ]]; then
    echo "OK (exists): $dest"
    return 0
  fi
  echo "Downloading: $dest"
  curl -L --fail -o "$dest" "$url"
}

# PoseC3D NTU120 XSub SlowOnly R50
download \
  "http://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_ntu120_xsub/joint.pth" \
  "pretrained_model/posec3d/joint.pth"

# CTR-GCN NTU120 XSub 3D joint (pyskl format; rename path for --ctrgcn-weights if needed)
CTRGCN_DEST="pretrained_model/CTRGCN_NTU120_CSub_joint_84.9/runs-58-57072.pt"
if [[ ! -f "$CTRGCN_DEST" ]]; then
  echo "CTR-GCN: downloading pyskl NTU120 joint checkpoint to ${CTRGCN_DEST}.pyskl"
  download \
    "http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu120_xsub_3dkp/j.pth" \
    "${CTRGCN_DEST}.pyskl"
  echo "NOTE: If inference fails to load state dict, copy your CTR-GCN runs-58-57072.pt to:"
  echo "  $CTRGCN_DEST"
fi

# YOLO11l pose (optional prefetch)
download \
  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-pose.pt" \
  "yolo_models/yolo11l-pose.pt"

echo ""
echo "Done. Optional: add yolo_models/best.pt for a custom fine-tuned pose model."
