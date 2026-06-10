# Pretrained model weights

These files are **not** stored in git. Download them after cloning the repository.

## Quick download

From the project root:

```bash
bash scripts/download_models.sh
```

## Manual download

### PoseC3D (required for `--action-model posec3d`)

```bash
mkdir -p pretrained_model/posec3d
curl -L -o pretrained_model/posec3d/joint.pth \
  "http://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_ntu120_xsub/joint.pth"
```

### CTR-GCN NTU120 joint (required for `--action-model ctrgcn`)

This pipeline uses the CTR-GCN NTU120 cross-subject **joint** checkpoint:

```
pretrained_model/CTRGCN_NTU120_CSub_joint_84.9/runs-58-57072.pt
```

Obtain from the [CTR-GCN repository](https://github.com/Uason-Chen/CTR-GCN) releases/checkpoints, or copy from your existing training export. The OpenMMLab pyskl alternative (different filename) is:

```
http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu120_xsub_3dkp/j.pth
```

If you use the pyskl `.pth` file, pass it explicitly:

```bash
python3 src/main.py --video clip.mp4 --ctrgcn-weights path/to/j.pth
```

### Other bundled folders

Additional checkpoint folders under `pretrained_model/` (bone, motion, NTU60) are optional and not required for inference.
