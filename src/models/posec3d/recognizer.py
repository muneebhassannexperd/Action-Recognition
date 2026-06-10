"""PoseC3D Recognizer3D + I3DHead (inference-only)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet3d import ResNet3dSlowOnly


class I3DHead(nn.Module):
    def __init__(self, num_classes: int, in_channels: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            x = F.adaptive_avg_pool3d(x, 1).view(x.shape[0], -1)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc_cls(x)


class PoseC3DRecognizer(nn.Module):
    """SlowOnly R50 PoseC3D for NTU120 (17-channel pose heatmaps)."""

    def __init__(self, num_classes: int = 120, in_channels: int = 17) -> None:
        super().__init__()
        self.backbone = ResNet3dSlowOnly(
            depth=50,
            in_channels=in_channels,
            base_channels=32,
            num_stages=3,
            stage_blocks=(4, 6, 3),
            out_indices=(2,),
            conv1_stride=(1, 1),
            pool1_stride=(1, 1),
            inflate=(0, 1, 1),
            spatial_strides=(2, 2, 2),
            temporal_strides=(1, 1, 2),
        )
        self.cls_head = I3DHead(num_classes=num_classes, in_channels=512, dropout=0.5)
        self.test_cfg = dict(average_clips="prob")

    def extract_feat(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.backbone(imgs)

    def average_clip(self, cls_score: torch.Tensor) -> torch.Tensor:
        """Average over temporal segments with softmax (test_cfg average_clips='prob')."""
        if cls_score.ndim == 2:
            return F.softmax(cls_score, dim=1)
        assert cls_score.ndim == 3
        return F.softmax(cls_score, dim=2).mean(dim=1)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            imgs: (N, num_segs, C, T, H, W) or (N, C, T, H, W)
        Returns:
            Class probabilities, shape (N, num_classes).
        """
        if imgs.ndim == 5:
            imgs = imgs.unsqueeze(1)

        batches, num_segs = imgs.shape[:2]
        x = imgs.reshape((-1,) + imgs.shape[2:])
        feat = self.extract_feat(x)
        cls_score = self.cls_head(feat)
        cls_score = cls_score.reshape(batches, num_segs, cls_score.shape[-1])
        return self.average_clip(cls_score)
