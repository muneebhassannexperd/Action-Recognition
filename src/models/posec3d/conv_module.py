"""Minimal Conv3d + BN3d + ReLU block matching mmcv ConvModule layout."""

from __future__ import annotations

import torch.nn as nn


class ConvModule(nn.Module):
    """Conv-BN-ReLU with ``.conv`` and ``.bn`` submodules for checkpoint compatibility."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        bias: bool = False,
        act: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.activate = nn.ReLU(inplace=True) if act else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate is not None:
            x = self.activate(x)
        return x
