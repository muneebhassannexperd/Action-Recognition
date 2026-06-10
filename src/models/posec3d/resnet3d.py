"""ResNet3dSlowOnly backbone for PoseC3D (ported from pyskl/mmaction)."""

from __future__ import annotations

import torch.nn as nn
from torch.nn.modules.utils import _triple

from .conv_module import ConvModule


class BasicBlock3d(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: tuple[int, int] = (1, 1),
        downsample: nn.Module | None = None,
        inflate: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = ConvModule(
            inplanes,
            planes,
            3 if inflate else (1, 3, 3),
            stride=(stride[0], stride[1], stride[1]),
            padding=1 if inflate else (0, 1, 1),
        )
        self.conv2 = ConvModule(
            planes,
            planes * self.expansion,
            3 if inflate else (1, 3, 3),
            stride=1,
            padding=1 if inflate else (0, 1, 1),
            act=False,
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class Bottleneck3d(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: tuple[int, int] = (1, 1),
        downsample: nn.Module | None = None,
        inflate: bool = True,
        inflate_style: str = "3x1x1",
    ) -> None:
        super().__init__()
        assert inflate_style in ("3x1x1", "3x3x3")
        mode = "no_inflate" if not inflate else inflate_style
        conv1_kernel_size = {"no_inflate": 1, "3x1x1": (3, 1, 1), "3x3x3": 1}
        conv1_padding = {"no_inflate": 0, "3x1x1": (1, 0, 0), "3x3x3": 0}
        conv2_kernel_size = {"no_inflate": (1, 3, 3), "3x1x1": (1, 3, 3), "3x3x3": 3}
        conv2_padding = {"no_inflate": (0, 1, 1), "3x1x1": (0, 1, 1), "3x3x3": 1}

        self.conv1 = ConvModule(
            inplanes,
            planes,
            conv1_kernel_size[mode],
            stride=1,
            padding=conv1_padding[mode],
        )
        self.conv2 = ConvModule(
            planes,
            planes,
            conv2_kernel_size[mode],
            stride=(stride[0], stride[1], stride[1]),
            padding=conv2_padding[mode],
        )
        self.conv3 = ConvModule(planes, planes * self.expansion, 1, act=False)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class ResNet3d(nn.Module):
    arch_settings = {
        50: (Bottleneck3d, (3, 4, 6, 3)),
    }

    def __init__(
        self,
        depth: int = 50,
        in_channels: int = 17,
        base_channels: int = 32,
        num_stages: int = 3,
        stage_blocks: tuple[int, ...] | None = (4, 6, 3),
        out_indices: tuple[int, ...] = (2,),
        spatial_strides: tuple[int, ...] = (2, 2, 2),
        temporal_strides: tuple[int, ...] = (1, 1, 2),
        conv1_kernel: tuple[int, int, int] = (1, 7, 7),
        conv1_stride: tuple[int, int] = (1, 1),
        pool1_stride: tuple[int, int] = (1, 1),
        inflate: tuple[int, ...] = (0, 1, 1),
        inflate_style: str = "3x1x1",
    ) -> None:
        super().__init__()
        block, default_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks or default_blocks[:num_stages]
        self.out_indices = out_indices
        self.inplanes = base_channels

        self.conv1 = ConvModule(
            in_channels,
            base_channels,
            kernel_size=conv1_kernel,
            stride=(conv1_stride[0], conv1_stride[1], conv1_stride[1]),
            padding=tuple((k - 1) // 2 for k in _triple(conv1_kernel)),
        )
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(pool1_stride[0], pool1_stride[1], pool1_stride[1]),
            padding=(0, 1, 1),
        )

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            planes = base_channels * 2**i
            stride = (temporal_strides[i], spatial_strides[i])
            layer = self._make_layer(
                block,
                planes,
                num_blocks,
                stride=stride,
                inflate=inflate[i],
                inflate_style=inflate_style,
            )
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, layer)
            self.res_layers.append(layer_name)
            self.inplanes = planes * block.expansion

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: tuple[int, int],
        inflate: int,
        inflate_style: str,
    ) -> nn.Sequential:
        downsample = None
        if stride[1] != 1 or self.inplanes != planes * block.expansion:
            downsample = ConvModule(
                self.inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=(stride[0], stride[1], stride[1]),
                act=False,
            )

        inflate_flags = (inflate == 1,) * blocks
        layers = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                inflate=inflate_flags[0],
                inflate_style=inflate_style,
            )
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    inflate=inflate_flags[i],
                    inflate_style=inflate_style,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            x = getattr(self, layer_name)(x)
            if i in self.out_indices:
                outs.append(x)
        return outs[0] if len(outs) == 1 else tuple(outs)


class ResNet3dSlowOnly(ResNet3d):
    def __init__(self, conv1_kernel=(1, 7, 7), inflate=(0, 0, 1, 1), **kwargs):
        super().__init__(conv1_kernel=conv1_kernel, inflate=inflate, **kwargs)
