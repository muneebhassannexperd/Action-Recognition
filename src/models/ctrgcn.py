"""
CTR-GCN model architecture (inference-only).

Vendored from https://github.com/Uason-Chen/CTR-GCN (model/ctrgcn.py) to load
official pretrained checkpoints with keys ``l1``..``l10`` and ``fc``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .ntu_graph import Graph


def conv_init(conv: nn.Conv2d) -> None:
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn: nn.BatchNorm2d, scale: float) -> None:
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class MultiScale_TemporalConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        dilations=(1, 2),
        residual=True,
        residual_kernel_size=1,
    ):
        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if isinstance(kernel_size, list):
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(branch_channels, branch_channels, kernel_size=ks, stride=stride, dilation=d),
            )
            for ks, d in zip(kernel_size, dilations)
        ])
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels),
        ))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels),
        ))

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

    def forward(self, x):
        res = self.residual(x)
        branch_outs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outs, dim=1)
        return out + res


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels in (3, 9):
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        self.conv1 = nn.Conv2d(in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x, a=None, alpha=1):
        x1 = self.conv1(x).mean(-2)
        x2 = self.conv2(x).mean(-2)
        x3 = self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (a.unsqueeze(0).unsqueeze(0) if a is not None else 0)
        return torch.einsum("ncuv,nctv->nctu", x1, x3)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.bn(self.conv(x))


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, a, coff_embedding=4, adaptive=True, residual=True):
        super().__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = a.shape[0]
        self.convs = nn.ModuleList([CTRGC(in_channels, out_channels) for _ in range(self.num_subset)])

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0

        self.PA = nn.Parameter(torch.from_numpy(a.astype(np.float32)))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = None
        a = self.PA
        for i, conv in enumerate(self.convs):
            z = conv(x, a[i], self.alpha)
            y = z if y is None else y + z
        y = self.bn(y)
        y = self.relu(y + self.down(x))
        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, a, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=(1, 2)):
        super().__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, a, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(
            out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations, residual=False,
        )
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        return self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))


class Model(nn.Module):
    """Official CTR-GCN classifier. Input: (N, C, T, V, M)."""

    def __init__(self, num_class=120, num_point=25, num_person=2, in_channels=3, graph=None, graph_args=None, drop_out=0, adaptive=True):
        super().__init__()
        graph_args = graph_args or {"labeling_mode": "spatial"}
        self.graph = Graph(**graph_args)

        a = self.graph.A
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, a, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, a, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, a, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, a, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel * 2, a, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel * 2, base_channel * 2, a, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel * 2, base_channel * 2, a, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel * 2, base_channel * 4, a, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel * 4, base_channel * 4, a, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel * 4, base_channel * 4, a, adaptive=adaptive)
        self.fc = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        bn_init(self.data_bn, 1)
        self.drop_out = nn.Dropout(drop_out) if drop_out else (lambda x: x)

    def forward(self, x):
        if len(x.shape) == 3:
            n, t, vc = x.shape
            x = x.view(n, t, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)

        n, c, t, v, m = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(n, m * v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)

        for layer in (self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7, self.l8, self.l9, self.l10):
            x = layer(x)

        c_new = x.size(1)
        x = x.view(n, m, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        return self.fc(x)
