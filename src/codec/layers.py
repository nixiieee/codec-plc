from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Conv1d):
    """1D convolution with left padding only."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left_pad = (self.kernel_size[0] - 1) * self.dilation[0]
        return super().forward(F.pad(x, (left_pad, 0)))


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size, dilation=dilation),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            CausalConv1d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)
