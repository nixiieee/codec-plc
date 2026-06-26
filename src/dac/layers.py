from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import weight_norm


def wn_conv1d(*args, **kwargs) -> nn.Conv1d:
    return weight_norm(nn.Conv1d(*args, **kwargs))


def wn_conv_transpose1d(*args, **kwargs) -> nn.ConvTranspose1d:
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def wn_conv2d(*args, act: bool = True, **kwargs) -> nn.Module:
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


@torch.jit.script
def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    y = x.reshape(shape[0], shape[1], -1)
    y = y + (alpha + 1e-9).reciprocal() * torch.sin(alpha * y).pow(2)
    return y.reshape(shape)


class Snake1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake(x, self.alpha)
