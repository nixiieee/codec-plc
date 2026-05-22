from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Quantized:
    z_q: torch.Tensor
    indices: list[torch.Tensor]
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor


class ScalarQuantizer(nn.Module):
    """Small trainable scalar quantizer using straight-through rounding."""

    def __init__(self, dim: int, bins: int = 16) -> None:
        super().__init__()
        self.bins = bins
        self.to_scalar = nn.Conv1d(dim, dim, 1)
        self.from_scalar = nn.Conv1d(dim, dim, 1)

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scaled = torch.tanh(self.to_scalar(residual))
        levels = self.bins - 1
        rounded = torch.round((scaled + 1.0) * 0.5 * levels) / levels * 2.0 - 1.0
        quantized_scalar = scaled + (rounded - scaled).detach()
        quantized = self.from_scalar(quantized_scalar)
        indices = torch.clamp(((rounded + 1.0) * 0.5 * levels).long(), 0, levels)
        return quantized, indices


class VectorQuantizer(nn.Module):
    def __init__(self, dim: int, codebook_size: int = 256) -> None:
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [B, C, T] -> [B*T, C]
        flat = residual.permute(0, 2, 1).reshape(-1, residual.shape[1])
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1)
        )
        indices = distances.argmin(dim=1)
        z_q = self.codebook(indices).view(residual.shape[0], residual.shape[2], residual.shape[1])
        z_q = z_q.permute(0, 2, 1).contiguous()
        z_q_st = residual + (z_q - residual).detach()
        loss = F.mse_loss(z_q, residual.detach()) + 0.25 * F.mse_loss(z_q.detach(), residual)
        return z_q_st, indices.view(residual.shape[0], residual.shape[2]), loss


class ResidualScalarVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim: int,
        scalar_quantizers: int = 2,
        vector_quantizers: int = 6,
        codebook_size: int = 256,
    ) -> None:
        super().__init__()
        self.scalar = nn.ModuleList([ScalarQuantizer(dim) for _ in range(scalar_quantizers)])
        self.vector = nn.ModuleList(
            [VectorQuantizer(dim, codebook_size=codebook_size) for _ in range(vector_quantizers)]
        )

    @property
    def num_quantizers(self) -> int:
        return len(self.scalar) + len(self.vector)

    def forward(self, z: torch.Tensor, n_active: int | None = None) -> Quantized:
        n_active = self.num_quantizers if n_active is None else n_active
        if n_active < 1 or n_active > self.num_quantizers:
            raise ValueError(f"n_active must be in [1, {self.num_quantizers}], got {n_active}")

        z_q = torch.zeros_like(z)
        residual = z
        indices: list[torch.Tensor] = []
        losses = []

        for quantizer in self.scalar[: min(n_active, len(self.scalar))]:
            q, idx = quantizer(residual)
            z_q = z_q + q
            residual = residual - q
            indices.append(idx)

        remaining = n_active - len(indices)
        for quantizer in self.vector[:remaining]:
            q, idx, loss = quantizer(residual)
            z_q = z_q + q
            residual = residual - q
            indices.append(idx)
            losses.append(loss)

        zero = z.sum() * 0.0
        codebook_loss = torch.stack(losses).mean() if losses else zero
        commitment_loss = F.mse_loss(z_q.detach(), z)
        return Quantized(z_q=z_q, indices=indices, commitment_loss=commitment_loss, codebook_loss=codebook_loss)
