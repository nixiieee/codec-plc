from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .layers import wn_conv1d


@dataclass
class DacQuantized:
    z_q: torch.Tensor
    codes: torch.Tensor
    latents: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor


class VectorQuantize(nn.Module):
    """Factorized, L2-normalized VQ layer used by DAC."""

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim)
        self.in_proj = wn_conv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = wn_conv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.in_proj(residual)
        z_q, indices = self.decode_latents(z_e)
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean(dim=(1, 2))
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean(dim=(1, 2))
        z_q = z_e + (z_q - z_e).detach()
        z_q = self.out_proj(z_q)
        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.codebook.weight)

    def decode_code(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embed_code(indices).transpose(1, 2).contiguous()

    def decode_latents(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, dim, frames = latents.shape
        encodings = latents.permute(0, 2, 1).reshape(batch * frames, dim)
        encodings = F.normalize(encodings, dim=1)
        codebook = F.normalize(self.codebook.weight, dim=1)
        similarity = encodings @ codebook.t()
        indices = similarity.argmax(dim=1).view(batch, frames)
        return self.decode_code(indices), indices


class ResidualVectorQuantize(nn.Module):
    """SoundStream-style residual vector quantizer as used by DAC."""

    def __init__(
        self,
        input_dim: int,
        n_codebooks: int = 12,
        codebook_size: int = 1024,
        codebook_dim: int | list[int] = 8,
        quantizer_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dims = [int(codebook_dim)] * int(n_codebooks)
        else:
            codebook_dims = [int(value) for value in codebook_dim]
        if len(codebook_dims) != int(n_codebooks):
            raise ValueError("codebook_dim list length must match n_codebooks")

        self.n_codebooks = int(n_codebooks)
        self.codebook_size = int(codebook_size)
        self.codebook_dim = codebook_dims
        self.quantizer_dropout = float(quantizer_dropout)
        self.quantizers = nn.ModuleList(
            [VectorQuantize(input_dim, codebook_size, dim) for dim in codebook_dims]
        )

    def forward(self, z: torch.Tensor, n_quantizers: int | None = None) -> DacQuantized:
        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        n_quantizers = int(n_quantizers)
        if n_quantizers < 1 or n_quantizers > self.n_codebooks:
            raise ValueError(f"n_quantizers must be in [1, {self.n_codebooks}], got {n_quantizers}")

        if self.training and self.quantizer_dropout > 0:
            active_per_batch = torch.full((z.shape[0],), self.n_codebooks + 1, device=z.device)
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],), device=z.device)
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            active_per_batch[:n_dropout] = dropout[:n_dropout]
        else:
            active_per_batch = torch.full((z.shape[0],), n_quantizers, device=z.device)

        z_q = torch.zeros_like(z)
        residual = z
        commitment_loss = z.sum() * 0.0
        codebook_loss = z.sum() * 0.0
        codebook_indices: list[torch.Tensor] = []
        latents: list[torch.Tensor] = []

        for index, quantizer in enumerate(self.quantizers):
            if not self.training and index >= n_quantizers:
                break
            z_q_i, commitment_i, codebook_i, indices_i, z_e_i = quantizer(residual)
            mask = (torch.full((z.shape[0],), index, device=z.device) < active_per_batch).to(z.dtype)
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i
            commitment_loss = commitment_loss + (commitment_i * mask).mean()
            codebook_loss = codebook_loss + (codebook_i * mask).mean()
            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        return DacQuantized(
            z_q=z_q,
            codes=torch.stack(codebook_indices, dim=1),
            latents=torch.cat(latents, dim=1),
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
        )

    def from_codes(self, codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_q: torch.Tensor | float = 0.0
        projected = []
        for index in range(codes.shape[1]):
            z_p_i = self.quantizers[index].decode_code(codes[:, index, :])
            projected.append(z_p_i)
            z_q = z_q + self.quantizers[index].out_proj(z_p_i)
        return z_q, torch.cat(projected, dim=1), codes
