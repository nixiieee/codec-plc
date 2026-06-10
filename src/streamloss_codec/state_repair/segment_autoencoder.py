from __future__ import annotations

import torch
from torch import nn

from streamloss_codec.codec.layers import CausalConv1d, CausalResidualBlock


class SegmentRepairEncoder(nn.Module):
    """Encode one 20 ms DRED-restored segment into a 768-value packet embedding."""

    def __init__(self, channels: int = 136, latent_dim: int = 96, latent_frames: int = 8) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_frames = latent_frames
        self.net = nn.Sequential(
            CausalConv1d(1, channels, 7),
            nn.SiLU(),
            CausalResidualBlock(channels, dilation=1),
            CausalConv1d(channels, channels, 5, stride=2),
            nn.SiLU(),
            CausalResidualBlock(channels, dilation=1),
            CausalConv1d(channels, channels, 5, stride=2),
            nn.SiLU(),
            CausalConv1d(channels, channels, 5, stride=2),
            nn.SiLU(),
            CausalConv1d(channels, latent_dim, 5, stride=5),
        )

    @property
    def embedding_dim(self) -> int:
        return self.latent_dim * self.latent_frames

    def encode_latent(self, segment: torch.Tensor) -> torch.Tensor:
        if segment.dim() == 2:
            segment = segment.unsqueeze(1)
        if segment.dim() != 3:
            raise ValueError(f"segment must have shape [batch, samples] or [batch, 1, samples], got {segment.shape}")
        latent = self.net(segment)
        if latent.shape[-1] != self.latent_frames:
            raise ValueError(f"expected {self.latent_frames} latent frames, got {latent.shape[-1]}")
        return latent

    def forward(self, segment: torch.Tensor) -> torch.Tensor:
        return self.encode_latent(segment).flatten(start_dim=1)


class SegmentRepairAutoencoder(nn.Module):
    """Pretraining autoencoder for the segment repair encoder.

    The decoder predicts a correction by default. Adding that correction to the
    DRED input makes the initial model behave like passthrough DRED, instead of
    forcing the network to synthesize a whole packet from scratch.
    """

    def __init__(self, channels: int = 136, latent_dim: int = 96, latent_frames: int = 8, residual: bool = True) -> None:
        super().__init__()
        self.residual = residual
        self.encoder = SegmentRepairEncoder(channels=channels, latent_dim=latent_dim, latent_frames=latent_frames)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, channels, 5, stride=5),
            nn.SiLU(),
            CausalResidualBlock(channels, dilation=1),
            nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1),
            nn.SiLU(),
            CausalResidualBlock(channels, dilation=1),
            nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1),
            nn.SiLU(),
            CausalConv1d(channels, 1, 7),
        )
        self._init_residual_output()

    def _init_residual_output(self) -> None:
        final = self.decoder[-1]
        if isinstance(final, nn.Conv1d):
            nn.init.zeros_(final.weight)
            if final.bias is not None:
                nn.init.zeros_(final.bias)

    @property
    def embedding_dim(self) -> int:
        return self.encoder.embedding_dim

    def decode_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.dim() == 2:
            latent = embedding.view(embedding.shape[0], self.encoder.latent_dim, self.encoder.latent_frames)
        elif embedding.dim() == 3:
            latent = embedding
        else:
            raise ValueError(f"embedding must have shape [batch, 768] or [batch, channels, frames], got {embedding.shape}")
        return self.decoder(latent).squeeze(1)

    def forward(self, segment: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(segment)
        correction = self.decode_embedding(embedding)
        if self.residual:
            if segment.dim() == 3:
                segment = segment.squeeze(1)
            reconstructed = (segment[:, : correction.shape[-1]] + correction).clamp(-1.0, 1.0)
        else:
            reconstructed = correction.tanh()
        return reconstructed, embedding
