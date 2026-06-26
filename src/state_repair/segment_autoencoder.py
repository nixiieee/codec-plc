from __future__ import annotations

import torch
from torch import nn

from codec.layers import CausalConv1d, CausalResidualBlock


class SegmentRepairEncoder(nn.Module):
    """Encode one 20 ms DRED-restored segment into a packet embedding."""

    def __init__(
        self,
        channels: int = 136,
        latent_dim: int = 96,
        latent_frames: int = 8,
        encoder_rates: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.latent_frames = int(latent_frames)
        self.encoder_rates = [2, 2, 2, 5] if encoder_rates is None else [int(rate) for rate in encoder_rates]
        if any(rate <= 0 for rate in self.encoder_rates):
            raise ValueError(f"encoder_rates must be positive, got {self.encoder_rates}")

        layers: list[nn.Module] = [
            CausalConv1d(1, channels, 7),
            nn.SiLU(),
            CausalResidualBlock(channels, dilation=1),
        ]
        in_channels = channels
        for index, rate in enumerate(self.encoder_rates):
            out_channels = self.latent_dim if index == len(self.encoder_rates) - 1 else channels
            layers.extend([CausalConv1d(in_channels, out_channels, max(3, rate), stride=rate), nn.SiLU()])
            in_channels = out_channels
            if index < len(self.encoder_rates) - 1:
                layers.append(CausalResidualBlock(in_channels, dilation=1))
        self.net = nn.Sequential(*layers)

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
    """Pretraining autoencoder for the segment repair encoder."""

    def __init__(
        self,
        channels: int = 136,
        latent_dim: int = 96,
        latent_frames: int = 8,
        residual: bool = True,
        encoder_rates: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.residual = bool(residual)
        rates = [2, 2, 2, 5] if encoder_rates is None else [int(rate) for rate in encoder_rates]
        self.encoder = SegmentRepairEncoder(
            channels=channels,
            latent_dim=latent_dim,
            latent_frames=latent_frames,
            encoder_rates=rates,
        )
        decoder_layers: list[nn.Module] = []
        in_channels = int(latent_dim)
        for rate in reversed(rates):
            decoder_layers.extend(
                [
                    nn.ConvTranspose1d(in_channels, channels, kernel_size=rate, stride=rate),
                    nn.SiLU(),
                    CausalResidualBlock(channels, dilation=1),
                ]
            )
            in_channels = channels
        decoder_layers.append(CausalConv1d(channels, 1, 7))
        self.decoder = nn.Sequential(*decoder_layers)
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
            raise ValueError(f"embedding must have shape [batch, embedding] or [batch, channels, frames], got {embedding.shape}")
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
