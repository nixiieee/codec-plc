from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .layers import Snake1d, wn_conv1d, wn_conv_transpose1d
from .quantizer import DacQuantized, ResidualVectorQuantize


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv1d):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int = 1) -> None:
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            wn_conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            wn_conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            wn_conv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DacEncoder(nn.Module):
    def __init__(self, d_model: int = 64, strides: list[int] | None = None, d_latent: int | None = None) -> None:
        super().__init__()
        strides = [2, 4, 5, 8] if strides is None else [int(stride) for stride in strides]
        if d_latent is None:
            d_latent = d_model * (2 ** len(strides))
        layers: list[nn.Module] = [wn_conv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            layers.append(EncoderBlock(d_model, stride=stride))
        layers.extend([Snake1d(d_model), wn_conv1d(d_model, d_latent, kernel_size=3, padding=1)])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            wn_conv_transpose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DacDecoder(nn.Module):
    def __init__(self, input_dim: int, channels: int = 1536, rates: list[int] | None = None) -> None:
        super().__init__()
        rates = [8, 5, 4, 2] if rates is None else [int(rate) for rate in rates]
        layers: list[nn.Module] = [wn_conv1d(input_dim, channels, kernel_size=7, padding=3)]
        output_dim = channels
        for index, stride in enumerate(rates):
            input_channels = channels // 2**index
            output_dim = channels // 2 ** (index + 1)
            layers.append(DecoderBlock(input_channels, output_dim, stride=stride))
        layers.extend([Snake1d(output_dim), wn_conv1d(output_dim, 1, kernel_size=7, padding=3), nn.Tanh()])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DAC(nn.Module):
    """Local tensor-only implementation of the Descript Audio Codec generator."""

    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: list[int] | None = None,
        latent_dim: int | None = None,
        decoder_dim: int = 1536,
        decoder_rates: list[int] | None = None,
        n_codebooks: int = 12,
        codebook_size: int = 1024,
        codebook_dim: int | list[int] = 8,
        quantizer_dropout: float = 0.5,
        sample_rate: int = 16_000,
    ) -> None:
        super().__init__()
        self.encoder_dim = int(encoder_dim)
        self.encoder_rates = [2, 4, 5, 8] if encoder_rates is None else [int(rate) for rate in encoder_rates]
        self.decoder_rates = list(reversed(self.encoder_rates)) if decoder_rates is None else [int(rate) for rate in decoder_rates]
        self.decoder_dim = int(decoder_dim)
        self.sample_rate = int(sample_rate)
        self.hop_length = math.prod(self.encoder_rates)
        if latent_dim is None:
            latent_dim = self.encoder_dim * (2 ** len(self.encoder_rates))
        self.latent_dim = int(latent_dim)
        self.n_codebooks = int(n_codebooks)
        self.codebook_size = int(codebook_size)
        self.codebook_dim = codebook_dim

        self.encoder = DacEncoder(self.encoder_dim, self.encoder_rates, self.latent_dim)
        self.quantizer = ResidualVectorQuantize(
            input_dim=self.latent_dim,
            n_codebooks=self.n_codebooks,
            codebook_size=self.codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=float(quantizer_dropout),
        )
        self.decoder = DacDecoder(self.latent_dim, self.decoder_dim, self.decoder_rates)
        self.apply(init_weights)

    def preprocess(self, audio_data: torch.Tensor, sample_rate: int | None = None) -> torch.Tensor:
        if sample_rate is not None and int(sample_rate) != self.sample_rate:
            raise ValueError(f"expected sample_rate={self.sample_rate}, got {sample_rate}")
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        return F.pad(audio_data, (0, right_pad))

    def encode(self, audio_data: torch.Tensor, n_quantizers: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(audio_data)
        quantized: DacQuantized = self.quantizer(z, n_quantizers=n_quantizers)
        return quantized.z_q, quantized.codes, quantized.latents, quantized.commitment_loss, quantized.codebook_loss

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int | None = None,
        n_quantizers: int | None = None,
    ) -> dict[str, torch.Tensor]:
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z, codes, latents, commitment_loss, codebook_loss = self.encode(audio_data, n_quantizers=n_quantizers)
        audio = self.decode(z)
        return {
            "audio": audio[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }
