from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .layers import CausalConv1d, CausalResidualBlock
from .quantizer import Quantized, ResidualScalarVectorQuantizer


@dataclass
class Packet:
    z_q: torch.Tensor
    indices: list[torch.Tensor]
    active_quantizers: int


@dataclass
class DecoderState:
    gru_h: torch.Tensor

    def detach(self) -> "DecoderState":
        return DecoderState(gru_h=self.gru_h.detach())


class CausalEncoder(nn.Module):
    def __init__(self, channels: int = 96, latent_dim: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(1, channels, 7),
            nn.SiLU(),
            CausalResidualBlock(channels, dilation=1),
            CausalResidualBlock(channels, dilation=2),
            CausalConv1d(channels, channels, 4, stride=2),
            nn.SiLU(),
            CausalResidualBlock(channels, dilation=1),
            CausalConv1d(channels, latent_dim, 4, stride=2),
        )

    def forward(self, chunk: torch.Tensor) -> torch.Tensor:
        if chunk.dim() == 2:
            chunk = chunk.unsqueeze(1)
        return self.net(chunk)


class CausalDecoder(nn.Module):
    def __init__(self, latent_dim: int = 96, hidden: int = 128) -> None:
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(latent_dim, hidden, batch_first=True)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(hidden, 96, 4, stride=2, padding=1),
            nn.SiLU(),
            CausalResidualBlock(96, dilation=1),
            nn.ConvTranspose1d(96, 48, 4, stride=2, padding=1),
            nn.SiLU(),
            CausalConv1d(48, 1, 7),
            nn.Tanh(),
        )

    def initial_state(self, batch: int, device: torch.device) -> DecoderState:
        return DecoderState(gru_h=torch.zeros(1, batch, self.hidden, device=device))

    def forward(self, z_q: torch.Tensor, state: DecoderState | None = None) -> tuple[torch.Tensor, DecoderState]:
        batch = z_q.shape[0]
        if state is None:
            state = self.initial_state(batch, z_q.device)
        seq = z_q.transpose(1, 2)
        decoded_seq, gru_h = self.gru(seq, state.gru_h)
        decoded = self.net(decoded_seq.transpose(1, 2)).squeeze(1)
        return decoded, DecoderState(gru_h=gru_h)


class StreamingSpeechCodec(nn.Module):
    """20 ms streaming codec with explicit decoder state."""

    def __init__(
        self,
        channels: int = 96,
        latent_dim: int = 96,
        decoder_hidden: int = 128,
        scalar_quantizers: int = 2,
        vector_quantizers: int = 6,
        codebook_size: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = CausalEncoder(channels=channels, latent_dim=latent_dim)
        self.quantizer = ResidualScalarVectorQuantizer(
            dim=latent_dim,
            scalar_quantizers=scalar_quantizers,
            vector_quantizers=vector_quantizers,
            codebook_size=codebook_size,
        )
        self.decoder = CausalDecoder(latent_dim=latent_dim, hidden=decoder_hidden)

    def encode_chunk(self, chunk: torch.Tensor, active_quantizers: int | None = None) -> tuple[Packet, Quantized]:
        z = self.encoder(chunk)
        quantized = self.quantizer(z, n_active=active_quantizers)
        packet = Packet(
            z_q=quantized.z_q.detach(),
            indices=[idx.detach() for idx in quantized.indices],
            active_quantizers=active_quantizers or self.quantizer.num_quantizers,
        )
        return packet, quantized

    def decode_packet(
        self,
        packet: Packet,
        state: DecoderState | None = None,
    ) -> tuple[torch.Tensor, DecoderState]:
        return self.decoder(packet.z_q, state)

    def forward(
        self,
        chunk: torch.Tensor,
        active_quantizers: int | None = None,
        state: DecoderState | None = None,
    ) -> tuple[torch.Tensor, DecoderState, Quantized]:
        _, quantized = self.encode_chunk(chunk, active_quantizers=active_quantizers)
        audio, next_state = self.decoder(quantized.z_q, state)
        return audio, next_state, quantized
