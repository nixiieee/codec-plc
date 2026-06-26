from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import torch
from torch import nn
from torch.nn import functional as F

from codec import chunk_samples, frame_audio
from state_repair import SegmentRepairEncoder

from .model import DAC


@dataclass
class DacPacketOutput:
    audio: torch.Tensor
    z_received: torch.Tensor
    z_repair: torch.Tensor
    z_mixed: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    lost_frames: int


def _product(values: list[int]) -> int:
    return int(reduce(mul, values, 1))


OFFICIAL_16KHZ_DAC_URL = "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.5/weights_16khz.pth"


def _official_dac_cache_path() -> Path:
    return Path.home() / ".cache" / "descript" / "dac" / "weights_16khz_8kbps_0.0.5.pth"


def _download_official_16khz_dac() -> Path:
    path = _official_dac_cache_path()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(OFFICIAL_16KHZ_DAC_URL, timeout=120) as response:
        path.write_bytes(response.read())
    return path


def _resolve_pretrained_checkpoint(pretrained_checkpoint: str | None) -> str | None:
    if pretrained_checkpoint in {None, "", "none", "None"}:
        return None
    if pretrained_checkpoint in {"official:16khz", "16khz", "official_16khz"}:
        return str(_download_official_16khz_dac())
    return str(pretrained_checkpoint)


def _extract_dac_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    state = checkpoint
    for key in ("dac_state_dict", "state_dict", "model_state_dict", "generator_state_dict"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            state = checkpoint[key]
            break
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise ValueError("pretrained DAC checkpoint does not contain a state dict")
    cleaned = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        clean_key = str(key)
        for prefix in ("module.", "generator.", "model.", "dac."):
            clean_key = clean_key.removeprefix(prefix)
        cleaned[clean_key] = value
    if not cleaned:
        raise ValueError("pretrained DAC checkpoint state dict is empty")
    return cleaned


def create_official_dac_model(
    *,
    sample_rate: int = 16_000,
    latent_dim: int | None = None,
    encoder_dim: int = 64,
    encoder_rates: list[int] | None = None,
    decoder_dim: int = 1536,
    n_codebooks: int = 12,
    codebook_size: int = 1024,
    codebook_dim: int = 8,
    quantizer_dropout: float = 0.5,
    pretrained_checkpoint: str | None = None,
) -> nn.Module:
    """Instantiate the local DAC architecture configured for 20 ms packets."""

    rates = [2, 4, 5, 8] if encoder_rates is None else [int(rate) for rate in encoder_rates]
    if _product(rates) <= 0:
        raise ValueError(f"encoder_rates must be positive, got {rates}")
    model = DAC(
        encoder_dim=int(encoder_dim),
        encoder_rates=rates,
        latent_dim=None if latent_dim is None else int(latent_dim),
        decoder_dim=int(decoder_dim),
        decoder_rates=list(reversed(rates)),
        n_codebooks=int(n_codebooks),
        codebook_size=int(codebook_size),
        codebook_dim=int(codebook_dim),
        quantizer_dropout=float(quantizer_dropout),
        sample_rate=int(sample_rate),
    )
    pretrained_checkpoint = _resolve_pretrained_checkpoint(pretrained_checkpoint)
    if pretrained_checkpoint:
        checkpoint = torch.load(pretrained_checkpoint, map_location="cpu", weights_only=False)
        cleaned = _extract_dac_state_dict(checkpoint)
        model.load_state_dict(cleaned, strict=False)
    return model


class OfficialDacPacketCodec(nn.Module):
    """DAC packet wrapper with frozen segment-encoder fill for lost packets."""

    def __init__(
        self,
        frozen_encoder: SegmentRepairEncoder,
        dac_model: nn.Module | None = None,
        *,
        sample_rate: int = 16_000,
        chunk_ms: int = 20,
        latent_dim: int = 1024,
        latent_frames: int = 1,
        dac_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.chunk_ms = int(chunk_ms)
        self.frame_size = chunk_samples(self.sample_rate, self.chunk_ms)
        self.latent_dim = int(latent_dim)
        self.latent_frames = int(latent_frames)
        self.frozen_encoder = frozen_encoder
        self.dac = dac_model or create_official_dac_model(
            sample_rate=self.sample_rate,
            latent_dim=self.latent_dim,
            **(dac_kwargs or {}),
        )
        if _product(getattr(self.dac, "encoder_rates", [self.frame_size])) != self.frame_size:
            raise ValueError(
                f"DAC encoder_rates must produce one latent packet per {self.frame_size} samples, "
                f"got rates={getattr(self.dac, 'encoder_rates', None)}"
            )
        self._freeze_segment_encoder()

    @property
    def packet_embedding_dim(self) -> int:
        return self.latent_dim * self.latent_frames

    def _freeze_segment_encoder(self) -> None:
        self.frozen_encoder.eval()
        for parameter in self.frozen_encoder.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_encoder.eval()
        return self

    def _encode_received(self, frames: torch.Tensor, n_quantizers: int | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.dac.encode(frames.unsqueeze(1), n_quantizers=n_quantizers)
        if isinstance(encoded, dict):
            z = encoded["z"]
            commitment = encoded.get("vq/commitment_loss", z.sum() * 0.0)
            codebook = encoded.get("vq/codebook_loss", z.sum() * 0.0)
        else:
            z, _codes, _latents, commitment, codebook = encoded
        expected = (frames.shape[0], self.latent_dim, self.latent_frames)
        if tuple(z.shape) != expected:
            raise ValueError(f"DAC encode must return z shape {expected}, got {tuple(z.shape)}")
        return z, commitment, codebook

    def _decode_packets(self, z: torch.Tensor) -> torch.Tensor:
        decoded = self.dac.decode(z)
        if isinstance(decoded, dict):
            decoded = decoded["audio"]
        if decoded.dim() == 3:
            decoded = decoded.squeeze(1)
        if decoded.shape[-1] < self.frame_size:
            decoded = F.pad(decoded, (0, self.frame_size - decoded.shape[-1]))
        return decoded[:, : self.frame_size]

    def forward(
        self,
        audio: torch.Tensor,
        dred_audio: torch.Tensor,
        loss_mask: torch.Tensor,
        *,
        n_quantizers: int | None = None,
    ) -> DacPacketOutput:
        clean_frames = frame_audio(audio, self.frame_size)
        dred_frames = frame_audio(dred_audio, self.frame_size)
        if loss_mask.shape != clean_frames.shape[:2]:
            raise ValueError(f"loss_mask must have shape {clean_frames.shape[:2]}, got {loss_mask.shape}")
        batch, frames_count, frame_size = clean_frames.shape
        flat_clean = clean_frames.reshape(batch * frames_count, frame_size)
        flat_dred = dred_frames.reshape(batch * frames_count, frame_size)

        z_received, commitment_loss, codebook_loss = self._encode_received(flat_clean, n_quantizers)
        with torch.no_grad():
            z_repair = self.frozen_encoder.encode_latent(flat_dred)
        if tuple(z_repair.shape) != tuple(z_received.shape):
            raise ValueError(
                f"frozen encoder returned {tuple(z_repair.shape)}, expected {tuple(z_received.shape)}. "
                "Train or load a segment repair checkpoint with matching DAC-native latent_dim/latent_frames."
            )

        lost = loss_mask.to(audio.device).bool().reshape(batch * frames_count, 1, 1)
        z_mixed = torch.where(lost, z_repair.to(z_received.dtype), z_received)
        decoded_frames = self._decode_packets(z_mixed)
        reconstructed = decoded_frames.reshape(batch, frames_count, frame_size).reshape(batch, -1)
        reconstructed = reconstructed[:, : audio.shape[-1]]
        return DacPacketOutput(
            audio=reconstructed,
            z_received=z_received.reshape(batch, frames_count, self.latent_dim, self.latent_frames),
            z_repair=z_repair.reshape(batch, frames_count, self.latent_dim, self.latent_frames),
            z_mixed=z_mixed.reshape(batch, frames_count, self.latent_dim, self.latent_frames),
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            lost_frames=int(loss_mask.bool().sum().item()),
        )
