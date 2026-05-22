from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from streamloss_codec.codec import StreamingSpeechCodec, chunk_samples, frame_audio
from streamloss_codec.dred import DredProvider
from streamloss_codec.loss_sim import PacketLossConfig, make_loss_mask
from streamloss_codec.state_repair import StateRepairMiniEncoder


@dataclass
class LossBreakdown:
    total: torch.Tensor
    reconstruction: torch.Tensor
    quantizer: torch.Tensor
    repair: torch.Tensor


def _backward(total: torch.Tensor, accelerator: object | None = None) -> None:
    if accelerator is None:
        total.backward()
    else:
        accelerator.backward(total)


def _wrapped_module(module):
    return module.module if hasattr(module, "module") else module


def _zero_parameter_anchor(*modules: torch.nn.Module) -> torch.Tensor | None:
    anchors = []
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad:
                anchors.append(parameter.reshape(-1)[0] * 0.0)
    if not anchors:
        return None
    return torch.stack(anchors).sum()


def train_base_step(
    codec: StreamingSpeechCodec,
    optimizer: torch.optim.Optimizer,
    audio: torch.Tensor,
    active_quantizers: int,
    accelerator: object | None = None,
) -> LossBreakdown:
    reconstructed, _, quantized = codec(audio, active_quantizers=active_quantizers)
    target = audio[:, : reconstructed.shape[-1]]
    recon_loss = F.l1_loss(reconstructed, target)
    quantizer_loss = quantized.commitment_loss + quantized.codebook_loss
    total = recon_loss + quantizer_loss
    anchor = _zero_parameter_anchor(codec)
    if anchor is not None:
        total = total + anchor
    optimizer.zero_grad(set_to_none=True)
    _backward(total, accelerator)
    optimizer.step()
    zero = total.detach() * 0.0
    return LossBreakdown(total.detach(), recon_loss.detach(), quantizer_loss.detach(), zero)


def train_repair_sequence_step(
    codec: StreamingSpeechCodec,
    repair: StateRepairMiniEncoder,
    optimizer: torch.optim.Optimizer,
    audio: torch.Tensor,
    dred_provider: DredProvider,
    loss_config: PacketLossConfig,
    active_quantizers: int,
    sample_rate: int = 16_000,
    chunk_ms: int = 20,
    generator: torch.Generator | None = None,
    accelerator: object | None = None,
) -> LossBreakdown:
    frame_size = chunk_samples(sample_rate, chunk_ms)
    frames = frame_audio(audio, frame_size)
    loss_mask = make_loss_mask(frames.shape[1], loss_config, generator=generator)

    state = None
    outputs = []
    repair_losses = []
    decoder_module = _wrapped_module(codec.decoder)
    for frame_idx in range(frames.shape[1]):
        target = frames[:, frame_idx]
        if loss_mask[frame_idx]:
            if state is None:
                state = decoder_module.initial_state(target.shape[0], target.device)
            start = frame_idx * frame_size
            dred_audio = dred_provider.reconstruct(audio, start, frame_size).to(target.device)
            state = repair(dred_audio, state)
            # Feed a silence-equivalent latent for the missing packet after repairing state.
            z_lost = torch.zeros(
                target.shape[0],
                codec.quantizer.scalar[0].from_scalar.out_channels,
                frame_size // 4,
                device=target.device,
            )
            decoded, state = codec.decoder(z_lost, state)
            repair_losses.append(F.l1_loss(decoded[:, :frame_size], target))
        else:
            packet, _ = codec.encode_chunk(target, active_quantizers=active_quantizers)
            decoded, state = codec.decode_packet(packet, state)
        outputs.append(decoded[:, :frame_size])

    reconstructed = torch.cat(outputs, dim=-1)
    target_audio = audio[:, : reconstructed.shape[-1]]
    recon_loss = F.l1_loss(reconstructed, target_audio)
    repair_loss = torch.stack(repair_losses).mean() if repair_losses else recon_loss * 0.0
    total = recon_loss + repair_loss
    anchor = _zero_parameter_anchor(codec.decoder, repair)
    if anchor is not None:
        total = total + anchor

    optimizer.zero_grad(set_to_none=True)
    _backward(total, accelerator)
    optimizer.step()
    zero = total.detach() * 0.0
    return LossBreakdown(total.detach(), recon_loss.detach(), zero, repair_loss.detach())



def train_cached_repair_step(
    codec: StreamingSpeechCodec,
    repair: StateRepairMiniEncoder,
    optimizer: torch.optim.Optimizer,
    audio: torch.Tensor,
    dred_audio: torch.Tensor,
    loss_mask: torch.Tensor,
    active_quantizers: int,
    sample_rate: int = 16_000,
    chunk_ms: int = 20,
    accelerator: object | None = None,
) -> LossBreakdown:
    """Train repair module from an offline Opus DRED cache.

    `loss_mask` has shape [batch, frames] and marks lost 20 ms packets. Received
    samples follow normal packet decoding; lost samples repair decoder state from
    cached DRED audio and then decode a silence-equivalent latent.
    """
    frame_size = chunk_samples(sample_rate, chunk_ms)
    frames = frame_audio(audio, frame_size)
    dred_frames = frame_audio(dred_audio, frame_size)
    if loss_mask.shape != frames.shape[:2]:
        raise ValueError(f"loss_mask must have shape {frames.shape[:2]}, got {loss_mask.shape}")

    state = None
    outputs = []
    repair_losses = []
    decoder_module = _wrapped_module(codec.decoder)
    latent_dim = codec.quantizer.scalar[0].from_scalar.out_channels
    latent_frames = frame_size // 4

    for frame_idx in range(frames.shape[1]):
        target = frames[:, frame_idx]
        packet, _ = codec.encode_chunk(target, active_quantizers=active_quantizers)
        normal_decoded, normal_state = codec.decode_packet(packet, state)
        if state is None:
            state = decoder_module.initial_state(target.shape[0], target.device)

        lost = loss_mask[:, frame_idx].to(target.device).bool()
        if lost.any():
            repaired_state = repair(dred_frames[:, frame_idx].to(target.device), state)
            z_lost = torch.zeros(target.shape[0], latent_dim, latent_frames, device=target.device)
            lost_decoded, lost_state = codec.decoder(z_lost, repaired_state)

            lost_view = lost.view(1, -1, 1)
            next_h = torch.where(lost_view, lost_state.gru_h, normal_state.gru_h)
            state = type(normal_state)(gru_h=next_h)
            decoded = torch.where(lost.view(-1, 1), lost_decoded[:, :frame_size], normal_decoded[:, :frame_size])
            repair_losses.append(F.l1_loss(lost_decoded[lost, :frame_size], target[lost]))
        else:
            state = normal_state
            decoded = normal_decoded[:, :frame_size]
        outputs.append(decoded)

    reconstructed = torch.cat(outputs, dim=-1)
    target_audio = audio[:, : reconstructed.shape[-1]]
    recon_loss = F.l1_loss(reconstructed, target_audio)
    repair_loss = torch.stack(repair_losses).mean() if repair_losses else recon_loss * 0.0
    total = recon_loss + repair_loss
    anchor = _zero_parameter_anchor(codec.decoder, repair)
    if anchor is not None:
        total = total + anchor

    optimizer.zero_grad(set_to_none=True)
    _backward(total, accelerator)
    optimizer.step()
    zero = total.detach() * 0.0
    return LossBreakdown(total.detach(), recon_loss.detach(), zero, repair_loss.detach())



@torch.no_grad()
def evaluate_base_batch(
    codec: StreamingSpeechCodec,
    audio: torch.Tensor,
    active_quantizers: int,
) -> LossBreakdown:
    was_training = codec.training
    codec.eval()
    reconstructed, _, quantized = codec(audio, active_quantizers=active_quantizers)
    target = audio[:, : reconstructed.shape[-1]]
    recon_loss = F.l1_loss(reconstructed, target)
    quantizer_loss = quantized.commitment_loss + quantized.codebook_loss
    total = recon_loss + quantizer_loss
    zero = total.detach() * 0.0
    if was_training:
        codec.train()
    return LossBreakdown(total.detach(), recon_loss.detach(), quantizer_loss.detach(), zero)


@torch.no_grad()
def evaluate_cached_repair_batch(
    codec: StreamingSpeechCodec,
    repair: StateRepairMiniEncoder,
    audio: torch.Tensor,
    dred_audio: torch.Tensor,
    loss_mask: torch.Tensor,
    active_quantizers: int,
    sample_rate: int = 16_000,
    chunk_ms: int = 20,
) -> LossBreakdown:
    was_codec_training = codec.training
    was_repair_training = repair.training
    codec.eval()
    repair.eval()

    frame_size = chunk_samples(sample_rate, chunk_ms)
    frames = frame_audio(audio, frame_size)
    dred_frames = frame_audio(dred_audio, frame_size)
    if loss_mask.shape != frames.shape[:2]:
        raise ValueError(f"loss_mask must have shape {frames.shape[:2]}, got {loss_mask.shape}")

    state = None
    outputs = []
    repair_losses = []
    decoder_module = _wrapped_module(codec.decoder)
    latent_dim = codec.quantizer.scalar[0].from_scalar.out_channels
    latent_frames = frame_size // 4

    for frame_idx in range(frames.shape[1]):
        target = frames[:, frame_idx]
        packet, _ = codec.encode_chunk(target, active_quantizers=active_quantizers)
        normal_decoded, normal_state = codec.decode_packet(packet, state)
        if state is None:
            state = decoder_module.initial_state(target.shape[0], target.device)

        lost = loss_mask[:, frame_idx].to(target.device).bool()
        if lost.any():
            repaired_state = repair(dred_frames[:, frame_idx].to(target.device), state)
            z_lost = torch.zeros(target.shape[0], latent_dim, latent_frames, device=target.device)
            lost_decoded, lost_state = codec.decoder(z_lost, repaired_state)
            lost_view = lost.view(1, -1, 1)
            next_h = torch.where(lost_view, lost_state.gru_h, normal_state.gru_h)
            state = type(normal_state)(gru_h=next_h)
            decoded = torch.where(lost.view(-1, 1), lost_decoded[:, :frame_size], normal_decoded[:, :frame_size])
            repair_losses.append(F.l1_loss(lost_decoded[lost, :frame_size], target[lost]))
        else:
            state = normal_state
            decoded = normal_decoded[:, :frame_size]
        outputs.append(decoded)

    reconstructed = torch.cat(outputs, dim=-1)
    target_audio = audio[:, : reconstructed.shape[-1]]
    recon_loss = F.l1_loss(reconstructed, target_audio)
    repair_loss = torch.stack(repair_losses).mean() if repair_losses else recon_loss * 0.0
    total = recon_loss + repair_loss
    zero = total.detach() * 0.0

    if was_codec_training:
        codec.train()
    if was_repair_training:
        repair.train()
    return LossBreakdown(total.detach(), recon_loss.detach(), zero, repair_loss.detach())
