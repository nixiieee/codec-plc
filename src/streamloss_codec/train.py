from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from streamloss_codec.codec import StreamingSpeechCodec, chunk_samples, frame_audio
from streamloss_codec.dred import DredProvider
from streamloss_codec.loss_sim import PacketLossConfig, make_loss_mask
from streamloss_codec.state_repair import SegmentRepairAutoencoder, StateRepairMiniEncoder




@dataclass
class SegmentAELoss:
    total: torch.Tensor
    mse: torch.Tensor
    l1: torch.Tensor
    stft: torch.Tensor
    lost_frames: int



def _multi_scale_stft_loss(reconstructed: torch.Tensor, target: torch.Tensor, fft_sizes: tuple[int, ...] = (64, 128, 256)) -> torch.Tensor:
    losses = []
    for n_fft in fft_sizes:
        if reconstructed.shape[-1] < n_fft:
            continue
        window = torch.hann_window(n_fft, device=reconstructed.device, dtype=reconstructed.dtype)
        pred_spec = torch.stft(
            reconstructed,
            n_fft=n_fft,
            hop_length=max(1, n_fft // 4),
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        ).abs()
        target_spec = torch.stft(
            target,
            n_fft=n_fft,
            hop_length=max(1, n_fft // 4),
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        ).abs()
        mag_loss = F.l1_loss(pred_spec, target_spec)
        log_loss = F.l1_loss(torch.log1p(pred_spec), torch.log1p(target_spec))
        losses.append(mag_loss + log_loss)
    if not losses:
        return F.l1_loss(reconstructed, target)
    return torch.stack(losses).mean()


def _select_lost_segments(
    audio: torch.Tensor,
    dred_audio: torch.Tensor,
    loss_mask: torch.Tensor,
    sample_rate: int = 16_000,
    chunk_ms: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_size = chunk_samples(sample_rate, chunk_ms)
    clean_frames = frame_audio(audio, frame_size)
    dred_frames = frame_audio(dred_audio, frame_size)
    if loss_mask.shape != clean_frames.shape[:2]:
        raise ValueError(f"loss_mask must have shape {clean_frames.shape[:2]}, got {loss_mask.shape}")
    lost = loss_mask.to(clean_frames.device).bool()
    if not lost.any():
        empty = clean_frames.reshape(-1, frame_size)[:0]
        return empty, empty
    return dred_frames[lost], clean_frames[lost]


def train_segment_repair_ae_step(
    model: SegmentRepairAutoencoder,
    optimizer: torch.optim.Optimizer,
    audio: torch.Tensor,
    dred_audio: torch.Tensor,
    loss_mask: torch.Tensor,
    sample_rate: int = 16_000,
    chunk_ms: int = 20,
    mse_weight: float = 1.0,
    l1_weight: float = 0.5,
    stft_weight: float = 0.5,
    accelerator: object | None = None,
) -> SegmentAELoss:
    inputs, targets = _select_lost_segments(audio, dred_audio, loss_mask, sample_rate=sample_rate, chunk_ms=chunk_ms)
    if inputs.numel() == 0:
        zero = _zero_parameter_anchor(model)
        if zero is None:
            zero = audio.sum() * 0.0
        return SegmentAELoss(zero.detach(), zero.detach(), zero.detach(), zero.detach(), 0)

    reconstructed, _ = model(inputs)
    targets = targets[:, : reconstructed.shape[-1]]
    mse = F.mse_loss(reconstructed, targets)
    l1 = F.l1_loss(reconstructed, targets)
    stft = _multi_scale_stft_loss(reconstructed, targets)
    total = mse * mse_weight + l1 * l1_weight + stft * stft_weight
    anchor = _zero_parameter_anchor(model)
    if anchor is not None:
        total = total + anchor

    optimizer.zero_grad(set_to_none=True)
    _backward(total, accelerator)
    optimizer.step()
    return SegmentAELoss(total.detach(), mse.detach(), l1.detach(), stft.detach(), int(inputs.shape[0]))


@torch.no_grad()
def evaluate_segment_repair_ae_batch(
    model: SegmentRepairAutoencoder,
    audio: torch.Tensor,
    dred_audio: torch.Tensor,
    loss_mask: torch.Tensor,
    sample_rate: int = 16_000,
    chunk_ms: int = 20,
) -> SegmentAELoss:
    was_training = model.training
    model.eval()
    inputs, targets = _select_lost_segments(audio, dred_audio, loss_mask, sample_rate=sample_rate, chunk_ms=chunk_ms)
    if inputs.numel() == 0:
        zero = audio.sum() * 0.0
        if was_training:
            model.train()
        return SegmentAELoss(zero.detach(), zero.detach(), zero.detach(), zero.detach(), 0)
    reconstructed, _ = model(inputs)
    targets = targets[:, : reconstructed.shape[-1]]
    mse = F.mse_loss(reconstructed, targets)
    l1 = F.l1_loss(reconstructed, targets)
    stft = _multi_scale_stft_loss(reconstructed, targets)
    total = mse + l1 * 0.5 + stft * 0.5
    if was_training:
        model.train()
    return SegmentAELoss(total.detach(), mse.detach(), l1.detach(), stft.detach(), int(inputs.shape[0]))


@torch.no_grad()
def reconstruct_lost_segments_with_ae(
    model: SegmentRepairAutoencoder,
    dred_audio: torch.Tensor,
    loss_mask: torch.Tensor,
    sample_rate: int = 16_000,
    chunk_ms: int = 20,
) -> torch.Tensor:
    frame_size = chunk_samples(sample_rate, chunk_ms)
    frames = frame_audio(dred_audio, frame_size).clone()
    if loss_mask.shape != frames.shape[:2]:
        raise ValueError(f"loss_mask must have shape {frames.shape[:2]}, got {loss_mask.shape}")
    lost = loss_mask.to(frames.device).bool()
    if lost.any():
        reconstructed, _ = model(frames[lost])
        frames[lost] = reconstructed[:, :frame_size]
    return frames.reshape(frames.shape[0], -1)[:, : dred_audio.shape[-1]]

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
