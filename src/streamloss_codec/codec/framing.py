from __future__ import annotations

import torch


def chunk_samples(sample_rate: int = 16_000, chunk_ms: int = 20) -> int:
    return sample_rate * chunk_ms // 1000


def frame_audio(audio: torch.Tensor, frame_size: int, hop_size: int | None = None) -> torch.Tensor:
    """Frame audio as [batch, frames, frame_size], padding the tail with zeros."""
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.dim() != 2:
        raise ValueError(f"audio must have shape [samples] or [batch, samples], got {audio.shape}")
    hop_size = frame_size if hop_size is None else hop_size
    pad = (hop_size - (audio.shape[-1] - frame_size) % hop_size) % hop_size
    if audio.shape[-1] < frame_size:
        pad = frame_size - audio.shape[-1]
    if pad:
        audio = torch.nn.functional.pad(audio, (0, pad))
    return audio.unfold(dimension=-1, size=frame_size, step=hop_size)


def overlap_add(frames: torch.Tensor, hop_size: int) -> torch.Tensor:
    """Reconstruct [batch, samples] from [batch, frames, frame_size]."""
    if frames.dim() != 3:
        raise ValueError(f"frames must have shape [batch, frames, frame_size], got {frames.shape}")
    batch, n_frames, frame_size = frames.shape
    out_len = hop_size * (n_frames - 1) + frame_size
    out = frames.new_zeros(batch, out_len)
    weight = frames.new_zeros(batch, out_len)
    for idx in range(n_frames):
        start = idx * hop_size
        out[:, start : start + frame_size] += frames[:, idx]
        weight[:, start : start + frame_size] += 1
    return out / weight.clamp_min(1)
