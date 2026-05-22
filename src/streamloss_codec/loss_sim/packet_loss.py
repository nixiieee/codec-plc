from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PacketLossConfig:
    random_loss_p: float = 0.05
    burst_loss_p: float = 0.02
    min_burst: int = 1
    max_burst: int = 5


def make_loss_mask(
    n_frames: int,
    config: PacketLossConfig = PacketLossConfig(),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return bool tensor where True means packet is lost."""
    if n_frames < 0:
        raise ValueError("n_frames must be non-negative")
    mask = torch.rand(n_frames, generator=generator) < config.random_loss_p
    idx = 0
    while idx < n_frames:
        if torch.rand((), generator=generator).item() < config.burst_loss_p:
            burst_len = int(
                torch.randint(
                    low=config.min_burst,
                    high=config.max_burst + 1,
                    size=(),
                    generator=generator,
                ).item()
            )
            mask[idx : min(n_frames, idx + burst_len)] = True
            idx += burst_len
        else:
            idx += 1
    return mask
