from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from dred import DredProvider


def load_cache_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest = Path(path)
    with manifest.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass(frozen=True)
class DredCacheItem:
    audio: torch.Tensor
    dred_audio: torch.Tensor
    loss_mask: torch.Tensor
    segment_index: int
    metadata: dict[str, Any]


class CachedDredDataset(Dataset):
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.rows = load_cache_manifest(self.manifest_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> DredCacheItem:
        row = self.rows[item]
        cache_path = Path(row["cache_path"])
        if not cache_path.is_absolute():
            cache_path = self.root / cache_path
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return DredCacheItem(
            audio=payload["audio"].float(),
            dred_audio=payload["dred_audio"].float(),
            loss_mask=payload["loss_mask"].bool(),
            segment_index=int(payload["segment_index"]),
            metadata=dict(payload.get("metadata", {})),
        )


def collate_cached_dred(items: list[DredCacheItem]) -> dict[str, torch.Tensor | list[dict[str, Any]]]:
    return {
        "audio": torch.stack([item.audio for item in items], dim=0),
        "dred_audio": torch.stack([item.dred_audio for item in items], dim=0),
        "loss_mask": torch.stack([item.loss_mask for item in items], dim=0),
        "segment_index": torch.tensor([item.segment_index for item in items], dtype=torch.long),
        "metadata": [item.metadata for item in items],
    }


class CachedDredProvider(DredProvider):
    """DRED provider backed by one cached decoded/reconstructed segment batch."""

    def __init__(self, dred_audio: torch.Tensor) -> None:
        if dred_audio.dim() == 1:
            dred_audio = dred_audio.unsqueeze(0)
        self.dred_audio = dred_audio

    def reconstruct(self, audio_context: torch.Tensor, start_sample: int, num_samples: int) -> torch.Tensor:
        del audio_context
        end = start_sample + num_samples
        if end > self.dred_audio.shape[-1]:
            return torch.nn.functional.pad(self.dred_audio[:, start_sample:], (0, end - self.dred_audio.shape[-1]))
        return self.dred_audio[:, start_sample:end]
