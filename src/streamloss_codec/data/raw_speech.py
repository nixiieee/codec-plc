from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class RawSpeechConfig:
    speech_path: str
    sample_rate: int = 16_000
    segment_seconds: float = 2.0
    split: str = "train"
    val_fraction: float = 0.02
    split_seed: int = 1234
    split_mode: str = "random"
    max_segments: int | None = None


@dataclass(frozen=True)
class RawSpeechSegment:
    index: int
    start_sample: int
    num_samples: int
    split: str


def load_raw_int16_audio(path: str | Path) -> torch.Tensor:
    data = np.fromfile(path, dtype="<i2")
    if data.size == 0:
        raise ValueError(f"raw speech file is empty: {path}")
    audio = torch.from_numpy(data.astype(np.float32)) / 32768.0
    return audio.clamp(-1.0, 1.0)


def build_segment_index(
    num_samples: int,
    segment_samples: int,
    split: str,
    val_fraction: float = 0.02,
    split_seed: int = 1234,
    max_segments: int | None = None,
    split_mode: str = "random",
) -> list[RawSpeechSegment]:
    if split not in {"train", "val", "all"}:
        raise ValueError(f"split must be train, val, or all, got {split!r}")
    if segment_samples <= 0:
        raise ValueError("segment_samples must be positive")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")

    total = num_samples // segment_samples
    val_count = int(round(total * val_fraction))
    if split_mode == "random":
        order = torch.randperm(total, generator=torch.Generator().manual_seed(split_seed)).tolist()
        val_ids = set(order[:val_count])
    elif split_mode == "tail":
        val_ids = set(range(max(0, total - val_count), total))
    else:
        raise ValueError(f"split_mode must be random or tail, got {split_mode!r}")

    segments = []
    for idx in range(total):
        item_split = "val" if idx in val_ids else "train"
        if split != "all" and item_split != split:
            continue
        segments.append(
            RawSpeechSegment(
                index=idx,
                start_sample=idx * segment_samples,
                num_samples=segment_samples,
                split=item_split,
            )
        )
        if max_segments is not None and len(segments) >= max_segments:
            break
    return segments


class RawSpeechDataset(Dataset):
    """Dataset for Opus DRED's raw 16 kHz int16 speech source.

    The official speech source is tens of GB, so this class uses a memmap and
    only materializes the requested segment.
    """

    def __init__(self, config: RawSpeechConfig) -> None:
        self.config = config
        self.path = Path(config.speech_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        byte_size = self.path.stat().st_size
        if byte_size == 0 or byte_size % 2:
            raise ValueError(f"raw speech file must be non-empty int16 PCM: {self.path}")
        self.num_samples = byte_size // 2
        self.audio = np.memmap(self.path, dtype="<i2", mode="r")
        self.segment_samples = int(round(config.sample_rate * config.segment_seconds))
        if self.segment_samples % (config.sample_rate // 50) != 0:
            raise ValueError("segment_seconds must be aligned to 20 ms frames")
        self.segments = build_segment_index(
            self.num_samples,
            self.segment_samples,
            config.split,
            config.val_fraction,
            config.split_seed,
            config.max_segments,
            config.split_mode,
        )

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | int | str]:
        segment = self.segments[item]
        start = segment.start_sample
        end = start + segment.num_samples
        data = np.asarray(self.audio[start:end], dtype=np.float32) / 32768.0
        audio = torch.from_numpy(data).clamp(-1.0, 1.0)
        return {
            "audio": audio,
            "segment_index": segment.index,
            "start_sample": segment.start_sample,
            "split": segment.split,
        }
