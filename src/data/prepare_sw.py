from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True)


def raw_sw_duration_seconds(*, src: Path, sample_rate: int, channels: int = 1) -> float:
    bytes_per_sample = 2
    bytes_per_frame = bytes_per_sample * channels
    byte_size = src.stat().st_size
    if byte_size == 0 or byte_size % bytes_per_frame:
        raise ValueError(f"raw speech file must be non-empty int16 PCM: {src}")
    return byte_size / bytes_per_frame / sample_rate


def segment_raw_sw(
    *,
    src: Path,
    output_pattern: Path,
    sample_rate: int,
    segment_seconds: float,
    channels: int = 1,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> Path:
    output_pattern.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
    ]
    if start_seconds is not None:
        command.extend(["-ss", f"{start_seconds:.6f}"])
    command.extend(["-i", str(src)])
    if duration_seconds is not None:
        command.extend(["-t", f"{duration_seconds:.6f}"])
    command.extend(
        [
            "-f",
            "segment",
            "-segment_time",
            f"{segment_seconds:g}",
            "-c",
            "copy",
            "-reset_timestamps",
            "1",
            str(output_pattern),
        ]
    )
    _run(command)
    return output_pattern.parent


def prepare_sw_with_ffmpeg(
    *,
    src: Path,
    out_dir: Path,
    sample_rate: int,
    train_fraction: float = 0.8,
    segment_seconds: float = 10.0,
    channels: int = 1,
) -> dict[str, Any]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if segment_seconds <= 0.0:
        raise ValueError("segment_seconds must be positive")

    duration = raw_sw_duration_seconds(src=src, sample_rate=sample_rate, channels=channels)
    split_seconds = duration * train_fraction
    train_pattern = out_dir / "train" / "train_%06d.wav"
    test_pattern = out_dir / "test" / "test_%06d.wav"

    train_dir = segment_raw_sw(
        src=src,
        output_pattern=train_pattern,
        sample_rate=sample_rate,
        channels=channels,
        segment_seconds=segment_seconds,
        duration_seconds=split_seconds,
    )
    test_dir = segment_raw_sw(
        src=src,
        output_pattern=test_pattern,
        sample_rate=sample_rate,
        channels=channels,
        segment_seconds=segment_seconds,
        start_seconds=split_seconds,
    )

    return {
        "train_dir": train_dir,
        "test_dir": test_dir,
        "duration_seconds": duration,
        "split_seconds": split_seconds,
        "segment_seconds": segment_seconds,
    }
