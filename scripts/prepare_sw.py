from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from data.prepare_sw import prepare_sw_with_ffmpeg  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare raw 16 kHz int16 .sw audio through ffmpeg into train/test WAV chunks.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--src", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--segment-time", type=float, default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--sample-rate", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    result = prepare_sw_with_ffmpeg(
        src=args.src or Path(dataset_cfg["speech_path"]),
        out_dir=args.out_dir or Path(dataset_cfg.get("wav_dir", "data/wav")),
        sample_rate=int(args.sample_rate or cfg.get("sample_rate", 16_000)),
        train_fraction=float(args.train_fraction if args.train_fraction is not None else dataset_cfg.get("prepare_train_fraction", 0.8)),
        segment_seconds=float(args.segment_time if args.segment_time is not None else dataset_cfg.get("prepare_segment_seconds", 10.0)),
    )
    print({key: str(value) if isinstance(value, Path) else value for key, value in result.items()})


if __name__ == "__main__":
    main()
