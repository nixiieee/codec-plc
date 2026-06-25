from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streamloss_codec.cache import OpusDredCacheConfig, build_opus_dred_cache  # noqa: E402
from streamloss_codec.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline Opus DRED cache from raw 16 kHz speech.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default=None, choices=["train", "val", "all"])
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel Opus workers for cache generation.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    cache_cfg = cfg["dred_cache"]
    loss_cfg = cfg["packet_loss"]
    config = OpusDredCacheConfig(
        speech_path=dataset_cfg["speech_path"],
        cache_dir=cache_cfg["cache_dir"],
        opus_demo_path=cache_cfg["opus_demo_path"],
        opus_root=cache_cfg.get("opus_root"),
        dred_checkpoint_dir=cache_cfg.get("dred_checkpoint_dir", "checkpoints/dred"),
        sample_rate=cfg["sample_rate"],
        segment_seconds=dataset_cfg["segment_seconds"],
        split=args.split or dataset_cfg.get("split", "train"),
        val_fraction=dataset_cfg.get("val_fraction", 0.02),
        split_seed=dataset_cfg.get("split_seed", 1234),
        split_mode=dataset_cfg.get("split_mode", "random"),
        max_segments=args.max_segments if args.max_segments is not None else dataset_cfg.get("max_segments"),
        bitrate=cache_cfg.get("bitrate", 64000),
        dred_frames_10ms=cache_cfg.get("dred_frames_10ms", 100),
        random_loss_p=loss_cfg.get("random_loss_p", 0.05),
        burst_loss_p=loss_cfg.get("burst_loss_p", 0.02),
        min_burst=loss_cfg.get("min_burst", 1),
        max_burst=loss_cfg.get("max_burst", 5),
        dry_run=args.dry_run or cache_cfg.get("dry_run", False),
        show_progress=not args.no_progress,
        num_workers=args.workers if args.workers is not None else cache_cfg.get("num_workers", 1),
    )
    manifest = build_opus_dred_cache(config)
    print({"manifest": str(manifest)})


if __name__ == "__main__":
    main()
