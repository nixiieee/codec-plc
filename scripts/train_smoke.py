from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codec import StreamingSpeechCodec  # noqa: E402
from config import load_config  # noqa: E402
from state_repair import SegmentRepairAutoencoder  # noqa: E402
from train import train_base_step, train_segment_repair_ae_step  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    codec = StreamingSpeechCodec(**{k: v for k, v in model_cfg.items() if k != "active_quantizers_train"})

    repair_cfg = cfg.get("segment_repair", {})
    repair = SegmentRepairAutoencoder(
        channels=int(repair_cfg.get("channels", model_cfg.get("channels", 96))),
        latent_dim=int(repair_cfg.get("latent_dim", 96)),
        latent_frames=int(repair_cfg.get("latent_frames", 8)),
    )

    audio = torch.randn(2, cfg["chunk_samples"] * 8).clamp(-1.0, 1.0)
    dred_audio = (audio + 0.01 * torch.randn_like(audio)).clamp(-1.0, 1.0)
    loss_mask = torch.zeros(2, 8, dtype=torch.bool)
    loss_mask[:, [1, 4]] = True

    opt_codec = torch.optim.Adam(codec.parameters(), lr=cfg["training"]["base_lr"])
    base = train_base_step(codec, opt_codec, audio[:, : cfg["chunk_samples"]], active_quantizers=2)

    opt_repair = torch.optim.Adam(repair.parameters(), lr=cfg["training"]["repair_lr"])
    repair_loss = train_segment_repair_ae_step(
        repair,
        opt_repair,
        audio,
        dred_audio,
        loss_mask,
        sample_rate=cfg["sample_rate"],
        chunk_ms=cfg["chunk_ms"],
    )
    print(
        {
            "base_total": float(base.total),
            "repair_mse": float(repair_loss.mse),
            "repair_lost_frames": repair_loss.lost_frames,
        }
    )


if __name__ == "__main__":
    main()
