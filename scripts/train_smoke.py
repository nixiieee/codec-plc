from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streamloss_codec.codec import StreamingSpeechCodec  # noqa: E402
from streamloss_codec.config import load_config  # noqa: E402
from streamloss_codec.dred import PassthroughDredProvider  # noqa: E402
from streamloss_codec.loss_sim import PacketLossConfig  # noqa: E402
from streamloss_codec.state_repair import StateRepairMiniEncoder  # noqa: E402
from streamloss_codec.train import train_base_step, train_repair_sequence_step  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    codec = StreamingSpeechCodec(**{k: v for k, v in model_cfg.items() if k != "active_quantizers_train"})
    repair = StateRepairMiniEncoder(decoder_hidden=model_cfg["decoder_hidden"])

    audio = torch.randn(2, cfg["chunk_samples"] * 8)
    opt_codec = torch.optim.Adam(codec.parameters(), lr=cfg["training"]["base_lr"])
    base = train_base_step(codec, opt_codec, audio[:, : cfg["chunk_samples"]], active_quantizers=2)

    opt_repair = torch.optim.Adam(
        list(codec.decoder.parameters()) + list(repair.parameters()),
        lr=cfg["training"]["repair_lr"],
    )
    repair_loss = train_repair_sequence_step(
        codec,
        repair,
        opt_repair,
        audio,
        PassthroughDredProvider(),
        PacketLossConfig(random_loss_p=0.2, burst_loss_p=0.1),
        active_quantizers=2,
    )
    print(
        {
            "base_total": float(base.total),
            "repair_total": float(repair_loss.total),
            "repair_loss": float(repair_loss.repair),
        }
    )


if __name__ == "__main__":
    main()
