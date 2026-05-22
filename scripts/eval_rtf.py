from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streamloss_codec.codec import StreamingSpeechCodec  # noqa: E402
from streamloss_codec.config import load_config  # noqa: E402
from streamloss_codec.eval import measure_rtf  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    codec = StreamingSpeechCodec(**{k: v for k, v in model_cfg.items() if k != "active_quantizers_train"})
    codec.eval()
    chunk = torch.randn(1, cfg["chunk_samples"])

    state = None

    def step() -> None:
        nonlocal state
        packet, _ = codec.encode_chunk(chunk, active_quantizers=2)
        _, state = codec.decode_packet(packet, state)
        state = state.detach()

    rtf = measure_rtf(step, audio_seconds=cfg["chunk_ms"] / 1000, runs=args.runs)
    print({"rtf": rtf})


if __name__ == "__main__":
    main()
