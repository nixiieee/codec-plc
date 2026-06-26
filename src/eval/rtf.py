from __future__ import annotations

import time
from collections.abc import Callable

import torch


@torch.inference_mode()
def measure_rtf(fn: Callable[[], object], audio_seconds: float, warmup: int = 5, runs: int = 20) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    elapsed = time.perf_counter() - start
    return elapsed / (runs * audio_seconds)
