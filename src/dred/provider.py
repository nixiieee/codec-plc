from __future__ import annotations

import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import soundfile as sf
import torch


class DredProvider(ABC):
    @abstractmethod
    def reconstruct(self, audio_context: torch.Tensor, start_sample: int, num_samples: int) -> torch.Tensor:
        """Return DRED reconstruction for a lost span as [batch, num_samples]."""


class PassthroughDredProvider(DredProvider):
    """Oracle-like provider useful for smoke tests and isolating state repair."""

    def reconstruct(self, audio_context: torch.Tensor, start_sample: int, num_samples: int) -> torch.Tensor:
        if audio_context.dim() == 1:
            audio_context = audio_context.unsqueeze(0)
        end = start_sample + num_samples
        if end > audio_context.shape[-1]:
            audio_context = torch.nn.functional.pad(audio_context, (0, end - audio_context.shape[-1]))
        return audio_context[:, start_sample:end]


class ExternalDredProvider(DredProvider):
    """Adapter for Opus/RDOVAE tooling kept outside this repo.

    The command must support:
    `<cmd> --input in.wav --start-sample N --num-samples M --output out.wav`.
    """

    def __init__(self, command: list[str], sample_rate: int = 16_000) -> None:
        if not command:
            raise ValueError("command must not be empty")
        self.command = command
        self.sample_rate = sample_rate

    def reconstruct(self, audio_context: torch.Tensor, start_sample: int, num_samples: int) -> torch.Tensor:
        if audio_context.dim() == 1:
            audio_context = audio_context.unsqueeze(0)
        outputs = []
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for batch_idx in range(audio_context.shape[0]):
                in_wav = tmp_dir / f"in_{batch_idx}.wav"
                out_wav = tmp_dir / f"out_{batch_idx}.wav"
                sf.write(in_wav, audio_context[batch_idx].detach().cpu().numpy(), self.sample_rate)
                cmd = [
                    *self.command,
                    "--input",
                    str(in_wav),
                    "--start-sample",
                    str(start_sample),
                    "--num-samples",
                    str(num_samples),
                    "--output",
                    str(out_wav),
                ]
                subprocess.run(cmd, check=True)
                wav, _ = sf.read(out_wav, dtype="float32")
                tensor = torch.from_numpy(wav).to(audio_context.device)
                if tensor.numel() < num_samples:
                    tensor = torch.nn.functional.pad(tensor, (0, num_samples - tensor.numel()))
                outputs.append(tensor[:num_samples])
        return torch.stack(outputs, dim=0)
