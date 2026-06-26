from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _as_waveform(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.squeeze(1)
    if x.dim() != 2:
        raise ValueError(f"audio must have shape [batch, samples] or [batch, 1, samples], got {x.shape}")
    return x


def _stft_mag(x: torch.Tensor, n_fft: int) -> torch.Tensor:
    x = _as_waveform(x)
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    return torch.stft(
        x,
        n_fft=n_fft,
        hop_length=max(1, n_fft // 4),
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    ).abs()


class MultiScaleSTFTLoss(nn.Module):
    def __init__(
        self,
        window_lengths: tuple[int, ...] = (2048, 512),
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        clamp_eps: float = 1e-5,
        pow: float = 2.0,
    ) -> None:
        super().__init__()
        self.window_lengths = tuple(int(value) for value in window_lengths)
        self.mag_weight = float(mag_weight)
        self.log_weight = float(log_weight)
        self.clamp_eps = float(clamp_eps)
        self.pow = float(pow)

    def forward(self, estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        loss = estimate.sum() * 0.0
        for n_fft in self.window_lengths:
            if min(estimate.shape[-1], reference.shape[-1]) < n_fft:
                continue
            est = _stft_mag(estimate, n_fft)
            ref = _stft_mag(reference, n_fft)
            loss = loss + self.mag_weight * F.l1_loss(est, ref)
            loss = loss + self.log_weight * F.l1_loss(
                est.clamp_min(self.clamp_eps).pow(self.pow).log10(),
                ref.clamp_min(self.clamp_eps).pow(self.pow).log10(),
            )
        return loss


class MelSpectrogramLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16_000,
        n_mels: tuple[int, ...] = (5, 10, 20, 40, 80, 160, 320),
        window_lengths: tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048),
        clamp_eps: float = 1e-5,
        pow: float = 1.0,
        mag_weight: float = 0.0,
        log_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if len(n_mels) != len(window_lengths):
            raise ValueError("n_mels and window_lengths must have the same length")
        self.sample_rate = int(sample_rate)
        self.n_mels = tuple(int(value) for value in n_mels)
        self.window_lengths = tuple(int(value) for value in window_lengths)
        self.clamp_eps = float(clamp_eps)
        self.pow = float(pow)
        self.mag_weight = float(mag_weight)
        self.log_weight = float(log_weight)
        self._fbanks: dict[tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}

    def _mel_filterbank(self, n_fft: int, n_mels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (n_fft, n_mels, device, dtype)
        if key not in self._fbanks:
            try:
                from torchaudio.functional import melscale_fbanks

                fbank = melscale_fbanks(
                    n_freqs=n_fft // 2 + 1,
                    f_min=0.0,
                    f_max=float(self.sample_rate // 2),
                    n_mels=n_mels,
                    sample_rate=self.sample_rate,
                ).to(device=device, dtype=dtype)
            except Exception:
                fbank = torch.eye(n_fft // 2 + 1, n_mels, device=device, dtype=dtype)
            self._fbanks[key] = fbank
        return self._fbanks[key]

    def forward(self, estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        loss = estimate.sum() * 0.0
        for n_fft, n_mels in zip(self.window_lengths, self.n_mels, strict=True):
            if min(estimate.shape[-1], reference.shape[-1]) < n_fft:
                continue
            est = _stft_mag(estimate, n_fft)
            ref = _stft_mag(reference, n_fft)
            fbank = self._mel_filterbank(n_fft, n_mels, est.device, est.dtype)
            est_mel = torch.matmul(est.transpose(1, 2), fbank).transpose(1, 2)
            ref_mel = torch.matmul(ref.transpose(1, 2), fbank).transpose(1, 2)
            loss = loss + self.mag_weight * F.l1_loss(est_mel, ref_mel)
            loss = loss + self.log_weight * F.l1_loss(
                est_mel.clamp_min(self.clamp_eps).pow(self.pow).log10(),
                ref_mel.clamp_min(self.clamp_eps).pow(self.pow).log10(),
            )
        return loss


class GANLoss(nn.Module):
    def __init__(self, discriminator: nn.Module) -> None:
        super().__init__()
        self.discriminator = discriminator

    def discriminator_loss(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        d_fake = self.discriminator(fake.detach())
        d_real = self.discriminator(real)
        loss = fake.sum() * 0.0
        for fake_features, real_features in zip(d_fake, d_real, strict=True):
            loss = loss + fake_features[-1].pow(2).mean()
            loss = loss + (1.0 - real_features[-1]).pow(2).mean()
        return loss

    def generator_loss(self, fake: torch.Tensor, real: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d_fake = self.discriminator(fake)
        with torch.no_grad():
            d_real = self.discriminator(real)
        gen_loss = fake.sum() * 0.0
        feature_loss = fake.sum() * 0.0
        for fake_features, real_features in zip(d_fake, d_real, strict=True):
            gen_loss = gen_loss + (1.0 - fake_features[-1]).pow(2).mean()
            for fake_map, real_map in zip(fake_features[:-1], real_features[:-1], strict=True):
                feature_loss = feature_loss + F.l1_loss(fake_map, real_map.detach())
        return gen_loss, feature_loss
