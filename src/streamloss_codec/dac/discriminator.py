from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .layers import wn_conv1d, wn_conv2d


BANDS: list[tuple[float, float]] = [
    (0.0, 0.1),
    (0.1, 0.25),
    (0.25, 0.5),
    (0.5, 0.75),
    (0.75, 1.0),
]


def _as_waveform(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() != 3:
        raise ValueError(f"audio must have shape [batch, samples] or [batch, 1, samples], got {x.shape}")
    return x


class MPD(nn.Module):
    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = int(period)
        self.convs = nn.ModuleList(
            [
                wn_conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
                wn_conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
                wn_conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
                wn_conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
                wn_conv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
            ]
        )
        self.conv_post = wn_conv2d(1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = _as_waveform(x)
        remainder = x.shape[-1] % self.period
        if remainder:
            x = F.pad(x, (0, self.period - remainder), mode="reflect")
        x = x.reshape(x.shape[0], x.shape[1], x.shape[-1] // self.period, self.period)
        fmap = []
        for layer in self.convs:
            x = layer(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return fmap


class MRD(nn.Module):
    def __init__(
        self,
        window_length: int,
        hop_factor: float = 0.25,
        sample_rate: int = 16_000,
        bands: list[tuple[float, float]] | None = None,
    ) -> None:
        super().__init__()
        self.window_length = int(window_length)
        self.hop_length = int(window_length * hop_factor)
        self.sample_rate = int(sample_rate)
        n_fft = self.window_length // 2 + 1
        bands = BANDS if bands is None else bands
        self.bands = [(int(low * n_fft), int(high * n_fft)) for low, high in bands]
        channels = 32
        self.band_convs = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        wn_conv2d(2, channels, (3, 9), (1, 1), padding=(1, 4)),
                        wn_conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4)),
                        wn_conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4)),
                        wn_conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4)),
                        wn_conv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1)),
                    ]
                )
                for _ in self.bands
            ]
        )
        self.conv_post = wn_conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def _spectrogram_bands(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = _as_waveform(x).squeeze(1)
        window = torch.hann_window(self.window_length, device=x.device, dtype=x.dtype)
        spec = torch.stft(
            x,
            n_fft=self.window_length,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=window,
            center=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec).permute(0, 3, 2, 1).contiguous()
        return [spec[..., low:high] for low, high in self.bands if high > low]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        bands = self._spectrogram_bands(x)
        fmap = []
        outputs = []
        for band, layers in zip(bands, self.band_convs, strict=False):
            y = band
            for layer in layers:
                y = layer(y)
                fmap.append(y)
            outputs.append(y)
        y = torch.cat(outputs, dim=-1)
        y = self.conv_post(y)
        fmap.append(y)
        return fmap


class Discriminator(nn.Module):
    def __init__(
        self,
        periods: list[int] | None = None,
        fft_sizes: list[int] | None = None,
        sample_rate: int = 16_000,
        bands: list[tuple[float, float]] | None = None,
    ) -> None:
        super().__init__()
        periods = [2, 3, 5, 7, 11] if periods is None else [int(period) for period in periods]
        fft_sizes = [2048, 1024, 512] if fft_sizes is None else [int(size) for size in fft_sizes]
        discriminators: list[nn.Module] = [MPD(period) for period in periods]
        discriminators.extend(MRD(size, sample_rate=sample_rate, bands=bands) for size in fft_sizes)
        self.discriminators = nn.ModuleList(discriminators)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_waveform(x)
        x = x - x.mean(dim=-1, keepdim=True)
        return 0.8 * x / (x.abs().amax(dim=-1, keepdim=True) + 1e-9)

    def forward(self, x: torch.Tensor) -> list[list[torch.Tensor]]:
        x = self.preprocess(x)
        return [discriminator(x) for discriminator in self.discriminators]
