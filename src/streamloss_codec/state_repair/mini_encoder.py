from __future__ import annotations

import torch
from torch import nn

from streamloss_codec.codec import DecoderState
from streamloss_codec.codec.layers import CausalConv1d


class StateRepairMiniEncoder(nn.Module):
    """Predict an additive GRU-state delta from DRED audio and previous decoder state."""

    def __init__(self, decoder_hidden: int = 128, channels: int = 64) -> None:
        super().__init__()
        self.audio_encoder = nn.Sequential(
            CausalConv1d(1, channels, 7),
            nn.SiLU(),
            CausalConv1d(channels, channels, 5, stride=2),
            nn.SiLU(),
            CausalConv1d(channels, channels, 5, stride=2),
            nn.SiLU(),
            CausalConv1d(channels, channels, 80, stride=80),
        )
        self.state_proj = nn.Linear(decoder_hidden, channels)
        self.delta = nn.Sequential(
            nn.Linear(channels * 2, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, decoder_hidden),
        )

    def forward(self, dred_audio: torch.Tensor, previous_state: DecoderState) -> DecoderState:
        if dred_audio.dim() == 2:
            dred_audio = dred_audio.unsqueeze(1)
        audio_feat = self.audio_encoder(dred_audio).squeeze(-1)
        prev_h = previous_state.gru_h[-1]
        state_feat = self.state_proj(prev_h)
        delta = self.delta(torch.cat([audio_feat, state_feat], dim=-1))
        repaired_h = previous_state.gru_h.clone()
        repaired_h[-1] = repaired_h[-1] + delta
        return DecoderState(gru_h=repaired_h)
