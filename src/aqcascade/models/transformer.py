"""A compact Transformer encoder that forecasts PM2.5 from a station's own
recent history + weather -- not included for appearance's sake: it's the
one model in this project that learns directly from the raw time series
instead of hand-engineered lag/rolling summaries, and is evaluated on
exactly the same targets and test period as the Phase 5 baselines.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PollutionTransformer(nn.Module):
    """Encoder-only Transformer over a [batch, seq_len, n_channels] window.

    No causal mask is applied, and none is needed: the prediction head only
    ever reads out the *last* position's representation, and there is
    nothing after the last position for it to attend to in the first place
    (the window itself already contains only at-or-before-T data — see
    sequence_data.py). A causal mask would restrict every *earlier*
    position's attention too, which would have zero effect on the one
    representation that's actually used and only add dead complexity — the
    kind of decoration the model is explicitly not supposed to carry.

    The final timestep's representation (i.e. "now") is fed through a small
    MLP head to produce one prediction per horizon.
    """

    def __init__(
        self,
        n_channels: int,
        n_horizons: int,
        seq_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        last_step = h[:, -1, :]
        return self.head(last_step)
