"""FT-Transformer baseline (Gorishniy et al., 2021).

Feature Tokenizer + L independent pre-norm Transformer layers + [CLS] head.
Uses the same tokenizer as LoopTab so the comparison isolates the effect of
looping, gating and dense aggregation.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ..tokenizer import FeatureTokenizer


class FTTransformer(nn.Module):
    def __init__(self, n_num: int, cat_cardinalities: Sequence[int] = (), d_model: int = 128,
                 n_heads: int = 8, n_layers: int = 4, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num, cat_cardinalities, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x_num, x_cat=None):
        z = self.encoder(self.tokenizer(x_num, x_cat))
        return self.head(z[:, 0]).squeeze(-1)
