"""TabTransformer baseline (Huang et al., 2020).

Self-attention runs over categorical embeddings only. The contextual
categorical embeddings are flattened, concatenated with layer-normed
numerical features and passed to an MLP. With no categorical features it
degenerates to an MLP on the numericals, as in the original paper.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ..tokenizer import FeatureTokenizer


class TabTransformer(nn.Module):
    def __init__(self, n_num: int, cat_cardinalities: Sequence[int] = (), d_model: int = 32,
                 n_heads: int = 8, n_layers: int = 6, d_ff: int | None = None,
                 mlp_hidden: Sequence[int] = (256, 128), dropout: float = 0.1):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(cat_cardinalities)
        if self.n_cat:
            self.cat_tokenizer = FeatureTokenizer(0, cat_cardinalities, d_model, cls_token=False)
            layer = nn.TransformerEncoderLayer(
                d_model, n_heads, d_ff or 4 * d_model, dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.num_norm = nn.LayerNorm(n_num) if n_num else None
        in_dim = self.n_cat * d_model + n_num
        layers, d = [], in_dim
        for h in mlp_hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat=None):
        parts = []
        if self.n_cat:
            parts.append(self.encoder(self.cat_tokenizer(None, x_cat)).flatten(1))
        if self.n_num:
            parts.append(self.num_norm(x_num))
        return self.mlp(torch.cat(parts, dim=1)).squeeze(-1)
