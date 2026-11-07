"""SAINT baseline (Somepalli et al., 2021), supervised variant.

Each stage = column (self-)attention over feature tokens followed by
intersample attention, where each row attends to the other rows in the batch
over its flattened token representation. Contrastive pre-training from the
original paper is not included.

Intersample attention makes predictions depend on batch composition; at
evaluation time rows are scored in the fixed-order batches the loader emits.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ..tokenizer import FeatureTokenizer


class _PreNormAttn(nn.Module):
    def __init__(self, d: int, heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d))

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


class SAINT(nn.Module):
    def __init__(self, n_num: int, cat_cardinalities: Sequence[int] = (), d_model: int = 32,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1,
                 intersample_heads: int = 4, intersample_dim: int = 256):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num, cat_cardinalities, d_model)
        n_tok = self.tokenizer.n_tokens
        flat = n_tok * d_model
        self.col_layers = nn.ModuleList(
            [_PreNormAttn(d_model, n_heads, 4 * d_model, dropout) for _ in range(n_layers)]
        )
        # project flattened rows down before intersample attention to bound cost
        self.row_in = nn.ModuleList([nn.Linear(flat, intersample_dim) for _ in range(n_layers)])
        self.row_layers = nn.ModuleList(
            [_PreNormAttn(intersample_dim, intersample_heads, 2 * intersample_dim, dropout)
             for _ in range(n_layers)]
        )
        self.row_out = nn.ModuleList([nn.Linear(intersample_dim, flat) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x_num, x_cat=None):
        z = self.tokenizer(x_num, x_cat)  # (B, T, d)
        B, T, d = z.shape
        for col, rin, row, rout in zip(self.col_layers, self.row_in, self.row_layers, self.row_out):
            z = col(z)
            r = rin(z.reshape(B, T * d)).unsqueeze(0)  # (1, B, d_r): rows as a sequence
            r = row(r).squeeze(0)
            z = z + rout(r).view(B, T, d)
        return self.head(z[:, 0]).squeeze(-1)
