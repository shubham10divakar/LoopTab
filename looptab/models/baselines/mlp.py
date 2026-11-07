"""Plain MLP baseline over [numerical features, categorical embeddings]."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_num: int, cat_cardinalities: Sequence[int] = (), cat_emb_dim: int = 8,
                 hidden: Sequence[int] = (256, 256), dropout: float = 0.1):
        super().__init__()
        self.n_num = n_num
        self.cat_embs = nn.ModuleList([nn.Embedding(c + 1, cat_emb_dim) for c in cat_cardinalities])
        layers, d = [], n_num + cat_emb_dim * len(cat_cardinalities)
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat=None):
        parts = [x_num] if self.n_num else []
        parts += [emb(x_cat[:, j]) for j, emb in enumerate(self.cat_embs)]
        return self.net(torch.cat(parts, dim=1)).squeeze(-1)
