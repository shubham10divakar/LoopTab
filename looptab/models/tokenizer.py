"""Feature Tokenizer (design doc Section 2.2).

Every numerical feature gets its own (non-shared) affine projection to a
d-dimensional token, and every categorical feature gets its own embedding
table. A learned [CLS] token is prepended to the sequence.

The per-feature projections are stored as one weight tensor of shape
(N_num, d) instead of N_num separate ``nn.Linear(1, d)`` modules. The two are
mathematically identical (``z_j = x_j * W_j + b_j``) but the vectorised form
is far faster for datasets with hundreds of features.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """Convert raw tabular features into a token sequence.

    Args:
        n_num: number of numerical features.
        cat_cardinalities: cardinality of each categorical feature. Index 0 of
            every categorical column is reserved for missing / unseen values,
            so valid codes are ``0 .. cardinality`` inclusive.
        d_model: token dimension d.
        cls_token: whether to prepend a learned [CLS] token.
    """

    def __init__(
        self,
        n_num: int,
        cat_cardinalities: Sequence[int],
        d_model: int,
        cls_token: bool = True,
    ):
        super().__init__()
        self.n_num = n_num
        self.cat_cardinalities = list(cat_cardinalities)
        self.n_cat = len(self.cat_cardinalities)
        self.d_model = d_model
        if self.n_num + self.n_cat == 0:
            raise ValueError("FeatureTokenizer needs at least one feature")

        if n_num > 0:
            self.num_weight = nn.Parameter(torch.empty(n_num, d_model))
            self.num_bias = nn.Parameter(torch.empty(n_num, d_model))
            bound = 1.0 / math.sqrt(d_model)
            nn.init.uniform_(self.num_weight, -bound, bound)
            nn.init.uniform_(self.num_bias, -bound, bound)

        if self.n_cat > 0:
            # One table with per-feature offsets == one table per feature.
            sizes = [c + 1 for c in self.cat_cardinalities]  # +1 for missing
            offsets = torch.tensor([0] + sizes[:-1]).cumsum(0)
            self.register_buffer("cat_offsets", offsets, persistent=False)
            self.cat_embedding = nn.Embedding(sum(sizes), d_model)
            nn.init.normal_(self.cat_embedding.weight, std=1.0 / math.sqrt(d_model))

        self.cls_token = (
            nn.Parameter(torch.randn(1, 1, d_model) / math.sqrt(d_model))
            if cls_token
            else None
        )

    @property
    def n_tokens(self) -> int:
        return self.n_num + self.n_cat + (1 if self.cls_token is not None else 0)

    def forward(self, x_num: torch.Tensor | None, x_cat: torch.Tensor | None) -> torch.Tensor:
        """
        Args:
            x_num: (B, N_num) float tensor, or None if there are no numericals.
            x_cat: (B, N_cat) long tensor of codes in [0, cardinality], or None.
        Returns:
            (B, F[+1], d) token tensor.
        """
        tokens = []
        if self.n_num > 0:
            if x_num is None:
                raise ValueError("model expects numerical features")
            tokens.append(x_num.unsqueeze(-1) * self.num_weight + self.num_bias)
        if self.n_cat > 0:
            if x_cat is None:
                raise ValueError("model expects categorical features")
            tokens.append(self.cat_embedding(x_cat + self.cat_offsets))
        z = torch.cat(tokens, dim=1)
        if self.cls_token is not None:
            z = torch.cat([self.cls_token.expand(z.size(0), -1, -1), z], dim=1)
        return z
