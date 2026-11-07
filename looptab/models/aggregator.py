"""Dense Residual Aggregator (design doc Section 2.4).

    Z_agg = sum_{k=0}^{K} alpha_k Z^k,   alpha = softmax(a_0, ..., a_K)

The raw logits start equal, so alpha starts uniform at 1/(K+1).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class DenseResidualAggregator(nn.Module):
    def __init__(self, n_loops: int):
        super().__init__()
        self.alpha_logits = nn.Parameter(torch.zeros(n_loops + 1))

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.alpha_logits, dim=0)

    def forward(self, loop_outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(loop_outputs) != self.alpha_logits.numel():
            raise ValueError(
                f"expected {self.alpha_logits.numel()} loop outputs, got {len(loop_outputs)}"
            )
        stacked = torch.stack(list(loop_outputs), dim=0)  # (K+1, B, T, d)
        w = self.weights.view(-1, *([1] * (stacked.dim() - 1)))
        return (w * stacked).sum(dim=0)
