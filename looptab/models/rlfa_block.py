"""Residual-Loop Feature Attention (RLFA) block (design doc Sections 2.3, 4.1).

Formal definition implemented here (one loop iteration k):

    A_bar^k = LN(Z^k)
    A^k     = MHSA(A_bar^k)
    G^k     = sigmoid(W_g A_bar^k + U_g A^k + b_g)   # per-token, per-dim gate
    Z~^k    = Z^k + G^k * A^k                         # gated residual
    Z^{k+1} = Z~^k + FFN(LN(Z~^k))                    # standard residual

Note: the design doc's Section 2.3.2 feeds the raw Z^k into the gate while
Section 4.1 and the reference code feed LN(Z^k). We follow Section 4.1.

With ``use_gate=False`` the gate is fixed to 1, giving the plain residual
update used by the LoopTab-NoGate ablation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class ResidualGate(nn.Module):
    """G = sigmoid(W_g z + U_g a + b_g), at full (tokens x d) resolution."""

    def __init__(self, d_model: int, bias_init: float = 0.0):
        super().__init__()
        self.W_g = nn.Linear(d_model, d_model, bias=True)  # carries b_g
        self.U_g = nn.Linear(d_model, d_model, bias=False)
        nn.init.constant_(self.W_g.bias, bias_init)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.W_g(z) + self.U_g(a))


@dataclass
class RLFAOutput:
    z: torch.Tensor
    attn: torch.Tensor | None  # (B, h, T, T) when requested
    gate: torch.Tensor | None  # (B, T, d) when gating is on


class RLFABlock(nn.Module):
    """One RLFA iteration. Reused K times when weights are tied."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attn_dropout: float | None = None,
        use_gate: bool = True,
        gate_bias_init: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.use_gate = use_gate
        self.ln1 = nn.LayerNorm(d_model)
        self.mhsa = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout if attn_dropout is None else attn_dropout,
            batch_first=True,
        )
        self.attn_out_dropout = nn.Dropout(dropout)
        self.gate = ResidualGate(d_model, gate_bias_init) if use_gate else None
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, z: torch.Tensor, return_attn: bool = False) -> RLFAOutput:
        z_norm = self.ln1(z)
        a, attn = self.mhsa(
            z_norm,
            z_norm,
            z_norm,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        a = self.attn_out_dropout(a)
        if self.gate is not None:
            g = self.gate(z_norm, a)
            z = z + g * a
        else:
            g = None
            z = z + a
        z = z + self.ffn(self.ln2(z))
        return RLFAOutput(z=z, attn=attn, gate=g)
