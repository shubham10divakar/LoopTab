"""Full LoopTab model and named variants (design doc Sections 2, 3, 6.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn

from .aggregator import DenseResidualAggregator
from .rlfa_block import RLFABlock
from .tokenizer import FeatureTokenizer


@dataclass
class LoopTabOutput:
    logits: torch.Tensor                                  # (B,)
    attn_maps: list[torch.Tensor] = field(default_factory=list)  # K x (B, h, T, T)
    gates: list[torch.Tensor] = field(default_factory=list)      # K x (B, T, d)
    alpha: torch.Tensor | None = None                     # (K+1,)


class LoopTab(nn.Module):
    """LoopTab: Residual-Loop Feature Attention for tabular fraud detection.

    Args:
        n_num: number of numerical features.
        cat_cardinalities: cardinality of each categorical feature.
        d_model: token dimension d.
        n_heads: attention heads h.
        n_loops: number of RLFA iterations K.
        d_ff: FFN hidden size.
        dropout: dropout rate.
        tied_weights: share one RLFA block across all loops (LoopTab-S/D) or
            use K independent blocks (LoopTab-U).
        use_gate: residual gating on (False -> LoopTab-NoGate).
        use_dense: dense aggregation across loops (False -> LoopTab-NoDense,
            which classifies from Z^K only).
    """

    def __init__(
        self,
        n_num: int,
        cat_cardinalities: Sequence[int] = (),
        d_model: int = 128,
        n_heads: int = 8,
        n_loops: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        tied_weights: bool = True,
        use_gate: bool = True,
        use_dense: bool = True,
        gate_bias_init: float = 0.0,
    ):
        super().__init__()
        if n_loops < 1:
            raise ValueError("n_loops must be >= 1")
        self.n_loops = n_loops
        self.tied_weights = tied_weights
        self.use_dense = use_dense

        self.tokenizer = FeatureTokenizer(n_num, cat_cardinalities, d_model)

        def make_block() -> RLFABlock:
            return RLFABlock(
                d_model, n_heads, d_ff, dropout,
                use_gate=use_gate, gate_bias_init=gate_bias_init,
            )

        n_blocks = 1 if tied_weights else n_loops
        self.blocks = nn.ModuleList([make_block() for _ in range(n_blocks)])
        self.aggregator = DenseResidualAggregator(n_loops) if use_dense else None
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def block(self, k: int) -> RLFABlock:
        return self.blocks[0] if self.tied_weights else self.blocks[k]

    def forward(
        self,
        x_num: torch.Tensor | None,
        x_cat: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | LoopTabOutput:
        z = self.tokenizer(x_num, x_cat)
        loop_outputs = [z]
        attn_maps, gates = [], []
        for k in range(self.n_loops):
            out = self.block(k)(z, return_attn=return_aux)
            z = out.z
            loop_outputs.append(z)
            if return_aux:
                attn_maps.append(out.attn)
                if out.gate is not None:
                    gates.append(out.gate)

        z_final = self.aggregator(loop_outputs) if self.aggregator is not None else z
        logits = self.head(z_final[:, 0, :]).squeeze(-1)

        if not return_aux:
            return logits
        return LoopTabOutput(
            logits=logits,
            attn_maps=attn_maps,
            gates=gates,
            alpha=self.aggregator.weights.detach() if self.aggregator is not None else None,
        )


# Variant presets from Sections 3 and 6.4. Values override LoopTab defaults.
VARIANTS: dict[str, dict] = {
    "looptab-s": dict(d_model=64, n_heads=4, n_loops=2, d_ff=256),
    "looptab-d": dict(d_model=128, n_heads=8, n_loops=4, d_ff=512),
    "looptab-u": dict(d_model=128, n_heads=8, n_loops=4, d_ff=512, tied_weights=False),
    "looptab-nogate": dict(d_model=128, n_heads=8, n_loops=4, d_ff=512, use_gate=False),
    "looptab-noloop": dict(d_model=128, n_heads=8, n_loops=1, d_ff=512),
    "looptab-nodense": dict(d_model=128, n_heads=8, n_loops=4, d_ff=512, use_dense=False),
}


def build_looptab(
    variant: str,
    n_num: int,
    cat_cardinalities: Sequence[int] = (),
    **overrides,
) -> LoopTab:
    key = variant.lower()
    if key not in VARIANTS:
        raise KeyError(f"unknown LoopTab variant {variant!r}; choose from {sorted(VARIANTS)}")
    kwargs = {**VARIANTS[key], **overrides}
    return LoopTab(n_num, cat_cardinalities, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
