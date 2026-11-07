"""Losses for imbalanced binary classification (design doc Section 2.5)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss  FL(p_t) = -alpha_t (1 - p_t)^gamma log(p_t).

    Args:
        alpha: weight of the positive (fraud) class; negatives get 1 - alpha.
        gamma: focusing parameter; gamma = 0 recovers alpha-weighted BCE.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)  # p if y = 1, 1 - p if y = 0
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedBCELoss(nn.Module):
    """BCE with a positive-class weight (LoopTab-NoFocal ablation)."""

    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=self.pos_weight
        )


def build_loss(cfg: dict, pos_rate: float | None = None) -> nn.Module:
    """Build a loss from a config dict like ``{"type": "focal", "alpha": .75}``.

    For ``weighted_bce`` the positive weight defaults to (1 - pi) / pi where pi
    is the training fraud rate, unless ``pos_weight`` is given.
    """
    kind = cfg.get("type", "focal").lower()
    if kind == "focal":
        return FocalLoss(cfg.get("alpha", 0.75), cfg.get("gamma", 2.0))
    if kind in ("weighted_bce", "bce"):
        pw = cfg.get("pos_weight")
        if pw is None:
            pw = 1.0 if kind == "bce" or not pos_rate else (1 - pos_rate) / pos_rate
        return WeightedBCELoss(pw)
    raise ValueError(f"unknown loss type {kind!r}")
