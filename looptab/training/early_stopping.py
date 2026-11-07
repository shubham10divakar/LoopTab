"""Patience-based early stopping that keeps the best model weights in memory."""

from __future__ import annotations

import copy

import torch.nn as nn


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 0.0):
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.best_epoch = -1
        self.best_state: dict | None = None
        self.counter = 0

    def _improved(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        """Record ``score``; return True when training should stop."""
        if self._improved(score):
            self.best_score = score
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
