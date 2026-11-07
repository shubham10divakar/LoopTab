"""LR schedules: cosine annealing with linear warmup, or warmup + reduce-on-plateau."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau


def cosine_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    warmup_steps = max(0, warmup_steps)
    total_steps = max(total_steps, warmup_steps + 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


class WarmupPlateau:
    """Linear warmup per optimiser step, then ``ReduceLROnPlateau`` on the validation metric.

    ``step()`` is called after every optimiser step (like the cosine scheduler);
    ``step_epoch(score)`` once per epoch with the monitored validation score.
    """

    def __init__(self, optimizer: Optimizer, warmup_steps: int, mode: str = "max",
                 factor: float = 0.5, patience: int = 3, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.warmup_steps = max(0, warmup_steps)
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_count = 0
        self.plateau = ReduceLROnPlateau(optimizer, mode=mode, factor=factor,
                                         patience=patience, min_lr=min_lr)
        if self.warmup_steps:
            self._scale(1.0 / self.warmup_steps)

    def _scale(self, s: float) -> None:
        for g, lr in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = lr * s

    def step(self) -> None:
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            self._scale((self.step_count + 1) / self.warmup_steps)
        elif self.step_count == self.warmup_steps:
            self._scale(1.0)

    def step_epoch(self, score: float) -> None:
        if self.step_count >= self.warmup_steps:
            self.plateau.step(score)

    def state_dict(self) -> dict:
        return {"step_count": self.step_count, "base_lrs": self.base_lrs,
                "plateau": self.plateau.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.step_count = state["step_count"]
        self.base_lrs = state["base_lrs"]
        self.plateau.load_state_dict(state["plateau"])
