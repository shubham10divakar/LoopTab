"""Build any neural model (LoopTab variant or baseline) by name."""

from __future__ import annotations

from typing import Sequence

import torch.nn as nn

from .baselines import GBM_MODELS, NEURAL_BASELINES
from .looptab import VARIANTS, build_looptab


def available_models() -> list[str]:
    return sorted(VARIANTS) + sorted(NEURAL_BASELINES) + list(GBM_MODELS)


def is_tree_model(name: str) -> bool:
    return name.lower() in GBM_MODELS


def build_model(name: str, n_num: int, cat_cardinalities: Sequence[int] = (), **kwargs) -> nn.Module:
    key = name.lower()
    if key in VARIANTS:
        return build_looptab(key, n_num, cat_cardinalities, **kwargs)
    if key in NEURAL_BASELINES:
        return NEURAL_BASELINES[key](n_num, cat_cardinalities, **kwargs)
    if key in GBM_MODELS:
        raise ValueError(f"{name} is a tree model; use looptab.models.baselines.fit_gbm")
    raise KeyError(f"unknown model {name!r}; choose from {available_models()}")
