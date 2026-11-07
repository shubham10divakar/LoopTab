"""Array containers and PyTorch loaders for preprocessed tabular data."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


@dataclass
class TabularSplit:
    """One split of encoded data. ``x_num`` float32, ``x_cat`` int64 codes."""

    x_num: np.ndarray  # (N, n_num)
    x_cat: np.ndarray  # (N, n_cat)
    y: np.ndarray      # (N,)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def pos_rate(self) -> float:
        return float(self.y.mean()) if len(self.y) else 0.0

    def subset(self, idx: np.ndarray) -> "TabularSplit":
        return TabularSplit(self.x_num[idx], self.x_cat[idx], self.y[idx])


@dataclass
class TabularData:
    """Train/val/test splits plus the schema the model needs."""

    train: TabularSplit
    val: TabularSplit
    test: TabularSplit
    num_names: list[str]
    cat_names: list[str]
    cat_cardinalities: list[int]
    name: str = "dataset"
    meta: dict = field(default_factory=dict)

    @property
    def n_num(self) -> int:
        return len(self.num_names)

    @property
    def n_cat(self) -> int:
        return len(self.cat_names)

    @property
    def feature_names(self) -> list[str]:
        return self.num_names + self.cat_names

    def summary(self) -> str:
        rows = [f"{self.name}: {self.n_num} numerical + {self.n_cat} categorical features"]
        for s in ("train", "val", "test"):
            sp = getattr(self, s)
            rows.append(f"  {s:5s} n={len(sp):>9,d}  fraud={sp.pos_rate:.4%}")
        return "\n".join(rows)


class TabularDataset(Dataset):
    def __init__(self, split: TabularSplit):
        self.x_num = torch.as_tensor(split.x_num, dtype=torch.float32)
        self.x_cat = torch.as_tensor(split.x_cat, dtype=torch.long)
        self.y = torch.as_tensor(split.y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return self.x_num[i], self.x_cat[i], self.y[i]


class FastTensorLoader:
    """Shuffled minibatches straight from (optionally on-GPU) tensors.

    Much faster than ``DataLoader`` with per-item ``__getitem__`` for the
    small-row, many-row tensors typical of tabular data.
    """

    def __init__(
        self,
        split: TabularSplit,
        batch_size: int,
        shuffle: bool = False,
        device: str | torch.device = "cpu",
        drop_last: bool = False,
        seed: int | None = None,
    ):
        self.x_num = torch.as_tensor(split.x_num, dtype=torch.float32, device=device)
        self.x_cat = torch.as_tensor(split.x_cat, dtype=torch.long, device=device)
        self.y = torch.as_tensor(split.y, dtype=torch.float32, device=device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = torch.Generator(device="cpu")
        if seed is not None:
            self.generator.manual_seed(seed)

    def __len__(self) -> int:
        n = len(self.y)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        n = len(self.y)
        if self.shuffle:
            idx = torch.randperm(n, generator=self.generator).to(self.y.device)
        else:
            idx = torch.arange(n, device=self.y.device)
        for b in range(len(self)):
            sl = idx[b * self.batch_size:(b + 1) * self.batch_size]
            yield self.x_num[sl], self.x_cat[sl], self.y[sl]


def make_loader(
    split: TabularSplit,
    batch_size: int,
    shuffle: bool = False,
    balanced: bool = False,
    num_workers: int = 0,
    device: str | torch.device = "cpu",
    seed: int | None = None,
):
    """Build a loader for a split.

    ``balanced=True`` returns a ``DataLoader`` with a ``WeightedRandomSampler``
    that draws both classes equally often (an over-sampling alternative to the
    focal loss); otherwise a ``FastTensorLoader`` is used.
    """
    if not balanced:
        return FastTensorLoader(split, batch_size, shuffle=shuffle, device=device, seed=seed)
    y = split.y.astype(int)
    counts = np.bincount(y, minlength=2).astype(float)
    weights = 1.0 / counts[y]
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(y),
        replacement=True, generator=gen,
    )
    return DataLoader(
        TabularDataset(split), batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=str(device).startswith("cuda"),
    )
