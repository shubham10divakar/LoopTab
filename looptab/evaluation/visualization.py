"""Interpretability diagnostics and paper figures (design doc Section 8.3).

``collect_diagnostics`` runs a trained LoopTab over a split and aggregates,
per loop k, separately for fraud and legitimate rows:
    * mean gate activation G^k (overall and per token),
    * mean attention map (averaged over heads and rows),
plus the learned dense-aggregation weights alpha.

The ``plot_*`` helpers turn those (and experiment results) into figures.
matplotlib is imported lazily so the rest of the package does not need it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..data.dataloader import TabularSplit, make_loader
from ..models.looptab import LoopTab


@torch.no_grad()
def collect_diagnostics(
    model: LoopTab,
    split: TabularSplit,
    device: str | torch.device = "cpu",
    batch_size: int = 1024,
    max_rows: int | None = 50_000,
    seed: int = 0,
) -> dict:
    model = model.to(device).eval()
    if max_rows is not None and len(split) > max_rows:
        # keep every fraud row (they are rare) and fill with legitimate rows
        rng = np.random.default_rng(seed)
        pos = np.flatnonzero(split.y == 1)
        neg = np.flatnonzero(split.y == 0)
        n_neg = max(0, max_rows - len(pos))
        idx = np.concatenate([pos, rng.choice(neg, size=min(n_neg, len(neg)), replace=False)])
        split = split.subset(np.sort(idx))

    K = model.n_loops
    sums = {c: {"gate_tok": None, "attn": None, "n": 0} for c in (0, 1)}
    for x_num, x_cat, y in make_loader(split, batch_size, device=device):
        out = model(x_num, x_cat, return_aux=True)
        for c in (0, 1):
            m = y == c
            n = int(m.sum())
            if n == 0:
                continue
            s = sums[c]
            if out.gates:
                g = torch.stack([gk[m].mean(-1).sum(0) for gk in out.gates])  # (K, T)
                s["gate_tok"] = g if s["gate_tok"] is None else s["gate_tok"] + g
            a = torch.stack([ak[m].mean(1).sum(0) for ak in out.attn_maps])  # (K, T, T)
            s["attn"] = a if s["attn"] is None else s["attn"] + a
            s["n"] += n

    diag: dict = {"n_loops": K, "alpha": None}
    if model.aggregator is not None:
        diag["alpha"] = model.aggregator.weights.cpu().numpy()
    for c, label in ((0, "legit"), (1, "fraud")):
        s = sums[c]
        if s["n"] == 0:
            continue
        diag[f"attn_{label}"] = (s["attn"] / s["n"]).cpu().numpy()
        if s["gate_tok"] is not None:
            gt = (s["gate_tok"] / s["n"]).cpu().numpy()
            diag[f"gate_tok_{label}"] = gt          # (K, T)
            diag[f"gate_{label}"] = gt.mean(axis=1)  # (K,)
        diag[f"n_{label}"] = s["n"]
    return diag


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: str | Path | None):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_gate_activation(diag: dict, path=None, title: str = "Mean gate activation per loop"):
    """Analysis 1: mean G^k across loops, fraud vs legitimate."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ks = np.arange(1, diag["n_loops"] + 1)
    for label, style in (("legit", "o-"), ("fraud", "s--")):
        if f"gate_{label}" in diag:
            ax.plot(ks, diag[f"gate_{label}"], style, label=label)
    ax.set_xlabel("loop k")
    ax.set_ylabel("mean gate value")
    ax.set_xticks(ks)
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def plot_gate_heatmap(diag: dict, feature_names: Sequence[str], path=None, label: str = "fraud"):
    """Per-feature gate activation across loops (fraud minus legit if both)."""
    plt = _plt()
    g = diag[f"gate_tok_{label}"]
    names = ["[CLS]"] + list(feature_names)
    fig, ax = plt.subplots(figsize=(max(6, 0.22 * len(names)), 2 + 0.3 * diag["n_loops"]))
    im = ax.imshow(g, aspect="auto", cmap="viridis")
    ax.set_yticks(range(diag["n_loops"]), [f"k={k + 1}" for k in range(diag["n_loops"])])
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=6)
    ax.set_title(f"Gate activation per feature ({label})")
    fig.colorbar(im, ax=ax)
    return _save(fig, path)


def plot_attention_evolution(diag: dict, feature_names: Sequence[str], path=None,
                             label: str = "fraud", loops: Sequence[int] | None = None):
    """Analysis 3: attention heatmaps at selected loops (default first & last)."""
    plt = _plt()
    attn = diag[f"attn_{label}"]
    K = attn.shape[0]
    loops = list(loops) if loops is not None else sorted({0, K - 1})
    names = ["[CLS]"] + list(feature_names)
    fig, axes = plt.subplots(1, len(loops), figsize=(5 * len(loops), 4.5), squeeze=False)
    for ax, k in zip(axes[0], loops):
        im = ax.imshow(attn[k], cmap="magma")
        ax.set_title(f"loop k={k + 1} ({label})")
        if len(names) <= 40:
            ax.set_xticks(range(len(names)), names, rotation=90, fontsize=6)
            ax.set_yticks(range(len(names)), names, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, path)


def plot_alpha(alphas: dict[str, np.ndarray], path=None):
    """Analysis 4: learned dense-aggregation weights per dataset."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for name, a in alphas.items():
        ax.plot(np.arange(len(a)), a, "o-", label=name)
    ax.set_xlabel("loop output Z^k")
    ax.set_ylabel("alpha_k")
    ax.set_title("Dense aggregation weights")
    ax.legend()
    return _save(fig, path)


def plot_curve(xs, series: dict[str, Sequence[float]], xlabel: str, ylabel: str, path=None,
               title: str = "", logx: bool = False):
    """Generic line plot (loop-depth sensitivity, imbalance robustness)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for name, ys in series.items():
        ax.plot(xs, ys, "o-", label=name)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def plot_param_efficiency(points: dict[str, tuple[int, float]], path=None, metric: str = "AUC-ROC"):
    """Analysis 5: parameter count vs metric."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for name, (n, v) in points.items():
        ax.scatter(n, v, s=40)
        ax.annotate(name, (n, v), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel(metric)
    ax.set_title("Parameter efficiency")
    return _save(fig, path)
