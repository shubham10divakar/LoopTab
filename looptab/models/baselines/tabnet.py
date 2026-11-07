"""Compact TabNet baseline (Arik & Pfister, 2021).

Sequential attentive feature selection with sparsemax masks, a shared + step
specific GLU feature transformer, prior-scale relaxation (gamma) and ghost
batch norm. Categorical features enter through small embeddings. The sparsity
regulariser is exposed as ``self.last_entropy`` but, for a like-for-like
comparison, the shared Trainer only optimises the classification loss.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    k = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim()
    shape[dim] = -1
    k = k.view(shape)
    cssv = z_sorted.cumsum(dim) - 1
    support = (k * z_sorted) > cssv
    k_z = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = cssv.gather(dim, k_z - 1) / k_z.to(z.dtype)
    return torch.clamp(z - tau, min=0)


class GhostBatchNorm(nn.Module):
    def __init__(self, dim: int, virtual_batch_size: int = 128, momentum: float = 0.02):
        super().__init__()
        self.vbs = virtual_batch_size
        self.bn = nn.BatchNorm1d(dim, momentum=momentum)

    def forward(self, x):
        if not self.training or x.size(0) <= self.vbs:
            return self.bn(x)
        return torch.cat([self.bn(c) for c in x.split(self.vbs, 0)], dim=0)


class GLULayer(nn.Module):
    def __init__(self, d_in: int, d_out: int, fc: nn.Linear | None = None, vbs: int = 128):
        super().__init__()
        self.fc = fc or nn.Linear(d_in, 2 * d_out, bias=False)
        self.bn = GhostBatchNorm(2 * d_out, vbs)
        self.d_out = d_out

    def forward(self, x):
        h = self.bn(self.fc(x))
        return h[:, : self.d_out] * torch.sigmoid(h[:, self.d_out:])


class FeatureTransformer(nn.Module):
    SCALE = 0.5 ** 0.5

    def __init__(self, d_in: int, d_out: int, shared: nn.ModuleList, n_independent: int, vbs: int):
        super().__init__()
        layers = []
        for i, fc in enumerate(shared):
            layers.append(GLULayer(d_in if i == 0 else d_out, d_out, fc, vbs))
        for i in range(n_independent):
            first = not layers and i == 0
            layers.append(GLULayer(d_in if first else d_out, d_out, vbs=vbs))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            x = (x + layer(x)) * self.SCALE
        return x


class TabNet(nn.Module):
    def __init__(self, n_num: int, cat_cardinalities: Sequence[int] = (), n_d: int = 32,
                 n_a: int = 32, n_steps: int = 4, gamma: float = 1.5, n_shared: int = 2,
                 n_independent: int = 2, cat_emb_dim: int = 4, virtual_batch_size: int = 128):
        super().__init__()
        self.n_num = n_num
        self.cat_embs = nn.ModuleList([nn.Embedding(c + 1, cat_emb_dim) for c in cat_cardinalities])
        d_in = n_num + cat_emb_dim * len(cat_cardinalities)
        self.d_in, self.n_d, self.n_a, self.n_steps, self.gamma = d_in, n_d, n_a, n_steps, gamma
        self.initial_bn = nn.BatchNorm1d(d_in, momentum=0.01)
        d_h = n_d + n_a
        shared = nn.ModuleList(
            [nn.Linear(d_in if i == 0 else d_h, 2 * d_h, bias=False) for i in range(n_shared)]
        )
        self.initial_ft = FeatureTransformer(d_in, d_h, shared, n_independent, virtual_batch_size)
        self.step_ft = nn.ModuleList(
            [FeatureTransformer(d_in, d_h, shared, n_independent, virtual_batch_size)
             for _ in range(n_steps)]
        )
        self.attentive = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_a, d_in, bias=False), GhostBatchNorm(d_in, virtual_batch_size))
             for _ in range(n_steps)]
        )
        self.head = nn.Linear(n_d, 1)
        self.last_entropy = torch.tensor(0.0)

    def forward(self, x_num, x_cat=None):
        parts = [x_num] if self.n_num else []
        parts += [emb(x_cat[:, j]) for j, emb in enumerate(self.cat_embs)]
        x = self.initial_bn(torch.cat(parts, dim=1))
        prior = torch.ones_like(x)
        a = self.initial_ft(x)[:, self.n_d:]
        out = 0.0
        entropy = 0.0
        for step in range(self.n_steps):
            mask = sparsemax(self.attentive[step](a) * prior)
            prior = prior * (self.gamma - mask)
            entropy = entropy + (-mask * torch.log(mask + 1e-10)).sum(1).mean()
            h = self.step_ft[step](mask * x)
            out = out + torch.relu(h[:, : self.n_d])
            a = h[:, self.n_d:]
        self.last_entropy = entropy / self.n_steps
        return self.head(out).squeeze(-1)
