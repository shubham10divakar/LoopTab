"""Training loop for LoopTab and the deep-learning baselines (Section 7.4).

Any model with signature ``model(x_num, x_cat) -> logits (B,)`` works.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..data.dataloader import TabularData, TabularSplit, make_loader
from ..evaluation.metrics import compute_metrics
from .early_stopping import EarlyStopping
from .focal_loss import build_loss
from .scheduler import WarmupPlateau, cosine_warmup_scheduler


@dataclass
class TrainConfig:
    batch_size: int = 512
    eval_batch_size: int = 4096
    epochs: int = 100
    early_stopping_patience: int = 10
    monitor: str = "auc_pr"          # validation metric for early stopping
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler: str = "cosine_warmup"  # "cosine_warmup" | "plateau" | "none"
    warmup_epochs: int = 5
    min_lr: float = 0.0               # floor for cosine and plateau
    plateau_factor: float = 0.5       # plateau: multiply LR by this when val stalls
    plateau_patience: int = 3         # plateau: epochs without improvement before cutting LR
    resume: bool = True               # continue from <out_dir>/last.pt if it exists
    save_last: bool = True            # write <out_dir>/last.pt every epoch
    grad_clip: float | None = 1.0
    balanced_sampling: bool = False
    amp: bool = False
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    log_every: int = 1
    loss: dict = field(default_factory=lambda: {"type": "focal", "alpha": 0.75, "gamma": 2.0})

    @classmethod
    def from_dict(cls, training: dict | None = None, loss: dict | None = None) -> "TrainConfig":
        training = dict(training or {})
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(training) - known
        if unknown:
            raise ValueError(f"unknown training options: {sorted(unknown)}")
        cfg = cls(**training)
        if loss is not None:
            cfg.loss = dict(loss)
        return cfg


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    split: TabularSplit,
    batch_size: int = 4096,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    model.eval()
    loader = make_loader(split, batch_size, shuffle=False, device=device)
    probs = [torch.sigmoid(model(x_num, x_cat)).float().cpu() for x_num, x_cat, _ in loader]
    return torch.cat(probs).numpy() if probs else np.zeros(0)


# Options that may change between a run and its resume without invalidating it.
_RESUME_IGNORED = {"resume", "save_last", "device", "log_every", "num_workers", "eval_batch_size"}


class Trainer:
    """Trains with early stopping; with ``out_dir`` it checkpoints and auto-resumes.

    Files in ``out_dir``: ``best.pt`` (best weights by the monitored val metric),
    ``last.pt`` (full training state after the latest epoch), ``history.json``
    (per-epoch log, rewritten every epoch) and ``result.json`` (final metrics).
    ``signature`` identifies the run (model / data); resuming from a
    ``last.pt`` with a different signature or config raises instead.
    """

    def __init__(self, model: nn.Module, data: TabularData, cfg: TrainConfig,
                 out_dir: str | Path | None = None, verbose: bool = True,
                 signature: dict | None = None):
        self.cfg = cfg
        self.data = data
        self.device = resolve_device(cfg.device)
        self.model = model.to(self.device)
        self.out_dir = Path(out_dir) if out_dir else None
        self.verbose = verbose
        self.history: list[dict] = []
        cfg_sig = {k: v for k, v in asdict(cfg).items() if k not in _RESUME_IGNORED}
        self.signature = json.dumps({"config": cfg_sig, "run": signature or {}},
                                    sort_keys=True, default=str)

        self.criterion = build_loss(cfg.loss, pos_rate=data.train.pos_rate).to(self.device)
        self.optimizer = self._build_optimizer()
        self.train_loader = make_loader(
            data.train, cfg.batch_size, shuffle=True, balanced=cfg.balanced_sampling,
            num_workers=cfg.num_workers, device=self.device, seed=cfg.seed,
        )
        steps_per_epoch = max(1, len(self.train_loader))
        warmup = cfg.warmup_epochs * steps_per_epoch
        if cfg.scheduler == "cosine_warmup":
            self.scheduler = cosine_warmup_scheduler(
                self.optimizer, warmup_steps=warmup, total_steps=cfg.epochs * steps_per_epoch,
                min_lr_ratio=cfg.min_lr / cfg.learning_rate,
            )
        elif cfg.scheduler == "plateau":
            self.scheduler = WarmupPlateau(
                self.optimizer, warmup_steps=warmup, mode="max", factor=cfg.plateau_factor,
                patience=cfg.plateau_patience, min_lr=cfg.min_lr,
            )
        elif cfg.scheduler == "none":
            self.scheduler = None
        else:
            raise ValueError(f"unknown scheduler {cfg.scheduler!r}")
        self.use_amp = cfg.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        # No weight decay on biases, norms, embeddings, CLS and aggregator logits.
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or any(k in name for k in ("norm", "ln", "embedding", "cls_token",
                                                     "alpha_logits", "num_bias")):
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [
            {"params": decay, "weight_decay": self.cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        opt = self.cfg.optimizer.lower()
        if opt == "adamw":
            return torch.optim.AdamW(groups, lr=self.cfg.learning_rate)
        if opt == "adam":
            return torch.optim.Adam(groups, lr=self.cfg.learning_rate)
        raise ValueError(f"unknown optimizer {self.cfg.optimizer!r}")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def train_epoch(self) -> float:
        self.model.train()
        total, n = 0.0, 0
        for x_num, x_cat, y in self.train_loader:
            x_num = x_num.to(self.device, non_blocking=True)
            x_cat = x_cat.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(x_num, x_cat)
            loss = self.criterion(logits.float(), y)
            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            total += loss.item() * len(y)
            n += len(y)
        return total / max(n, 1)

    def evaluate(self, split: TabularSplit, threshold: float | None = None) -> dict:
        probs = predict_proba(self.model, split, self.cfg.eval_batch_size, self.device)
        return compute_metrics(split.y, probs, threshold=threshold)

    # ---- checkpointing --------------------------------------------------------

    def _path(self, name: str) -> Path | None:
        return self.out_dir / name if self.out_dir is not None else None

    def _loader_generator(self) -> torch.Generator | None:
        gen = getattr(self.train_loader, "generator", None)
        if gen is None:
            gen = getattr(getattr(self.train_loader, "sampler", None), "generator", None)
        return gen

    @staticmethod
    def _atomic_save(obj, path: Path) -> None:
        tmp = path.with_name(path.name + ".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, path)  # a crash mid-write never corrupts the previous file

    def _save_last(self, epoch: int, stopper: EarlyStopping, finished: bool, elapsed: float) -> None:
        if self.out_dir is None or not self.cfg.save_last:
            return
        gen = self._loader_generator()
        state = {
            "epoch": epoch,
            "finished": finished,
            "elapsed_s": elapsed,
            "signature": self.signature,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict(),
            "stopper": stopper.state_dict(),
            "history": self.history,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "loader": gen.get_state() if gen is not None else None,
            },
        }
        self._atomic_save(state, self._path("last.pt"))

    def _write_history(self) -> None:
        if self.out_dir is not None:
            self._path("history.json").write_text(json.dumps(self.history, indent=2, default=float))

    def _try_resume(self, stopper: EarlyStopping) -> tuple[int, bool, float]:
        """Load ``last.pt`` if present; return (next epoch, finished, elapsed seconds)."""
        last = self._path("last.pt")
        if last is None or not self.cfg.resume or not last.exists():
            return 1, False, 0.0
        state = torch.load(last, map_location=self.device, weights_only=False)
        if state.get("signature") != self.signature:
            raise RuntimeError(
                f"{last} was written by a different model/config; delete the folder or "
                "pass --fresh to start over"
            )
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state["scheduler"] is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        stopper.load_state_dict(state["stopper"])
        self.history = list(state["history"])
        best = self._path("best.pt")
        if best.exists():
            stopper.best_state = torch.load(best, map_location=self.device, weights_only=True)
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"] is not None and torch.cuda.is_available() \
                and len(rng["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        gen = self._loader_generator()
        if gen is not None and rng["loader"] is not None:
            gen.set_state(rng["loader"].cpu())
        self._log(
            f"resumed from {last}: epoch {state['epoch']} done, best epoch {stopper.best_epoch} "
            f"({self.cfg.monitor} {stopper.best_score:.4f}), "
            f"lr {self.optimizer.param_groups[0]['lr']:.2e}"
            + (" [training already finished]" if state["finished"] else "")
        )
        return state["epoch"] + 1, bool(state["finished"]), float(state.get("elapsed_s", 0.0))

    # ---- main loop ----------------------------------------------------------------

    def fit(self) -> dict:
        set_seed(self.cfg.seed)
        stopper = EarlyStopping(self.cfg.early_stopping_patience, mode="max")
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        start_epoch, finished, elapsed = self._try_resume(stopper)
        t0 = time.time()
        for epoch in range(start_epoch, self.cfg.epochs + 1):
            if finished:
                break
            te = time.time()
            train_loss = self.train_epoch()
            val = self.evaluate(self.data.val)
            score = val.get(self.cfg.monitor, float("nan"))
            if np.isnan(score):
                score = -train_loss  # degenerate val set: fall back to loss
            lr = self.optimizer.param_groups[0]["lr"]
            rec = {"epoch": epoch, "train_loss": train_loss, "lr": lr, "time_s": time.time() - te,
                   **{f"val_{k}": v for k, v in val.items()}}
            self.history.append(rec)

            stop = stopper.step(score, self.model, epoch)
            improved = stopper.best_epoch == epoch
            if improved and self.out_dir is not None:
                self._atomic_save(stopper.best_state, self._path("best.pt"))
            if isinstance(self.scheduler, WarmupPlateau):
                self.scheduler.step_epoch(score)

            if epoch % self.cfg.log_every == 0:
                self._log(
                    f"epoch {epoch:3d} | loss {train_loss:.5f} | lr {lr:.2e} | val auc_roc "
                    f"{val.get('auc_roc', float('nan')):.4f} auc_pr {val.get('auc_pr', float('nan')):.4f}"
                    f" | {rec['time_s']:.1f}s" + (" *best" if improved else "")
                )
            finished = stop or epoch == self.cfg.epochs
            self._save_last(epoch, stopper, finished, elapsed + time.time() - t0)
            self._write_history()
            if stop:
                self._log(f"early stopping at epoch {epoch} (best epoch {stopper.best_epoch})")
                break
        stopper.restore(self.model)

        val = self.evaluate(self.data.val)
        test = self.evaluate(self.data.test, threshold=val.get("threshold"))
        result = {
            "best_epoch": stopper.best_epoch,
            "train_time_s": elapsed + time.time() - t0,
            "val": val,
            "test": test,
            "history": self.history,
            "config": asdict(self.cfg),
        }
        if self.out_dir is not None:
            self.save(result)
        return result

    def save(self, result: dict) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_save(self.model.state_dict(), self._path("best.pt"))
        with open(self.out_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2, default=float)
