"""Fraud-detection metrics (design doc Section 5.3)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def best_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Return (max F1 over thresholds, threshold achieving it)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    p, r = precision[:-1], recall[:-1]  # last point has no threshold
    denom = p + r
    f1 = np.divide(2 * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    if f1.size == 0:
        return 0.0, 0.5
    i = int(np.argmax(f1))
    return float(f1[i]), float(thresholds[i])


def f1_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    pred = y_score >= threshold
    tp = float(np.sum(pred & (y_true == 1)))
    fp = float(np.sum(pred & (y_true == 0)))
    fn = float(np.sum(~pred & (y_true == 1)))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    k = min(k, len(y_true))
    if k <= 0:
        return 0.0
    top = np.argsort(-y_score, kind="stable")[:k]
    return float(np.mean(y_true[top]))


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.01) -> float:
    """Highest TPR achievable with FPR <= max_fpr."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def compute_metrics(
    y_true,
    y_score,
    threshold: float | None = None,
    ks: tuple[int, ...] = (100, 500),
    fpr: float = 0.01,
) -> dict[str, float]:
    """Compute all metrics.

    If ``threshold`` is given (e.g. tuned on the validation set), F1 is also
    reported at that fixed threshold as ``f1_at_val_thr``, which is the
    leakage-free number to quote for test sets.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    out: dict[str, float] = {"n": float(len(y_true)), "pos_rate": float(y_true.mean())}
    if len(np.unique(y_true)) < 2:
        nan = float("nan")
        out.update(auc_roc=nan, auc_pr=nan, f1=nan, threshold=nan, recall_at_fpr=nan)
        return out
    f1, thr = best_f1(y_true, y_score)
    out.update(
        auc_roc=float(roc_auc_score(y_true, y_score)),
        auc_pr=float(average_precision_score(y_true, y_score)),
        f1=f1,
        threshold=thr,
        recall_at_fpr=recall_at_fpr(y_true, y_score, fpr),
    )
    for k in ks:
        out[f"precision_at_{k}"] = precision_at_k(y_true, y_score, k)
    if threshold is not None:
        out["f1_at_val_thr"] = f1_at_threshold(y_true, y_score, threshold)
    return out
