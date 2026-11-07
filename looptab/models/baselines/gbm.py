"""Tree-ensemble baselines: XGBoost, LightGBM, Random Forest (Section 6.1).

They consume the same encoded arrays as the neural models: numerical columns
as-is and categorical codes as integer features. XGBoost / LightGBM use the
validation set for early stopping.
"""

from __future__ import annotations

import time

import numpy as np

from ...data.dataloader import TabularData, TabularSplit
from ...evaluation.metrics import compute_metrics


def _X(split: TabularSplit) -> np.ndarray:
    return np.concatenate([split.x_num, split.x_cat.astype(np.float32)], axis=1)


def _scale_pos_weight(y: np.ndarray) -> float:
    pos = max(float(y.sum()), 1.0)
    return float((len(y) - pos) / pos)


def fit_gbm(name: str, data: TabularData, params: dict | None = None, seed: int = 42,
            use_gpu: bool = False) -> dict:
    """Fit a tree baseline and return a result dict shaped like Trainer.fit()."""
    params = dict(params or {})
    Xtr, Xva, Xte = _X(data.train), _X(data.val), _X(data.test)
    ytr, yva = data.train.y, data.val.y
    t0 = time.time()
    if name == "xgboost":
        import xgboost as xgb

        defaults = dict(n_estimators=2000, learning_rate=0.05, max_depth=8, subsample=0.8,
                        colsample_bytree=0.8, eval_metric="aucpr", early_stopping_rounds=100,
                        tree_method="hist", random_state=seed, n_jobs=-1,
                        scale_pos_weight=_scale_pos_weight(ytr))
        if use_gpu:
            defaults["device"] = "cuda"
        model = xgb.XGBClassifier(**{**defaults, **params})
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    elif name == "lightgbm":
        import lightgbm as lgb

        defaults = dict(n_estimators=2000, learning_rate=0.05, num_leaves=127, subsample=0.8,
                        subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=-1,
                        scale_pos_weight=_scale_pos_weight(ytr), verbose=-1)
        model = lgb.LGBMClassifier(**{**defaults, **params})
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    elif name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        defaults = dict(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
                        n_jobs=-1, random_state=seed)
        model = RandomForestClassifier(**{**defaults, **params})
        model.fit(Xtr, ytr)
    else:
        raise KeyError(f"unknown tree baseline {name!r}")

    val = compute_metrics(yva, model.predict_proba(Xva)[:, 1])
    test = compute_metrics(data.test.y, model.predict_proba(Xte)[:, 1], threshold=val["threshold"])
    return {"model_name": name, "train_time_s": time.time() - t0, "val": val, "test": test,
            "params": params}


GBM_MODELS = ("xgboost", "lightgbm", "random_forest")
