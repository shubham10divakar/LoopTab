import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from looptab.data import TabularPreprocessor, make_synthetic, prepare_dataset, split_frame
from looptab.evaluation.metrics import compute_metrics, precision_at_k, recall_at_fpr
from looptab.models import build_looptab
from looptab.training import EarlyStopping, FocalLoss, Trainer, TrainConfig, cosine_warmup_scheduler


# ---- losses -----------------------------------------------------------------

def test_focal_gamma0_equals_alpha_weighted_bce():
    logits = torch.randn(64)
    y = (torch.rand(64) < 0.3).float()
    fl = FocalLoss(alpha=0.5, gamma=0.0)(logits, y)
    bce = 0.5 * F.binary_cross_entropy_with_logits(logits, y)
    assert torch.allclose(fl, bce, atol=1e-6)


def test_focal_downweights_easy_examples():
    y = torch.tensor([0.0, 0.0])
    loss = FocalLoss(alpha=0.5, gamma=2.0, reduction="none")(torch.tensor([-6.0, 0.0]), y)
    assert loss[0] < loss[1] * 1e-3


# ---- scheduler / early stopping --------------------------------------------

def test_cosine_warmup_shape():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sch = cosine_warmup_scheduler(opt, warmup_steps=10, total_steps=110)
    lrs = []
    for _ in range(110):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    assert lrs[0] == pytest.approx(0.1)
    assert lrs[9] == pytest.approx(1.0)
    assert lrs[-1] < 0.01


def test_early_stopping_restores_best():
    m = torch.nn.Linear(1, 1)
    es = EarlyStopping(patience=2)
    with torch.no_grad():
        m.weight.fill_(1.0)
    assert not es.step(0.9, m, 1)
    with torch.no_grad():
        m.weight.fill_(2.0)
    assert not es.step(0.5, m, 2)
    assert es.step(0.4, m, 3)
    es.restore(m)
    assert m.weight.item() == 1.0 and es.best_epoch == 1


# ---- metrics ----------------------------------------------------------------

def test_metrics_perfect_ranking():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    m = compute_metrics(y, s, ks=(2,))
    assert m["auc_roc"] == 1.0 and m["auc_pr"] == 1.0 and m["f1"] == 1.0
    assert m["precision_at_2"] == 1.0 and m["recall_at_fpr"] == 1.0


def test_metrics_helpers():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.1, 0.0])
    assert precision_at_k(y, s, 2) == 0.5
    assert recall_at_fpr(y, s, 0.0) == 0.5


def test_metrics_single_class_is_nan():
    m = compute_metrics(np.zeros(5), np.random.rand(5))
    assert np.isnan(m["auc_roc"])


# ---- preprocessing ----------------------------------------------------------

def test_preprocessor_fit_on_train_only():
    tr = pd.DataFrame({"a": [1.0, 2.0, np.nan, 3.0], "c": ["x", "y", "x", None]})
    te = pd.DataFrame({"a": [np.nan, 100.0], "c": ["z", "y"]})
    pre = TabularPreprocessor(["a"], ["c"], scaler="none").fit(tr)
    x_num, x_cat = pre.transform(te)
    assert x_num[0, 0] == 2.0           # train median imputation
    assert x_cat[0, 0] == 0             # unseen -> 0
    assert x_cat[1, 0] == pre.vocabs["c"]["y"]
    assert pre.cat_cardinalities == [2]


def test_temporal_split_is_ordered():
    df = pd.DataFrame({"t": np.random.permutation(100), "y": np.r_[np.zeros(90), np.ones(10)]})
    tr, va, te = split_frame(df, "y", "temporal", time_col="t")
    assert tr["t"].max() < va["t"].min() and va["t"].max() < te["t"].min()
    assert (len(tr), len(va), len(te)) == (70, 15, 15)


def test_prepare_synthetic():
    data = prepare_dataset("synthetic", loader_kwargs={"n": 2000})
    assert data.n_num == 8 and data.n_cat == 3
    assert len(data.train) + len(data.val) + len(data.test) == 2000
    assert np.isfinite(data.train.x_num).all()
    for j, card in enumerate(data.cat_cardinalities):
        assert data.test.x_cat[:, j].max() <= card


# ---- end-to-end -------------------------------------------------------------

def test_trainer_learns_synthetic(tmp_path):
    torch.manual_seed(0)
    data = prepare_dataset("synthetic", loader_kwargs={"n": 6000, "fraud_rate": 0.05})
    model = build_looptab("looptab-s", data.n_num, data.cat_cardinalities, d_model=32)
    cfg = TrainConfig(batch_size=256, epochs=8, warmup_epochs=1, early_stopping_patience=4,
                      device="cpu")
    res = Trainer(model, data, cfg, out_dir=tmp_path, verbose=False).fit()
    assert res["test"]["auc_roc"] > 0.8
    assert (tmp_path / "model.pt").exists() and (tmp_path / "result.json").exists()
    assert "f1_at_val_thr" in res["test"]


def test_trainer_weighted_bce_and_balanced_sampling():
    data = prepare_dataset("synthetic", loader_kwargs={"n": 1500})
    model = build_looptab("looptab-s", data.n_num, data.cat_cardinalities, d_model=16)
    cfg = TrainConfig(batch_size=128, epochs=1, warmup_epochs=0, device="cpu",
                      balanced_sampling=True, loss={"type": "weighted_bce"})
    res = Trainer(model, data, cfg, verbose=False).fit()
    assert np.isfinite(res["history"][0]["train_loss"])


def test_make_synthetic_rate():
    raw = make_synthetic(n=10000, fraud_rate=0.02)
    assert abs(raw.df["label"].mean() - 0.02) < 0.002
