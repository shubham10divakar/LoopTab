import numpy as np
import pytest
import torch

from looptab.data import prepare_dataset
from looptab.models.baselines import NEURAL_BASELINES, fit_gbm
from looptab.models.baselines.tabnet import sparsemax
from looptab.models.registry import available_models, build_model, is_tree_model

B, N_NUM, CARDS = 16, 5, [3, 7]


def _inputs():
    x_num = torch.randn(B, N_NUM)
    x_cat = torch.stack([torch.randint(0, c + 1, (B,)) for c in CARDS], dim=1)
    return x_num, x_cat


@pytest.mark.parametrize("name", sorted(NEURAL_BASELINES))
def test_neural_baselines_forward_backward(name):
    model = build_model(name, N_NUM, CARDS)
    out = model(*_inputs())
    assert out.shape == (B,)
    out.sum().backward()
    model.eval()
    with torch.no_grad():
        assert model(*_inputs()).shape == (B,)


def test_tab_transformer_numerical_only():
    assert build_model("tab_transformer", N_NUM)(torch.randn(B, N_NUM)).shape == (B,)


def test_sparsemax_properties():
    p = sparsemax(torch.tensor([[1.0, 2.0, 5.0], [0.0, 0.0, 0.0]]))
    assert torch.allclose(p.sum(-1), torch.ones(2))
    assert p[0, 0] == 0 and p[0, 2] == 1  # sparse
    assert torch.allclose(p[1], torch.full((3,), 1 / 3))


def test_registry():
    names = available_models()
    assert "looptab-d" in names and "xgboost" in names
    assert is_tree_model("xgboost") and not is_tree_model("looptab-d")
    with pytest.raises(ValueError):
        build_model("xgboost", 3)


@pytest.mark.parametrize("name", ["xgboost", "random_forest", "lightgbm"])
def test_tree_baselines(name):
    pytest.importorskip({"xgboost": "xgboost", "lightgbm": "lightgbm"}.get(name, "sklearn"))
    data = prepare_dataset("synthetic", loader_kwargs={"n": 3000, "fraud_rate": 0.05})
    params = {"n_estimators": 50}
    res = fit_gbm(name, data, params)
    assert res["test"]["auc_roc"] > 0.7
    assert np.isfinite(res["test"]["auc_pr"])
