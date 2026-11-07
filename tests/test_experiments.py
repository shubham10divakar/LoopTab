import numpy as np
import pytest

from looptab.data import prepare_dataset
from looptab.evaluation.ablation import resample_to_rate, run_ablation
from looptab.evaluation.reporting import main_table_latex, records_to_frame, summarise
from looptab.evaluation.visualization import collect_diagnostics, plot_attention_evolution, plot_gate_activation
from looptab.experiments.runner import CONFIG_DIR, deep_update, load_config, resolve_model, run_experiment
from looptab.models import LoopTab


def _cfg(epochs=1):
    cfg = load_config("synthetic")
    cfg = deep_update(cfg, {"training": {"epochs": epochs, "warmup_epochs": 0, "device": "cpu",
                                         "batch_size": 256},
                            "model": {"d_model": 16, "n_heads": 2, "d_ff": 32}})
    return cfg


@pytest.fixture(scope="module")
def data():
    return prepare_dataset("synthetic", loader_kwargs={"n": 2000, "fraud_rate": 0.05})


@pytest.mark.parametrize("name", sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")))
def test_all_configs_load(name):
    cfg = load_config(name)
    assert {"data", "model", "training", "loss"} <= set(cfg)
    resolve_model(cfg, None)


def test_resolve_ablations_flip_one_switch():
    cfg = load_config("ieee_cis")
    _, base, _ = resolve_model(cfg, "looptab")
    for name, key, val in [("looptab-nogate", "use_gate", False), ("looptab-noloop", "n_loops", 1),
                           ("looptab-u", "tied_weights", False), ("looptab-nodense", "use_dense", False)]:
        _, kw, _ = resolve_model(cfg, name)
        assert kw[key] == val
        assert {k: v for k, v in kw.items() if k != key} == {k: v for k, v in base.items() if k != key}
    _, _, eff = resolve_model(cfg, "looptab-nofocal")
    assert eff["loss"]["type"] == "weighted_bce" and cfg["loss"]["type"] == "focal"


def test_run_experiment_neural_and_tree(data, tmp_path):
    res = run_experiment(_cfg(), "looptab", data=data, out_dir=tmp_path / "lt", verbose=False)
    assert isinstance(res["_model"], LoopTab) and res["n_params"] > 0
    assert (tmp_path / "lt" / "result.json").exists()
    res = run_experiment(_cfg(), "xgboost", data=data, verbose=False)
    assert res["test"]["auc_roc"] > 0.5


def test_ablation_writes_table(data, tmp_path):
    summ = run_ablation(_cfg(), data, tmp_path, seeds=[0],
                        variants=["looptab", "looptab-nogate"], verbose=False)
    assert len(summ) == 2
    assert "\\begin{table}" in (tmp_path / "ablation_table.tex").read_text()


def test_resample_to_rate(data):
    for rate in (0.01, 0.2):
        s = resample_to_rate(data.train, rate)
        assert abs(s.pos_rate - rate) < 0.01


def test_diagnostics_and_plots(data, tmp_path):
    pytest.importorskip("matplotlib")
    res = run_experiment(_cfg(), "looptab", data=data, verbose=False)
    diag = collect_diagnostics(res["_model"], data.test)
    K = res["_model"].n_loops
    assert diag["gate_fraud"].shape == (K,) and diag["attn_legit"].shape[0] == K
    assert np.isclose(diag["alpha"].sum(), 1.0)
    plot_gate_activation(diag, tmp_path / "g.png")
    plot_attention_evolution(diag, data.feature_names, tmp_path / "a.png")
    assert (tmp_path / "g.png").exists() and (tmp_path / "a.png").exists()


def test_main_table_latex():
    recs = [{"dataset": "d1", "model": m, "seed": s, "test": {"auc_roc": a, "auc_pr": a / 2}}
            for m, a in (("A", 0.9), ("B", 0.8)) for s in (0, 1)]
    tex = main_table_latex(summarise(records_to_frame(recs)))
    assert "\\textbf{0.9000}" in tex and "B &" in tex
