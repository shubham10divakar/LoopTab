"""Config-driven single-run experiment logic shared by all CLIs.

Model naming used throughout the experiments:

* ``looptab-s`` / ``looptab-d``: the presets from design doc Section 3.
* ``looptab`` (or ``None``): the architecture given in the config's ``model:``.
* ablations (Section 6.4) take the config architecture and flip one switch:
  ``looptab-nogate``, ``looptab-noloop`` (K=1), ``looptab-u`` (untied),
  ``looptab-nodense``, ``looptab-nofocal`` (weighted BCE instead of focal).
* neural baselines (``ft_transformer``, ``tabnet``, ...) read kwargs from the
  optional ``baselines: {name: {...}}`` config section.
* tree baselines (``xgboost``, ``lightgbm``, ``random_forest``).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from ..data import TabularData, prepare_dataset
from ..models import count_parameters
from ..models.baselines import fit_gbm
from ..models.looptab import VARIANTS
from ..models.registry import build_model, is_tree_model
from ..training import Trainer, TrainConfig, set_seed

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

ABLATIONS: dict[str, dict] = {
    "looptab-nogate": {"model": {"use_gate": False}},
    "looptab-noloop": {"model": {"n_loops": 1}},
    "looptab-u": {"model": {"tied_weights": False}},
    "looptab-nodense": {"model": {"use_dense": False}},
    "looptab-nofocal": {"loss": {"type": "weighted_bce"}},
}
PRESETS = ("looptab-s", "looptab-d")
LOOPTAB_ARCH_KEYS = ("d_model", "n_heads", "n_loops", "d_ff", "dropout", "tied_weights",
                     "use_gate", "use_dense", "gate_bias_init")


def load_config(path_or_name: str | Path) -> dict:
    p = Path(path_or_name)
    if not p.exists():
        p = CONFIG_DIR / f"{path_or_name}.yaml"
    with open(p) as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(p)
    return cfg


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_data(cfg: dict, root_override: str | None = None) -> TabularData:
    d = dict(cfg["data"])
    d.pop("num_workers", None)
    name = d.pop("name")
    root = root_override or d.pop("root", None)
    d.pop("root", None)
    d.pop("test_ratio", None)
    seed = cfg.get("training", {}).get("seed", 42)
    return prepare_dataset(name, root=root, seed=seed, **d)


def resolve_model(cfg: dict, model_name: str | None) -> tuple[str, dict, dict]:
    """Return (model key, model kwargs, effective config) for a run."""
    model_cfg = dict(cfg.get("model", {}))
    base_variant = model_cfg.pop("variant", "looptab-d")
    arch = {k: v for k, v in model_cfg.items() if k in LOOPTAB_ARCH_KEYS}
    name = (model_name or "looptab").lower()

    if name == "looptab":
        return base_variant, {**VARIANTS[base_variant], **arch}, cfg
    if name in ABLATIONS:
        patch = ABLATIONS[name]
        eff = deep_update(cfg, {k: v for k, v in patch.items() if k != "model"})
        kwargs = {**VARIANTS[base_variant], **arch, **patch.get("model", {})}
        return "looptab-d", kwargs, eff
    if name in PRESETS:
        kwargs = dict(VARIANTS[name])
        if "dropout" in arch:
            kwargs["dropout"] = arch["dropout"]
        return name, kwargs, cfg
    return name, dict(cfg.get("baselines", {}).get(name, {})), cfg


def run_experiment(
    cfg: dict,
    model_name: str | None = None,
    data: TabularData | None = None,
    out_dir: str | Path | None = None,
    seed: int | None = None,
    model_overrides: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Train + evaluate one model on one dataset; returns a result dict."""
    if seed is not None:
        cfg = deep_update(cfg, {"training": {"seed": seed}})
    if data is None:
        data = load_data(cfg)
    key, kwargs, eff = resolve_model(cfg, model_name)
    kwargs.update(model_overrides or {})
    train_cfg = eff.get("training", {})
    run_seed = train_cfg.get("seed", 42)
    label = model_name or "looptab"

    if is_tree_model(key):
        res = fit_gbm(key, data, kwargs, seed=run_seed)
        res["n_params"] = None
    else:
        set_seed(run_seed)
        model = build_model(key, data.n_num, data.cat_cardinalities, **kwargs)
        tcfg = TrainConfig.from_dict(train_cfg, eff.get("loss"))
        if verbose:
            print(f"[{data.name}] {label}: {count_parameters(model):,} params", flush=True)
        res = Trainer(model, data, tcfg, out_dir=out_dir, verbose=verbose).fit()
        res["n_params"] = count_parameters(model)
        res["model_kwargs"] = kwargs
        res["_model"] = model  # not serialised; used by analysis scripts

    res.update(dataset=data.name, model=label, seed=run_seed)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(out_dir) / "result.json", "w") as f:
            json.dump(serialisable(res), f, indent=2, default=float)
    return res


def serialisable(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}
