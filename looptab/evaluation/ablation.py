"""Ablations and sweeps (design doc Sections 6.4, 8.2, 8.3).

    python -m looptab.evaluation.ablation --config ieee_cis --study ablation
    python -m looptab.evaluation.ablation --config ieee_cis --study depth --ks 1 2 3 4 5 6
    python -m looptab.evaluation.ablation --config ieee_cis --study imbalance \
        --rates 0.001 0.005 0.01 0.035 0.05 0.1 --models looptab xgboost ft_transformer
    python -m looptab.evaluation.ablation --config ieee_cis --study focal

Each study appends records to ``<out>/<study>.jsonl`` and writes a summary CSV
(and LaTeX for the ablation table) plus a figure where applicable.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from ..data.dataloader import TabularData, TabularSplit
from ..experiments.runner import load_config, load_data, run_experiment, serialisable
from .reporting import ablation_table_latex, append_jsonl, records_to_frame, summarise

ABLATION_VARIANTS = ["looptab", "looptab-nogate", "looptab-noloop", "looptab-u",
                     "looptab-nodense", "looptab-nofocal"]
ABLATION_LABELS = {
    "looptab": "LoopTab (full)",
    "looptab-nogate": "- Gating",
    "looptab-noloop": "- Loop (K=1)",
    "looptab-u": "- Weight tying",
    "looptab-nodense": "- Dense aggregation",
    "looptab-nofocal": "- Focal loss",
}


def resample_to_rate(split: TabularSplit, rate: float, seed: int = 0) -> TabularSplit:
    """Down-sample one class so the split has fraud rate ``rate``.

    Below the natural rate positives are dropped; above it negatives are.
    """
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(split.y == 1), np.flatnonzero(split.y == 0)
    cur = len(pos) / max(len(split), 1)
    if rate < cur:
        n_pos = max(1, int(round(rate * len(neg) / (1 - rate))))
        pos = rng.choice(pos, size=min(n_pos, len(pos)), replace=False)
    elif rate > cur:
        n_neg = int(round(len(pos) * (1 - rate) / rate))
        neg = rng.choice(neg, size=min(n_neg, len(neg)), replace=False)
    return split.subset(np.sort(np.concatenate([pos, neg])))


def resample_data(data: TabularData, rate: float, seed: int = 0) -> TabularData:
    return TabularData(
        train=resample_to_rate(data.train, rate, seed),
        val=resample_to_rate(data.val, rate, seed + 1),
        test=resample_to_rate(data.test, rate, seed + 2),
        num_names=data.num_names, cat_names=data.cat_names,
        cat_cardinalities=data.cat_cardinalities, name=data.name, meta=data.meta,
    )


def _record(res: dict, out_jsonl: Path, **extra) -> dict:
    rec = {**serialisable(res), **extra}
    rec.pop("history", None)
    append_jsonl(out_jsonl, rec)
    return rec


def run_ablation(cfg, data, out: Path, seeds, variants=ABLATION_VARIANTS, verbose=True):
    jsonl = out / "ablation.jsonl"
    recs = []
    for seed, v in itertools.product(seeds, variants):
        res = run_experiment(cfg, v, data=data, seed=seed, verbose=verbose)
        recs.append(_record(res, jsonl))
    df = records_to_frame(recs)
    df["model"] = df["model"].map(lambda m: ABLATION_LABELS.get(m, m))
    summ = summarise(df)
    summ.to_csv(out / "ablation_summary.csv", index=False)
    (out / "ablation_table.tex").write_text(ablation_table_latex(
        summ, caption=f"Ablation study on {data.name}."))
    return summ


def run_depth(cfg, data, out: Path, seeds, ks=(1, 2, 3, 4, 5, 6), verbose=True):
    from .visualization import plot_curve

    jsonl = out / "depth.jsonl"
    recs = []
    for seed, k in itertools.product(seeds, ks):
        res = run_experiment(cfg, "looptab", data=data, seed=seed,
                             model_overrides={"n_loops": k}, verbose=verbose)
        res["model"] = f"K={k}"
        recs.append(_record(res, jsonl, n_loops=k))
    df = records_to_frame(recs)
    df["K"] = [r["n_loops"] for r in recs]
    summ = df.groupby("K")[["auc_roc", "auc_pr"]].agg(["mean", "std"])
    summ.to_csv(out / "depth_summary.csv")
    plot_curve(list(summ.index), {"AUC-ROC": summ[("auc_roc", "mean")].tolist(),
                                  "AUC-PR": summ[("auc_pr", "mean")].tolist()},
               "number of loops K", "test metric", out / "depth.png",
               title=f"Loop depth sensitivity ({data.name})")
    return summ


def run_imbalance(cfg, data, out: Path, seeds, rates, models=("looptab",), verbose=True):
    from .visualization import plot_curve

    jsonl = out / "imbalance.jsonl"
    recs = []
    for seed, rate in itertools.product(seeds, rates):
        d = resample_data(data, rate, seed)
        for m in models:
            res = run_experiment(cfg, m, data=d, seed=seed, verbose=verbose)
            recs.append(_record(res, jsonl, fraud_rate=rate))
    df = records_to_frame(recs)
    df["fraud_rate"] = [r["fraud_rate"] for r in recs]
    summ = df.groupby(["model", "fraud_rate"])["auc_pr"].mean().unstack(0)
    summ.to_csv(out / "imbalance_summary.csv")
    plot_curve(list(summ.index), {m: summ[m].tolist() for m in summ.columns},
               "fraud rate", "test AUC-PR", out / "imbalance.png",
               title=f"Imbalance robustness ({data.name})", logx=True)
    return summ


def run_focal_sweep(cfg, data, out: Path, seeds, alphas=(0.25, 0.5, 0.75),
                    gammas=(0.5, 1.0, 2.0, 3.0), verbose=True):
    """Risk-mitigation sweep from Section 12 (grid; use Optuna for finer search)."""
    from ..experiments.runner import deep_update

    jsonl = out / "focal.jsonl"
    recs = []
    for seed, a, g in itertools.product(seeds, alphas, gammas):
        c = deep_update(cfg, {"loss": {"type": "focal", "alpha": a, "gamma": g}})
        res = run_experiment(c, "looptab", data=data, seed=seed, verbose=verbose)
        res["model"] = f"alpha={a},gamma={g}"
        recs.append(_record(res, jsonl, alpha=a, gamma=g))
    summ = summarise(records_to_frame(recs))
    summ.to_csv(out / "focal_summary.csv", index=False)
    return summ


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--study", required=True, choices=["ablation", "depth", "imbalance", "focal"])
    p.add_argument("--root", default=None, help="override dataset root directory")
    p.add_argument("--out", default="looptab/experiments/results")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--rates", type=float, nargs="+", default=[0.001, 0.005, 0.01, 0.035, 0.05, 0.1])
    p.add_argument("--models", nargs="+", default=["looptab", "xgboost", "ft_transformer"])
    p.add_argument("--epochs", type=int, default=None, help="override training epochs")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    data = load_data(cfg, args.root)
    print(data.summary())
    out = Path(args.out) / data.name
    out.mkdir(parents=True, exist_ok=True)
    v = not args.quiet
    if args.study == "ablation":
        print(run_ablation(cfg, data, out, args.seeds, verbose=v).to_string())
    elif args.study == "depth":
        print(run_depth(cfg, data, out, args.seeds, args.ks, verbose=v).to_string())
    elif args.study == "imbalance":
        print(run_imbalance(cfg, data, out, args.seeds, args.rates, args.models, verbose=v).to_string())
    else:
        print(run_focal_sweep(cfg, data, out, args.seeds, verbose=v).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
