"""Run the main comparison (design doc Section 8.1) across datasets and models.

    python -m looptab.experiments.run_all                     # everything
    python -m looptab.experiments.run_all --configs ccfraud --seeds 42 43 44
    python -m looptab.experiments.run_all --configs synthetic --models looptab-s xgboost --epochs 5

Writes per-run records to ``<out>/main.jsonl`` (appended, so interrupted runs
can be resumed with ``--resume``), a summary CSV, the LaTeX main table in
``looptab/paper/tables`` and the alpha-weight figure in ``looptab/paper/figures``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..evaluation.reporting import (
    append_jsonl,
    main_table_latex,
    read_jsonl,
    records_to_frame,
    summarise,
)
from .runner import load_config, load_data, run_experiment, serialisable

DEFAULT_CONFIGS = ["ieee_cis", "ccfraud", "paysim", "fraud_ecom"]
DEFAULT_MODELS = ["xgboost", "lightgbm", "random_forest", "ft_transformer", "tab_transformer",
                  "tabnet", "saint", "looptab-s", "looptab-d"]
PAPER_DIR = Path(__file__).resolve().parents[1] / "paper"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--data-root", default=None,
                   help="parent dir holding one subdir per dataset (overrides config roots)")
    p.add_argument("--out", default="looptab/experiments/results")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--resume", action="store_true", help="skip (dataset, model, seed) already in main.jsonl; "
                   "interrupted runs always continue from their last.pt")
    p.add_argument("--fresh", action="store_true",
                   help="ignore an existing last.pt checkpoint and train from epoch 1")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    out = Path(args.out)
    jsonl = out / "main.jsonl"
    done = set()
    if args.resume and jsonl.exists():
        done = {(r["dataset"], r["model"], r["seed"]) for r in read_jsonl(jsonl)}

    alphas = {}
    for cfg_name in args.configs:
        cfg = load_config(cfg_name)
        if args.epochs is not None:
            cfg["training"]["epochs"] = args.epochs
        if args.fresh:
            cfg["training"]["resume"] = False
        root = str(Path(args.data_root) / cfg["data"]["name"]) if args.data_root else None
        try:
            data = load_data(cfg, root)
        except FileNotFoundError as e:
            print(f"[skip] {cfg_name}: {e}")
            continue
        print(data.summary())
        for seed in args.seeds:
            for model in args.models:
                if (data.name, model, seed) in done:
                    print(f"[resume] skipping {data.name}/{model}/seed{seed}")
                    continue
                try:
                    res = run_experiment(cfg, model, data=data, seed=seed, verbose=not args.quiet,
                                         out_dir=out / data.name / f"{model}_seed{seed}")
                except Exception as e:  # keep the suite going; record the failure
                    print(f"[error] {data.name}/{model}/seed{seed}: {e!r}")
                    continue
                rec = serialisable(res)
                rec.pop("history", None)
                append_jsonl(jsonl, rec)
                m = res.get("_model")
                if model == "looptab-d" and getattr(m, "aggregator", None) is not None:
                    alphas[data.name] = m.aggregator.weights.detach().cpu().numpy()
                t = res["test"]
                print(f"  -> {data.name}/{model}: AUC-ROC {t['auc_roc']:.4f}  AUC-PR {t['auc_pr']:.4f}")

    if not jsonl.exists():
        print("no results")
        return 1
    summ = summarise(records_to_frame(read_jsonl(jsonl)))
    summ.to_csv(out / "main_summary.csv", index=False)
    (PAPER_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (PAPER_DIR / "tables" / "main_results.tex").write_text(main_table_latex(summ))
    if alphas:
        from ..evaluation.visualization import plot_alpha

        plot_alpha(alphas, PAPER_DIR / "figures" / "alpha_weights.png")
    print(summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
