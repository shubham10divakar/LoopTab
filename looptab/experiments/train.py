"""Train and evaluate one model on one dataset.

    python -m looptab.experiments.train --config synthetic
    python -m looptab.experiments.train --config ieee_cis --model looptab-d --analyze
    python -m looptab.experiments.train --config ccfraud --model xgboost
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..models.looptab import LoopTab
from .runner import load_config, load_data, run_experiment, serialisable


def analyze(model: LoopTab, data, out_dir: Path, device: str = "cpu") -> None:
    from ..evaluation.visualization import (
        collect_diagnostics,
        plot_attention_evolution,
        plot_gate_activation,
        plot_gate_heatmap,
    )

    diag = collect_diagnostics(model, data.test, device=device)
    summary = {k: (v.tolist() if hasattr(v, "tolist") else v)
               for k, v in diag.items() if not k.startswith(("attn_", "gate_tok_"))}
    (out_dir / "diagnostics.json").write_text(json.dumps(summary, indent=2))
    if "gate_fraud" in diag:
        plot_gate_activation(diag, out_dir / "gate_activation.png")
        plot_gate_heatmap(diag, data.feature_names, out_dir / "gate_heatmap_fraud.png")
    if "attn_fraud" in diag:
        plot_attention_evolution(diag, data.feature_names, out_dir / "attention_fraud.png")
    print("diagnostics:", json.dumps(summary))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="config path or name in looptab/configs")
    p.add_argument("--model", default=None, help="model name (default: config's LoopTab)")
    p.add_argument("--root", default=None, help="override dataset root directory")
    p.add_argument("--out", default="looptab/experiments/results")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--analyze", action="store_true", help="gate / attention diagnostics + figures")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.device is not None:
        cfg["training"]["device"] = args.device
    data = load_data(cfg, args.root)
    print(data.summary())

    name = args.model or "looptab"
    seed = args.seed if args.seed is not None else cfg["training"].get("seed", 42)
    out_dir = Path(args.out) / data.name / f"{name}_seed{seed}"
    res = run_experiment(cfg, args.model, data=data, out_dir=out_dir, seed=seed)

    print("\nTEST:", json.dumps({k: round(v, 5) for k, v in res["test"].items()}))
    if args.analyze and isinstance(res.get("_model"), LoopTab):
        from ..training import resolve_device

        analyze(res["_model"], data, out_dir, str(resolve_device(cfg["training"].get("device", "auto"))))
    print(f"saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
