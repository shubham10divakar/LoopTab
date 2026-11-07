"""Collect result records into DataFrames and LaTeX tables (Section 8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

METRIC_LABELS = {
    "auc_roc": "AUC-ROC",
    "auc_pr": "AUC-PR",
    "f1_at_val_thr": "F1",
    "recall_at_fpr": "Recall@1%FPR",
}


def append_jsonl(path: str | Path, record: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=float) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def records_to_frame(records: Iterable[dict], split: str = "test") -> pd.DataFrame:
    rows = []
    for r in records:
        row = {"dataset": r["dataset"], "model": r["model"], "seed": r.get("seed"),
               "n_params": r.get("n_params"), "train_time_s": r.get("train_time_s")}
        row.update({k: v for k, v in r[split].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, metrics=("auc_roc", "auc_pr", "f1_at_val_thr"),
              by=("dataset", "model")) -> pd.DataFrame:
    """Mean and std over seeds."""
    present = [m for m in metrics if m in df.columns]
    g = df.groupby(list(by), sort=False)[present]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=0).add_suffix("_std"))
    return out.reset_index()


def _fmt(mean: float, std: float, bold: bool) -> str:
    if mean is None or np.isnan(mean):
        return "--"
    s = f"{mean:.4f}"
    if std and not np.isnan(std) and std > 0:
        s += f"$\\pm${std:.4f}"
    return f"\\textbf{{{s}}}" if bold else s


def main_table_latex(summary: pd.DataFrame, metrics=("auc_roc", "auc_pr"),
                     caption: str = "Fraud detection performance (AUC-ROC / AUC-PR).",
                     label: str = "tab:main") -> str:
    """Models as rows, datasets as columns, 'AUC-ROC / AUC-PR' per cell; best bold."""
    datasets = list(dict.fromkeys(summary["dataset"]))
    models = list(dict.fromkeys(summary["model"]))
    best = {(d, m): summary.loc[summary.dataset == d, f"{m}_mean"].max()
            for d in datasets for m in metrics}
    lines = [
        "\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l" + "c" * len(datasets) + "}", "\\toprule",
        "Model & " + " & ".join(d.replace("_", "-") for d in datasets) + " \\\\", "\\midrule",
    ]
    for model in models:
        cells = []
        for d in datasets:
            row = summary[(summary.dataset == d) & (summary.model == model)]
            if row.empty:
                cells.append("--")
                continue
            r = row.iloc[0]
            parts = [_fmt(r[f"{m}_mean"], r.get(f"{m}_std", np.nan),
                          np.isclose(r[f"{m}_mean"], best[(d, m)])) for m in metrics]
            cells.append(" / ".join(parts))
        lines.append(model + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines)


def ablation_table_latex(summary: pd.DataFrame, metrics=("auc_roc", "auc_pr", "f1_at_val_thr"),
                         caption: str = "Ablation study.", label: str = "tab:ablation") -> str:
    lines = [
        "\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "c" * len(metrics) + "}", "\\toprule",
        "Variant & " + " & ".join(METRIC_LABELS.get(m, m) for m in metrics) + " \\\\",
        "\\midrule",
    ]
    best = {m: summary[f"{m}_mean"].max() for m in metrics}
    for _, r in summary.iterrows():
        cells = [_fmt(r[f"{m}_mean"], r.get(f"{m}_std", np.nan),
                      np.isclose(r[f"{m}_mean"], best[m])) for m in metrics]
        lines.append(f"{r['model']} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)
