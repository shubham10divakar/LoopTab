# LoopTab: Residual-Loop Feature Attention for Tabular Fraud Detection

PyTorch implementation of **LoopTab**. It applies one weight-tied attention block **K times** over
the feature tokens of a transaction. A learned **per-token, per-dimension residual gate** controls
each update, and a **dense residual aggregator** combines the outputs of all loops before a
[CLS] classification head trained with focal loss.

The full design is in [`docs/looptab_design_doc.md`](docs/looptab_design_doc.md).

```
features -> FeatureTokenizer -> [RLFA block] x K (shared weights, gated residual)
                                     |  Z^0 ... Z^K
                                     v
                         Dense Residual Aggregator (softmax alpha)
                                     v
                           [CLS] -> Linear -> fraud logit (focal loss)
```

One RLFA iteration (design doc Section 4.1):

```
A_bar = LN(Z^k);  A = MHSA(A_bar)
G     = sigmoid(W_g A_bar + U_g A + b_g)
Z~    = Z^k + G * A
Z^k+1 = Z~ + FFN(LN(Z~))
```

## Install

```bash
pip install -r requirements.txt        # or: pip install -e ".[baselines,extras,dev]"
pytest                                  # 47 tests, ~15 s on CPU
```

## Quick start (no download needed)

```bash
python -m looptab.experiments.train --config synthetic --analyze
```

This trains LoopTab-S on a synthetic dataset with higher-order interactions. It prints the test
metrics and writes the checkpoint, `result.json`, gate and attention diagnostics, and figures to
`looptab/experiments/results/synthetic/`.

## Datasets

```bash
# needs a Kaggle API token in ~/.kaggle/kaggle.json; accept the IEEE-CIS rules on Kaggle first
python -m looptab.data.download --dataset all --out datasets
```

| config       | expected files in `datasets/<name>/`                | split      |
|--------------|-----------------------------------------------------|------------|
| `ieee_cis`   | `train_transaction.csv`, `train_identity.csv`       | temporal   |
| `ccfraud`    | `creditcard.csv`                                    | stratified |
| `paysim`     | `PS_*.csv` (1M-row stratified subsample)            | stratified |
| `fraud_ecom` | `Fraud_Data.csv`                                    | stratified |

All preprocessing statistics are fitted on the training split only: medians, scalers,
category vocabularies, frequency encodings, and the IEEE-CIS V-column selection.
Categorical code `0` is reserved for missing, unseen, and rare values.

## Experiments

```bash
# one model
python -m looptab.experiments.train --config ieee_cis                    # config's LoopTab
python -m looptab.experiments.train --config ieee_cis --model xgboost

# main comparison table (Section 8.1) -> results/main_summary.csv, paper/tables/main_results.tex
python -m looptab.experiments.run_all --seeds 42 43 44 --resume

# ablation table (Section 8.2) and analysis sweeps (Section 8.3)
python -m looptab.evaluation.ablation --config ieee_cis --study ablation --seeds 42 43 44
python -m looptab.evaluation.ablation --config ieee_cis --study depth --ks 1 2 3 4 5 6
python -m looptab.evaluation.ablation --config ieee_cis --study imbalance
python -m looptab.evaluation.ablation --config ieee_cis --study focal
```

### Running one model at a time

Running `run_all` with one dataset and one model adds the result to `results/main.jsonl` and rebuilds the summary
table each time. With `--resume`, models that already finished are skipped, so you can stop and restart at any point.
Datasets from fastest to slowest: `fraud_ecom`, `ccfraud`, `paysim`, `ieee_cis`.

```powershell
$models = "xgboost","lightgbm","random_forest","ft_transformer","tab_transformer","tabnet","saint","looptab-s","looptab-d"
foreach ($m in $models) { python -m looptab.experiments.run_all --configs fraud_ecom --models $m --seeds 42 --resume }
```

Measured on an RTX 3060, one epoch of all six neural models takes about 30 s on Fraud eCom, 68 s on CCFraud,
180 s on PaySim and 353 s on IEEE-CIS. The full single-seed suite takes roughly 5–9 hours, because early stopping
usually ends a run after 30–50 epochs.

### LoopTab variants

Parameter counts are measured at IEEE-CIS scale (67 features).

| name | K | d / heads / FFN | tied | params | what it is |
|------|---|-----------------|------|--------|------------|
| `looptab` | config | config | config | | default: the config's `model:` section (LoopTab-D for the four real datasets) |
| `looptab-s` | 2 | 64 / 4 / 256 | yes | ~76K | shallow preset |
| `looptab-d` | 4 | 128 / 8 / 512 | yes | ~267K | main model |
| `looptab-u` | 4 | 128 / 8 / 512 | **no** | ~961K | untied: one block per loop (the "− weight tying" ablation) |
| `looptab-nogate` | 4 | config | yes | ~234K | gate removed, `Z = Z + A` |
| `looptab-noloop` | **1** | config | yes | ~267K | single pass |
| `looptab-nodense` | 4 | config | yes | ~267K | classifies from `Z^K` only, no aggregation |
| `looptab-nofocal` | 4 | config | yes | ~267K | weighted BCE instead of focal loss |

Each ablation changes exactly one thing on the config architecture. Loop depth can also be swept with
`--study depth --ks 1 2 3 4 5 6`. The graph-augmented LoopTab-G is not implemented.

### Baselines

| name | type |
|------|------|
| `ft_transformer`, `tab_transformer`, `tabnet`, `saint`, `mlp` | neural baselines |
| `xgboost`, `lightgbm`, `random_forest` | tree baselines |

### Metrics

AUC-ROC, AUC-PR (average precision), F1, Precision@{100,500} and Recall@1%FPR.
Choose test F1 with care:

- `f1_at_val_thr` uses the threshold tuned on the validation set. This is the number to report.
- `f1` is the best F1 over thresholds on the test set itself, which is optimistic.

Early stopping monitors validation AUC-PR.

## Layout

```
looptab/
  configs/            dataset YAMLs (model / training / loss sections)
  data/               download.py, preprocess.py, dataloader.py
  models/             tokenizer.py, rlfa_block.py, aggregator.py, looptab.py, registry.py
    baselines/        ft_transformer, tab_transformer, tabnet, saint, mlp, gbm
  training/           trainer.py, focal_loss.py, scheduler.py, early_stopping.py
  evaluation/         metrics.py, ablation.py, visualization.py, reporting.py
  experiments/        train.py, run_all.py, runner.py, results/
  paper/              figures/, tables/ (generated)
tests/
docs/looptab_design_doc.md
```

## Implementation notes and deviations from the design doc

- **Gate input.** Section 2.3.2 feeds the raw `Z^k` into the gate, while Section 4.1 and the
  reference code feed `LN(Z^k)`. This code follows Section 4.1.
- **Parameter counts.** The doc's estimates (~150K / 600K / 2.4M) are higher than the real counts
  for this architecture. With 51 numerical and 16 categorical features (IEEE-CIS scale), the
  measured counts are:
  - LoopTab-S: 76K
  - LoopTab-D: 267K
  - LoopTab-U: 961K
  - 4-layer FT-Transformer: 829K

  The ~3.6x saving from weight tying still holds.
- **IEEE-CIS 67 features.** The exact FDB feature list is not bundled. The default uses 47 fixed
  core columns (amount, card/addr, C, D, M, email domains, device), then fills up to 67 with the
  V columns that have the lowest missing rate on the training split. To use the FDB list, pass it
  with `loader_kwargs.features: {num: [...], cat: [...]}`. Setting `n_features: null` uses every
  column.
- **PaySim.** `nameOrig` and `nameDest` are near-unique IDs, so only their C/M type prefix is
  kept. Set `drop_balance: true` for the leakage-free variant.
- **Fraud eCom.** `device_id` and `ip_address` are frequency-encoded using train counts. The
  loader also adds `time_delta` and the purchase hour and day of week.
- **The tokenizer is vectorized.** It is mathematically identical to one `nn.Linear(1, d)` per
  numerical feature, with one embedding table per categorical feature.
- **Not implemented yet:**
  - ResNet-LSTM-Attention and TabConFraud baselines
  - SAINT contrastive pre-training (SAINT is supervised only)
  - LoopTab-G (graph-augmented)
  - Optuna search (a grid sweep is provided instead)
  - wandb / tensorboard logging (results are logged as JSON / JSONL)
