# LoopTab: Residual-Loop Feature Attention for Tabular Fraud Detection

## Full Design Document — FSI (FinTech and Sustainable Innovation) Paper

---

## 1. Motivation & Research Gap

### 1.1 Problem Statement

Financial transaction fraud detection on tabular data faces three compounding challenges:

1. **Feature interaction complexity**: Fraud signals emerge from non-obvious cross-feature interactions (e.g., transaction amount × time-of-day × merchant category × device fingerprint). Single-pass attention captures pairwise interactions but misses higher-order combinatorial patterns.
2. **Extreme class imbalance**: Fraud rates range from 0.13% (PaySim) to 3.5% (IEEE-CIS), meaning models are overwhelmed by legitimate transactions and fail to learn subtle fraud representations.
3. **Adversarial drift**: Fraud patterns evolve as attackers adapt, requiring models that learn robust, generalizable feature relationships rather than brittle surface patterns.

### 1.2 What Exists (State of the Art, 2025–2026)

| Approach | Key Work | Limitation |
|----------|----------|------------|
| FT-Transformer | Gorishniy et al., 2021 | Single-pass attention; no iterative refinement |
| TabTransformer | Huang et al., 2020 | Only attends over categorical features |
| Graph Attention (GAT/GraphSAGE) | Semi-GNN, GTAN, ATGAT | Requires pre-built transaction graph; not end-to-end on raw tabular |
| ResNet-LSTM + Attention | Umaru et al., 2025 | Bolts attention onto ResNet/LSTM; attention is auxiliary, not the core mechanism |
| GraphSAGE + DistilBERT cross-attention | ScienceDirect 2026 | Needs NLP embeddings; heavy LLM dependency |
| SpiderNet | Itsy Bitsy SpiderNet | Skip connections on CNN rules; no self-attention |
| TabConFraud | ACM 2025 | Contrastive learning focus; attention is vanilla self-attention |
| Transformer–BDI | MDPI 2025 | Knowledge distillation wrapper; standard 4-head transformer inside |
| Mix-of-Experts Hybrid | Vallarino, 2025 | MoE routing overhead; no novel attention |

### 1.3 The Gap We Fill

**No existing work iterates attention over tabular features with residual gating.** Current transformers for fraud detection apply attention in a single pass (or stack identical layers). We propose *LoopTab*: a mechanism that **loops attention computation K times** with **learned residual gates** that progressively refine feature representations from low-order to high-order interactions, while **residual shortcuts** (inspired by ResNet) ensure gradient flow and prevent representation collapse.

This directly addresses all three challenges:
- **Iterative loops** capture progressively higher-order feature interactions
- **Residual gating** prevents the model from being overwhelmed by majority-class patterns (the gate learns to preserve minority-class signals)
- **Dense residual aggregation** across loops makes the model robust to adversarial drift by maintaining multi-scale feature views

---

## 2. The LoopTab Architecture

### 2.1 High-Level Overview

```
Input (tabular features)
    │
    ▼
┌─────────────────────┐
│  Feature Tokenizer   │  ← Each feature → d-dim token
│  (Numerical + Cat.)  │
└─────────┬───────────┘
          │
          ▼
    ┌───────────┐
    │ Loop k=1  │◄──────────────────────┐
    │  RLFA     │                       │
    │  Block    │──── residual gate ────►│ Dense Residual
    └─────┬─────┘                       │ Aggregation
          │                             │
    ┌─────▼─────┐                       │
    │ Loop k=2  │                       │
    │  RLFA     │──── residual gate ────►│
    │  Block    │                       │
    └─────┬─────┘                       │
          │                             │
         ...                            │
          │                             │
    ┌─────▼─────┐                       │
    │ Loop k=K  │                       │
    │  RLFA     │──── residual gate ────►│
    │  Block    │                       │
    └─────┬─────┘                       │
          │                             │
          ▼                             ▼
    ┌───────────────────────────────────────┐
    │     Dense Residual Aggregator         │
    │  (weighted sum of all K loop outputs) │
    └───────────────────┬───────────────────┘
                        │
                        ▼
              ┌───────────────────┐
              │  Classification   │
              │  Head (Focal)     │
              └───────────────────┘
```

### 2.2 Component 1: Feature Tokenizer

Converts raw tabular features into a sequence of d-dimensional tokens.

**Numerical features:**
```
z_num_j = W_num_j · x_j + b_num_j     (per-feature linear projection)
```
Each numerical feature gets its own linear projection (not shared), producing a d-dimensional token. This is critical for tabular data where features have heterogeneous scales and semantics.

**Categorical features:**
```
z_cat_j = Embedding_j(x_j)            (per-feature learned embedding table)
```
Each categorical feature has its own embedding lookup table of size (cardinality_j × d).

**Combined token sequence:**
```
Z⁰ = [z_1, z_2, ..., z_F] ∈ ℝ^{F × d}
```
where F = total number of features (numerical + categorical). A learned [CLS] token is prepended:
```
Z⁰ = [z_CLS, z_1, z_2, ..., z_F] ∈ ℝ^{(F+1) × d}
```

**Design choice**: Unlike TabTransformer (which only attends over categoricals), we tokenize ALL features. Unlike FT-Transformer (which does the same), we then ITERATE over them.

### 2.3 Component 2: Residual-Loop Feature Attention (RLFA) Block

This is the core novel contribution. Each RLFA block contains:

#### 2.3.1 Multi-Head Self-Attention (MHSA)

Standard scaled dot-product attention across feature tokens:

```
Q = Z^k · W_Q,  K = Z^k · W_K,  V = Z^k · W_V

Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V

MHSA(Z^k) = Concat(head_1, ..., head_h) · W_O
```

where h = number of attention heads, d_k = d / h.

**Key detail**: Q, K, V weights are **shared across all K loops** (weight-tied iteration). This is what makes it a true "loop" rather than just stacking layers — the same attention function is applied iteratively, but it receives progressively refined inputs. This dramatically reduces parameters compared to stacking K independent transformer layers.

#### 2.3.2 The Residual Gate (novel)

After attention, a learned gate controls how much new information flows through:

```
A^k = MHSA(LayerNorm(Z^k))                          # Attention output
G^k = σ(W_g · Z^k + U_g · A^k + b_g)               # Gate ∈ [0, 1]
Z^{k+1} = Z^k + G^k ⊙ A^k                          # Gated residual update
```

where:
- σ is the sigmoid function
- W_g, U_g ∈ ℝ^{d × d} are learned gate parameters
- ⊙ is element-wise multiplication
- G^k is computed **per-feature, per-dimension** (it's a full F×d gate)

**Why this matters**: The gate learns, per feature and per sample, whether the attention output at this loop iteration is informative. For majority-class (legitimate) samples, early loops may suffice and gates close early. For minority-class (fraud) samples, later loops may activate higher-order feature interactions that reveal the fraud pattern.

**ResNet connection**: When G^k = 1 everywhere, this reduces to the standard residual connection Z^{k+1} = Z^k + A^k (identity shortcut). When G^k = 0, the loop is a no-op pass-through. The gate interpolates between these extremes.

#### 2.3.3 Feed-Forward Network (FFN) with Residual

After the gated attention update:

```
F^k = FFN(LayerNorm(Z^{k+1}))
Z^{k+1} = Z^{k+1} + F^k                            # Standard residual
```

where FFN = Linear(d → 4d) → GELU → Dropout → Linear(4d → d).

#### 2.3.4 Full RLFA Block Pseudocode

```python
def rlfa_block(Z_k, shared_attn, shared_ffn, gate_params):
    # Step 1: Gated Multi-Head Self-Attention
    A_k = shared_attn(layer_norm(Z_k))           # MHSA
    G_k = sigmoid(W_g @ Z_k + U_g @ A_k + b_g)  # Gate
    Z_mid = Z_k + G_k * A_k                      # Gated residual

    # Step 2: Feed-Forward with standard residual
    F_k = shared_ffn(layer_norm(Z_mid))
    Z_next = Z_mid + F_k

    return Z_next
```

### 2.4 Component 3: Dense Residual Aggregator

Instead of using only the final loop output (which discards early-loop representations), we collect outputs from ALL K loops and aggregate:

```
Z_agg = Σ_{k=0}^{K} α_k · Z^k
```

where α_k are learned scalar weights (initialized to 1/(K+1)) passed through softmax to ensure they sum to 1:

```
α = softmax([α_0, α_1, ..., α_K])     (learnable parameters)
```

**Intuition**: Different fraud patterns may be detectable at different loop depths. Simple velocity-based fraud (many small transactions) might be captured at k=1. Sophisticated identity-spoofing fraud might need k=4. The aggregator lets the model use ALL depths simultaneously.

**DenseNet parallel**: This is analogous to DenseNet's concatenation of feature maps from all layers, but applied across attention loop iterations with learned weighted summation instead of concatenation (to keep dimensionality constant).

### 2.5 Component 4: Classification Head

```
z_cls = Z_agg[0]                               # [CLS] token representation
logit = Linear(d → 1)(z_cls)                   # Binary fraud/not-fraud
p = sigmoid(logit)
```

**Loss function**: Focal Loss (Lin et al., 2017) to handle class imbalance:

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```

where:
- α_t = class weight (higher for fraud class)
- γ = focusing parameter (default 2.0)
- p_t = model's estimated probability for the true class

This is superior to weighted BCE for extreme imbalance because it down-weights easy negatives dynamically.

---

## 3. Model Variants

### 3.1 LoopTab-S (Shallow)
- K = 2 loops
- d = 64, h = 4 heads
- FFN hidden = 256
- ~150K parameters
- **Use case**: Baseline, fast training, resource-constrained

### 3.2 LoopTab-D (Deep)
- K = 4 loops
- d = 128, h = 8 heads
- FFN hidden = 512
- ~600K parameters
- **Use case**: Main model for paper results

### 3.3 LoopTab-U (Untied / Ablation)
- Same as LoopTab-D but **weights are NOT shared** across loops
- Each loop k has its own W_Q, W_K, W_V, W_O, FFN parameters
- ~2.4M parameters
- **Use case**: Ablation study to demonstrate that weight-tying is both more parameter-efficient AND comparably effective

### 3.4 LoopTab-G (Graph-Augmented) — Stretch variant
- Same as LoopTab-D but adds a graph attention pre-processing step
- Build a transaction graph where nodes = accounts, edges = transactions
- Use 1-layer GAT to get node (account) embeddings
- Concatenate account embedding to each transaction's feature token sequence
- **Use case**: Show extensibility; if time permits

---

## 4. Mathematical Formulation (Paper-Ready)

### 4.1 Formal Definition

**Definition (Residual-Loop Feature Attention)**: Given an input token matrix Z⁰ ∈ ℝ^{(F+1)×d}, the RLFA mechanism applies K iterations of gated self-attention with shared parameters:

```
For k = 0, 1, ..., K-1:
    Ā^k = LN(Z^k)
    A^k = MHSA(Ā^k; Θ_attn)
    G^k = σ(W_g · Ā^k + U_g · A^k + b_g)
    Z̃^k = Z^k + G^k ⊙ A^k
    Z^{k+1} = Z̃^k + FFN(LN(Z̃^k); Θ_ffn)
```

where Θ_attn, Θ_ffn, W_g, U_g, b_g are shared across all K iterations.

**Theorem (Gradient Flow)**: The gradient of the loss with respect to the input Z⁰ passes through at least one unbroken additive path (the identity shortcut) per loop, preventing vanishing gradients regardless of K:

```
∂Z^K / ∂Z⁰ = I + Σ_{k} (terms involving ∂G^k/∂Z⁰ and ∂A^k/∂Z⁰)
```

This follows directly from the ResNet gradient flow theorem (He et al., 2016), extended to gated residuals.

### 4.2 Computational Complexity

Per-loop complexity: O(F² · d) for self-attention + O(F · d²) for FFN

Total: O(K · (F² · d + F · d²))

Since K is small (2–4) and F is moderate (67–393 for fraud datasets), this is efficient compared to graph-based methods that scale with |V| + |E| of the transaction graph.

Memory: O(K · F · d) for storing all loop outputs (for dense aggregation)

### 4.3 Why Weight-Tying Works

Weight-tied iteration is inspired by:
1. **Universal Transformers** (Dehghani et al., 2019): Showed that sharing weights across transformer layers with adaptive halting achieves strong results
2. **Recurrent attention** (conceptually): Each loop is like an RNN step over the feature tokens, but with parallel attention instead of sequential processing
3. **Parameter efficiency**: K=4 loops with shared weights uses the SAME parameters as K=1, but gets 4× the representational depth

This connects directly to the author's prior work on iterative and loop-carried attention mechanisms (recursive attention CNNs, LoGA).

---

## 5. Datasets & Benchmarks

### 5.1 Primary Datasets

| Dataset | Samples | Features | Fraud Rate | Type | Source |
|---------|---------|----------|-----------|------|--------|
| IEEE-CIS | 590,540 | 393 (67 after selection) | 3.50% | Real e-commerce | Kaggle/Vesta |
| CCFraud (European) | 284,807 | 30 (28 PCA + Time + Amount) | 0.172% | Real card transactions | Kaggle/ULB |
| PaySim | 6,362,620 | 11 | 0.13% | Synthetic mobile money | Kaggle |
| Fraud eCommerce | 150,000 | 6 | 10.60% | Synthetic e-commerce | Kaggle |

### 5.2 Dataset-Specific Preprocessing

#### IEEE-CIS
- Merge transaction + identity tables on TransactionID
- Feature selection: Use 67-feature subset from FDB benchmark (Amazon)
- Handle missing values: Median imputation for numerical, "MISSING" category for categorical
- Time-based train/val/test split (no random split — preserves temporal ordering)
- Feature groups: C (counting), D (time delta), M (match), V (Vesta-engineered)

#### CCFraud
- 28 PCA-anonymized features (V1–V28) + Time + Amount
- Normalize Amount with StandardScaler
- Stratified 70/15/15 train/val/test split
- No feature engineering needed (already PCA-transformed)

#### PaySim
- Features: type, amount, nameOrig, oldbalanceOrg, newbalanceOrg, nameDest, oldbalanceDest, newbalanceDest, isFlaggedFraud
- **CAUTION**: Balance columns can leak — include but note in paper
- Subsample to 1M transactions for tractable training (stratified)
- Encode transaction type as categorical

#### Fraud eCommerce
- Features: user_id, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address, class
- Engineer: time_delta = purchase_time - signup_time
- Encode source, browser, sex as categorical

### 5.3 Evaluation Metrics

Primary:
- **AUC-ROC** (main comparison metric — standard in fraud literature)
- **AUC-PR** (better for extreme imbalance)
- **F1-score** (at optimal threshold)

Secondary:
- **Precision@k** (top-k suspected fraud)
- **Recall@FPR=1%** (how many frauds caught at 1% false positive rate — operational metric)

---

## 6. Baselines for Comparison

### 6.1 Traditional ML
1. **XGBoost** — Still the tabular SOTA in many benchmarks; strong baseline
2. **LightGBM** — Faster gradient boosting variant
3. **Random Forest** — Classic ensemble

### 6.2 Deep Learning (Tabular)
4. **FT-Transformer** (Gorishniy et al., 2021) — Feature Tokenizer + Transformer
5. **TabTransformer** (Huang et al., 2020) — Attention only on categoricals
6. **TabNet** (Arik & Pfister, 2021) — Attentive feature selection
7. **SAINT** (Somepalli et al., 2021) — Row + column attention with contrastive pre-training

### 6.3 Fraud-Specific
8. **ResNet-LSTM-Attention** (Umaru et al., 2025) — Hybrid with attention
9. **TabConFraud** (2025) — Contrastive learning for fraud

### 6.4 Ablation Models
10. **LoopTab-NoGate** — Remove gating; standard residual Z^{k+1} = Z^k + A^k
11. **LoopTab-NoLoop** — K=1 (reduces to single-pass FT-Transformer-like)
12. **LoopTab-U (Untied)** — No weight sharing across loops
13. **LoopTab-NoDense** — Use only final loop output Z^K, no aggregation
14. **LoopTab-NoFocal** — Replace Focal Loss with weighted BCE

---

## 7. Training Pipeline & Implementation

### 7.1 Framework & Libraries

```
Python 3.10+
PyTorch 2.x (with CUDA)
Libraries:
  - torch, torch.nn
  - pandas, numpy, scikit-learn
  - imbalanced-learn (for SMOTE baseline comparison)
  - optuna (hyperparameter tuning)
  - wandb or tensorboard (experiment tracking)
  - kaggle API (dataset download)
```

### 7.2 Project Structure

```
looptab/
├── configs/
│   ├── ieee_cis.yaml
│   ├── ccfraud.yaml
│   ├── paysim.yaml
│   └── fraud_ecom.yaml
├── data/
│   ├── download.py          # Kaggle API download scripts
│   ├── preprocess.py        # Dataset-specific preprocessing
│   └── dataloader.py        # PyTorch DataLoader with stratified sampling
├── models/
│   ├── tokenizer.py         # Feature Tokenizer (numerical + categorical)
│   ├── rlfa_block.py        # Residual-Loop Feature Attention block
│   ├── aggregator.py        # Dense Residual Aggregator
│   ├── looptab.py           # Full LoopTab model
│   └── baselines/
│       ├── ft_transformer.py
│       ├── tab_transformer.py
│       └── tabnet.py
├── training/
│   ├── trainer.py           # Training loop with focal loss
│   ├── focal_loss.py        # Focal Loss implementation
│   ├── scheduler.py         # Cosine annealing with warmup
│   └── early_stopping.py    # Patience-based early stopping
├── evaluation/
│   ├── metrics.py           # AUC-ROC, AUC-PR, F1, Precision@k
│   ├── ablation.py          # Run all ablation variants
│   └── visualization.py     # Attention heatmaps, gate activation plots
├── experiments/
│   ├── run_all.py           # Run full experiment suite
│   └── results/             # Saved results, checkpoints
└── paper/
    ├── figures/             # Generated figures for paper
    └── tables/              # Generated LaTeX tables
```

### 7.3 Core Model Implementation (Pseudocode → PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTokenizer(nn.Module):
    """Convert raw tabular features to d-dim tokens."""

    def __init__(self, num_features_info, cat_features_info, d_model):
        super().__init__()
        # num_features_info: list of numerical feature names
        # cat_features_info: dict {feature_name: cardinality}

        self.num_projections = nn.ModuleDict({
            name: nn.Linear(1, d_model)
            for name in num_features_info
        })

        self.cat_embeddings = nn.ModuleDict({
            name: nn.Embedding(card + 1, d_model)  # +1 for missing
            for name, card in cat_features_info.items()
        })

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x_num, x_cat):
        # x_num: (B, N_num)  x_cat: (B, N_cat) as integer indices
        tokens = []

        for i, name in enumerate(self.num_projections):
            t = self.num_projections[name](x_num[:, i:i+1])  # (B, d)
            tokens.append(t.unsqueeze(1))

        for i, name in enumerate(self.cat_embeddings):
            t = self.cat_embeddings[name](x_cat[:, i])  # (B, d)
            tokens.append(t.unsqueeze(1))

        tokens = torch.cat(tokens, dim=1)  # (B, F, d)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        return torch.cat([cls, tokens], dim=1)  # (B, F+1, d)


class ResidualGate(nn.Module):
    """Learned gate for residual attention update."""

    def __init__(self, d_model):
        super().__init__()
        self.W_g = nn.Linear(d_model, d_model)
        self.U_g = nn.Linear(d_model, d_model)

    def forward(self, z, a):
        # z: current state, a: attention output
        gate = torch.sigmoid(self.W_g(z) + self.U_g(a))
        return gate


class RLFABlock(nn.Module):
    """Residual-Loop Feature Attention Block (weight-tied across loops)."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mhsa = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.gate = ResidualGate(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, z):
        # Gated Multi-Head Self-Attention
        z_norm = self.ln1(z)
        a, attn_weights = self.mhsa(z_norm, z_norm, z_norm)
        g = self.gate(z_norm, a)          # (B, F+1, d)
        z = z + g * a                     # Gated residual

        # Feed-Forward with standard residual
        z = z + self.ffn(self.ln2(z))

        return z, attn_weights


class DenseResidualAggregator(nn.Module):
    """Aggregate outputs from all K loops with learned weights."""

    def __init__(self, n_loops):
        super().__init__()
        # K+1 weights (including initial Z⁰)
        self.alpha = nn.Parameter(torch.ones(n_loops + 1))

    def forward(self, loop_outputs):
        # loop_outputs: list of (B, F+1, d) tensors, length K+1
        weights = F.softmax(self.alpha, dim=0)
        aggregated = sum(w * z for w, z in zip(weights, loop_outputs))
        return aggregated


class LoopTab(nn.Module):
    """
    LoopTab: Residual-Loop Feature Attention for Tabular Fraud Detection.

    Args:
        num_features_info: list of numerical feature names
        cat_features_info: dict {feature_name: cardinality}
        d_model: token embedding dimension
        n_heads: number of attention heads
        n_loops: number of RLFA iterations (K)
        d_ff: FFN hidden dimension
        dropout: dropout rate
        tied_weights: whether to share RLFA weights across loops
    """

    def __init__(
        self,
        num_features_info,
        cat_features_info,
        d_model=128,
        n_heads=8,
        n_loops=4,
        d_ff=512,
        dropout=0.1,
        tied_weights=True,
    ):
        super().__init__()
        self.n_loops = n_loops
        self.tied_weights = tied_weights

        # Feature Tokenizer
        self.tokenizer = FeatureTokenizer(
            num_features_info, cat_features_info, d_model
        )

        # RLFA Block(s)
        if tied_weights:
            self.rlfa = RLFABlock(d_model, n_heads, d_ff, dropout)
        else:
            self.rlfa_blocks = nn.ModuleList([
                RLFABlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_loops)
            ])

        # Dense Residual Aggregator
        self.aggregator = DenseResidualAggregator(n_loops)

        # Classification Head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_num, x_cat):
        # Tokenize
        z = self.tokenizer(x_num, x_cat)  # (B, F+1, d)

        # Collect all loop outputs for dense aggregation
        loop_outputs = [z]
        attn_maps = []

        for k in range(self.n_loops):
            block = self.rlfa if self.tied_weights else self.rlfa_blocks[k]
            z, attn_w = block(z)
            loop_outputs.append(z)
            attn_maps.append(attn_w)

        # Dense Residual Aggregation
        z_agg = self.aggregator(loop_outputs)  # (B, F+1, d)

        # Classification from [CLS] token
        cls_repr = z_agg[:, 0, :]  # (B, d)
        logit = self.head(cls_repr).squeeze(-1)  # (B,)

        return logit, attn_maps


class FocalLoss(nn.Module):
    """Focal Loss for extreme class imbalance."""

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(),
                                                  reduction='none')
        p_t = torch.exp(-bce)  # p_t = p if y=1, 1-p if y=0
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()
```

### 7.4 Training Configuration

```yaml
# configs/ieee_cis.yaml (main experiment)
model:
  d_model: 128
  n_heads: 8
  n_loops: 4
  d_ff: 512
  dropout: 0.15
  tied_weights: true

training:
  batch_size: 512
  epochs: 100
  early_stopping_patience: 10
  optimizer: AdamW
  learning_rate: 1e-3
  weight_decay: 1e-5
  scheduler: cosine_warmup
  warmup_epochs: 5

loss:
  type: focal
  alpha: 0.75    # Weight for positive (fraud) class
  gamma: 2.0     # Focusing parameter

data:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  split_strategy: temporal  # Time-based split for IEEE-CIS
  num_workers: 4
```

### 7.5 Desktop GPU Training Feasibility

**Estimated memory footprint for LoopTab-D (K=4, d=128):**

- Model parameters: ~600K × 4 bytes = ~2.4 MB
- IEEE-CIS (67 features, batch=512): 
  - Activations per loop: 512 × 68 × 128 × 4 bytes = ~17 MB
  - Total for K=4 loops + aggregation: ~100 MB
  - Attention maps: 512 × 8 × 68 × 68 × 4 bytes × 4 loops = ~310 MB
- **Total GPU memory: ~500 MB–1 GB** (well within any modern GPU)
- **Training time estimate**: ~5–15 min/epoch on IEEE-CIS with RTX 3060/4060

Even LoopTab-U (untied, ~2.4M params) fits comfortably on a 6GB+ GPU.

---

## 8. Experiment Design

### 8.1 Main Experiments (Table for Paper)

**Table: Performance comparison on fraud detection benchmarks (AUC-ROC / AUC-PR)**

| Model | IEEE-CIS | CCFraud | PaySim | Fraud eCom |
|-------|----------|---------|--------|------------|
| XGBoost | — | — | — | — |
| LightGBM | — | — | — | — |
| FT-Transformer | — | — | — | — |
| TabTransformer | — | — | — | — |
| TabNet | — | — | — | — |
| SAINT | — | — | — | — |
| **LoopTab-S (K=2)** | — | — | — | — |
| **LoopTab-D (K=4)** | — | — | — | — |

### 8.2 Ablation Study

**Table: Ablation on IEEE-CIS (AUC-ROC / AUC-PR / F1)**

| Variant | AUC-ROC | AUC-PR | F1 | Description |
|---------|---------|--------|----|-------------|
| LoopTab-D (full) | — | — | — | Full model |
| − Gating | — | — | — | Standard residual |
| − Loop (K=1) | — | — | — | Single-pass |
| − Weight tying | — | — | — | Independent layers |
| − Dense aggregation | — | — | — | Only final output |
| − Focal Loss | — | — | — | Weighted BCE |

### 8.3 Analysis Experiments (Figures for Paper)

1. **Gate activation analysis**: Visualize mean gate values G^k across loops for fraud vs. legitimate samples. Hypothesis: fraud samples show higher gate activation in later loops (needs higher-order features).

2. **Loop depth sensitivity**: Plot AUC-ROC vs. K (1, 2, 3, 4, 5, 6) to find the sweet spot. Expect diminishing returns after K=4.

3. **Attention heatmaps**: Show which features attend to which, and how attention patterns evolve across loops (k=1 vs k=4). Expect early loops show broad attention, later loops show focused fraud-signal patterns.

4. **Dense aggregation weight analysis**: Plot learned α weights. Hypothesis: different datasets peak at different loop depths.

5. **Training efficiency**: Parameter count vs. AUC-ROC curve comparing LoopTab-D (tied, 600K params) vs. LoopTab-U (untied, 2.4M params) vs. stacked FT-Transformer (4 layers, ~2.4M params). Show that weight-tying achieves comparable performance with 4× fewer parameters.

6. **Class-imbalance robustness**: Plot AUC-PR as a function of artificially varied fraud rate (0.1%, 0.5%, 1%, 3.5%, 5%, 10%) on IEEE-CIS. Show LoopTab degrades less than baselines.

---

## 9. Paper Outline

### Title
**LoopTab: Residual-Loop Feature Attention for Tabular Financial Fraud Detection**

### Abstract (~250 words)
- Problem: Tabular fraud detection with extreme imbalance
- Gap: Existing transformers use single-pass attention; no iterative refinement
- Contribution: LoopTab — weight-tied iterative attention with residual gating and dense aggregation
- Results: Outperforms FT-Transformer, TabNet, XGBoost on 4 benchmarks
- Significance: Parameter-efficient, explainable gate activations, applicable to FinTech SDG goals

### 1. Introduction (1.5 pages)
- Financial fraud statistics and impact
- Why tabular data is hard for deep learning
- Limitations of current approaches
- Our contribution (3 bullet points)
- Paper organization

### 2. Related Work (2 pages)
- 2.1 Traditional ML for fraud detection (XGBoost, RF)
- 2.2 Deep learning on tabular data (FT-Transformer, TabTransformer, TabNet, SAINT)
- 2.3 Attention mechanisms for fraud (GAT, temporal attention, cross-attention)
- 2.4 Residual learning and iterative computation (ResNet, Universal Transformer, recurrent attention)
- 2.5 Class imbalance strategies (SMOTE, focal loss, contrastive learning)

### 3. Methodology (3 pages)
- 3.1 Problem Formulation
- 3.2 Feature Tokenizer
- 3.3 Residual-Loop Feature Attention (RLFA) Block
- 3.4 Dense Residual Aggregation
- 3.5 Classification with Focal Loss
- 3.6 Model Variants (LoopTab-S, LoopTab-D)
- 3.7 Complexity Analysis

### 4. Experiments (3 pages)
- 4.1 Datasets and Preprocessing
- 4.2 Baselines and Evaluation Metrics
- 4.3 Implementation Details
- 4.4 Main Results (comparison table)
- 4.5 Ablation Study
- 4.6 Analysis: Gate activations, attention evolution, aggregation weights

### 5. Discussion (1 page)
- Why iterative attention helps fraud detection
- Practical deployment considerations
- Connection to UN SDGs (financial inclusion, fraud prevention → SDG 8, 9, 16)
- Limitations

### 6. Conclusion and Future Work (0.5 page)
- Summary
- Future: LoopTab-G (graph variant), online/streaming adaptation, multi-task learning

### References (~40–50 references)

---

## 10. Key Novelty Claims (for Reviewer Arguments)

1. **First weight-tied iterative attention mechanism for tabular fraud detection**: Unlike stacking transformer layers (which multiplies parameters), LoopTab reuses the same attention block K times, achieving depth without parameter explosion.

2. **Residual gating per-feature, per-dimension, per-loop**: The gate is not a scalar (like GRU gates) — it operates at full (F×d) resolution, allowing fine-grained control of information flow per feature per loop iteration.

3. **Dense residual aggregation across loop iterations**: Inspired by DenseNet but applied to attention loop outputs, enabling multi-resolution fraud pattern detection.

4. **Built-in explainability**: Gate activation maps and attention evolution across loops provide interpretable fraud explanations without post-hoc methods (SHAP, LIME).

5. **Connects ResNet, Universal Transformer, and GRU principles** in a single unified mechanism for tabular data — a novel combination not explored in the fraud detection or tabular learning literature.

---

## 11. Timeline & Milestones

| Week | Task |
|------|------|
| 1 | Dataset download, preprocessing pipeline, feature tokenizer |
| 2 | RLFA block + LoopTab model implementation |
| 3 | Training loop, focal loss, baselines (XGBoost, LightGBM) |
| 4 | Deep learning baselines (FT-Transformer, TabNet, SAINT) |
| 5 | Main experiments on all 4 datasets |
| 6 | Ablation study + analysis experiments |
| 7 | Figures, tables, attention visualizations |
| 8 | Paper writing (Introduction, Related Work, Methodology) |
| 9 | Paper writing (Experiments, Discussion, Conclusion) |
| 10 | Revision, proofreading, submission |

---

## 12. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| LoopTab doesn't beat XGBoost on AUC-ROC | Focus on AUC-PR (where DL often wins on imbalanced data); emphasize interpretability and parameter efficiency |
| Weight-tying hurts performance | Include LoopTab-U variant; even if untied is better, the tied variant's parameter efficiency is a contribution |
| IEEE-CIS has too many features (393) | Use FDB's 67-feature subset; run secondary experiment with full 393 |
| PaySim balance leakage | Explicitly note in paper; run with and without balance columns |
| Focal loss hyperparameter sensitivity | Sweep α ∈ {0.25, 0.5, 0.75} and γ ∈ {0.5, 1.0, 2.0, 3.0} via Optuna |

---

## Appendix A: Connection to Prior Work (Author's Research Line)

This work extends the author's research on iterative and loop-carried attention mechanisms:

- **Recursive Additive Attention** (IEEE 2024): Introduced recursive attention for CNNs in plant disease classification
- **LoGA (Loop-Carried Graph Attention)**: Applied loop-carried computation to graph attention for vision
- **IMHA (Entropy-Guided Iterative Multi-Head Attention)**: Multi-head attention with entropy-based iteration control

LoopTab advances this line by:
1. Moving from vision/graph domains to **tabular financial data**
2. Replacing recursive/entropy-guided iteration control with **learned residual gating**
3. Adding **dense residual aggregation** across loop iterations (not present in prior mechanisms)
4. Integrating **class-imbalance-aware loss** directly with the iterative architecture

This represents the first application of the loop-carried attention paradigm to FinTech.
