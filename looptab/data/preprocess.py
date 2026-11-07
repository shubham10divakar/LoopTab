"""Dataset-specific loading, splitting and encoding (design doc Section 5.2).

Pipeline per dataset:
    raw files -> load_<name>() -> RawDataset (DataFrame + column roles)
              -> split (temporal or stratified)
              -> TabularPreprocessor fitted on TRAIN only
              -> TabularData (encoded numpy arrays + schema)

All statistics (medians, scaler moments, category vocabularies, frequency
encodings, V-column selection for IEEE-CIS) are computed on the training split
only, so nothing leaks from validation/test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

from .dataloader import TabularData, TabularSplit


# ----------------------------------------------------------------------------
# Generic pieces
# ----------------------------------------------------------------------------

@dataclass
class RawDataset:
    df: pd.DataFrame
    num_cols: list[str]
    cat_cols: list[str]
    target: str
    time_col: str | None = None          # used for temporal splits
    freq_cols: list[str] = field(default_factory=list)  # frequency-encoded -> numerical
    name: str = "dataset"
    meta: dict = field(default_factory=dict)


class TabularPreprocessor:
    """Fit-on-train encoder for numerical and categorical columns.

    Numerical: median imputation, then ``standard`` (z-score), ``quantile``
    (Gaussian quantile transform) or ``none`` scaling; clipped to +-clip.
    Categorical: values mapped to codes 1..C by train frequency; missing,
    unseen and rare (< min_frequency) values map to 0.
    Frequency columns: replaced by log1p(train count of the value) and treated
    as numerical.
    """

    def __init__(self, num_cols, cat_cols, freq_cols=(), scaler: str = "standard",
                 min_frequency: int = 1, max_cardinality: int | None = None,
                 clip: float | None = 10.0):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.freq_cols = list(freq_cols)
        self.scaler = scaler
        self.min_frequency = min_frequency
        self.max_cardinality = max_cardinality
        self.clip = clip

    @property
    def out_num_names(self) -> list[str]:
        return self.num_cols + [f"{c}_freq" for c in self.freq_cols]

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        self.freq_maps = {c: df[c].value_counts(dropna=True) for c in self.freq_cols}
        num = self._raw_num(df)
        self.medians = np.nanmedian(num, axis=0) if num.shape[1] else np.zeros(0)
        self.medians = np.where(np.isnan(self.medians), 0.0, self.medians)
        num = self._impute(num)
        if self.scaler == "standard":
            self.mean = num.mean(axis=0)
            std = num.std(axis=0)
            self.std = np.where(std < 1e-12, 1.0, std)
        elif self.scaler == "quantile":
            self.qt = QuantileTransformer(
                n_quantiles=min(1000, max(10, len(num))), output_distribution="normal",
                subsample=200_000, random_state=0,
            ).fit(num) if num.shape[1] else None
        elif self.scaler != "none":
            raise ValueError(f"unknown scaler {self.scaler!r}")

        self.vocabs: dict[str, dict] = {}
        for c in self.cat_cols:
            counts = df[c].dropna().astype(str).value_counts()
            counts = counts[counts >= self.min_frequency]
            if self.max_cardinality is not None:
                counts = counts.iloc[: self.max_cardinality]
            self.vocabs[c] = {v: i + 1 for i, v in enumerate(counts.index)}
        return self

    @property
    def cat_cardinalities(self) -> list[int]:
        return [len(self.vocabs[c]) for c in self.cat_cols]

    def _raw_num(self, df: pd.DataFrame) -> np.ndarray:
        cols = [pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64) for c in self.num_cols]
        for c in self.freq_cols:
            cnt = df[c].map(self.freq_maps[c]).fillna(0).to_numpy(dtype=np.float64)
            cols.append(np.log1p(cnt))
        if not cols:
            return np.zeros((len(df), 0))
        return np.stack(cols, axis=1)

    def _impute(self, num: np.ndarray) -> np.ndarray:
        if num.shape[1] == 0:
            return num
        mask = ~np.isfinite(num)
        if mask.any():
            num = num.copy()
            num[mask] = np.take(self.medians, np.where(mask)[1])
        return num

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        num = self._impute(self._raw_num(df))
        if num.shape[1]:
            if self.scaler == "standard":
                num = (num - self.mean) / self.std
            elif self.scaler == "quantile":
                num = self.qt.transform(num)
            if self.clip is not None:
                num = np.clip(num, -self.clip, self.clip)
        cats = [
            df[c].astype(str).where(df[c].notna(), None).map(self.vocabs[c]).fillna(0)
            .to_numpy(dtype=np.int64)
            for c in self.cat_cols
        ]
        x_cat = np.stack(cats, axis=1) if cats else np.zeros((len(df), 0), dtype=np.int64)
        return num.astype(np.float32), x_cat


def split_frame(
    df: pd.DataFrame,
    target: str,
    strategy: str = "stratified",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    time_col: str | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test. ``temporal`` sorts by ``time_col`` first."""
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1")
    if strategy == "temporal":
        if time_col is None:
            raise ValueError("temporal split needs a time column")
        df = df.sort_values(time_col, kind="stable")
        n = len(df)
        a, b = int(n * train_ratio), int(n * (train_ratio + val_ratio))
        return df.iloc[:a], df.iloc[a:b], df.iloc[b:]
    if strategy == "stratified":
        y = df[target]
        train, rest = train_test_split(df, train_size=train_ratio, stratify=y, random_state=seed)
        val, test = train_test_split(
            rest, train_size=val_ratio / (val_ratio + test_ratio),
            stratify=rest[target], random_state=seed,
        )
        return train, val, test
    if strategy == "random":
        train, rest = train_test_split(df, train_size=train_ratio, random_state=seed)
        val, test = train_test_split(rest, train_size=val_ratio / (val_ratio + test_ratio),
                                     random_state=seed)
        return train, val, test
    raise ValueError(f"unknown split strategy {strategy!r}")


def stratified_subsample(df: pd.DataFrame, target: str, n: int, seed: int = 42) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df
    sub, _ = train_test_split(df, train_size=n, stratify=df[target], random_state=seed)
    return sub


# ----------------------------------------------------------------------------
# Dataset loaders
# ----------------------------------------------------------------------------

def load_ccfraud(root: str | Path, **_) -> RawDataset:
    """ULB credit-card fraud: Time, V1..V28, Amount, Class."""
    df = pd.read_csv(Path(root) / "creditcard.csv")
    num = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    return RawDataset(df, num, [], "Class", time_col="Time", name="ccfraud")


PAYSIM_BALANCE_COLS = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]


def load_paysim(root: str | Path, subsample: int | None = 1_000_000,
                drop_balance: bool = False, seed: int = 42, **_) -> RawDataset:
    """PaySim mobile money.

    nameOrig / nameDest are near-unique account ids, so we keep only their
    type prefix (C = customer, M = merchant) as a categorical feature. Balance
    columns can leak the label (see design doc); ``drop_balance`` removes them.
    """
    root = Path(root)
    files = sorted(root.glob("PS_*.csv")) or sorted(root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no PaySim csv in {root}")
    df = pd.read_csv(files[0])
    df = stratified_subsample(df, "isFraud", subsample, seed)
    df["orig_type"] = df["nameOrig"].astype(str).str[0]
    df["dest_type"] = df["nameDest"].astype(str).str[0]
    df["hour_of_day"] = df["step"] % 24
    num = ["step", "hour_of_day", "amount"] + ([] if drop_balance else PAYSIM_BALANCE_COLS)
    cat = ["type", "orig_type", "dest_type", "isFlaggedFraud"]
    return RawDataset(df, num, cat, "isFraud", time_col="step", name="paysim",
                      meta={"drop_balance": drop_balance})


def load_fraud_ecom(root: str | Path, **_) -> RawDataset:
    """Fraud e-commerce (Fraud_Data.csv).

    Engineered: time_delta = purchase_time - signup_time (seconds), purchase
    hour / day-of-week. device_id and ip_address are frequency-encoded (how
    many training rows share them), a standard fraud signal.
    """
    root = Path(root)
    path = root / "Fraud_Data.csv"
    if not path.exists():
        cands = sorted(root.glob("*raud*.csv"))
        if not cands:
            raise FileNotFoundError(f"no Fraud_Data.csv in {root}")
        path = cands[0]
    df = pd.read_csv(path)
    signup = pd.to_datetime(df["signup_time"])
    purchase = pd.to_datetime(df["purchase_time"])
    df["time_delta"] = (purchase - signup).dt.total_seconds()
    df["purchase_hour"] = purchase.dt.hour
    df["purchase_dow"] = purchase.dt.dayofweek
    df["purchase_ts"] = purchase.astype("int64") // 10**9
    num = ["purchase_value", "age", "time_delta", "purchase_hour"]
    cat = ["source", "browser", "sex", "purchase_dow"]
    return RawDataset(df, num, cat, "class", time_col="purchase_ts",
                      freq_cols=["device_id", "ip_address"], name="fraud_ecom")


IEEE_CORE_NUM = (
    ["TransactionAmt", "dist1", "dist2", "card1", "card2", "card3", "card5", "addr1", "addr2"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in (1, 2, 3, 4, 10, 11, 15)]
)
IEEE_CORE_CAT = (
    ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"]
    + [f"M{i}" for i in range(1, 10)]
)


def load_ieee_cis(root: str | Path, n_features: int | None = 67,
                  features: dict | None = None, **_) -> RawDataset:
    """IEEE-CIS: merge transaction + identity on TransactionID.

    The FDB benchmark's exact 67-column list is not bundled here. By default we
    use a fixed core of 47 informative columns (amount, card/addr, C, D, M,
    e-mail domains, device) and fill up to ``n_features`` with the V columns
    of lowest missing rate on the training split (selected later in
    ``prepare_dataset``). Pass ``features={"num": [...], "cat": [...]}`` to use
    an explicit list instead (e.g. the FDB list), or ``n_features=None`` for
    all 393+ columns.
    """
    root = Path(root)
    tx = pd.read_csv(root / "train_transaction.csv")
    idp = root / "train_identity.csv"
    df = tx.merge(pd.read_csv(idp), on="TransactionID", how="left") if idp.exists() else tx
    df["hour_of_day"] = (df["TransactionDT"] // 3600) % 24

    if features is not None:
        num, cat = list(features.get("num", [])), list(features.get("cat", []))
        meta = {}
    elif n_features is None:
        excl = {"TransactionID", "isFraud", "TransactionDT"}
        cat = [c for c in df.columns if c not in excl and
               (df[c].dtype == object or c in IEEE_CORE_CAT or c.startswith("id_"))]
        num = [c for c in df.columns if c not in excl and c not in cat]
        meta = {}
    else:
        num = [c for c in IEEE_CORE_NUM + ["hour_of_day"] if c in df.columns]
        cat = [c for c in IEEE_CORE_CAT if c in df.columns]
        v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
        meta = {"fill_v_cols": v_cols, "n_features": n_features}
    return RawDataset(df, num, cat, "isFraud", time_col="TransactionDT",
                      name="ieee_cis", meta=meta)


def make_synthetic(n: int = 20_000, n_num: int = 8, cat_cards=(5, 12, 3),
                   fraud_rate: float = 0.03, seed: int = 0, **_) -> RawDataset:
    """Synthetic imbalanced data with higher-order interactions.

    Fraud depends on a 3-way interaction (x0 * x1 > 0 AND cat0 == 1) plus a
    noisy linear signal, so a model must combine features to do well.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_num))
    cats = np.stack([rng.integers(0, c, size=n) for c in cat_cards], axis=1)
    inter = (X[:, 0] * X[:, 1] > 0.5) & (cats[:, 0] == 1)
    score = 2.5 * inter + 0.8 * X[:, 2] - 0.5 * X[:, 3] * (cats[:, 1] % 2) + rng.normal(0, 1, n)
    thr = np.quantile(score, 1 - fraud_rate)
    y = (score >= thr).astype(int)
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(n_num)])
    for j in range(len(cat_cards)):
        df[f"cat_{j}"] = cats[:, j]
    df.loc[rng.random(n) < 0.02, "num_4"] = np.nan  # exercise imputation
    df["t"] = np.arange(n)
    df["label"] = y
    return RawDataset(df, [f"num_{i}" for i in range(n_num)],
                      [f"cat_{j}" for j in range(len(cat_cards))], "label",
                      time_col="t", name="synthetic")


LOADERS: dict[str, Callable[..., RawDataset]] = {
    "ccfraud": load_ccfraud,
    "paysim": load_paysim,
    "fraud_ecom": load_fraud_ecom,
    "ieee_cis": load_ieee_cis,
    "synthetic": make_synthetic,
}

DEFAULT_SPLIT = {
    "ccfraud": "stratified",
    "paysim": "stratified",
    "fraud_ecom": "stratified",
    "ieee_cis": "temporal",
    "synthetic": "stratified",
}


def _select_v_columns(train: pd.DataFrame, raw: RawDataset) -> None:
    """Fill IEEE-CIS feature budget with least-missing V columns (train only)."""
    v_cols = raw.meta.get("fill_v_cols")
    if not v_cols:
        return
    budget = raw.meta["n_features"] - len(raw.num_cols) - len(raw.cat_cols)
    if budget <= 0:
        return
    miss = train[v_cols].isna().mean()
    var = train[v_cols].var()
    ranked = miss[var > 0].sort_values(kind="stable").index.tolist()
    raw.num_cols = raw.num_cols + ranked[:budget]


def prepare_dataset(
    name: str,
    root: str | Path | None = None,
    split_strategy: str | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    scaler: str = "standard",
    min_frequency: int = 1,
    max_cardinality: int | None = None,
    seed: int = 42,
    loader_kwargs: dict | None = None,
    raw: RawDataset | None = None,
) -> TabularData:
    """Load, split and encode a dataset into a ``TabularData``."""
    if raw is None:
        if name not in LOADERS:
            raise KeyError(f"unknown dataset {name!r}; choose from {sorted(LOADERS)}")
        kwargs = dict(loader_kwargs or {})
        if name != "synthetic":
            if root is None:
                raise ValueError(f"dataset {name!r} needs a root directory")
            kwargs["root"] = root
        raw = LOADERS[name](**kwargs)
    strategy = split_strategy or DEFAULT_SPLIT.get(name, "stratified")
    df = raw.df.dropna(subset=[raw.target])
    tr, va, te = split_frame(df, raw.target, strategy, train_ratio, val_ratio, raw.time_col, seed)
    _select_v_columns(tr, raw)

    pre = TabularPreprocessor(raw.num_cols, raw.cat_cols, raw.freq_cols, scaler=scaler,
                              min_frequency=min_frequency, max_cardinality=max_cardinality).fit(tr)

    def encode(part: pd.DataFrame) -> TabularSplit:
        x_num, x_cat = pre.transform(part)
        return TabularSplit(x_num, x_cat, part[raw.target].to_numpy(dtype=np.float32))

    return TabularData(
        train=encode(tr), val=encode(va), test=encode(te),
        num_names=pre.out_num_names, cat_names=list(raw.cat_cols),
        cat_cardinalities=pre.cat_cardinalities, name=raw.name,
        meta={**raw.meta, "split_strategy": strategy, "scaler": scaler,
              "fill_v_cols": None},
    )
