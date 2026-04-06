#!/usr/bin/env python3
"""
R62 — Alternative Model as Feature (Stacking / Level-1 Meta)

Adds 1-2 meta-features computed from simpler models as new features
to the champion 31f LGB+XGB ensemble:

  p_lin — Logistic Regression on 8 simple features
           OOF predictions (5-fold temporal CV within each training window)
           Added as feature 32.

  p_seq — GRU micro-model (hidden=16) on 8-bar × 5-feat sequences
           OOF for train, full-fit for test. Added as feature 33.

Combos tested:
  1. baseline_31f      — no meta features (baseline)
  2. +p_lin            — 32 features
  3. +p_seq            — 32 features
  4. +p_lin+p_seq      — 33 features

Uses ORIGINAL_WINDOWS (with gaps) for comparability with Sharpe 1.66.
N_ROUNDS=600, EARLY_STOP=40, 5 seeds.

Dependencies:
  - torch (2.10.0+cu128 available on MLC, CPU inference)
  - sklearn LogisticRegression
"""

import sys
import warnings
from typing import Dict, List, Set, Tuple
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

from _research_round7 import SYM_35
from _research_r22_models import SEEDS, LEVERAGE, CAPITAL, log, cs_rank_cols
from _research_r30b_fixed import compute_regime_extended
from _research_r35_new_features import (
    add_r35_features, load_research_frame, MARKET_LEVEL_FEATURES,
)
from _research_r47_coinglass import (
    CANONICAL_EXEC_CFG, CHAMPION_FEAT_30,
    add_cg_features, compute_cg_features, load_cg_daily,
)

CHAMPION_FEAT_31 = CHAMPION_FEAT_30 + ["cg_taker_imb"]

TIER1_SYMS = {"BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"}
TIER3_SYMS = {
    "SAND/USDT", "LDO/USDT", "INJ/USDT", "APT/USDT", "ARB/USDT",
    "GALA/USDT", "FTM/USDT", "MATIC/USDT",
}
TIER2_SYMS = set(SYM_35) - TIER1_SYMS - TIER3_SYMS


def _cost_for_sym(sym: str) -> float:
    if sym in TIER1_SYMS:
        return 0.92 * (-0.0001) + 0.08 * 0.0007
    elif sym in TIER2_SYMS:
        return 0.75 * 0.0001 + 0.25 * 0.0007
    else:
        return 0.0005 + 0.0002


ORIGINAL_WINDOWS = [
    {"name": "W1",
     "train_end": "2024-06-01",
     "val_start": "2024-06-01", "val_end": "2024-09-30",
     "test_start": "2024-10-15", "test_end": "2025-01-31"},
    {"name": "W2",
     "train_end": "2025-01-01",
     "val_start": "2025-01-01", "val_end": "2025-04-30",
     "test_start": "2025-05-15", "test_end": "2025-08-31"},
    {"name": "W3",
     "train_end": "2025-07-01",
     "val_start": "2025-07-01", "val_end": "2025-10-31",
     "test_start": "2025-11-15", "test_end": "2026-03-17"},
]

PROD_CFG = {
    "n_long": 6, "n_short": 3, "rebal_hours": 12,
    "trend_cutoff": 0.9, "dyn_threshold": 0.7,
    "ema_alpha": 0.5, "hysteresis": 3,
}

LGB_PARAMS = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.03, "num_leaves": 63,
    "min_child_samples": 100, "subsample": 0.8,
    "colsample_bytree": 0.8, "lambda_l2": 1.0,
    "verbose": -1, "n_jobs": -1,
}
XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "learning_rate": 0.03, "max_depth": 6,
    "min_child_weight": 100, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_lambda": 1.0,
    "n_jobs": -1, "verbosity": 0,
}
N_ROUNDS = 600
EARLY_STOP = 40

# ── Level-0 model feature sets ─────────────────────────────────
LIN_FEATS = [
    "ret_12h", "ret_24h", "mom_z_24h", "oi_chg_12h",
    "taker_cvd_12h", "atr_14", "pct_coins_up_12h", "cg_taker_imb",
]
GRU_FEATS = ["ret_12h", "rvol_12h", "cg_taker_imb", "oi_chg_12h", "pct_coins_up_12h"]
GRU_SEQ_LEN = 8


# ══════════════════════════════════════════════════════════
#  GRU MICRO-MODEL
# ══════════════════════════════════════════════════════════

class GRUMicro(nn.Module):
    def __init__(self, n_feats: int = 5, hidden: int = 16):
        super().__init__()
        self.gru = nn.GRU(input_size=n_feats, hidden_size=hidden,
                          num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, seq_len, n_feats)
        _, h = self.gru(x)   # h: (1, B, hidden)
        return torch.sigmoid(self.head(h.squeeze(0)))  # (B, 1)


def build_gru_sequences(df: pd.DataFrame, feats: List[str], seq_len: int = 8
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (N, seq_len, n_feats) tensor from per-symbol time series.
    Returns (X_seq, idx_array) where idx_array is the row index of the
    LAST bar in each sequence (= prediction target row).
    Only returns sequences where all seq_len bars are non-NaN.
    """
    all_x = []
    all_idx = []
    for sym, grp in df.groupby("symbol"):
        grp = grp.sort_values("timestamp")
        vals = grp[feats].values.astype(np.float32)
        idxs = grp.index.values
        n = len(grp)
        for i in range(seq_len - 1, n):
            seq = vals[i - seq_len + 1: i + 1]
            if np.isnan(seq).any():
                continue
            all_x.append(seq)
            all_idx.append(idxs[i])
    if not all_x:
        return np.empty((0, seq_len, len(feats)), dtype=np.float32), np.array([])
    return np.stack(all_x), np.array(all_idx)


def train_gru(X_train: np.ndarray, y_train: np.ndarray, n_epochs: int = 15,
              batch_size: int = 2048) -> GRUMicro:
    """Train GRU on sequences."""
    model = GRUMicro(n_feats=X_train.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCELoss()
    model.train()
    n = len(X_train)
    perm = np.random.permutation(n)
    X_train, y_train = X_train[perm], y_train[perm]
    for epoch in range(n_epochs):
        for start in range(0, n, batch_size):
            xb = torch.tensor(X_train[start:start + batch_size])
            yb = torch.tensor(y_train[start:start + batch_size], dtype=torch.float32).unsqueeze(1)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


def predict_gru(model: GRUMicro, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start:start + batch_size])
            preds.append(model(xb).squeeze(1).numpy())
    return np.concatenate(preds) if preds else np.array([])


# ══════════════════════════════════════════════════════════
#  STACKING: COMPUTE META-FEATURES (OOF)
# ══════════════════════════════════════════════════════════

def compute_p_lin_oof(df: pd.DataFrame, feats: List[str],
                      n_folds: int = 5) -> pd.Series:
    """
    Returns p_lin as a Series indexed by df.index.
    Uses temporal 5-fold CV: splits by time, NOT random.
    NaN for rows where model is not trained yet (first fold).
    """
    avail = [f for f in feats if f in df.columns]
    df_sorted = df.sort_values("timestamp")
    timestamps = df_sorted["timestamp"].unique()
    timestamps.sort()
    fold_size = len(timestamps) // n_folds
    p_lin = pd.Series(np.nan, index=df.index, dtype=float)

    for fold in range(1, n_folds):  # fold 0 has no history
        train_ts = timestamps[:fold * fold_size]
        val_ts = timestamps[fold * fold_size: (fold + 1) * fold_size]

        tr = df_sorted[df_sorted["timestamp"].isin(set(train_ts))]
        va = df_sorted[df_sorted["timestamp"].isin(set(val_ts))]

        for d in [tr, va]:
            if len(d) < 100:
                continue

        tr_x = tr[avail].fillna(0).replace([np.inf, -np.inf], 0)
        tr_y = (tr["fwd_ret_12h"] > 0).astype(int)
        va_x = va[avail].fillna(0).replace([np.inf, -np.inf], 0)

        scaler = StandardScaler()
        tr_x_sc = scaler.fit_transform(tr_x)
        va_x_sc = scaler.transform(va_x)

        clf = LogisticRegression(C=0.1, max_iter=300, n_jobs=-1, random_state=0)
        clf.fit(tr_x_sc, tr_y)
        preds = clf.predict_proba(va_x_sc)[:, 1]
        p_lin.loc[va.index] = preds

    return p_lin


def compute_p_seq_oof(df: pd.DataFrame, feats: List[str], seq_len: int = 8,
                      n_folds: int = 5) -> pd.Series:
    """OOF predictions from GRU micro-model via temporal CV."""
    avail = [f for f in feats if f in df.columns]
    df_sorted = df.sort_values("timestamp").copy()
    timestamps = sorted(df_sorted["timestamp"].unique())
    fold_size = len(timestamps) // n_folds
    p_seq = pd.Series(np.nan, index=df.index, dtype=float)

    for fold in range(1, n_folds):
        train_ts = set(timestamps[:fold * fold_size])
        val_ts = set(timestamps[fold * fold_size: (fold + 1) * fold_size])

        tr_df = df_sorted[df_sorted["timestamp"].isin(train_ts)]
        va_df = df_sorted[df_sorted["timestamp"].isin(val_ts)]

        if len(tr_df) < 1000:
            continue

        # Normalize with train stats
        scaler = StandardScaler()
        tr_df = tr_df.copy()
        va_df = va_df.copy()
        tr_df[avail] = scaler.fit_transform(tr_df[avail].fillna(0).replace([np.inf, -np.inf], 0))
        va_df[avail] = scaler.transform(va_df[avail].fillna(0).replace([np.inf, -np.inf], 0))

        X_tr, idx_tr = build_gru_sequences(tr_df, avail, seq_len)
        X_va, idx_va = build_gru_sequences(va_df, avail, seq_len)

        if len(X_tr) < 500 or len(X_va) == 0:
            continue

        y_tr = (df_sorted.loc[idx_tr, "fwd_ret_12h"] > 0).astype(np.float32).values
        model = train_gru(X_tr, y_tr)
        preds_va = predict_gru(model, X_va)
        p_seq.loc[idx_va] = preds_va
        print(f"    GRU fold {fold}/{n_folds-1}: tr={len(X_tr):,} va={len(X_va):,}")

    return p_seq


# ══════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════

def load_data():
    print("=" * 70)
    print("  LOADING DATA")
    print("=" * 70)
    df, regime_df = load_research_frame()
    df, _ = add_r35_features(df)
    cg = load_cg_daily()
    cg_feats = compute_cg_features(cg)
    df, _, _ = add_cg_features(df, cg_feats)

    missing_31 = [f for f in CHAMPION_FEAT_31 if f not in df.columns]
    if missing_31:
        print(f"  WARNING: Missing features: {missing_31}")

    print(f"  Frame: {len(df):,} rows, {df['symbol'].nunique()} symbols")
    print(f"  Dates: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    return df, regime_df


# ══════════════════════════════════════════════════════════
#  MAIN ENSEMBLE TRAINING WITH META-FEATURES
# ══════════════════════════════════════════════════════════

def compute_meta_features_for_window(df: pd.DataFrame, w: dict,
                                     use_p_lin: bool, use_p_seq: bool):
    """
    For a given WF window, compute p_lin and/or p_seq as OOF on train,
    full-fit on test. Returns df with 'p_lin' and/or 'p_seq' columns added.
    """
    tz = df["timestamp"].dt.tz
    tr_end = pd.Timestamp(w["train_end"], tz=tz)
    te_start = pd.Timestamp(w["test_start"], tz=tz)
    te_end = pd.Timestamp(w["test_end"], tz=tz)

    train_df = df[df["timestamp"] < tr_end].copy()
    test_df = df[(df["timestamp"] >= te_start) & (df["timestamp"] <= te_end)].copy()

    df = df.copy()

    if use_p_lin:
        lin_avail = [f for f in LIN_FEATS if f in df.columns]
        print(f"    Computing p_lin OOF (5-fold temporal)...")
        p_lin_oof = compute_p_lin_oof(train_df, lin_avail, n_folds=5)

        # Full fit on train, predict test
        scaler = StandardScaler()
        tr_x = scaler.fit_transform(
            train_df[lin_avail].fillna(0).replace([np.inf, -np.inf], 0))
        tr_y = (train_df["fwd_ret_12h"] > 0).astype(int)
        clf = LogisticRegression(C=0.1, max_iter=300, n_jobs=-1, random_state=0)
        clf.fit(tr_x, tr_y)
        te_x = scaler.transform(
            test_df[lin_avail].fillna(0).replace([np.inf, -np.inf], 0))
        p_lin_test = pd.Series(
            clf.predict_proba(te_x)[:, 1], index=test_df.index)

        df.loc[train_df.index, "p_lin"] = p_lin_oof
        df.loc[test_df.index, "p_lin"] = p_lin_test
        df["p_lin"] = df["p_lin"].fillna(0.5)

    if use_p_seq:
        gru_avail = [f for f in GRU_FEATS if f in df.columns]
        print(f"    Computing p_seq OOF (5-fold temporal GRU)...")
        p_seq_oof = compute_p_seq_oof(train_df, gru_avail, seq_len=GRU_SEQ_LEN, n_folds=5)

        # Full fit on train, predict test
        scaler = StandardScaler()
        tr_df2 = train_df.copy()
        te_df2 = test_df.copy()
        tr_df2[gru_avail] = scaler.fit_transform(
            tr_df2[gru_avail].fillna(0).replace([np.inf, -np.inf], 0))
        te_df2[gru_avail] = scaler.transform(
            te_df2[gru_avail].fillna(0).replace([np.inf, -np.inf], 0))

        X_tr_full, idx_tr_full = build_gru_sequences(tr_df2, gru_avail, GRU_SEQ_LEN)
        X_te_full, idx_te_full = build_gru_sequences(te_df2, gru_avail, GRU_SEQ_LEN)

        if len(X_tr_full) > 500:
            y_tr_full = (train_df.loc[idx_tr_full, "fwd_ret_12h"] > 0).astype(np.float32).values
            model_full = train_gru(X_tr_full, y_tr_full, n_epochs=20)
            if len(X_te_full) > 0:
                preds_te = predict_gru(model_full, X_te_full)
                p_seq_te = pd.Series(preds_te, index=idx_te_full)
                df.loc[test_df.index, "p_seq"] = np.nan
                df.loc[p_seq_te.index, "p_seq"] = p_seq_te.values

        df.loc[train_df.index, "p_seq"] = p_seq_oof
        df["p_seq"] = df["p_seq"].fillna(0.5)

    return df


def train_ensemble(df: pd.DataFrame, feats: List[str], w: dict,
                   seeds=SEEDS, cs_rank_exclude=None):
    """Train LGB+XGB ensemble for a single window. Returns predictions df."""
    avail = [f for f in feats if f in df.columns]
    rank_exclude = set(cs_rank_exclude or [])
    rank_feats = [f for f in avail if f not in rank_exclude]
    tz = df["timestamp"].dt.tz

    te_end = pd.Timestamp(w["test_end"], tz=tz)
    te_start = pd.Timestamp(w["test_start"], tz=tz)
    tr_end = pd.Timestamp(w["train_end"], tz=tz)
    va_start = pd.Timestamp(w["val_start"], tz=tz)
    va_end = pd.Timestamp(w["val_end"], tz=tz)

    train_ = df[df["timestamp"] < tr_end].copy()
    val_ = df[(df["timestamp"] >= va_start) & (df["timestamp"] < va_end)].copy()
    test_ = df[(df["timestamp"] >= te_start) & (df["timestamp"] <= te_end)].copy()

    if len(train_) < 5000 or len(test_) < 200:
        return None

    if rank_feats:
        train_ = cs_rank_cols(train_, rank_feats)
        val_ = cs_rank_cols(val_, rank_feats)
        test_ = cs_rank_cols(test_, rank_feats)

    for d in [train_, val_, test_]:
        d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)

    for col in avail:
        for d in [train_, val_, test_]:
            if d[col].isna().any():
                d[col] = d[col].fillna(0)

    all_lgb, all_xgb = [], []
    fwd = test_[["timestamp", "symbol", "fwd_ret_12h"]].rename(
        columns={"fwd_ret_12h": "fwd_ret"}).dropna()

    for seed in seeds:
        p_lgb = {**LGB_PARAMS, "seed": seed}
        p_xgb = {**XGB_PARAMS, "seed": seed}

        tr = train_[avail + ["target_binary"]].dropna()
        va = val_[avail + ["target_binary"]].dropna()
        te = test_[avail + ["target_binary", "timestamp", "symbol"]].dropna()

        for d in [tr, va, te]:
            d.replace([np.inf, -np.inf], np.nan, inplace=True)
        tr, va, te = tr.dropna(), va.dropna(), te.dropna()
        if len(te) == 0:
            continue

        # LGB
        dt = lgb.Dataset(tr[avail], label=tr["target_binary"])
        dv = lgb.Dataset(va[avail], label=va["target_binary"])
        m = lgb.train(p_lgb, dt, num_boost_round=N_ROUNDS,
                      valid_sets=[dv],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                 lgb.log_evaluation(-1)])
        p = m.predict(te[avail])
        rec = te[["timestamp", "symbol"]].copy()
        rec["pred_lgb"] = p
        rec = rec.merge(fwd, on=["timestamp", "symbol"], how="inner")
        rec["window"] = w["name"]
        rec["seed"] = seed
        all_lgb.append(rec)

        # XGB
        dt_x = xgb.DMatrix(tr[avail], label=tr["target_binary"])
        dv_x = xgb.DMatrix(va[avail], label=va["target_binary"])
        m_x = xgb.train(p_xgb, dt_x, num_boost_round=N_ROUNDS,
                         evals=[(dv_x, "val")],
                         early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        p_x = m_x.predict(xgb.DMatrix(te[avail]))
        rec2 = te[["timestamp", "symbol"]].copy()
        rec2["pred_xgb"] = p_x
        rec2 = rec2.merge(fwd, on=["timestamp", "symbol"], how="inner")
        rec2["window"] = w["name"]
        rec2["seed"] = seed
        all_xgb.append(rec2)

    if not all_lgb:
        return None

    lgb_df = pd.concat(all_lgb)
    xgb_df = pd.concat(all_xgb)
    lgb_avg = lgb_df.groupby(["timestamp", "symbol"]).agg(
        pred_lgb=("pred_lgb", "mean"), fwd_ret=("fwd_ret", "first"),
        window=("window", "first")).reset_index()
    xgb_avg = xgb_df.groupby(["timestamp", "symbol"]).agg(
        pred_xgb=("pred_xgb", "mean")).reset_index()
    merged = lgb_avg.merge(xgb_avg, on=["timestamp", "symbol"], how="inner")
    merged["rank_lgb"] = merged.groupby("timestamp")["pred_lgb"].rank(pct=True) - 0.5
    merged["rank_xgb"] = merged.groupby("timestamp")["pred_xgb"].rank(pct=True) - 0.5
    merged["pred"] = 0.5 * merged["rank_lgb"] + 0.5 * merged["rank_xgb"]
    return merged[["timestamp", "symbol", "pred", "fwd_ret", "window"]]


# ══════════════════════════════════════════════════════════
#  SIMULATION (standard hybrid costs, same as R48)
# ══════════════════════════════════════════════════════════

def simulate_with_hybrid_costs(merged, regime_df, cfg):
    n_long = cfg.get("n_long", 6)
    n_short = cfg.get("n_short", 3)
    trend_cutoff = cfg.get("trend_cutoff", 0.9)
    rebal_hours = cfg.get("rebal_hours", 12)
    ema_alpha = cfg.get("ema_alpha", None)
    hysteresis = cfg.get("hysteresis", 0)
    funding_per_12h = 0.00008

    all_rets = []
    prev_longs: Set[str] = set()
    prev_shorts: Set[str] = set()
    prev_preds: Dict[str, float] = {}

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}
    rebal_timestamps = timestamps_sorted[::rebal_hours]

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        if trend_str > trend_cutoff:
            continue
        grp = grouped[ts].copy()
        n = len(grp)
        nl, ns = min(n_long, n // 3), min(n_short, n // 3)
        if nl == 0 and ns == 0:
            continue

        exposure = 1.0
        dyn_threshold = cfg.get("dyn_threshold", 0.5)
        if dyn_threshold is not None and trend_str > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) /
                           (trend_cutoff - dyn_threshold + 1e-10) * 0.5)

        if ema_alpha is not None and ema_alpha < 1.0:
            for idx, r in grp.iterrows():
                sym = r["symbol"]
                raw_pred = r["pred"]
                smoothed = ema_alpha * raw_pred + (1 - ema_alpha) * prev_preds.get(sym, raw_pred)
                prev_preds[sym] = smoothed
                grp.at[idx, "pred"] = smoothed

        grp["pred_rank"] = grp["pred"].rank(ascending=False)

        if hysteresis > 0 and (prev_longs or prev_shorts):
            new_longs: Set[str] = set()
            new_shorts: Set[str] = set()
            for idx, r in grp.iterrows():
                sym, rank = r["symbol"], r["pred_rank"]
                if sym in prev_longs and rank <= nl + hysteresis:
                    new_longs.add(sym)
                elif sym in prev_shorts and rank > (n - ns - hysteresis):
                    new_shorts.add(sym)
            for _, r in grp[~grp["symbol"].isin(new_longs | new_shorts)].sort_values("pred_rank").head(nl - len(new_longs)).iterrows():
                new_longs.add(r["symbol"])
            for _, r in grp[~grp["symbol"].isin(new_longs | new_shorts)].sort_values("pred_rank", ascending=False).head(ns - len(new_shorts)).iterrows():
                new_shorts.add(r["symbol"])
        else:
            new_longs = set(grp[grp["pred_rank"] <= nl]["symbol"].tolist()) if nl > 0 else set()
            new_shorts = set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist()) if ns > 0 else set()

        longs = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]
        long_ret = longs["fwd_ret"].mean() if len(longs) > 0 else 0
        short_ret = shorts["fwd_ret"].mean() if len(shorts) > 0 else 0

        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        total_positions = len(new_longs) + len(new_shorts)

        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            turnover_cost = sum(_cost_for_sym(sym) * avg_weight for sym in new_opened)
            turnover_cost += sum(_cost_for_sym(sym) * avg_weight for sym in closed)
            holding_cost = funding_per_12h * (rebal_hours / 12)
            total_cost = turnover_cost + holding_cost
        else:
            total_cost = 0.0

        nl_act, ns_act = len(new_longs), len(new_shorts)
        if nl_act > 0 and ns_act > 0:
            port_ret = 0.5 * long_ret - 0.5 * short_ret
        elif ns_act > 0:
            port_ret = -short_ret
        else:
            port_ret = long_ret

        port_ret *= exposure
        port_ret -= total_cost

        prev_longs = new_longs
        prev_shorts = new_shorts

        all_rets.append({
            "timestamp": ts,
            "portfolio_ret": port_ret,
            "n_long": nl_act,
            "n_short": ns_act,
            "cost": total_cost,
        })

    return pd.DataFrame(all_rets) if all_rets else pd.DataFrame()


def compute_metrics(port_df):
    if port_df.empty:
        return {"sharpe": 0, "total_return_pct": 0, "max_dd_pct": 0, "win_rate": 0}
    eq = (1 + port_df["portfolio_ret"]).cumprod()
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(2 * 365)
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    maxdd = (eq / eq.cummax() - 1).min()
    win_rate = (rets > 0).sum() / len(rets) * 100
    return {
        "sharpe": round(sharpe, 3),
        "total_return_pct": round(total_ret * 100, 1),
        "max_dd_pct": round(maxdd * 100, 1),
        "win_rate": round(win_rate, 1),
    }


def compute_window_metrics(port_df, preds):
    results = {}
    for wname in preds["window"].unique():
        wp = preds[preds["window"] == wname]
        ts_min, ts_max = wp["timestamp"].min(), wp["timestamp"].max()
        w_port = port_df[(port_df["timestamp"] >= ts_min) & (port_df["timestamp"] <= ts_max)]
        if len(w_port) < 2:
            results[wname] = 0
            continue
        eq = (1 + w_port["portfolio_ret"]).cumprod()
        rets = eq.pct_change().dropna()
        sh = rets.mean() / (rets.std() + 1e-10) * np.sqrt(2 * 365)
        results[wname] = round(sh, 2)
    return results


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("  R62 — ALT MODEL STACKING (+p_lin, +p_seq)")
    print("  Adding Level-0 meta-features to champion 31f")
    print("=" * 70)

    df, regime_df = load_data()
    base_no_rank = [f for f in CHAMPION_FEAT_31 if f in MARKET_LEVEL_FEATURES]

    experiments = [
        ("baseline_31f",  False, False, CHAMPION_FEAT_31),
        ("+p_lin_32f",    True,  False, CHAMPION_FEAT_31 + ["p_lin"]),
        ("+p_seq_32f",    False, True,  CHAMPION_FEAT_31 + ["p_seq"]),
        ("+p_lin+p_seq",  True,  True,  CHAMPION_FEAT_31 + ["p_lin", "p_seq"]),
    ]

    results = []

    print(f"\n  Running {len(experiments)} experiments × {len(ORIGINAL_WINDOWS)} windows...\n")

    for exp_name, use_lin, use_seq, feats in experiments:
        print(f"\n  {'─' * 65}")
        print(f"  Experiment: {exp_name}")
        t1 = time.time()

        all_preds = []

        for w in ORIGINAL_WINDOWS:
            print(f"\n  Window {w['name']} ({exp_name}):")

            # Compute meta-features (OOF within this window's trainset)
            if use_lin or use_seq:
                df_w = compute_meta_features_for_window(df, w, use_lin, use_seq)
            else:
                df_w = df

            avail = [f for f in feats if f in df_w.columns]
            meta_no_rank = ["p_lin", "p_seq"]  # never CS-rank meta probs
            no_rank = list(set(base_no_rank + meta_no_rank))

            preds_w = train_ensemble(df_w, avail, w, seeds=SEEDS,
                                     cs_rank_exclude=no_rank)
            if preds_w is not None:
                all_preds.append(preds_w)

        if not all_preds:
            print(f"  ⚠️  {exp_name}: no predictions")
            continue

        preds = pd.concat(all_preds)
        port = simulate_with_hybrid_costs(preds, regime_df, PROD_CFG)

        if port.empty:
            print(f"  ⚠️  {exp_name}: simulation empty")
            continue

        m = compute_metrics(port)
        wm = compute_window_metrics(port, preds)
        elapsed = time.time() - t1

        result = {
            "experiment": exp_name,
            "n_features": len([f for f in feats if f in df.columns or f in ["p_lin", "p_seq"]]),
            "use_p_lin": use_lin,
            "use_p_seq": use_seq,
            "sharpe": m["sharpe"],
            "total_ret%": m["total_return_pct"],
            "maxDD%": m["max_dd_pct"],
            "win_rate%": m["win_rate"],
            "W1": wm.get("W1", "?"),
            "W2": wm.get("W2", "?"),
            "W3": wm.get("W3", "?"),
        }
        results.append(result)

        baseline_sh = next((r["sharpe"] for r in results if r["experiment"] == "baseline_31f"), 0)
        delta = m["sharpe"] - baseline_sh
        print(f"  [{exp_name:20s}]  Sharpe={m['sharpe']:5.2f} (Δ{delta:+.2f})  "
              f"Ret={m['total_return_pct']:+6.1f}%  DD={m['max_dd_pct']:5.1f}%  "
              f"W1={wm.get('W1','?'):5.2f} W2={wm.get('W2','?'):5.2f} W3={wm.get('W3','?'):5.2f}  "
              f"({elapsed/60:.1f}min)")

    # ── Summary ───────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  R62 RESULTS SUMMARY")
    print("=" * 90)
    print(f"  {'Experiment':<22} {'Feats':>6} {'Sharpe':>7} {'Ret%':>7} {'MaxDD%':>7} "
          f"{'WR%':>6} {'W1':>6} {'W2':>6} {'W3':>6}")
    print("  " + "-" * 80)

    baseline_sharpe = next((r["sharpe"] for r in results if r["experiment"] == "baseline_31f"), 0)

    for r in sorted(results, key=lambda x: x["sharpe"], reverse=True):
        delta = r["sharpe"] - baseline_sharpe
        marker = f" (Δ{delta:+.2f})" if r["experiment"] != "baseline_31f" else " ← BASELINE"
        print(f"  {r['experiment']:<22} {r['n_features']:>6} {r['sharpe']:>7.2f} "
              f"{r['total_ret%']:>+7.1f} {r['maxDD%']:>7.1f} {r['win_rate%']:>6.1f} "
              f"{r['W1']:>6} {r['W2']:>6} {r['W3']:>6}{marker}")

    print(f"\n  Total elapsed: {(time.time()-t0)/60:.1f}min")
    print("\n  ✅ R62 COMPLETE")

    out_path = "/data/datasets/results_r62_stacking.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
