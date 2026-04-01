#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R27 — Multi-Horizon & Target Engineering

R26 best: F-6L3S-dt0.7 → Sh=3.39, Eq=$3632, Worst=-3.0%
R26 finding: model-type changes (focal, multi-class, CatBoost, interactions) failed.
             Only portfolio config (6L3S) marginally improved.
             → The 12h binary signal is near-optimal for this feature set.
             → Next alpha must come from TARGET DIVERSITY, not model diversity.

Available forward returns: fwd_ret_1h, fwd_ret_4h, fwd_ret_12h, fwd_ret_24h, fwd_ret_48h
Currently only 12h is used.

Experiments:
  A: Multi-Horizon Signal Blending (4h + 12h + 24h predictions)
  B: Temporal Sample Weighting (exponential decay, recent data matters more)
  C: Relative Return Target (predict vs BTC, not absolute)
  D: Multi-Horizon + Relative combined
  E: Meta-Stacking (L1 diverse models → L2 logistic)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import warnings, time, sys, argparse
warnings.filterwarnings("ignore")

from _research_round7 import (
    SYM_35, WINDOWS, compute_regime, simulate, eval_config, show,
)
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal
from _research_r22_models import (
    FEATURES_23, SEEDS, LEVERAGE, CAPITAL, CFG_BEST, DATA_DIR, SENT_DIR,
    log, build_r19_features, add_new_features, cs_rank_cols,
)

# R26 winner: 6L3S with dt=0.7
CFG_6L3S = {**CFG_BEST, "n_long": 6, "n_short": 3, "dyn_threshold": 0.7}
# R25 baseline for comparison
CFG_5L3S = {**CFG_BEST, "n_long": 5, "n_short": 3, "dyn_threshold": 0.5625}


# ═══════════════════════════════════════════════════════════════════════════════
#  Core trainers — horizon-aware
# ═══════════════════════════════════════════════════════════════════════════════

def train_lgb_horizon(df, feats, target_col="fwd_ret_12h", eval_col="fwd_ret_12h",
                      sample_weight_fn=None, seeds=SEEDS):
    """
    Train LGB binary classifier on arbitrary horizon.
    target_col: column used for binary target (> 0 → 1)
    eval_col: column used as fwd_ret in output (for portfolio sim — always 12h usually)
    sample_weight_fn: function(train_df) -> weight array, or None
    """
    avail = [f for f in feats if f in df.columns]
    all_preds = []
    tz = df["timestamp"].dt.tz

    for seed in seeds:
        seed_preds = []
        for w in WINDOWS:
            train = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz=tz)].copy()
            val = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz=tz)) &
                     (df["timestamp"] < pd.Timestamp(w["val_end"], tz=tz))].copy()
            test = df[(df["timestamp"] >= pd.Timestamp(w["test_start"], tz=tz)) &
                      (df["timestamp"] <= pd.Timestamp(w["test_end"], tz=tz))].copy()
            if len(train) < 5000 or len(test) < 200:
                continue
            train = cs_rank_cols(train, avail)
            val = cs_rank_cols(val, avail)
            test = cs_rank_cols(test, avail)

            for d in [train, val, test]:
                d["target_binary"] = (d[target_col] > 0).astype(int)

            need_cols = avail + ["target_binary"]
            train_c = train[need_cols].dropna()
            val_c = val[need_cols].dropna()

            dtrain = lgb.Dataset(train_c[avail], label=train_c["target_binary"])
            dval = lgb.Dataset(val_c[avail], label=val_c["target_binary"])

            # Apply sample weights if provided
            if sample_weight_fn is not None:
                w_train = sample_weight_fn(train.loc[train_c.index])
                dtrain.set_weight(w_train)

            model = lgb.train(
                {"objective": "binary", "metric": "auc",
                 "learning_rate": 0.03, "num_leaves": 63,
                 "min_child_samples": 100, "subsample": 0.8,
                 "colsample_bytree": 0.8, "lambda_l2": 1.0,
                 "verbose": -1, "n_jobs": -1, "seed": seed},
                dtrain, num_boost_round=600, valid_sets=[dval],
                callbacks=[lgb.early_stopping(40, verbose=False),
                           lgb.log_evaluation(-1)])

            test_c = test[avail + ["target_binary", "timestamp", "symbol"]].dropna()
            if len(test_c) == 0:
                continue
            preds = model.predict(test_c[avail])
            fwd = test[["timestamp", "symbol", eval_col]].rename(
                columns={eval_col: "fwd_ret"}).dropna()
            merged = test_c[["timestamp", "symbol"]].copy()
            merged["pred"] = preds
            merged = merged.merge(fwd, on=["timestamp", "symbol"], how="inner")
            merged["window"] = w["name"]
            seed_preds.append(merged)

        if seed_preds:
            all_preds.append(pd.concat(seed_preds, ignore_index=True))

    if not all_preds:
        return None
    combined = pd.concat(all_preds, ignore_index=True)
    return (combined.groupby(["timestamp", "symbol"])
            .agg(pred=("pred", "mean"), fwd_ret=("fwd_ret", "first"),
                 window=("window", "first"))
            .reset_index())


def train_xgb_horizon(df, feats, target_col="fwd_ret_12h", eval_col="fwd_ret_12h",
                      sample_weight_fn=None, seeds=SEEDS):
    """Train XGB binary classifier on arbitrary horizon."""
    avail = [f for f in feats if f in df.columns]
    all_preds = []
    tz = df["timestamp"].dt.tz

    for seed in seeds:
        seed_preds = []
        for w in WINDOWS:
            train = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz=tz)].copy()
            val = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz=tz)) &
                     (df["timestamp"] < pd.Timestamp(w["val_end"], tz=tz))].copy()
            test = df[(df["timestamp"] >= pd.Timestamp(w["test_start"], tz=tz)) &
                      (df["timestamp"] <= pd.Timestamp(w["test_end"], tz=tz))].copy()
            if len(train) < 5000 or len(test) < 200:
                continue
            train = cs_rank_cols(train, avail)
            val = cs_rank_cols(val, avail)
            test = cs_rank_cols(test, avail)

            for d in [train, val, test]:
                d["target_binary"] = (d[target_col] > 0).astype(int)

            need_cols = avail + ["target_binary"]
            train_c = train[need_cols].dropna()
            val_c = val[need_cols].dropna()

            dtrain = xgb.DMatrix(train_c[avail], label=train_c["target_binary"])
            dval = xgb.DMatrix(val_c[avail], label=val_c["target_binary"])

            if sample_weight_fn is not None:
                w_train = sample_weight_fn(train.loc[train_c.index])
                dtrain.set_weight(w_train)

            model = xgb.train(
                {"objective": "binary:logistic", "eval_metric": "auc",
                 "learning_rate": 0.03, "max_depth": 6,
                 "min_child_weight": 100, "subsample": 0.8,
                 "colsample_bytree": 0.8, "reg_lambda": 1.0,
                 "seed": seed, "n_jobs": -1, "verbosity": 0},
                dtrain, num_boost_round=600,
                evals=[(dval, "val")],
                early_stopping_rounds=40, verbose_eval=False)

            test_c = test[avail + ["target_binary", "timestamp", "symbol"]].dropna()
            if len(test_c) == 0:
                continue
            dtest = xgb.DMatrix(test_c[avail])
            preds = model.predict(dtest)
            fwd = test[["timestamp", "symbol", eval_col]].rename(
                columns={eval_col: "fwd_ret"}).dropna()
            merged = test_c[["timestamp", "symbol"]].copy()
            merged["pred"] = preds
            merged = merged.merge(fwd, on=["timestamp", "symbol"], how="inner")
            merged["window"] = w["name"]
            seed_preds.append(merged)

        if seed_preds:
            all_preds.append(pd.concat(seed_preds, ignore_index=True))

    if not all_preds:
        return None
    combined = pd.concat(all_preds, ignore_index=True)
    return (combined.groupby(["timestamp", "symbol"])
            .agg(pred=("pred", "mean"), fwd_ret=("fwd_ret", "first"),
                 window=("window", "first"))
            .reset_index())


# ═══════════════════════════════════════════════════════════════════════════════
#  Relative return target builder
# ═══════════════════════════════════════════════════════════════════════════════

def add_relative_returns(df):
    """Add fwd_ret_Xh_vs_btc = coin return - BTC return for each horizon."""
    btc = df[df["symbol"] == "BTC/USDT"][["timestamp", "fwd_ret_4h", "fwd_ret_12h", "fwd_ret_24h"]].copy()
    btc = btc.rename(columns={c: f"btc_{c}" for c in ["fwd_ret_4h", "fwd_ret_12h", "fwd_ret_24h"]})
    df = df.merge(btc, on="timestamp", how="left")
    for h in ["4h", "12h", "24h"]:
        df[f"fwd_ret_{h}_vs_btc"] = df[f"fwd_ret_{h}"] - df[f"btc_fwd_ret_{h}"]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def rank_normalize(p):
    """Rank-normalize predictions per timestamp → [-0.5, +0.5]."""
    out = p.copy()
    out["pred"] = out.groupby("timestamp")["pred"].rank(pct=True) - 0.5
    return out


def blend_predictions(preds_list, weights=None):
    """Blend multiple prediction DataFrames using rank-normalization."""
    if weights is None:
        weights = [1.0 / len(preds_list)] * len(preds_list)

    ranked = []
    for i, p in enumerate(preds_list):
        rn = rank_normalize(p).rename(columns={"pred": f"pred_{i}"})
        ranked.append(rn)

    merged = ranked[0][["timestamp", "symbol", "pred_0", "fwd_ret", "window"]]
    for i in range(1, len(ranked)):
        merged = merged.merge(
            ranked[i][["timestamp", "symbol", f"pred_{i}"]],
            on=["timestamp", "symbol"], how="inner")

    pred_cols = [f"pred_{i}" for i in range(len(ranked))]
    merged["pred"] = sum(w * merged[c] for w, c in zip(weights, pred_cols))
    return merged[["timestamp", "symbol", "pred", "fwd_ret", "window"]]


def eval_and_show(preds, regime_df, cfg, label):
    """Simulate + evaluate + show. Returns (label, result_dict) or None."""
    if preds is None:
        log(f"  !! {label}: no predictions")
        return None
    port = simulate(preds, regime_df, 12, cfg)
    r = eval_config(port, 12, label, LEVERAGE, CAPITAL)
    if r:
        show(r)
        return (label, r)
    return None


def make_temporal_weight_fn(half_life_days):
    """Return a sample_weight function with exponential decay."""
    decay_per_hour = np.log(2) / (half_life_days * 24)
    def weight_fn(train_df):
        ts = train_df["timestamp"]
        max_ts = ts.max()
        hours_ago = (max_ts - ts).dt.total_seconds() / 3600
        weights = np.exp(-decay_per_hour * hours_ago.values)
        return weights
    return weight_fn


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-A: Multi-Horizon Signal Blending
# ═══════════════════════════════════════════════════════════════════════════════

def exp_a(df, regime_df, avail):
    log("\n" + "=" * 80)
    log("  EXP-A: Multi-Horizon Signal Blending")
    log("=" * 80)

    results = []

    # Train single-horizon models (all evaluated on 12h returns)
    horizons = {
        "4h": "fwd_ret_4h",
        "12h": "fwd_ret_12h",
        "24h": "fwd_ret_24h",
    }

    lgb_preds = {}
    xgb_preds = {}

    for h_name, h_col in horizons.items():
        log(f"  Training LGB binary → {h_name}...")
        lgb_preds[h_name] = train_lgb_horizon(df, avail, target_col=h_col, eval_col="fwd_ret_12h")
        log(f"  Training XGB binary → {h_name}...")
        xgb_preds[h_name] = train_xgb_horizon(df, avail, target_col=h_col, eval_col="fwd_ret_12h")

    # Single-horizon ensembles (LGB+XGB per horizon)
    ens_preds = {}
    for h_name in horizons:
        if lgb_preds[h_name] is not None and xgb_preds[h_name] is not None:
            ens_preds[h_name] = blend_predictions([lgb_preds[h_name], xgb_preds[h_name]])
            r = eval_and_show(ens_preds[h_name], regime_df, CFG_6L3S, f"A-ens-{h_name}")
            if r:
                results.append(r)

    # R25 control (12h only, 6L3S)
    r = eval_and_show(ens_preds.get("12h"), regime_df, CFG_6L3S, "A-ctrl-6L3S")
    if r:
        results.append(r)

    # Multi-horizon blends
    blend_configs = [
        ("A-blend-4+12", ["4h", "12h"], [0.3, 0.7]),
        ("A-blend-12+24", ["12h", "24h"], [0.7, 0.3]),
        ("A-blend-4+12+24", ["4h", "12h", "24h"], [0.2, 0.5, 0.3]),
        ("A-blend-eq-3h", ["4h", "12h", "24h"], [1/3, 1/3, 1/3]),
        ("A-blend-4+12-eq", ["4h", "12h"], [0.5, 0.5]),
        ("A-blend-12+24-eq", ["12h", "24h"], [0.5, 0.5]),
    ]

    for label, h_names, weights in blend_configs:
        preds_to_blend = [ens_preds[h] for h in h_names if h in ens_preds]
        if len(preds_to_blend) == len(h_names):
            blended = blend_predictions(preds_to_blend, weights)
            r = eval_and_show(blended, regime_df, CFG_6L3S, label)
            if r:
                results.append(r)

    # Also try 24h model with 24h rebal
    if ens_preds.get("24h") is not None:
        # Need to re-train with eval_col=fwd_ret_24h for proper 24h rebal simulation
        log("  Training 24h ensemble for 24h rebalance...")
        lgb_24 = train_lgb_horizon(df, avail, target_col="fwd_ret_24h", eval_col="fwd_ret_24h")
        xgb_24 = train_xgb_horizon(df, avail, target_col="fwd_ret_24h", eval_col="fwd_ret_24h")
        if lgb_24 is not None and xgb_24 is not None:
            ens_24_full = blend_predictions([lgb_24, xgb_24])
            cfg_24h = {**CFG_6L3S, "rebal_hours": 24}
            port = simulate(ens_24_full, regime_df, 24, cfg_24h)
            r24 = eval_config(port, 24, "A-24h-rebal24", LEVERAGE, CAPITAL)
            if r24:
                show(r24)
                results.append(("A-24h-rebal24", r24))

    return results, ens_preds, lgb_preds, xgb_preds


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-B: Temporal Sample Weighting
# ═══════════════════════════════════════════════════════════════════════════════

def exp_b(df, regime_df, avail):
    log("\n" + "=" * 80)
    log("  EXP-B: Temporal Sample Weighting (exponential decay)")
    log("=" * 80)

    results = []

    for half_life in [90, 180, 360]:
        weight_fn = make_temporal_weight_fn(half_life)
        log(f"  Training LGB + XGB with half-life={half_life}d...")
        p_lgb = train_lgb_horizon(df, avail, sample_weight_fn=weight_fn)
        p_xgb = train_xgb_horizon(df, avail, sample_weight_fn=weight_fn)
        if p_lgb is not None and p_xgb is not None:
            ens = blend_predictions([p_lgb, p_xgb])
            r = eval_and_show(ens, regime_df, CFG_6L3S, f"B-decay-{half_life}d")
            if r:
                results.append(r)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-C: Relative Return Target (vs BTC)
# ═══════════════════════════════════════════════════════════════════════════════

def exp_c(df, regime_df, avail):
    log("\n" + "=" * 80)
    log("  EXP-C: Relative Return Target (coin - BTC)")
    log("=" * 80)

    results = []

    # Train on relative 12h returns
    log("  Training LGB + XGB on fwd_ret_12h_vs_btc...")
    p_lgb_rel = train_lgb_horizon(df, avail, target_col="fwd_ret_12h_vs_btc", eval_col="fwd_ret_12h")
    p_xgb_rel = train_xgb_horizon(df, avail, target_col="fwd_ret_12h_vs_btc", eval_col="fwd_ret_12h")
    if p_lgb_rel is not None and p_xgb_rel is not None:
        ens_rel = blend_predictions([p_lgb_rel, p_xgb_rel])
        r = eval_and_show(ens_rel, regime_df, CFG_6L3S, "C-rel-12h")
        if r:
            results.append(r)

    # Relative 24h
    log("  Training LGB + XGB on fwd_ret_24h_vs_btc...")
    p_lgb_r24 = train_lgb_horizon(df, avail, target_col="fwd_ret_24h_vs_btc", eval_col="fwd_ret_12h")
    p_xgb_r24 = train_xgb_horizon(df, avail, target_col="fwd_ret_24h_vs_btc", eval_col="fwd_ret_12h")
    if p_lgb_r24 is not None and p_xgb_r24 is not None:
        ens_r24 = blend_predictions([p_lgb_r24, p_xgb_r24])
        r = eval_and_show(ens_r24, regime_df, CFG_6L3S, "C-rel-24h")
        if r:
            results.append(r)

    # Blend absolute + relative
    if p_lgb_rel is not None and p_xgb_rel is not None:
        # Need absolute baseline preds
        log("  Training absolute baseline for blending...")
        p_lgb_abs = train_lgb_horizon(df, avail)
        p_xgb_abs = train_xgb_horizon(df, avail)
        if p_lgb_abs is not None and p_xgb_abs is not None:
            ens_abs = blend_predictions([p_lgb_abs, p_xgb_abs])

            for w_abs, w_rel, label in [
                (0.7, 0.3, "C-abs70+rel30"),
                (0.5, 0.5, "C-abs50+rel50"),
                (0.3, 0.7, "C-abs30+rel70"),
            ]:
                blended = blend_predictions([ens_abs, ens_rel], [w_abs, w_rel])
                r = eval_and_show(blended, regime_df, CFG_6L3S, label)
                if r:
                    results.append(r)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-D: Multi-Horizon + Relative combined (best of A + C)
# ═══════════════════════════════════════════════════════════════════════════════

def exp_d(df, regime_df, avail, ens_preds_a):
    """Combine multi-horizon (from EXP-A) with relative return models (EXP-C)."""
    log("\n" + "=" * 80)
    log("  EXP-D: Multi-Horizon + Relative Return combined")
    log("=" * 80)

    results = []

    # Get relative 12h predictions (re-train or reuse)
    log("  Training relative 12h LGB+XGB...")
    p_lgb_rel = train_lgb_horizon(df, avail, target_col="fwd_ret_12h_vs_btc", eval_col="fwd_ret_12h")
    p_xgb_rel = train_xgb_horizon(df, avail, target_col="fwd_ret_12h_vs_btc", eval_col="fwd_ret_12h")
    if p_lgb_rel is None or p_xgb_rel is None:
        return results
    ens_rel = blend_predictions([p_lgb_rel, p_xgb_rel])

    # Combine: absolute 12h-ens + relative 12h-ens + 4h-ens + 24h-ens
    combos = []
    if "12h" in ens_preds_a and "4h" in ens_preds_a:
        combos.append(("D-4+12+rel", [ens_preds_a["4h"], ens_preds_a["12h"], ens_rel],
                        [0.2, 0.5, 0.3]))
    if "12h" in ens_preds_a and "24h" in ens_preds_a:
        combos.append(("D-12+24+rel", [ens_preds_a["12h"], ens_preds_a["24h"], ens_rel],
                        [0.4, 0.3, 0.3]))
    if "4h" in ens_preds_a and "12h" in ens_preds_a and "24h" in ens_preds_a:
        combos.append(("D-all4", [ens_preds_a["4h"], ens_preds_a["12h"],
                                   ens_preds_a["24h"], ens_rel],
                        [0.15, 0.40, 0.20, 0.25]))
        combos.append(("D-all4-eq", [ens_preds_a["4h"], ens_preds_a["12h"],
                                      ens_preds_a["24h"], ens_rel],
                        [0.25, 0.25, 0.25, 0.25]))

    for label, preds_list, weights in combos:
        blended = blend_predictions(preds_list, weights)
        r = eval_and_show(blended, regime_df, CFG_6L3S, label)
        if r:
            results.append(r)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-E: Meta-Stacking (L1 diverse base → L2 logistic)
# ═══════════════════════════════════════════════════════════════════════════════

def exp_e(df, regime_df, avail, lgb_preds_a, xgb_preds_a):
    """
    L1: per-horizon LGB/XGB predictions (from EXP-A, reuse OOF)
    L2: LogisticRegression or Ridge learns optimal blend per-timestamp
    """
    log("\n" + "=" * 80)
    log("  EXP-E: Meta-Stacking (L1 → L2)")
    log("=" * 80)

    results = []

    # Also train relative models for stacking
    log("  Training L1 relative models...")
    lgb_rel = train_lgb_horizon(df, avail, target_col="fwd_ret_12h_vs_btc", eval_col="fwd_ret_12h")
    xgb_rel = train_xgb_horizon(df, avail, target_col="fwd_ret_12h_vs_btc", eval_col="fwd_ret_12h")

    # Collect all L1 predictions
    l1_models = {}
    for h in ["4h", "12h", "24h"]:
        if h in lgb_preds_a and lgb_preds_a[h] is not None:
            l1_models[f"lgb_{h}"] = lgb_preds_a[h]
        if h in xgb_preds_a and xgb_preds_a[h] is not None:
            l1_models[f"xgb_{h}"] = xgb_preds_a[h]
    if lgb_rel is not None:
        l1_models["lgb_rel12"] = lgb_rel
    if xgb_rel is not None:
        l1_models["xgb_rel12"] = xgb_rel

    if len(l1_models) < 4:
        log("  !! Not enough L1 models, skipping.")
        return results

    log(f"  L1 models: {len(l1_models)} — {list(l1_models.keys())}")

    # Merge all L1 predictions into a single DataFrame
    base_key = list(l1_models.keys())[0]
    merged = l1_models[base_key][["timestamp", "symbol", "fwd_ret", "window"]].copy()
    for name, p in l1_models.items():
        pr = p[["timestamp", "symbol", "pred"]].rename(columns={"pred": name})
        merged = merged.merge(pr, on=["timestamp", "symbol"], how="inner")

    pred_cols = list(l1_models.keys())
    merged = merged.dropna(subset=pred_cols)
    log(f"  Merged L1 rows: {len(merged):,}")

    # Rank-normalize L1 features per timestamp
    for c in pred_cols:
        merged[c] = merged.groupby("timestamp")[c].rank(pct=True) - 0.5

    # Walk-forward L2 training
    tz = merged["timestamp"].dt.tz
    all_l2_preds = []

    for w in WINDOWS:
        # Use first half of test window for L2 training, second half for L2 test
        # Actually, use val period for L2 training and test for evaluation
        l2_train = merged[(merged["timestamp"] >= pd.Timestamp(w["val_start"], tz=tz)) &
                          (merged["timestamp"] < pd.Timestamp(w["val_end"], tz=tz))]
        l2_test = merged[(merged["timestamp"] >= pd.Timestamp(w["test_start"], tz=tz)) &
                         (merged["timestamp"] <= pd.Timestamp(w["test_end"], tz=tz))]

        if len(l2_train) < 500 or len(l2_test) < 100:
            continue

        y_train = (l2_train["fwd_ret"] > 0).astype(int)
        X_train = l2_train[pred_cols].values
        X_test = l2_test[pred_cols].values

        # L2: Logistic Regression (regularized)
        lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        lr.fit(X_train, y_train)
        l2_preds = lr.predict_proba(X_test)[:, 1]

        out = l2_test[["timestamp", "symbol", "fwd_ret", "window"]].copy()
        out["pred"] = l2_preds
        all_l2_preds.append(out)

    if all_l2_preds:
        stacked = pd.concat(all_l2_preds, ignore_index=True)
        r = eval_and_show(stacked, regime_df, CFG_6L3S, "E-meta-stack")
        if r:
            results.append(r)

    # Compare with simple rank-average of all L1s
    merged["pred"] = merged[pred_cols].mean(axis=1)
    r = eval_and_show(merged[["timestamp", "symbol", "pred", "fwd_ret", "window"]],
                      regime_df, CFG_6L3S, "E-rank-avg-all")
    if r:
        results.append(r)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[], help="Experiments to skip (a b c d e)")
    args = ap.parse_args()
    skip = set(x.lower() for x in args.skip)

    t0 = time.time()
    log("=" * 80)
    log("  R27 — Multi-Horizon & Target Engineering")
    log(f"  Base: R26 winner 6L3S-dt0.7, Sh=3.39, Eq=$3632")
    log("=" * 80)

    # Load data
    log("\n  Loading data...")
    from _research_round7 import load_data
    df = load_data(SYM_35)
    regime_df = compute_regime(df)
    log(f"  Base: {len(df):,} rows, {df['symbol'].nunique()} symbols")

    log("\n  Building features...")
    df = build_r19_features(df)
    df, _ = add_new_features(df)
    avail = [f for f in FEATURES_23 if f in df.columns]
    log(f"  FEATURES_23: {len(avail)}/23")

    # Add relative return columns
    log("  Adding relative returns (vs BTC)...")
    df = add_relative_returns(df)
    for col in ["fwd_ret_4h_vs_btc", "fwd_ret_12h_vs_btc", "fwd_ret_24h_vs_btc"]:
        n_valid = df[col].notna().sum()
        log(f"    {col}: {n_valid:,} valid")

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-A: Multi-Horizon Signal Blending
    # ══════════════════════════════════════════════════════════════════════════
    ens_preds_a = {}
    lgb_preds_a = {}
    xgb_preds_a = {}
    results_a = []
    if "a" not in skip:
        results_a, ens_preds_a, lgb_preds_a, xgb_preds_a = exp_a(df, regime_df, avail)
    else:
        log("\n  [SKIP] EXP-A")

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-B: Temporal Sample Weighting
    # ══════════════════════════════════════════════════════════════════════════
    results_b = []
    if "b" not in skip:
        results_b = exp_b(df, regime_df, avail)
    else:
        log("\n  [SKIP] EXP-B")

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-C: Relative Return Target
    # ══════════════════════════════════════════════════════════════════════════
    results_c = []
    if "c" not in skip:
        results_c = exp_c(df, regime_df, avail)
    else:
        log("\n  [SKIP] EXP-C")

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-D: Multi-Horizon + Relative combined
    # ══════════════════════════════════════════════════════════════════════════
    results_d = []
    if "d" not in skip and ens_preds_a:
        results_d = exp_d(df, regime_df, avail, ens_preds_a)
    else:
        log("\n  [SKIP] EXP-D (needs EXP-A)")

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-E: Meta-Stacking
    # ══════════════════════════════════════════════════════════════════════════
    results_e = []
    if "e" not in skip and lgb_preds_a:
        results_e = exp_e(df, regime_df, avail, lgb_preds_a, xgb_preds_a)
    else:
        log("\n  [SKIP] EXP-E (needs EXP-A)")

    # ══════════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("  FINAL RANKINGS — R27 ALL EXPERIMENTS")
    log("=" * 80)

    all_results = []
    for bucket in [results_a, results_b, results_c, results_d, results_e]:
        if bucket:
            for label, r in bucket:
                all_results.append((label, r))

    all_results.sort(key=lambda x: x[1].get("sharpe", 0), reverse=True)

    log(f"\n  TOP-20 CONFIGURATIONS (of {len(all_results)}):")
    for i, (label, r) in enumerate(all_results[:20]):
        sh = r.get("sharpe", 0)
        eq = r.get("equity", 0)
        wm = r.get("worst_m", 0)
        win = r.get("win_months", 0)
        tot = r.get("total_months", 0)
        log(f"  #{i+1:2d}: {label:30s}  Sh={sh:+.2f}  Eq=${eq:.0f}  "
            f"Worst={wm*100:+.1f}%  WM={win}/{tot}")

    elapsed = (time.time() - t0) / 60
    log(f"\n  Total time: {elapsed:.1f} min")
    log("  DONE.")


if __name__ == "__main__":
    main()
