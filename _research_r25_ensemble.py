#!/usr/bin/env python3
"""
R25 — Ensemble & Feature Engineering for Classification

R24 best: E-5L-3S — LGB-binary-23f, 5L/3S → Sh=2.98, Eq=$2208
R24 findings: default params best, post-processing useless, absolute direction best

Experiments:
  A: Classifier Ensemble (LGB + XGB + CatBoost classifiers → blend probs)
  B: Feature Selection for Classification (importance-based pruning)
  C: Target Threshold (binary on ret > k instead of ret > 0)
  D: Multi-Horizon Classification (12h, 24h, 48h → blend signals)
  E: Sample Weighting (weight by |ret|, recency, vol-adjusted)
  F: 5L/3S Sweep (combine new winner with all promising R24 settings)
  G: Deeper/Shallower Trees + Regularization
  H: Proper Val-Based HPO (optimize on val, evaluate on test — no leakage)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from pathlib import Path
import warnings, time, sys
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from _research_round7 import (
    SYM_35, WINDOWS, compute_regime, simulate, eval_config, show,
)
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal
from _research_r22_models import (
    FEATURES_23, SEEDS, LEVERAGE, CAPITAL, CFG_BEST, DATA_DIR, SENT_DIR,
    log, build_r19_features, add_new_features, cs_rank_cols, run_eval,
)
from _research_r24_classification import train_cls

# R24 winner config: 5L/3S
CFG_5L3S = {**CFG_BEST, "n_long": 5, "n_short": 3}


# ═══════════════════════════════════════════════════════════════════════════════
#  XGBoost / CatBoost classifiers
# ═══════════════════════════════════════════════════════════════════════════════

def train_xgb_cls(df, feats, seeds=SEEDS):
    """Train XGBoost binary classifier, same walk-forward as train_cls."""
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
                d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)

            train_c = train[avail + ["target_binary"]].dropna()
            val_c = val[avail + ["target_binary"]].dropna()

            dtrain = xgb.DMatrix(train_c[avail], label=train_c["target_binary"])
            dval = xgb.DMatrix(val_c[avail], label=val_c["target_binary"])
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
            fwd = test[["timestamp", "symbol", "fwd_ret_12h"]].rename(
                columns={"fwd_ret_12h": "fwd_ret"}).dropna()
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


def train_cb_cls(df, feats, seeds=SEEDS):
    """Train CatBoost binary classifier, same walk-forward as train_cls."""
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
                d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)

            train_c = train[avail + ["target_binary"]].dropna()
            val_c = val[avail + ["target_binary"]].dropna()

            model = cb.CatBoostClassifier(
                loss_function="Logloss", eval_metric="AUC",
                learning_rate=0.03, depth=6,
                l2_leaf_reg=3.0, subsample=0.8,
                random_seed=seed, verbose=0,
                iterations=600, early_stopping_rounds=40)
            model.fit(train_c[avail], train_c["target_binary"],
                      eval_set=(val_c[avail], val_c["target_binary"]),
                      verbose=0)

            test_c = test[avail + ["target_binary", "timestamp", "symbol"]].dropna()
            if len(test_c) == 0:
                continue
            preds = model.predict_proba(test_c[avail])[:, 1]
            fwd = test[["timestamp", "symbol", "fwd_ret_12h"]].rename(
                columns={"fwd_ret_12h": "fwd_ret"}).dropna()
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
#  EXP-A: Classifier Ensemble (LGB + XGB + CatBoost)
# ═══════════════════════════════════════════════════════════════════════════════

def exp_a(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-A: Classifier Ensemble (LGB + XGB + CatBoost)")
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]

    log("  Training LGB classifier...")
    preds_lgb = train_cls(df, avail)
    log("  Training XGB classifier...")
    preds_xgb = train_xgb_cls(df, avail)
    log("  Training CatBoost classifier...")
    preds_cb = train_cb_cls(df, avail)

    results = []

    # Individual models with 5L/3S
    for label, preds in [("A-lgb-cls", preds_lgb), ("A-xgb-cls", preds_xgb),
                          ("A-cb-cls", preds_cb)]:
        if preds is None:
            log(f"  ⚠  {label}: no predictions")
            continue
        port = simulate(preds, regime_df, 12, CFG_5L3S)
        r = eval_config(port, 12, label, LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append((label, r))

    # Rank-normalize for blending
    def rank_normalize(p):
        out = p.copy()
        out["pred"] = out.groupby("timestamp")["pred"].rank(pct=True) - 0.5
        return out

    all_models = []
    for name, preds in [("lgb", preds_lgb), ("xgb", preds_xgb), ("cb", preds_cb)]:
        if preds is not None:
            rn = rank_normalize(preds)
            rn = rn.rename(columns={"pred": f"pred_{name}"})
            all_models.append((name, rn))

    if len(all_models) < 2:
        log("  ⚠  Need at least 2 models for ensemble")
        return results

    # Merge all predictions
    merged = all_models[0][1][["timestamp", "symbol", f"pred_{all_models[0][0]}",
                                "fwd_ret", "window"]]
    for name, rn in all_models[1:]:
        merged = merged.merge(
            rn[["timestamp", "symbol", f"pred_{name}"]],
            on=["timestamp", "symbol"], how="inner")

    # Equal-weight ensemble
    pred_cols = [f"pred_{name}" for name, _ in all_models]
    blend = merged.copy()
    blend["pred"] = blend[pred_cols].mean(axis=1)
    label = f"A-ens-eq-{len(all_models)}"
    port = simulate(blend, regime_df, 12, CFG_5L3S)
    r = eval_config(port, 12, label, LEVERAGE, CAPITAL)
    if r:
        show(r)
        results.append((label, r))

    # LGB-heavy ensemble (0.5 LGB + 0.25 XGB + 0.25 CB)
    if len(all_models) == 3:
        blend2 = merged.copy()
        blend2["pred"] = (0.5 * blend2["pred_lgb"] +
                          0.25 * blend2["pred_xgb"] +
                          0.25 * blend2["pred_cb"])
        label = "A-ens-lgb-heavy"
        port = simulate(blend2, regime_df, 12, CFG_5L3S)
        r = eval_config(port, 12, label, LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append((label, r))

        # LGB+XGB only (no CatBoost)
        blend3 = merged.copy()
        blend3["pred"] = 0.5 * blend3["pred_lgb"] + 0.5 * blend3["pred_xgb"]
        label = "A-ens-lgb-xgb"
        port = simulate(blend3, regime_df, 12, CFG_5L3S)
        r = eval_config(port, 12, label, LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append((label, r))

        # LGB+CB only
        blend4 = merged.copy()
        blend4["pred"] = 0.5 * blend4["pred_lgb"] + 0.5 * blend4["pred_cb"]
        label = "A-ens-lgb-cb"
        port = simulate(blend4, regime_df, 12, CFG_5L3S)
        r = eval_config(port, 12, label, LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append((label, r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-B: Feature Selection for Classification
# ═══════════════════════════════════════════════════════════════════════════════

def get_cls_feature_importance(df, feats):
    """Train one LGB binary classifier and return feature importance."""
    avail = [f for f in feats if f in df.columns]
    tz = df["timestamp"].dt.tz
    importance = np.zeros(len(avail))

    for w in WINDOWS:
        train = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz=tz)].copy()
        val = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz=tz)) &
                 (df["timestamp"] < pd.Timestamp(w["val_end"], tz=tz))].copy()
        train = cs_rank_cols(train, avail)
        val = cs_rank_cols(val, avail)
        for d in [train, val]:
            d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)
        train_c = train[avail + ["target_binary"]].dropna()
        val_c = val[avail + ["target_binary"]].dropna()
        dtrain = lgb.Dataset(train_c[avail], label=train_c["target_binary"])
        dval = lgb.Dataset(val_c[avail], label=val_c["target_binary"])
        model = lgb.train(
            {"objective": "binary", "metric": "auc",
             "learning_rate": 0.03, "num_leaves": 63,
             "min_child_samples": 100, "subsample": 0.8,
             "colsample_bytree": 0.8, "lambda_l2": 1.0,
             "seed": 42, "verbose": -1, "n_jobs": -1},
            dtrain, num_boost_round=600, valid_sets=[dval],
            callbacks=[lgb.early_stopping(40, verbose=False),
                       lgb.log_evaluation(-1)])
        importance += model.feature_importance(importance_type="gain")

    importance /= len(WINDOWS)
    imp_df = pd.DataFrame({"feature": avail, "importance": importance})
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return imp_df


def exp_b(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-B: Feature Selection for Classification")
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]

    log("  Computing feature importance for binary classifier...")
    imp_df = get_cls_feature_importance(df, avail)
    log("  Feature importance ranking:")
    for _, row in imp_df.iterrows():
        log(f"    {row['feature']:25s} {row['importance']:.0f}")

    results = []

    # Test subsets: top-N features
    for n in [10, 13, 16, 19, 23]:
        feats_n = imp_df["feature"].tolist()[:n]
        if n >= len(avail):
            feats_n = avail
        label = f"B-top{n}f"
        preds = train_cls(df, feats_n)
        port = simulate(preds, regime_df, 12, CFG_5L3S)
        r = eval_config(port, 12, label, LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append((label, r))

    # Drop bottom 3, 5 features
    for drop_n in [3, 5]:
        feats_drop = imp_df["feature"].tolist()[:len(avail) - drop_n]
        label = f"B-drop{drop_n}"
        preds = train_cls(df, feats_drop)
        port = simulate(preds, regime_df, 12, CFG_5L3S)
        r = eval_config(port, 12, label, LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append((label, r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-C: Target Threshold (binary on ret > k, not ret > 0)
# ═══════════════════════════════════════════════════════════════════════════════

def exp_c(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-C: Target Threshold (binary on ret > k)")
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]
    results = []

    for k in [-0.005, -0.002, 0.0, 0.002, 0.005, 0.01]:
        label = f"C-thr{k:+.3f}"
        target_fn = (lambda k_val: lambda d: (d["fwd_ret_12h"] > k_val).astype(int))(k)
        preds = train_cls(df, avail, target_fn=target_fn)
        port = simulate(preds, regime_df, 12, CFG_5L3S)
        r = eval_config(port, 12, label, LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append((k, r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-D: Multi-Horizon Classification (12h, 24h, 48h → blend)
# ═══════════════════════════════════════════════════════════════════════════════

def exp_d(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-D: Multi-Horizon Classification")
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]

    horizons_cols = {"12h": "fwd_ret_12h", "24h": "fwd_ret_24h", "48h": "fwd_ret_48h"}
    preds_by_h = {}

    for h, col in horizons_cols.items():
        if col not in df.columns:
            log(f"  ⚠  {col} not in df, skipping {h}")
            continue
        log(f"  Training classifier for {h}...")
        target_fn = (lambda c: lambda d: (d[c] > 0).astype(int))(col)
        preds = train_cls(df, avail, target_fn=target_fn)
        preds_by_h[h] = preds

        # Individual evaluation
        port = simulate(preds, regime_df, 12, CFG_5L3S)
        r = eval_config(port, 12, f"D-cls-{h}", LEVERAGE, CAPITAL)
        if r:
            show(r)

    results = []

    # Rank-normalize and blend
    def rank_normalize(p):
        out = p.copy()
        out["pred"] = out.groupby("timestamp")["pred"].rank(pct=True) - 0.5
        return out

    avail_h = {h: rank_normalize(p).rename(columns={"pred": f"pred_{h}"})
               for h, p in preds_by_h.items() if p is not None}

    if len(avail_h) < 2:
        log("  ⚠  Need at least 2 horizons")
        return results

    # Merge all horizons
    items = list(avail_h.items())
    merged = items[0][1][["timestamp", "symbol", f"pred_{items[0][0]}", "fwd_ret", "window"]]
    for h, rn in items[1:]:
        merged = merged.merge(
            rn[["timestamp", "symbol", f"pred_{h}"]],
            on=["timestamp", "symbol"], how="inner")

    # Equal weight blend
    pred_cols = [f"pred_{h}" for h in avail_h]
    blend_eq = merged.copy()
    blend_eq["pred"] = blend_eq[pred_cols].mean(axis=1)
    port = simulate(blend_eq, regime_df, 12, CFG_5L3S)
    r = eval_config(port, 12, "D-multi-eq", LEVERAGE, CAPITAL)
    if r:
        show(r)
        results.append(("D-multi-eq", r))

    # 12h-heavy: 0.6 * 12h + 0.2 * 24h + 0.2 * 48h
    if all(h in avail_h for h in ["12h", "24h", "48h"]):
        blend_12 = merged.copy()
        blend_12["pred"] = (0.6 * blend_12["pred_12h"] +
                            0.2 * blend_12["pred_24h"] +
                            0.2 * blend_12["pred_48h"])
        port = simulate(blend_12, regime_df, 12, CFG_5L3S)
        r = eval_config(port, 12, "D-multi-12h-heavy", LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append(("D-multi-12h-heavy", r))

        # 12h + 24h only
        blend_12_24 = merged.copy()
        blend_12_24["pred"] = 0.5 * blend_12_24["pred_12h"] + 0.5 * blend_12_24["pred_24h"]
        port = simulate(blend_12_24, regime_df, 12, CFG_5L3S)
        r = eval_config(port, 12, "D-multi-12h-24h", LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append(("D-multi-12h-24h", r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-E: Sample Weighting
# ═══════════════════════════════════════════════════════════════════════════════

def train_cls_weighted(df, feats, weight_fn, seeds=SEEDS):
    """Train LGB binary classifier with sample weights."""
    avail = [f for f in feats if f in df.columns]
    all_preds = []
    tz = df["timestamp"].dt.tz

    base_params = {
        "objective": "binary", "metric": "auc",
        "learning_rate": 0.03, "num_leaves": 63,
        "min_child_samples": 100, "subsample": 0.8,
        "colsample_bytree": 0.8, "lambda_l2": 1.0,
        "verbose": -1, "n_jobs": -1,
    }

    for seed in seeds:
        params = {**base_params, "seed": seed}
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
                d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)

            train_c = train[avail + ["target_binary", "timestamp", "fwd_ret_12h"]].dropna()
            val_c = val[avail + ["target_binary"]].dropna()

            # Compute weights
            weights = weight_fn(train_c)
            train_c = train_c[avail + ["target_binary"]]

            dtrain = lgb.Dataset(train_c[avail], label=train_c["target_binary"],
                                 weight=weights.values)
            dval = lgb.Dataset(val_c[avail], label=val_c["target_binary"])
            model = lgb.train(
                params, dtrain, num_boost_round=600, valid_sets=[dval],
                callbacks=[lgb.early_stopping(40, verbose=False),
                           lgb.log_evaluation(-1)])

            test_c = test[avail + ["target_binary", "timestamp", "symbol"]].dropna()
            if len(test_c) == 0:
                continue
            preds = model.predict(test_c[avail])
            fwd = test[["timestamp", "symbol", "fwd_ret_12h"]].rename(
                columns={"fwd_ret_12h": "fwd_ret"}).dropna()
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


def exp_e(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-E: Sample Weighting")
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]
    results = []

    # Weight by |return| — emphasize clear directional moves
    def weight_abs_ret(train_c):
        w = train_c["fwd_ret_12h"].abs()
        return (w / w.mean()).clip(0.1, 10.0)

    # Weight by sqrt(|return|) — less extreme weighting
    def weight_sqrt_ret(train_c):
        w = np.sqrt(train_c["fwd_ret_12h"].abs())
        return (w / w.mean()).clip(0.1, 10.0)

    # Exponential recency weighting (half-life 365 days)
    def weight_recency_365(train_c):
        max_ts = train_c["timestamp"].max()
        days_ago = (max_ts - train_c["timestamp"]).dt.total_seconds() / 86400
        w = np.exp(-np.log(2) * days_ago / 365)
        return w / w.mean()

    # Exponential recency weighting (half-life 180 days)
    def weight_recency_180(train_c):
        max_ts = train_c["timestamp"].max()
        days_ago = (max_ts - train_c["timestamp"]).dt.total_seconds() / 86400
        w = np.exp(-np.log(2) * days_ago / 180)
        return w / w.mean()

    configs = [
        ("E-wt-absret", weight_abs_ret),
        ("E-wt-sqrtret", weight_sqrt_ret),
        ("E-wt-recent365", weight_recency_365),
        ("E-wt-recent180", weight_recency_180),
    ]

    for label, wfn in configs:
        log(f"\n  {label}:")
        preds = train_cls_weighted(df, avail, wfn)
        port = simulate(preds, regime_df, 12, CFG_5L3S)
        r = eval_config(port, 12, label, LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append((label, r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-F: 5L/3S Sweep — combine best config with promising R24 settings
# ═══════════════════════════════════════════════════════════════════════════════

def exp_f(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-F: 5L/3S Combined Sweep")
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]
    preds_cls = train_cls(df, avail)
    results = []

    configs = [
        # Base 5L/3S
        ("F-5L3S-base", {**CFG_5L3S}),
        # 5L/3S + dyn_threshold variations
        ("F-5L3S-dyn0.4", {**CFG_5L3S, "dyn_threshold": 0.4}),
        ("F-5L3S-dyn0.5", {**CFG_5L3S, "dyn_threshold": 0.5}),
        ("F-5L3S-dyn0.65", {**CFG_5L3S, "dyn_threshold": 0.65}),
        ("F-5L3S-dyn0.75", {**CFG_5L3S, "dyn_threshold": 0.75}),
        # 5L/3S + trend_cutoff variations
        ("F-5L3S-cut0.85", {**CFG_5L3S, "trend_cutoff": 0.85}),
        ("F-5L3S-cut0.95", {**CFG_5L3S, "trend_cutoff": 0.95}),
        # 5L/3S with rebal 24h
        ("F-5L3S-reb24", {**CFG_5L3S, "rebal_hours": 24}),
        # 4L/3S and 5L/2S (close to winner)
        ("F-4L-3S", {**CFG_5L3S, "n_long": 4}),
        ("F-5L-2S", {**CFG_5L3S, "n_short": 2}),
        ("F-6L-2S", {**CFG_5L3S, "n_long": 6, "n_short": 2}),
    ]

    for label, cfg in configs:
        port = simulate(preds_cls, regime_df, cfg.get("rebal_hours", 12), cfg)
        r = eval_config(port, cfg.get("rebal_hours", 12), label, LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append((label, r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-G: Deeper/Shallower Trees + Regularization
# ═══════════════════════════════════════════════════════════════════════════════

def exp_g(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-G: Tree Depth + Regularization")
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]
    results = []

    configs = [
        ("G-leaves31", {"num_leaves": 31}),
        ("G-leaves63", {}),  # baseline
        ("G-leaves127", {"num_leaves": 127}),
        ("G-leaves255", {"num_leaves": 255}),
        ("G-minchild50", {"min_child_samples": 50}),
        ("G-minchild200", {"min_child_samples": 200}),
        ("G-l2-0.1", {"lambda_l2": 0.1}),
        ("G-l2-5.0", {"lambda_l2": 5.0}),
        ("G-l2-10.0", {"lambda_l2": 10.0}),
        ("G-lr0.01", {"learning_rate": 0.01}),
        ("G-lr0.05", {"learning_rate": 0.05}),
        ("G-lr0.1", {"learning_rate": 0.1}),
        ("G-sub0.6", {"subsample": 0.6}),
        ("G-col0.6", {"colsample_bytree": 0.6}),
        ("G-deep", {"num_leaves": 127, "min_child_samples": 50, "max_depth": 8}),
        ("G-shallow", {"num_leaves": 31, "min_child_samples": 200, "lambda_l2": 5.0}),
    ]

    for label, params in configs:
        preds = train_cls(df, avail, params_override=params if params else None)
        port = simulate(preds, regime_df, 12, CFG_5L3S)
        r = eval_config(port, 12, label, LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append((label, r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-H: Proper Val-Based HPO (no test leakage)
# ═══════════════════════════════════════════════════════════════════════════════

def train_cls_val_score(df, feats, params_override, seeds):
    """Train classifier and return val-set AUC (for HPO without leakage)."""
    avail = [f for f in feats if f in df.columns]
    tz = df["timestamp"].dt.tz
    val_aucs = []

    for seed in seeds:
        params = {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.03, "num_leaves": 63,
            "min_child_samples": 100, "subsample": 0.8,
            "colsample_bytree": 0.8, "lambda_l2": 1.0,
            "verbose": -1, "n_jobs": -1, "seed": seed,
        }
        if params_override:
            params.update(params_override)

        for w in WINDOWS:
            train = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz=tz)].copy()
            val = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz=tz)) &
                     (df["timestamp"] < pd.Timestamp(w["val_end"], tz=tz))].copy()
            if len(train) < 5000 or len(val) < 200:
                continue
            train = cs_rank_cols(train, avail)
            val = cs_rank_cols(val, avail)
            for d in [train, val]:
                d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)
            train_c = train[avail + ["target_binary"]].dropna()
            val_c = val[avail + ["target_binary"]].dropna()
            dtrain = lgb.Dataset(train_c[avail], label=train_c["target_binary"])
            dval = lgb.Dataset(val_c[avail], label=val_c["target_binary"])

            bst = lgb.train(
                params, dtrain, num_boost_round=600, valid_sets=[dval],
                callbacks=[lgb.early_stopping(40, verbose=False),
                           lgb.log_evaluation(-1)])
            val_aucs.append(bst.best_score["valid_0"]["auc"])

    return np.mean(val_aucs) if val_aucs else 0.5


def exp_h(df, regime_df, n_trials=30):
    log("\n" + "=" * 80)
    log("  EXP-H: Proper Val-Based HPO (no test leakage)")
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]
    quick_seeds = [0, 42]

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("lr", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child", 30, 300),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("l2", 0.01, 10.0, log=True),
            "lambda_l1": trial.suggest_float("l1", 0.0, 5.0),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
        }
        val_auc = train_cls_val_score(df, avail, params, quick_seeds)
        return val_auc

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    log(f"\n  Best trial: AUC={study.best_value:.4f}")
    log(f"  Best params: {study.best_params}")

    # Full eval on test with 5 seeds
    log("\n  Full eval with best params (5 seeds)...")
    params_map = {
        "lr": "learning_rate", "leaves": "num_leaves",
        "min_child": "min_child_samples", "subsample": "subsample",
        "colsample": "colsample_bytree", "l2": "lambda_l2",
        "l1": "lambda_l1", "max_depth": "max_depth",
    }
    full_params = {params_map.get(k, k): v for k, v in study.best_params.items()}

    preds = train_cls(df, avail, params_override=full_params, seeds=SEEDS)
    port = simulate(preds, regime_df, 12, CFG_5L3S)
    r = eval_config(port, 12, "H-val-hpo-best", LEVERAGE, CAPITAL)
    if r:
        show(r)
        for m in r.get("month_data", []):
            log(f"       {m['month']}   {m['ret']*100:+.1f}%  eq=${m['equity']:>8.0f}")

    # Control: default params
    log("\n  Control (default params, 5L/3S):")
    preds_ctrl = train_cls(df, avail)
    port_ctrl = simulate(preds_ctrl, regime_df, 12, CFG_5L3S)
    r_ctrl = eval_config(port_ctrl, 12, "H-default-5L3S", LEVERAGE, CAPITAL)
    if r_ctrl:
        show(r_ctrl)

    return r, r_ctrl


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 80)
    log("  R25 — ENSEMBLE & FEATURE ENGINEERING FOR CLASSIFICATION")
    log("=" * 80)
    log("  Base: R24-E — LGB-binary-23f, 5L/3S → Sh=2.98, Eq=$2208")
    log("  Experiments: A(ensemble) B(feat-select) C(threshold) D(multi-horizon)")
    log("               E(weighting) F(5L3S-sweep) G(tree-depth) H(val-HPO)")

    # ── Load data ─────────────────────────────────────────────────────────────
    log("\n  Loading data...")
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df = build_features_minimal(ohlcv, derivs)
    regime_df = compute_regime(df)
    log(f"  Base: {len(df):,} rows, {df['symbol'].nunique()} symbols")

    log("\n  Building features...")
    df = build_r19_features(df)
    df, _ = add_new_features(df)
    avail_23 = [f for f in FEATURES_23 if f in df.columns]
    log(f"  FEATURES_23: {len(avail_23)}/23")

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-F: 5L/3S sweep (fast, single train reuse)
    # ══════════════════════════════════════════════════════════════════════════
    results_f = exp_f(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-C: Target threshold (moderate — retrain for each threshold)
    # ══════════════════════════════════════════════════════════════════════════
    results_c = exp_c(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-B: Feature selection (moderate)
    # ══════════════════════════════════════════════════════════════════════════
    results_b = exp_b(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-E: Sample weighting (moderate)
    # ══════════════════════════════════════════════════════════════════════════
    results_e = exp_e(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-G: Tree depth + regularization (heavy — 16 configs × retraining)
    # ══════════════════════════════════════════════════════════════════════════
    results_g = exp_g(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-D: Multi-horizon classification (heavy — 3 classifiers)
    # ══════════════════════════════════════════════════════════════════════════
    results_d = exp_d(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-A: Classifier ensemble (heaviest — LGB + XGB + CatBoost)
    # ══════════════════════════════════════════════════════════════════════════
    results_a = exp_a(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-H: Val-based HPO (heaviest, last)
    # ══════════════════════════════════════════════════════════════════════════
    results_h = exp_h(df, regime_df, n_trials=30)

    # ══════════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("  FINAL RANKINGS — R25 ALL EXPERIMENTS")
    log("=" * 80)

    all_results = []

    if results_f:
        for _, r in results_f:
            all_results.append(r)
    if results_c:
        for _, r in results_c:
            all_results.append(r)
    if results_b:
        for _, r in results_b:
            all_results.append(r)
    if results_e:
        for _, r in results_e:
            all_results.append(r)
    if results_g:
        for _, r in results_g:
            all_results.append(r)
    if results_d:
        for _, r in results_d:
            all_results.append(r)
    if results_a:
        for _, r in results_a:
            all_results.append(r)
    if results_h:
        r_hpo, r_ctrl = results_h
        if r_hpo:
            all_results.append(r_hpo)
        if r_ctrl:
            all_results.append(r_ctrl)

    # Sort by sharpe
    all_results.sort(key=lambda x: x.get("sharpe", 0), reverse=True)

    log("\n  TOP-20 CONFIGURATIONS:")
    for i, r in enumerate(all_results[:20]):
        label = r.get("label", "?")
        sh = r.get("sharpe", 0)
        eq = r.get("equity", 0)
        wm = r.get("worst_m", 0)
        win = r.get("win_months", 0)
        tot = r.get("total_months", 0)
        log(f"  #{i+1:2d}: {label:30s}  Sh={sh:.2f}  Eq=${eq:.0f}  "
            f"Worst={wm*100:+.1f}%  WM={win}/{tot}")

    elapsed = (time.time() - t0) / 60
    log(f"\n  Total time: {elapsed:.1f} min")
    log("  DONE.")


if __name__ == "__main__":
    main()
