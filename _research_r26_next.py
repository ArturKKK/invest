#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R26 — Next-Generation Improvements for Classification Ensemble

R25 best: A-lgb+xgb — LGB+XGB binary cls ensemble, 5L/3S → Sh=3.36, Eq=$3371
R25 findings: LGB+XGB blend ⊳ 3-model, feature pruning useless, HPO useless

Experiments:
  A: Multi-Class Classification (3-class: strong-up / flat / strong-down)
  B: Focal Loss (focus on hard-to-classify samples)
  C: Feature Interaction Engineering (cross-products of top features)
  D: CatBoost-Huber + CLS Blend (classification + regression diversity)
  E: Larger Symbol Universe (50 coins vs 35)
  F: Asymmetric L/S Grid (n_long × n_short × threshold joint optimization)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from scipy.stats import rankdata
from pathlib import Path
import warnings, time, sys
warnings.filterwarnings("ignore")

from _research_round7 import (
    SYM_35, WINDOWS, compute_regime, simulate, eval_config, show,
)
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal
from _research_r22_models import (
    FEATURES_23, SEEDS, LEVERAGE, CAPITAL, CFG_BEST, DATA_DIR, SENT_DIR,
    log, build_r19_features, add_new_features, cs_rank_cols, run_eval,
)
from _research_r24_classification import train_cls
from _research_r25_ensemble import train_xgb_cls, train_cb_cls

# R25 winner config
CFG_5L3S = {**CFG_BEST, "n_long": 5, "n_short": 3}


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


# ═══════════════════════════════════════════════════════════════════════════════
#  Multi-class classifier
# ═══════════════════════════════════════════════════════════════════════════════

def train_multiclass_lgb(df, feats, n_classes=3, seeds=SEEDS):
    """Train LGB multi-class classifier (3 classes: strong-down / flat / strong-up)."""
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

            # Define 3-class target using quantiles of training fwd_ret
            for d in [train, val, test]:
                q33 = train["fwd_ret_12h"].quantile(0.33)
                q67 = train["fwd_ret_12h"].quantile(0.67)
                d["target_mc"] = np.where(d["fwd_ret_12h"] < q33, 0,
                                 np.where(d["fwd_ret_12h"] > q67, 2, 1)).astype(int)

            train_c = train[avail + ["target_mc"]].dropna()
            val_c = val[avail + ["target_mc"]].dropna()

            dtrain = lgb.Dataset(train_c[avail], label=train_c["target_mc"])
            dval = lgb.Dataset(val_c[avail], label=val_c["target_mc"])
            model = lgb.train(
                {"objective": "multiclass", "num_class": n_classes,
                 "metric": "multi_logloss",
                 "learning_rate": 0.03, "num_leaves": 63,
                 "min_child_samples": 100, "subsample": 0.8,
                 "colsample_bytree": 0.8, "lambda_l2": 1.0,
                 "verbose": -1, "n_jobs": -1, "seed": seed},
                dtrain, num_boost_round=600, valid_sets=[dval],
                callbacks=[lgb.early_stopping(40, verbose=False),
                           lgb.log_evaluation(-1)])

            test_c = test[avail + ["target_mc", "timestamp", "symbol"]].dropna()
            if len(test_c) == 0:
                continue
            probs = model.predict(test_c[avail])  # shape: (n, 3)
            # Signal = P(strong-up) - P(strong-down)
            preds = probs[:, 2] - probs[:, 0]

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


def train_multiclass_xgb(df, feats, n_classes=3, seeds=SEEDS):
    """Train XGB multi-class classifier."""
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
                q33 = train["fwd_ret_12h"].quantile(0.33)
                q67 = train["fwd_ret_12h"].quantile(0.67)
                d["target_mc"] = np.where(d["fwd_ret_12h"] < q33, 0,
                                 np.where(d["fwd_ret_12h"] > q67, 2, 1)).astype(int)

            train_c = train[avail + ["target_mc"]].dropna()
            val_c = val[avail + ["target_mc"]].dropna()

            dtrain = xgb.DMatrix(train_c[avail], label=train_c["target_mc"])
            dval = xgb.DMatrix(val_c[avail], label=val_c["target_mc"])
            model = xgb.train(
                {"objective": "multi:softprob", "num_class": n_classes,
                 "eval_metric": "mlogloss",
                 "learning_rate": 0.03, "max_depth": 6,
                 "min_child_weight": 100, "subsample": 0.8,
                 "colsample_bytree": 0.8, "reg_lambda": 1.0,
                 "seed": seed, "n_jobs": -1, "verbosity": 0},
                dtrain, num_boost_round=600,
                evals=[(dval, "val")],
                early_stopping_rounds=40, verbose_eval=False)

            test_c = test[avail + ["target_mc", "timestamp", "symbol"]].dropna()
            if len(test_c) == 0:
                continue
            dtest = xgb.DMatrix(test_c[avail])
            probs = model.predict(dtest).reshape(-1, n_classes)
            preds = probs[:, 2] - probs[:, 0]

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
#  Focal loss
# ═══════════════════════════════════════════════════════════════════════════════

def focal_loss_objective(gamma=2.0):
    """Return custom focal loss (grad, hess) for LGB fobj parameter."""
    def _focal(preds, dtrain):
        labels = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-preds))  # sigmoid
        p = np.clip(p, 1e-7, 1 - 1e-7)
        # focal weight
        pt = np.where(labels == 1, p, 1 - p)
        focal_w = (1 - pt) ** gamma
        # gradient & hessian of BCE weighted by focal
        grad = focal_w * (p - labels)
        hess = focal_w * p * (1 - p) * (1 + gamma * (1 - pt) * np.log(pt + 1e-10) / (1 - pt + 1e-10))
        # Simpler hessian that avoids instability
        hess = np.abs(hess) + 0.01
        return grad, hess
    return _focal


def train_focal_lgb(df, feats, gamma=2.0, seeds=SEEDS):
    """Train LGB binary classifier with focal loss."""
    avail = [f for f in feats if f in df.columns]
    all_preds = []
    tz = df["timestamp"].dt.tz
    fobj = focal_loss_objective(gamma)

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

            dtrain = lgb.Dataset(train_c[avail], label=train_c["target_binary"])
            dval = lgb.Dataset(val_c[avail], label=val_c["target_binary"])
            dtrain.set_init_score(np.zeros(len(train_c)))
            dval.set_init_score(np.zeros(len(val_c)))
            model = lgb.train(
                {"learning_rate": 0.03, "num_leaves": 63,
                 "min_child_samples": 100, "subsample": 0.8,
                 "colsample_bytree": 0.8, "lambda_l2": 1.0,
                 "verbose": -1, "n_jobs": -1, "seed": seed,
                 "objective": fobj},
                dtrain, num_boost_round=600, valid_sets=[dval],
                callbacks=[lgb.early_stopping(40, verbose=False),
                           lgb.log_evaluation(-1)])

            test_c = test[avail + ["target_binary", "timestamp", "symbol"]].dropna()
            if len(test_c) == 0:
                continue
            raw = model.predict(test_c[avail])
            preds = 1.0 / (1.0 + np.exp(-raw))  # sigmoid (focal loss outputs logits)

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
#  CatBoost Huber regression (for diversity with classification)
# ═══════════════════════════════════════════════════════════════════════════════

def train_cb_huber(df, feats, seeds=SEEDS):
    """Train CatBoost with Huber loss on rank target → regression signal."""
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

            # Rank target (same as R22 production CB)
            for d in [train, val, test]:
                d["target_rank"] = d.groupby("timestamp")["fwd_ret_12h"].rank(pct=True) - 0.5

            train_c = train[avail + ["target_rank"]].dropna()
            val_c = val[avail + ["target_rank"]].dropna()

            model = cb.CatBoostRegressor(
                loss_function="Huber:delta=0.5",
                learning_rate=0.03, depth=6,
                l2_leaf_reg=3.0, subsample=0.8,
                random_seed=seed, verbose=0,
                iterations=600, early_stopping_rounds=40)
            model.fit(train_c[avail], train_c["target_rank"],
                      eval_set=(val_c[avail], val_c["target_rank"]),
                      verbose=0)

            test_c = test[avail + ["target_rank", "timestamp", "symbol"]].dropna()
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature interaction builder
# ═══════════════════════════════════════════════════════════════════════════════

def add_interaction_features(df):
    """Add cross-product interaction features from domain knowledge."""
    interactions = [
        ("oi_chg_12h", "taker_cvd_12h", "ix_oi_x_flow"),      # positioning × flow
        ("ret_12h", "rvol_12h", "ix_ret_x_rvol"),              # vol-adjusted momentum
        ("ls_divergence", "oi_zscore", "ix_ls_x_oi"),          # crowding
        ("pct_coins_up_12h", "ret_12h", "ix_breadth_x_ret"),   # breadth × individual
        ("iv_rv_spread", "rvol_24h", "ix_iv_x_rvol"),          # vol premium × realized
        ("ret_24h", "ret_48h", "ix_mom24_x_mom48"),            # momentum consistency
        ("taker_cvd_12h", "ls_divergence", "ix_cvd_x_ls"),     # flow × positioning
        ("residual_12h", "oi_chg_12h", "ix_resid_x_oi"),       # idiosyncratic × OI
    ]
    added = []
    for c1, c2, name in interactions:
        if c1 in df.columns and c2 in df.columns:
            df[name] = df[c1] * df[c2]
            added.append(name)
    log(f"  [INTERACT] Added {len(added)} interaction features")
    return df, added


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-A: Multi-Class Classification
# ═══════════════════════════════════════════════════════════════════════════════

def exp_a(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-A: Multi-Class Classification (3-class)")
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]
    results = []

    # Multi-class LGB
    log("  Training multi-class LGB...")
    preds_mc_lgb = train_multiclass_lgb(df, avail)
    if preds_mc_lgb is not None:
        port = simulate(preds_mc_lgb, regime_df, 12, CFG_5L3S)
        r = eval_config(port, 12, "A-mc-lgb", LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append(("A-mc-lgb", r))

    # Multi-class XGB
    log("  Training multi-class XGB...")
    preds_mc_xgb = train_multiclass_xgb(df, avail)
    if preds_mc_xgb is not None:
        port = simulate(preds_mc_xgb, regime_df, 12, CFG_5L3S)
        r = eval_config(port, 12, "A-mc-xgb", LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append(("A-mc-xgb", r))

    # Multi-class ensemble (LGB + XGB)
    if preds_mc_lgb is not None and preds_mc_xgb is not None:
        blend = blend_predictions([preds_mc_lgb, preds_mc_xgb])
        port = simulate(blend, regime_df, 12, CFG_5L3S)
        r = eval_config(port, 12, "A-mc-ens", LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append(("A-mc-ens", r))

    # Also blend multi-class with R25 binary (cross-paradigm diversity)
    log("  Training binary LGB (R25 baseline for blending)...")
    preds_bin_lgb = train_cls(df, avail)
    preds_bin_xgb = train_xgb_cls(df, avail)

    if preds_mc_lgb is not None and preds_bin_lgb is not None and preds_bin_xgb is not None:
        # Binary ensemble (R25 method)
        preds_bin_ens = blend_predictions([preds_bin_lgb, preds_bin_xgb])
        port = simulate(preds_bin_ens, regime_df, 12, CFG_5L3S)
        r = eval_config(port, 12, "A-bin-ens-ctrl", LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append(("A-bin-ens-ctrl", r))

        # Cross-paradigm: binary-ens + multi-class-ens
        if preds_mc_xgb is not None:
            preds_mc_ens = blend_predictions([preds_mc_lgb, preds_mc_xgb])
            cross = blend_predictions([preds_bin_ens, preds_mc_ens])
            port = simulate(cross, regime_df, 12, CFG_5L3S)
            r = eval_config(port, 12, "A-cross-bin+mc", LEVERAGE, CAPITAL)
            if r:
                show(r)
                results.append(("A-cross-bin+mc", r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-B: Focal Loss
# ═══════════════════════════════════════════════════════════════════════════════

def exp_b(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-B: Focal Loss (γ sweep)")
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]
    results = []

    for gamma in [1.0, 2.0, 3.0, 5.0]:
        log(f"  Training focal LGB (γ={gamma})...")
        preds = train_focal_lgb(df, avail, gamma=gamma)
        if preds is not None:
            label = f"B-focal-g{gamma:.0f}"
            port = simulate(preds, regime_df, 12, CFG_5L3S)
            r = eval_config(port, 12, label, LEVERAGE, CAPITAL)
            if r:
                show(r)
                results.append((label, r))

    # Best focal + XGB ensemble
    if results:
        best_gamma = max(results, key=lambda x: x[1].get("sharpe", 0))
        best_g = float(best_gamma[0].split("g")[1])
        log(f"  Best focal γ={best_g}, blending with XGB...")
        preds_focal = train_focal_lgb(df, avail, gamma=best_g)
        preds_xgb = train_xgb_cls(df, avail)
        if preds_focal is not None and preds_xgb is not None:
            blend = blend_predictions([preds_focal, preds_xgb])
            port = simulate(blend, regime_df, 12, CFG_5L3S)
            r = eval_config(port, 12, "B-focal+xgb", LEVERAGE, CAPITAL)
            if r:
                show(r)
                results.append(("B-focal+xgb", r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-C: Feature Interactions
# ═══════════════════════════════════════════════════════════════════════════════

def exp_c(df_orig, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-C: Feature Interaction Engineering")
    log("=" * 80)

    df = df_orig.copy()
    df, ix_feats = add_interaction_features(df)

    feats_extended = [f for f in FEATURES_23 if f in df.columns] + ix_feats
    log(f"  Extended features: {len(feats_extended)} (23 + {len(ix_feats)} interactions)")

    results = []

    # LGB cls with interactions
    log("  Training LGB cls + interactions...")
    preds_lgb = train_cls(df, feats_extended)
    if preds_lgb is not None:
        port = simulate(preds_lgb, regime_df, 12, CFG_5L3S)
        r = eval_config(port, 12, "C-lgb-ix", LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append(("C-lgb-ix", r))

    # XGB cls with interactions
    log("  Training XGB cls + interactions...")
    preds_xgb = train_xgb_cls(df, feats_extended)
    if preds_xgb is not None:
        port = simulate(preds_xgb, regime_df, 12, CFG_5L3S)
        r = eval_config(port, 12, "C-xgb-ix", LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append(("C-xgb-ix", r))

    # Ensemble with interactions
    if preds_lgb is not None and preds_xgb is not None:
        blend = blend_predictions([preds_lgb, preds_xgb])
        port = simulate(blend, regime_df, 12, CFG_5L3S)
        r = eval_config(port, 12, "C-ens-ix", LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append(("C-ens-ix", r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-D: CatBoost Huber + CLS Blend
# ═══════════════════════════════════════════════════════════════════════════════

def exp_d(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-D: CatBoost-Huber + CLS Blend (cls × regression diversity)")
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]
    results = []

    # CB Huber standalone
    log("  Training CatBoost Huber (regression)...")
    preds_cb_h = train_cb_huber(df, avail)
    if preds_cb_h is not None:
        port = simulate(preds_cb_h, regime_df, 12, CFG_5L3S)
        r = eval_config(port, 12, "D-cb-huber", LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append(("D-cb-huber", r))

    # R25 binary ensemble (LGB+XGB cls)
    log("  Training R25 binary ensemble...")
    preds_lgb = train_cls(df, avail)
    preds_xgb = train_xgb_cls(df, avail)

    if preds_lgb is not None and preds_xgb is not None:
        preds_cls_ens = blend_predictions([preds_lgb, preds_xgb])

        # Blend cls-ens + CB-Huber (50/50)
        if preds_cb_h is not None:
            blend5050 = blend_predictions([preds_cls_ens, preds_cb_h])
            port = simulate(blend5050, regime_df, 12, CFG_5L3S)
            r = eval_config(port, 12, "D-cls+huber-50", LEVERAGE, CAPITAL)
            if r:
                show(r)
                results.append(("D-cls+huber-50", r))

            # CLS-heavy (67/33)
            blend6733 = blend_predictions([preds_cls_ens, preds_cb_h], weights=[0.67, 0.33])
            port = simulate(blend6733, regime_df, 12, CFG_5L3S)
            r = eval_config(port, 12, "D-cls+huber-67", LEVERAGE, CAPITAL)
            if r:
                show(r)
                results.append(("D-cls+huber-67", r))

            # Huber-heavy (33/67)
            blend3367 = blend_predictions([preds_cls_ens, preds_cb_h], weights=[0.33, 0.67])
            port = simulate(blend3367, regime_df, 12, CFG_5L3S)
            r = eval_config(port, 12, "D-cls+huber-33", LEVERAGE, CAPITAL)
            if r:
                show(r)
                results.append(("D-cls+huber-33", r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-E: Larger Symbol Universe
# ═══════════════════════════════════════════════════════════════════════════════

SYM_50 = set(SYM_35) | {
    "IMX/USDT", "SUI/USDT", "SEI/USDT", "TIA/USDT", "JUP/USDT",
    "WLD/USDT", "ORDI/USDT", "STX/USDT", "RENDER/USDT", "FET/USDT",
    "PEPE/USDT", "WIF/USDT", "BONK/USDT", "FLOKI/USDT", "ETC/USDT",
}

def exp_e(regime_df):
    log("\n" + "=" * 80)
    log("  EXP-E: Larger Symbol Universe (50 coins)")
    log("=" * 80)

    results = []

    # Load fresh data with 50 symbols
    log("  Loading data for 50 symbols...")
    from _research_round7 import load_data
    df50 = load_data(SYM_50)
    log(f"  Loaded: {len(df50):,} rows, {df50['symbol'].nunique()} symbols")

    df50 = build_r19_features(df50)
    df50, _ = add_new_features(df50)
    avail = [f for f in FEATURES_23 if f in df50.columns]
    log(f"  features: {len(avail)}/23")

    # Compute regime from 50-coin data
    regime_df_50 = compute_regime(df50)

    # LGB+XGB ensemble on 50 coins, test with various L/S configs
    log("  Training LGB cls (50 symbols)...")
    preds_lgb = train_cls(df50, avail)
    log("  Training XGB cls (50 symbols)...")
    preds_xgb = train_xgb_cls(df50, avail)

    if preds_lgb is None or preds_xgb is None:
        log("  ⚠  Training failed for 50-symbol universe")
        return results

    preds_ens = blend_predictions([preds_lgb, preds_xgb])

    # Test with various L/S sizes (more coins → more positions?)
    for nl, ns in [(5, 3), (7, 4), (8, 5), (10, 5)]:
        cfg = {**CFG_5L3S, "n_long": nl, "n_short": ns}
        label = f"E-50sym-{nl}L{ns}S"
        port = simulate(preds_ens, regime_df_50, 12, cfg)
        r = eval_config(port, 12, label, LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append((label, r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-F: Asymmetric L/S Grid Optimization
# ═══════════════════════════════════════════════════════════════════════════════

def exp_f(df, regime_df, preds_cls_ens):
    log("\n" + "=" * 80)
    log("  EXP-F: Asymmetric L/S Grid (n_long × n_short × threshold)")
    log("=" * 80)

    results = []

    for nl in [3, 4, 5, 6, 7]:
        for ns in [2, 3, 4, 5]:
            for dt in [0.4, 0.5625, 0.7]:
                cfg = {**CFG_5L3S, "n_long": nl, "n_short": ns, "dyn_threshold": dt}
                label = f"F-{nl}L{ns}S-dt{dt}"
                port = simulate(preds_cls_ens, regime_df, 12, cfg)
                r = eval_config(port, 12, label, LEVERAGE, CAPITAL)
                if r:
                    results.append((label, r))

    # Sort and show top-10
    results.sort(key=lambda x: x[1].get("sharpe", 0), reverse=True)
    log(f"\n  Top-10 L/S configs (of {len(results)}):")
    for i, (label, r) in enumerate(results[:10]):
        show(r)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="Skip EXP-D,A,C (already done)")
    args = ap.parse_args()

    t0 = time.time()
    log("=" * 80)
    log("  R26 — Next-Generation Improvements")
    log("  Base: R25 A-lgb+xgb, Sh=3.36, Eq=$3371")
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
    avail_23 = [f for f in FEATURES_23 if f in df.columns]
    log(f"  FEATURES_23: {len(avail_23)}/23")

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-D: CatBoost-Huber + CLS Blend (high ROI, low risk)
    # ══════════════════════════════════════════════════════════════════════════
    results_d = exp_d(df, regime_df) if not args.resume else []

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-A: Multi-Class Classification
    # ══════════════════════════════════════════════════════════════════════════
    results_a = exp_a(df, regime_df) if not args.resume else []

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-C: Feature Interactions
    # ══════════════════════════════════════════════════════════════════════════
    results_c = exp_c(df, regime_df) if not args.resume else []

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-B: Focal Loss
    # ══════════════════════════════════════════════════════════════════════════
    results_b = exp_b(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-E: Larger Symbol Universe (50 coins)
    # ══════════════════════════════════════════════════════════════════════════
    results_e = exp_e(regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-F: Asymmetric L/S Grid (reuses preds from EXP-A or EXP-D)
    # ══════════════════════════════════════════════════════════════════════════
    # Use the R25-control preds from EXP-A (binary LGB+XGB ens)
    avail = [f for f in FEATURES_23 if f in df.columns]
    log("\n  Training R25 binary ensemble for L/S grid (EXP-F)...")
    preds_lgb = train_cls(df, avail)
    preds_xgb = train_xgb_cls(df, avail)
    if preds_lgb is not None and preds_xgb is not None:
        preds_cls_ens = blend_predictions([preds_lgb, preds_xgb])
        results_f = exp_f(df, regime_df, preds_cls_ens)
    else:
        results_f = []

    # ══════════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("  FINAL RANKINGS — R26 ALL EXPERIMENTS")
    log("=" * 80)

    all_results = []
    for bucket in [results_d, results_a, results_c, results_b, results_e, results_f]:
        if bucket:
            for _, r in bucket:
                all_results.append(r)

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
