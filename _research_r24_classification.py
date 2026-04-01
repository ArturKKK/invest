#!/usr/bin/env python3
"""
R24 — Building on Classification Breakthrough

R23 key finding: binary classification (P(ret>0)) → Sh=2.94 beats regression Sh=2.80
R24 systematically explores classification variations.

Base: R23-G — LGB-binary-23f, cutoff=0.9, 12h rebal, 6L/3S → Sh=2.94, Eq=$1997

Experiments:
  A: HPO for classification (Optuna, tune binary-specific params)
  B: Classification + EMA smoothing combos
  C: Blend regression + classification predictions
  D: LambdaRank (LGB learning-to-rank objective)
  E: Classification with different n_long / n_short
  F: Quantile regression (predict median)
  G: Classification + stickiness / conviction weighting
  H: Classification with cross-sectional binary (ret > cs_median, not 0)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
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
    log, build_r19_features, add_new_features, cs_rank_cols,
    train_lgb, run_eval,
)
from _research_r23_signals import train_lgb_classification


# ═══════════════════════════════════════════════════════════════════════════════
#  Generic classification trainer with custom params
# ═══════════════════════════════════════════════════════════════════════════════

def train_cls(df, feats, params_override=None, seeds=SEEDS, target_fn=None):
    """
    Train LGB binary classifier with optional param overrides.
    target_fn: function(df) -> Series of 0/1 labels. Default: (fwd_ret_12h > 0).
    """
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
    if params_override:
        base_params.update(params_override)

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
                if target_fn is not None:
                    d["target_binary"] = target_fn(d)
                else:
                    d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)

            train_c = train[avail + ["target_binary"]].dropna()
            val_c = val[avail + ["target_binary"]].dropna()

            dtrain = lgb.Dataset(train_c[avail], label=train_c["target_binary"])
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


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-A: HPO for Classification (Optuna)
# ═══════════════════════════════════════════════════════════════════════════════

def exp_a(df, regime_df, n_trials=30):
    log("\n" + "=" * 80)
    log("  EXP-A: Classification HPO (Optuna, n=%d)" % n_trials)
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]
    quick_seeds = [0, 42]  # Faster trials with 2 seeds

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
        preds = train_cls(df, avail, params_override=params, seeds=quick_seeds)
        r = run_eval(preds, regime_df, f"HPO-{trial.number}", verbose_months=False)
        if r is None:
            return -99
        return r["sharpe"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    log(f"\n  Best trial: Sh={study.best_value:.2f}")
    log(f"  Best params: {study.best_params}")

    # Full evaluation with 5 seeds using best params
    log("\n  Full eval with best params (5 seeds)...")
    best_params = study.best_params
    params_map = {
        "lr": "learning_rate", "leaves": "num_leaves",
        "min_child": "min_child_samples", "subsample": "subsample",
        "colsample": "colsample_bytree", "l2": "lambda_l2",
        "l1": "lambda_l1", "max_depth": "max_depth",
    }
    full_params = {params_map.get(k, k): v for k, v in best_params.items()}
    preds_best = train_cls(df, avail, params_override=full_params, seeds=SEEDS)
    r_best = run_eval(preds_best, regime_df, "A-hpo-cls-best")
    return r_best


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-B: Classification + EMA smoothing combos
# ═══════════════════════════════════════════════════════════════════════════════

def exp_b(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-B: Classification + EMA / Stickiness / Conviction")
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]
    preds_cls = train_cls(df, avail)
    results = []

    configs = [
        ("B-cls-base", {}),
        ("B-cls-ema2", {"signal_ema": 2}),
        ("B-cls-ema3", {"signal_ema": 3}),
        ("B-cls-ema4", {"signal_ema": 4}),
        ("B-cls-stick0.02", {"stickiness": 0.02}),
        ("B-cls-stick0.05", {"stickiness": 0.05}),
        ("B-cls-conviction", {"conviction_weight": True}),
        ("B-cls-ema3-stick0.02", {"signal_ema": 3, "stickiness": 0.02}),
        ("B-cls-ema3-conv", {"signal_ema": 3, "conviction_weight": True}),
    ]

    for label, extra_cfg in configs:
        cfg = {**CFG_BEST, **extra_cfg}
        port = simulate(preds_cls, regime_df, 12, cfg)
        r = eval_config(port, 12, label, LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append((label, r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-C: Blend regression + classification
# ═══════════════════════════════════════════════════════════════════════════════

def exp_c(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-C: Blend Regression + Classification")
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]

    log("  Training regression...")
    preds_reg = train_lgb(df, avail)
    log("  Training classification...")
    preds_cls = train_cls(df, avail)

    if preds_reg is None or preds_cls is None:
        log("  ⚠  Missing predictions")
        return []

    results = []

    # Rank-normalize both for fair blending
    def rank_normalize(preds_df):
        p = preds_df.copy()
        p["pred"] = p.groupby("timestamp")["pred"].rank(pct=True) - 0.5
        return p

    preds_reg_r = rank_normalize(preds_reg)
    preds_cls_r = rank_normalize(preds_cls)

    # Merge on timestamp+symbol
    merged = preds_reg_r[["timestamp", "symbol", "pred", "fwd_ret", "window"]].rename(
        columns={"pred": "pred_reg"})
    merged = merged.merge(
        preds_cls_r[["timestamp", "symbol", "pred"]].rename(columns={"pred": "pred_cls"}),
        on=["timestamp", "symbol"], how="inner")

    for alpha in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        blend = merged.copy()
        blend["pred"] = alpha * blend["pred_cls"] + (1 - alpha) * blend["pred_reg"]
        label = f"C-blend-cls{alpha:.1f}"
        port = simulate(blend, regime_df, 12, CFG_BEST)
        r = eval_config(port, 12, label, LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append((alpha, r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-D: LambdaRank (learning to rank)
# ═══════════════════════════════════════════════════════════════════════════════

def train_lambdarank(df, feats, seeds=SEEDS):
    """Train LGB with lambdarank objective."""
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

            # For lambdarank: relevance labels = quintile bins (0-4)
            for d in [train, val, test]:
                d["target_rank"] = d.groupby("timestamp")["fwd_ret_12h"].transform(
                    lambda x: pd.qcut(x, q=5, labels=False, duplicates="drop")
                ).fillna(2).astype(int)

            train_c = train[avail + ["target_rank", "timestamp"]].dropna()
            val_c = val[avail + ["target_rank", "timestamp"]].dropna()

            # Query groups: how many items per timestamp
            train_groups = train_c.groupby("timestamp").size().values
            val_groups = val_c.groupby("timestamp").size().values

            dtrain = lgb.Dataset(train_c[avail], label=train_c["target_rank"],
                                 group=train_groups)
            dval = lgb.Dataset(val_c[avail], label=val_c["target_rank"],
                               group=val_groups)
            model = lgb.train(
                {"objective": "lambdarank", "metric": "ndcg",
                 "ndcg_eval_at": [3, 6, 9],
                 "learning_rate": 0.03, "num_leaves": 63,
                 "min_child_samples": 100, "subsample": 0.8,
                 "colsample_bytree": 0.8, "lambda_l2": 1.0,
                 "seed": seed, "verbose": -1, "n_jobs": -1},
                dtrain, num_boost_round=600, valid_sets=[dval],
                callbacks=[lgb.early_stopping(40, verbose=False),
                           lgb.log_evaluation(-1)])

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


def exp_d(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-D: LambdaRank (Learning to Rank)")
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]
    preds = train_lambdarank(df, avail)
    r = run_eval(preds, regime_df, "D-lambdarank")
    return r


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-E: Classification with different n_long / n_short
# ═══════════════════════════════════════════════════════════════════════════════

def exp_e(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-E: Classification with Different Position Counts")
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]
    preds_cls = train_cls(df, avail)
    results = []

    configs = [
        ("E-3L-2S", 3, 2),
        ("E-4L-2S", 4, 2),
        ("E-5L-3S", 5, 3),
        ("E-6L-3S", 6, 3),  # baseline
        ("E-6L-4S", 6, 4),
        ("E-7L-3S", 7, 3),
        ("E-7L-4S", 7, 4),
        ("E-8L-4S", 8, 4),
        ("E-8L-5S", 8, 5),
        ("E-9L-4S", 9, 4),
        ("E-10L-5S", 10, 5),
        ("E-5L-5S", 5, 5),  # symmetric
        ("E-4L-4S", 4, 4),
    ]

    for label, nl, ns in configs:
        cfg = {**CFG_BEST, "n_long": nl, "n_short": ns}
        port = simulate(preds_cls, regime_df, 12, cfg)
        r = eval_config(port, 12, label, LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append((label, r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-F: Quantile Regression
# ═══════════════════════════════════════════════════════════════════════════════

def train_quantile(df, feats, alpha=0.5, seeds=SEEDS):
    """Train LGB with quantile regression (predict median by default)."""
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
                d["target_rank"] = d.groupby("timestamp")["fwd_ret_12h"].rank(pct=True) - 0.5

            train_c = train[avail + ["target_rank"]].dropna()
            val_c = val[avail + ["target_rank"]].dropna()

            dtrain = lgb.Dataset(train_c[avail], label=train_c["target_rank"])
            dval = lgb.Dataset(val_c[avail], label=val_c["target_rank"])
            model = lgb.train(
                {"objective": "quantile", "alpha": alpha,
                 "metric": "quantile",
                 "learning_rate": 0.03, "num_leaves": 63,
                 "min_child_samples": 100, "subsample": 0.8,
                 "colsample_bytree": 0.8, "lambda_l2": 1.0,
                 "seed": seed, "verbose": -1, "n_jobs": -1},
                dtrain, num_boost_round=600, valid_sets=[dval],
                callbacks=[lgb.early_stopping(40, verbose=False),
                           lgb.log_evaluation(-1)])

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


def exp_f(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-F: Quantile Regression")
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]
    results = []

    for alpha in [0.25, 0.5, 0.6, 0.75]:
        log(f"\n  F-quantile-{alpha}:")
        preds = train_quantile(df, avail, alpha=alpha)
        r = run_eval(preds, regime_df, f"F-quantile-{alpha}")
        if r:
            results.append((alpha, r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-G: Classification + Stickiness and Conviction
#  (tested in B already, but here we also try with different cutoff/threshold)
# ═══════════════════════════════════════════════════════════════════════════════

def exp_g(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-G: Classification + Cutoff / Threshold Variations")
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]
    preds_cls = train_cls(df, avail)
    results = []

    configs = [
        ("G-cut0.85", {"trend_cutoff": 0.85}),
        ("G-cut0.90", {"trend_cutoff": 0.90}),  # baseline
        ("G-cut0.95", {"trend_cutoff": 0.95}),
        ("G-cut1.0", {"trend_cutoff": 1.0}),  # no cutoff
        ("G-dyn0.4", {"dyn_threshold": 0.4}),
        ("G-dyn0.5", {"dyn_threshold": 0.5}),
        ("G-dyn0.5625", {"dyn_threshold": 0.5625}),  # baseline
        ("G-dyn0.65", {"dyn_threshold": 0.65}),
        ("G-dyn0.75", {"dyn_threshold": 0.75}),
        ("G-cut0.95-dyn0.5", {"trend_cutoff": 0.95, "dyn_threshold": 0.5}),
        ("G-cut0.95-dyn0.65", {"trend_cutoff": 0.95, "dyn_threshold": 0.65}),
        ("G-cut1.0-dyn0.7", {"trend_cutoff": 1.0, "dyn_threshold": 0.7}),
    ]

    for label, extra_cfg in configs:
        cfg = {**CFG_BEST, **extra_cfg}
        port = simulate(preds_cls, regime_df, 12, cfg)
        r = eval_config(port, 12, label, LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append((label, r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-H: Cross-sectional binary (ret > cs_median instead of ret > 0)
# ═══════════════════════════════════════════════════════════════════════════════

def exp_h(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-H: Cross-Sectional Binary Classification")
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]

    def target_cs_median(d):
        """Target: 1 if ret > cross-sectional median at that timestamp."""
        return d.groupby("timestamp")["fwd_ret_12h"].transform(
            lambda x: (x > x.median()).astype(int))

    def target_cs_q60(d):
        """Target: 1 if ret > 60th percentile (top 40%)."""
        return d.groupby("timestamp")["fwd_ret_12h"].transform(
            lambda x: (x > x.quantile(0.6)).astype(int))

    def target_cs_q40(d):
        """Target: 1 if ret > 40th percentile (top 60%)."""
        return d.groupby("timestamp")["fwd_ret_12h"].transform(
            lambda x: (x > x.quantile(0.4)).astype(int))

    results = []

    for label, fn in [
        ("H-cs-median", target_cs_median),
        ("H-cs-q60", target_cs_q60),
        ("H-cs-q40", target_cs_q40),
    ]:
        log(f"\n  {label}:")
        preds = train_cls(df, avail, target_fn=fn)
        r = run_eval(preds, regime_df, label)
        if r:
            results.append((label, r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 80)
    log("  R24 — BUILDING ON CLASSIFICATION BREAKTHROUGH")
    log("=" * 80)
    log("  Base: R23-G — LGB-binary-23f → Sh=2.94, Eq=$1997")
    log("  Experiments: A(HPO) B(EMA/stick) C(blend) D(lambdarank)")
    log("               E(positions) F(quantile) G(cutoff/thresh) H(cs-binary)")

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
    # EXP-B: Classification + EMA/stickiness (fast, no retraining)
    # ══════════════════════════════════════════════════════════════════════════
    results_b = exp_b(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-E: Position count variations (fast, no retraining)
    # ══════════════════════════════════════════════════════════════════════════
    results_e = exp_e(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-G: Cutoff / threshold variations (fast, no retraining)
    # ══════════════════════════════════════════════════════════════════════════
    results_g = exp_g(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-H: Cross-sectional binary targets
    # ══════════════════════════════════════════════════════════════════════════
    results_h = exp_h(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-D: LambdaRank
    # ══════════════════════════════════════════════════════════════════════════
    r_d = exp_d(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-F: Quantile regression
    # ══════════════════════════════════════════════════════════════════════════
    results_f = exp_f(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-C: Blend regression + classification
    # ══════════════════════════════════════════════════════════════════════════
    results_c = exp_c(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-A: HPO (heaviest, last)
    # ══════════════════════════════════════════════════════════════════════════
    r_a = exp_a(df, regime_df, n_trials=30)

    # ══════════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("  FINAL RANKINGS — R24 ALL EXPERIMENTS")
    log("=" * 80)

    all_results = []
    if results_b:
        for _, r in results_b:
            all_results.append(r)
    if results_e:
        for _, r in results_e:
            all_results.append(r)
    if results_g:
        for _, r in results_g:
            all_results.append(r)
    if results_h:
        for _, r in results_h:
            all_results.append(r)
    if r_d:
        all_results.append(r_d)
    if results_f:
        for _, r in results_f:
            all_results.append(r)
    if results_c:
        for _, r in results_c:
            all_results.append(r)
    if r_a:
        all_results.append(r_a)

    if all_results:
        ranked = sorted(all_results, key=lambda r: -r["sharpe"])
        for i, r in enumerate(ranked, 1):
            delta = r["sharpe"] - 2.94  # vs R23-G baseline
            flag = "✅" if delta > 0 else ("⚠️" if delta > -0.10 else "❌")
            log(f"  #{i:2d} {flag} {r['name']:<50s} "
                f"Sh={r['sharpe']:+.2f} Eq=${r['equity']:.0f} "
                f"WM={r['win_months']}/{r['total_months']} "
                f"Wr={r['worst_m']*100:+.1f}% Δ={delta:+.2f}")

    log(f"\n  R23-G baseline: Sh=2.94, Eq=$1997")
    log(f"  R20-C baseline: Sh=2.80, Eq=$2096")

    elapsed = time.time() - t0
    log(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
