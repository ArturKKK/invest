#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R28c — Forward Feature Selection (add ONE feature at a time)

R28/R28b proved that adding 5+ features destroys performance.
This script tests adding EXACTLY ONE candidate feature to FEATURES_23.

If no single feature improves Sh=3.39, the conclusion is definitive:
FEATURES_23 is the local optimum for tree models on this data.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path
import warnings, time, sys
warnings.filterwarnings("ignore")

from _research_round7 import (
    SYM_35, WINDOWS, compute_regime, simulate, eval_config, show,
)
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal
from _research_r22_models import (
    FEATURES_23, SEEDS, LEVERAGE, CAPITAL, CFG_BEST, DATA_DIR, SENT_DIR,
    log, build_r19_features, add_new_features, cs_rank_cols,
)

CFG_6L3S = {**CFG_BEST, "n_long": 6, "n_short": 3, "dyn_threshold": 0.7}


# Candidates: per-coin features NOT in FEATURES_23, excluding toxic funding/market-level
CANDIDATES = [
    # Derivatives (per-coin)
    "top_ls_ratio_zscore",
    "global_ls_ratio_zscore",
    "premium_zscore",
    "taker_zscore",
    "oi_ret_diverge",
    # TA (per-coin)
    "adx",
    "rsi_14",
    "bb_pband_20",
    "mfi_14",
    "ret_skew_24h",
    "ret_kurt_24h",
    # Momentum (per-coin)
    "ret_168h",
    "ret_1h",
    "ret_4h",
    "dist_from_high_24h",
    # Vol (per-coin)
    "vol_of_vol",
    "rvol_168h",
    "vol_ratio_24h",
    # Interactions (per-coin)
    "premium_zscore_12h",
    "oi_velocity",
    "taker_imb_z",
    "obv_ma_ratio_24",
    "vwap_dev_24h",
]


def add_extra_features(df):
    """Ensure reversal and vol_crush exist."""
    if "ret_12h" in df.columns and "ret_48h" in df.columns:
        df["reversal_12v48"] = df["ret_12h"] - df["ret_48h"]
    if "rvol_12h" in df.columns and "rvol_168h" in df.columns:
        df["vol_crush"] = df["rvol_12h"] / (df["rvol_168h"] + 1e-10)
    return df


def train_lgb_cls(df, feats, seeds=SEEDS):
    avail = [f for f in feats if f in df.columns]
    all_preds = []
    tz = df["timestamp"].dt.tz

    for seed in seeds:
        params = {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.03, "num_leaves": 63,
            "min_child_samples": 100, "subsample": 0.8,
            "colsample_bytree": 0.8, "lambda_l2": 1.0,
            "verbose": -1, "n_jobs": -1, "seed": seed,
        }
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
            model = lgb.train(params, dtrain, num_boost_round=600,
                              valid_sets=[dval],
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


def train_xgb_cls(df, feats, seeds=SEEDS):
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


def ensemble_preds(lgb_preds, xgb_preds):
    if lgb_preds is None or xgb_preds is None:
        return lgb_preds if lgb_preds is not None else xgb_preds
    merged = lgb_preds.rename(columns={"pred": "pred_lgb"}).merge(
        xgb_preds[["timestamp", "symbol", "pred"]].rename(columns={"pred": "pred_xgb"}),
        on=["timestamp", "symbol"], how="inner")
    merged["rank_lgb"] = merged.groupby("timestamp")["pred_lgb"].rank(pct=True) - 0.5
    merged["rank_xgb"] = merged.groupby("timestamp")["pred_xgb"].rank(pct=True) - 0.5
    merged["pred"] = 0.5 * merged["rank_lgb"] + 0.5 * merged["rank_xgb"]
    return merged[["timestamp", "symbol", "pred", "fwd_ret", "window"]]


def run_forward_selection(df, regime_df):
    """Test adding ONE candidate at a time to FEATURES_23."""
    results = []

    # Baseline
    log("\n  [BASELINE] FEATURES_23 (23 features)")
    p_lgb = train_lgb_cls(df, FEATURES_23)
    p_xgb = train_xgb_cls(df, FEATURES_23)
    p_ens = ensemble_preds(p_lgb, p_xgb)
    port = simulate(p_ens, regime_df, 12, CFG_6L3S)
    r = eval_config(port, 12, "baseline-23f", LEVERAGE, CAPITAL)
    if r:
        show(r)
        results.append(("BASELINE (23f)", r["sharpe"], r["equity"]))

    # Test each candidate
    for feat in CANDIDATES:
        if feat not in df.columns:
            log(f"\n  [SKIP] {feat} not in data")
            results.append((f"+{feat}", None, None))
            continue

        feats_24 = FEATURES_23 + [feat]
        log(f"\n  [TEST] +{feat} ({len(feats_24)}f)")
        t0 = time.time()

        p_lgb = train_lgb_cls(df, feats_24)
        p_xgb = train_xgb_cls(df, feats_24)
        p_ens = ensemble_preds(p_lgb, p_xgb)
        port = simulate(p_ens, regime_df, 12, CFG_6L3S)
        r = eval_config(port, 12, f"+{feat}", LEVERAGE, CAPITAL)
        if r:
            show(r)
            delta = r["sharpe"] - results[0][1] if results[0][1] else 0
            log(f"  Delta vs baseline: {delta:+.2f}")
            results.append((f"+{feat}", r["sharpe"], r["equity"]))
        else:
            results.append((f"+{feat}", None, None))
        log(f"  Time: {time.time()-t0:.0f}s")

    return results


def main():
    log("=" * 80)
    log("  R28c — Forward Feature Selection")
    log(f"  Date: {pd.Timestamp.now()}")
    log("=" * 80)

    t_start = time.time()
    log("\n  Loading data...")
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()

    df = build_features_minimal(ohlcv, derivs)
    df = build_r19_features(df)
    df, _ = add_new_features(df)
    df = add_extra_features(df)
    df = df[df["symbol"].isin(SYM_35)].copy()
    log(f"  Final: {len(df):,} rows, {len(df.columns)} cols")

    # Check availability
    avail_cands = [f for f in CANDIDATES if f in df.columns]
    miss_cands = [f for f in CANDIDATES if f not in df.columns]
    log(f"  Candidates: {len(avail_cands)}/{len(CANDIDATES)} available")
    if miss_cands:
        log(f"  Missing: {miss_cands}")

    regime_df = compute_regime(df)
    log(f"  Load time: {time.time()-t_start:.0f}s")

    results = run_forward_selection(df, regime_df)

    # Summary
    log("\n" + "=" * 80)
    log("  R28c FORWARD SELECTION RESULTS")
    log("=" * 80)
    baseline_sh = results[0][1] if results else None
    for name, sh, eq in results:
        if sh is not None:
            delta = sh - baseline_sh if baseline_sh and name != "BASELINE (23f)" else 0
            marker = "***" if delta > 0 else ""
            log(f"  {name:35s}  Sh={sh:.2f}  Eq=${eq:.0f}  delta={delta:+.2f} {marker}")
        else:
            log(f"  {name:35s}  SKIP/FAIL")

    # Best additions
    additions = [(n, s, e) for n, s, e in results if s is not None and n != "BASELINE (23f)"]
    if additions:
        additions.sort(key=lambda x: x[1], reverse=True)
        log("\n  TOP 5 features by Sharpe:")
        for n, s, e in additions[:5]:
            log(f"    {n:35s}  Sh={s:.2f}  delta={s - baseline_sh:+.2f}")

    total = time.time() - t_start
    log(f"\n  Total runtime: {total/60:.1f} min")
    log("  Done.")


if __name__ == "__main__":
    main()
