#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R28b — Careful Feature Expansion (lessons from R28)

R28 finding: naive expansion destroys signal because:
  1. funding_rate_binance + cum_funding_* (5 correlated funding features)
     dominate the model with 4× importance of genuine features → overfitting
  2. market-level features (fng_zscore, vix_zscore, dvol_zscore) become NOISE
     after cs_rank() because all coins have the same value → random tied ranks

Strategy: add ONLY genuinely per-coin features in small, uncorrelated groups.
           No raw funding. No market-level features.

Experiments:
  F: BASELINE — FEATURES_23 (control, Sh=3.39)
  G: +5 per-coin derivatives (top_ls_zscore, global_ls_zscore, premium_zscore,
                               taker_zscore, oi_ret_diverge)
  H: +5 TA/vol per-coin (adx, rsi_14, bb_pband_20, ret_skew_24h, mfi_14)
  I: +5 momentum extras (ret_168h, ret_1h, ret_4h, dist_from_high_24h, reversal_12v48)
  J: +5 vol/risk (vol_of_vol, vol_crush, rvol_168h, btc_beta_168h, vol_ratio_24h)
  K: G+H combined (best of deriv + TA)
  L: G+H+I combined (deriv + TA + momentum)
  M: G+H+I+J combined (all safe per-coin)
  N: Best subset + stronger regularization (min_child=200, num_leaves=31)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
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

CFG_6L3S = {**CFG_BEST, "n_long": 6, "n_short": 3, "dyn_threshold": 0.7}


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature groups — all genuinely PER-COIN cross-sectional features
# ═══════════════════════════════════════════════════════════════════════════════

GROUP_G = [  # derivatives per-coin
    "top_ls_ratio_zscore", "global_ls_ratio_zscore",
    "premium_zscore", "taker_zscore", "oi_ret_diverge",
]

GROUP_H = [  # TA per-coin
    "adx", "rsi_14", "bb_pband_20", "ret_skew_24h", "mfi_14",
]

GROUP_I = [  # momentum per-coin
    "ret_168h", "ret_1h", "ret_4h", "dist_from_high_24h", "reversal_12v48",
]

GROUP_J = [  # vol/risk per-coin
    "vol_of_vol", "vol_crush", "rvol_168h", "btc_beta_168h", "vol_ratio_24h",
]

FEATS_G = FEATURES_23 + GROUP_G      # 28
FEATS_H = FEATURES_23 + GROUP_H      # 28
FEATS_I = FEATURES_23 + GROUP_I      # 28
FEATS_J = FEATURES_23 + GROUP_J      # 28
FEATS_K = FEATURES_23 + GROUP_G + GROUP_H      # 33
FEATS_L = FEATURES_23 + GROUP_G + GROUP_H + GROUP_I  # 38
FEATS_M = FEATURES_23 + GROUP_G + GROUP_H + GROUP_I + GROUP_J  # 43


def add_r28b_features(df):
    """Add extra per-coin features on top of add_new_features output."""
    added = []

    # Reversal: short-term vs long-term momentum
    if "ret_12h" in df.columns and "ret_48h" in df.columns:
        df["reversal_12v48"] = df["ret_12h"] - df["ret_48h"]
        added.append("reversal_12v48")

    # Vol crush: short vol / long vol — low = mean-reversion regime
    if "rvol_12h" in df.columns and "rvol_168h" in df.columns:
        df["vol_crush"] = df["rvol_12h"] / (df["rvol_168h"] + 1e-10)
        added.append("vol_crush")

    added = [f for f in added if f in df.columns]
    log(f"  [R28b] Extra features added: {len(added)}: {added}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  Trainers
# ═══════════════════════════════════════════════════════════════════════════════

def train_lgb_cls(df, feats, seeds=SEEDS, params_override=None):
    """LGB binary classifier, walk-forward, 5 seeds."""
    avail = [f for f in feats if f in df.columns]
    log(f"    LGB features: {len(avail)}/{len(feats)}")
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
    """XGB binary classifier, walk-forward, 5 seeds."""
    avail = [f for f in feats if f in df.columns]
    log(f"    XGB features: {len(avail)}/{len(feats)}")
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


def ensemble_preds(lgb_preds, xgb_preds, w_lgb=0.5):
    if lgb_preds is None or xgb_preds is None:
        return lgb_preds if lgb_preds is not None else xgb_preds
    merged = lgb_preds.rename(columns={"pred": "pred_lgb"}).merge(
        xgb_preds[["timestamp", "symbol", "pred"]].rename(columns={"pred": "pred_xgb"}),
        on=["timestamp", "symbol"], how="inner")
    merged["rank_lgb"] = merged.groupby("timestamp")["pred_lgb"].rank(pct=True) - 0.5
    merged["rank_xgb"] = merged.groupby("timestamp")["pred_xgb"].rank(pct=True) - 0.5
    merged["pred"] = w_lgb * merged["rank_lgb"] + (1 - w_lgb) * merged["rank_xgb"]
    return merged[["timestamp", "symbol", "pred", "fwd_ret", "window"]]


def run_eval(preds, regime_df, label, cfg=None):
    if preds is None:
        log(f"  ⚠  {label}: no predictions")
        return None
    if cfg is None:
        cfg = CFG_6L3S
    port = simulate(preds, regime_df, 12, cfg)
    if port is None:
        log(f"  ⚠  {label}: simulate returned None")
        return None
    r = eval_config(port, 12, label, LEVERAGE, CAPITAL)
    if r:
        show(r)
    return r


def run_exp(df, regime_df, feats, name, params_override=None):
    """Run a single experiment: LGB + XGB + ensemble."""
    avail = [f for f in feats if f in df.columns]
    log(f"\n{'='*80}")
    log(f"  {name}: {len(avail)} features")
    log(f"{'='*80}")
    log(f"  Features: {avail}")

    t0 = time.time()
    preds_lgb = train_lgb_cls(df, avail, params_override=params_override)
    r_lgb = run_eval(preds_lgb, regime_df, f"{name}-lgb")

    preds_xgb = train_xgb_cls(df, avail)
    r_xgb = run_eval(preds_xgb, regime_df, f"{name}-xgb")

    preds_ens = ensemble_preds(preds_lgb, preds_xgb)
    r_ens = run_eval(preds_ens, regime_df, f"{name}-ens")

    log(f"  {name} runtime: {time.time()-t0:.0f}s")
    return r_ens


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="*", default=None,
                        help="Experiments: f g h i j k l m n (default: all)")
    args = parser.parse_args()

    exps_to_run = set(args.exp) if args.exp else set("fghijklmn")

    log("=" * 80)
    log("  R28b — Careful Feature Expansion")
    log(f"  Date: {pd.Timestamp.now()}")
    log("=" * 80)

    # ── Load data ────────────────────────────────────────────────
    t_start = time.time()
    log("\n  Loading data...")
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    log(f"  OHLCV: {len(ohlcv):,} rows, {ohlcv['symbol'].nunique()} symbols")

    log("  Building features...")
    df = build_features_minimal(ohlcv, derivs)
    df = build_r19_features(df)
    df, _ = add_new_features(df)
    df = add_r28b_features(df)
    df = df[df["symbol"].isin(SYM_35)].copy()
    log(f"  Final: {len(df):,} rows, {len(df.columns)} cols")

    # Check availability
    all_cols = set(df.columns)
    for name, flist in [("FEATS_G", FEATS_G), ("FEATS_H", FEATS_H),
                        ("FEATS_I", FEATS_I), ("FEATS_J", FEATS_J),
                        ("FEATS_K", FEATS_K), ("FEATS_L", FEATS_L),
                        ("FEATS_M", FEATS_M)]:
        avail = [f for f in flist if f in all_cols]
        miss = [f for f in flist if f not in all_cols]
        log(f"  {name}: {len(avail)}/{len(flist)}" + (f" miss={miss}" if miss else ""))

    regime_df = compute_regime(df)
    log(f"  Load time: {time.time()-t_start:.0f}s\n")

    # ── Experiments ──────────────────────────────────────────────
    results = {}

    if "f" in exps_to_run:
        results["F"] = run_exp(df, regime_df, FEATURES_23, "F-baseline-23f")

    if "g" in exps_to_run:
        results["G"] = run_exp(df, regime_df, FEATS_G, "G-deriv-28f")

    if "h" in exps_to_run:
        results["H"] = run_exp(df, regime_df, FEATS_H, "H-ta-28f")

    if "i" in exps_to_run:
        results["I"] = run_exp(df, regime_df, FEATS_I, "I-mom-28f")

    if "j" in exps_to_run:
        results["J"] = run_exp(df, regime_df, FEATS_J, "J-vol-28f")

    if "k" in exps_to_run:
        results["K"] = run_exp(df, regime_df, FEATS_K, "K-deriv+ta-33f")

    if "l" in exps_to_run:
        results["L"] = run_exp(df, regime_df, FEATS_L, "L-deriv+ta+mom-38f")

    if "m" in exps_to_run:
        results["M"] = run_exp(df, regime_df, FEATS_M, "M-all-safe-43f")

    # N: best expanded set but with stronger regularization
    if "n" in exps_to_run:
        strong_reg = {"num_leaves": 31, "min_child_samples": 200, "lambda_l2": 3.0}
        results["N-43f-reg"] = run_exp(df, regime_df, FEATS_M, "N-all-reg-43f",
                                       params_override=strong_reg)
        results["N-38f-reg"] = run_exp(df, regime_df, FEATS_L, "N-38f-reg",
                                       params_override=strong_reg)
        results["N-33f-reg"] = run_exp(df, regime_df, FEATS_K, "N-33f-reg",
                                       params_override=strong_reg)

    # ── Summary ──────────────────────────────────────────────────
    log("\n" + "=" * 80)
    log("  R28b RESULTS SUMMARY")
    log("=" * 80)
    for name, r in results.items():
        if r and "sharpe" in r:
            log(f"  {name:25s}  Sh={r['sharpe']:.2f}  Eq=${r['equity']:.0f}  "
                f"Worst={r.get('worst_month', 0)*100:.1f}%")
        else:
            log(f"  {name:25s}  FAILED")

    total = time.time() - t_start
    log(f"\n  Total R28b runtime: {total/60:.1f} min")
    log("  Done.")


if __name__ == "__main__":
    main()
