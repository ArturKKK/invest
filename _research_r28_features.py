#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R28 — Feature Expansion: Use the ~80 features we already have

R27 finding: all target/model changes failed. Signal plateau at Sh=3.39.
Reality: CLS model uses only 23 of ~80 available features in research pipeline.
→ Expand the feature set. Zero cost — data already downloaded.

Build pipeline:  build_features_minimal()  →  ~55 cols
                 build_r19_features()      →  +10 cols (atr, dvol, breadth, calendar)
                 add_new_features()        →  +15 cols (macro, fng, TA, vol-of-vol)
                 add_r28_features()        →  +20 cols (funding, premium, reversal, extra)

Feature Groups (cumulative):
  FEATURES_23: baseline (23 features)
  FEATURES_35: + derivatives extras (funding, premium, oi velocity, taker z)
  FEATURES_50: + macro/sentiment (VIX, FNG, DXY) + TA (rsi, adx, bbands, skew)
  FEATURES_65: + volume/momentum extras (vol_of_vol, reversal, btc_beta, dist_from_high)

Experiments:
  A: FEATURES_35 — add derivative features already computed
  B: FEATURES_50 — add macro + TA 
  C: FEATURES_65 — full expansion
  D: Feature selection via importance pruning from C
  E: (A-D) with XGB ensemble
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path
import warnings, time, sys, argparse, json
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
#  Extended feature lists
# ═══════════════════════════════════════════════════════════════════════════════

# Group 1: derivative extras (already in build_features_minimal output)
DERIV_EXTRAS = [
    "funding_rate_binance", "funding_zscore", "cum_funding_24h", "cum_funding_72h",
    "premium_index", "premium_zscore",
    "taker_imbalance", "taker_zscore", "taker_cvd_4h",
    "oi_chg_1h", "oi_chg_4h", "oi_ret_diverge",
    "top_ls_ratio_zscore", "global_ls_ratio_zscore",
]

# Group 2: macro + sentiment + TA (from add_new_features / build_r19_features)
MACRO_TA = [
    "vix_close", "vix_zscore", "dxy_ret_7d",
    "fng_value", "fng_zscore",
    "rsi_14", "bb_pband_20", "adx", "mfi_14",
    "ret_skew_24h", "ret_kurt_24h",
    "dvol_zscore",
]

# Group 3: momentum + volume + extra (from build_features_minimal + add_new_features)
EXTRA_MOM_VOL = [
    "ret_1h", "ret_4h", "ret_168h",
    "rvol_168h", "vol_ratio_12h", "vol_ratio_24h",
    "mom_z_12h",
    "dist_from_high_24h",
    "btc_ret_1h", "btc_ret_4h", "btc_ret_12h", "btc_ret_24h",
    "btc_beta_168h", "btc_outperform",
    "premium_zscore_12h", "oi_velocity", "taker_imb_z", "vol_of_vol",
    "vwap_dev_24h", "obv_ma_ratio_24",
    "cum_funding_168h",
    "funding_x_mom_12h",
]

FEATURES_35 = FEATURES_23 + DERIV_EXTRAS
FEATURES_50 = FEATURES_35 + MACRO_TA
FEATURES_65 = FEATURES_50 + EXTRA_MOM_VOL


def add_r28_features(df):
    """Add extra features on top of add_new_features output."""
    added = []

    # Funding rate change (acceleration)
    if "funding_rate_binance" in df.columns:
        df["funding_accel"] = df.groupby("symbol")["funding_rate_binance"].diff()
        added.append("funding_accel")

    # OI momentum (12h vs 24h OI change ratio)
    if "oi_chg_12h" in df.columns and "oi_chg_24h" in df.columns:
        df["oi_momentum"] = df["oi_chg_12h"] - df["oi_chg_24h"] / 2
        added.append("oi_momentum")

    # Taker persistence (ratio of cvd_12h to cvd_24h — 1.0 = all recent)
    if "taker_cvd_12h" in df.columns and "taker_cvd_24h" in df.columns:
        df["taker_persistence"] = df["taker_cvd_12h"] / (df["taker_cvd_24h"] + 1e-10)
        df["taker_persistence"] = df["taker_persistence"].clip(-5, 5)
        added.append("taker_persistence")

    # Funding × OI divergence: high funding + rising OI = leverage buildup
    if "funding_zscore" in df.columns and "oi_chg_12h" in df.columns:
        df["funding_oi_interact"] = df["funding_zscore"] * df["oi_chg_12h"]
        added.append("funding_oi_interact")

    # Reversal signals: short-term vs long-term momentum
    if "ret_4h" in df.columns and "ret_24h" in df.columns:
        df["reversal_4v24"] = df["ret_4h"] - df["ret_24h"]
        added.append("reversal_4v24")
    if "ret_12h" in df.columns and "ret_48h" in df.columns:
        df["reversal_12v48"] = df["ret_12h"] - df["ret_48h"]
        added.append("reversal_12v48")

    # Vol crush: short vol / long vol — low = mean-reversion regime
    if "rvol_12h" in df.columns and "rvol_168h" in df.columns:
        df["vol_crush"] = df["rvol_12h"] / (df["rvol_168h"] + 1e-10)
        added.append("vol_crush")

    # Cross-sectional ranks (will be ranked later but raw rank has info)
    if "ret_12h" in df.columns:
        df["ret_12h_raw_rank"] = df.groupby("timestamp")["ret_12h"].rank(pct=True) - 0.5
        added.append("ret_12h_raw_rank")

    added = [f for f in added if f in df.columns]
    log(f"  [R28] Extra features added: {len(added)}: {added}")
    return df


# All features including R28 extras
FEATURES_R28_EXTRAS = [
    "funding_accel", "oi_momentum", "taker_persistence",
    "funding_oi_interact", "reversal_4v24", "reversal_12v48",
    "vol_crush", "ret_12h_raw_rank",
]

FEATURES_75 = FEATURES_65 + FEATURES_R28_EXTRAS


# ═══════════════════════════════════════════════════════════════════════════════
#  Trainers (LGB + XGB binary classification — same as R25/R26)
# ═══════════════════════════════════════════════════════════════════════════════

def train_lgb_cls(df, feats, seeds=SEEDS):
    """LGB binary classifier, walk-forward, 5 seeds."""
    avail = [f for f in feats if f in df.columns]
    log(f"    LGB features: {len(avail)}/{len(feats)}")
    all_preds = []
    importances = {}
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

            params = {
                "objective": "binary", "metric": "auc",
                "learning_rate": 0.03, "num_leaves": 63,
                "min_child_samples": 100, "subsample": 0.8,
                "colsample_bytree": 0.8, "lambda_l2": 1.0,
                "seed": seed, "verbose": -1, "n_jobs": -1,
            }

            dtrain = lgb.Dataset(train_c[avail], label=train_c["target_binary"])
            dval = lgb.Dataset(val_c[avail], label=val_c["target_binary"])
            model = lgb.train(params, dtrain, num_boost_round=600,
                              valid_sets=[dval],
                              callbacks=[lgb.early_stopping(40, verbose=False),
                                         lgb.log_evaluation(-1)])

            # Collect importances
            imp = dict(zip(model.feature_name(), model.feature_importance("gain")))
            for k, v in imp.items():
                importances[k] = importances.get(k, 0) + v

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
        return None, importances
    combined = pd.concat(all_preds, ignore_index=True)
    result = (combined.groupby(["timestamp", "symbol"])
              .agg(pred=("pred", "mean"), fwd_ret=("fwd_ret", "first"),
                   window=("window", "first"))
              .reset_index())
    return result, importances


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
    """Combine LGB + XGB predictions via rank averaging."""
    if lgb_preds is None or xgb_preds is None:
        return lgb_preds if lgb_preds is not None else xgb_preds

    merged = lgb_preds.rename(columns={"pred": "pred_lgb"}).merge(
        xgb_preds[["timestamp", "symbol", "pred"]].rename(columns={"pred": "pred_xgb"}),
        on=["timestamp", "symbol"], how="inner")

    # Rank-normalize each model's predictions per timestamp
    merged["rank_lgb"] = merged.groupby("timestamp")["pred_lgb"].rank(pct=True) - 0.5
    merged["rank_xgb"] = merged.groupby("timestamp")["pred_xgb"].rank(pct=True) - 0.5
    merged["pred"] = w_lgb * merged["rank_lgb"] + (1 - w_lgb) * merged["rank_xgb"]

    return merged[["timestamp", "symbol", "pred", "fwd_ret", "window"]]


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def run_eval(preds, regime_df, label, cfg=None, verbose_months=False):
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


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════

def exp_a(df, regime_df):
    """EXP-A: FEATURES_35 — add derivative extras to baseline 23."""
    log("\n" + "=" * 80)
    log("  EXP-A: FEATURES_35 (23 + 12 derivative extras)")
    log("=" * 80)
    avail = [f for f in FEATURES_35 if f in df.columns]
    log(f"  Available: {len(avail)}/{len(FEATURES_35)}: {avail}")

    t0 = time.time()
    preds_lgb, imp = train_lgb_cls(df, avail)
    r_lgb = run_eval(preds_lgb, regime_df, "A-lgb-35f")

    preds_xgb = train_xgb_cls(df, avail)
    r_xgb = run_eval(preds_xgb, regime_df, "A-xgb-35f")

    preds_ens = ensemble_preds(preds_lgb, preds_xgb)
    r_ens = run_eval(preds_ens, regime_df, "A-ens-35f")

    log(f"  A runtime: {time.time()-t0:.0f}s")
    log(f"  Top features (gain): {sorted(imp.items(), key=lambda x:-x[1])[:10]}")
    return r_ens, imp


def exp_b(df, regime_df):
    """EXP-B: FEATURES_50 — add macro + TA."""
    log("\n" + "=" * 80)
    log("  EXP-B: FEATURES_50 (35 + 12 macro/TA)")
    log("=" * 80)
    avail = [f for f in FEATURES_50 if f in df.columns]
    log(f"  Available: {len(avail)}/{len(FEATURES_50)}: {avail}")

    t0 = time.time()
    preds_lgb, imp = train_lgb_cls(df, avail)
    r_lgb = run_eval(preds_lgb, regime_df, "B-lgb-50f")

    preds_xgb = train_xgb_cls(df, avail)
    r_xgb = run_eval(preds_xgb, regime_df, "B-xgb-50f")

    preds_ens = ensemble_preds(preds_lgb, preds_xgb)
    r_ens = run_eval(preds_ens, regime_df, "B-ens-50f")

    log(f"  B runtime: {time.time()-t0:.0f}s")
    log(f"  Top features (gain): {sorted(imp.items(), key=lambda x:-x[1])[:15]}")
    return r_ens, imp


def exp_c(df, regime_df):
    """EXP-C: FEATURES_65 — full expansion."""
    log("\n" + "=" * 80)
    log("  EXP-C: FEATURES_65 (50 + 15 momentum/volume extras)")
    log("=" * 80)
    avail = [f for f in FEATURES_65 if f in df.columns]
    log(f"  Available: {len(avail)}/{len(FEATURES_65)}: {avail}")

    t0 = time.time()
    preds_lgb, imp = train_lgb_cls(df, avail)
    r_lgb = run_eval(preds_lgb, regime_df, "C-lgb-65f")

    preds_xgb = train_xgb_cls(df, avail)
    r_xgb = run_eval(preds_xgb, regime_df, "C-xgb-65f")

    preds_ens = ensemble_preds(preds_lgb, preds_xgb)
    r_ens = run_eval(preds_ens, regime_df, "C-ens-65f")

    log(f"  C runtime: {time.time()-t0:.0f}s")
    log(f"  Top features (gain): {sorted(imp.items(), key=lambda x:-x[1])[:20]}")
    return r_ens, imp


def exp_d(df, regime_df):
    """EXP-D: FEATURES_75 — full + R28 engineered extras."""
    log("\n" + "=" * 80)
    log("  EXP-D: FEATURES_75 (65 + 8 R28 engineered)")
    log("=" * 80)
    avail = [f for f in FEATURES_75 if f in df.columns]
    log(f"  Available: {len(avail)}/{len(FEATURES_75)}: {avail}")

    t0 = time.time()
    preds_lgb, imp = train_lgb_cls(df, avail)
    r_lgb = run_eval(preds_lgb, regime_df, "D-lgb-75f")

    preds_xgb = train_xgb_cls(df, avail)
    r_xgb = run_eval(preds_xgb, regime_df, "D-xgb-75f")

    preds_ens = ensemble_preds(preds_lgb, preds_xgb)
    r_ens = run_eval(preds_ens, regime_df, "D-ens-75f")

    log(f"  D runtime: {time.time()-t0:.0f}s")
    log(f"  Top features (gain): {sorted(imp.items(), key=lambda x:-x[1])[:20]}")
    return r_ens, imp


def exp_e(df, regime_df, best_imp):
    """EXP-E: Importance-pruned feature set from best expansion."""
    log("\n" + "=" * 80)
    log("  EXP-E: Importance-pruned features (top N from best)")
    log("=" * 80)

    # Sort by importance, try top 30 and top 40
    sorted_feats = sorted(best_imp.items(), key=lambda x: -x[1])
    log(f"  Full importance ranking:")
    for i, (f, v) in enumerate(sorted_feats):
        log(f"    {i+1:3d}. {f:30s} = {v:.0f}")

    results = {}
    for top_n in [25, 30, 35, 40, 45]:
        feats = [f for f, _ in sorted_feats[:top_n]]
        if len(feats) < top_n:
            log(f"  Only {len(feats)} features available, skipping top-{top_n}")
            continue
        log(f"\n  --- Top-{top_n} features ---")

        preds_lgb, _ = train_lgb_cls(df, feats)
        preds_xgb = train_xgb_cls(df, feats)
        preds_ens = ensemble_preds(preds_lgb, preds_xgb)
        r = run_eval(preds_ens, regime_df, f"E-ens-top{top_n}")
        results[top_n] = r

    return results


def exp_f(df, regime_df):
    """EXP-F: Baseline — FEATURES_23 (control)."""
    log("\n" + "=" * 80)
    log("  EXP-F: BASELINE (FEATURES_23 — control)")
    log("=" * 80)
    avail = [f for f in FEATURES_23 if f in df.columns]
    log(f"  Available: {len(avail)}/{len(FEATURES_23)}")

    t0 = time.time()
    preds_lgb, imp = train_lgb_cls(df, avail)
    r_lgb = run_eval(preds_lgb, regime_df, "F-lgb-23f-baseline")

    preds_xgb = train_xgb_cls(df, avail)
    r_xgb = run_eval(preds_xgb, regime_df, "F-xgb-23f-baseline")

    preds_ens = ensemble_preds(preds_lgb, preds_xgb)
    r_ens = run_eval(preds_ens, regime_df, "F-ens-23f-baseline")

    log(f"  F runtime: {time.time()-t0:.0f}s")
    return r_ens, imp


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="*", default=None,
                        help="Experiments to run: a b c d e f (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed experiments")
    args = parser.parse_args()

    exps_to_run = set(args.exp) if args.exp else {"a", "b", "c", "d", "e", "f"}

    log("=" * 80)
    log("  R28 — Feature Expansion Experiments")
    log(f"  Date: {pd.Timestamp.now()}")
    log("=" * 80)

    # ── Load data ────────────────────────────────────────────────
    t_start = time.time()
    log("\n  Loading data...")
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    log(f"  OHLCV: {len(ohlcv):,} rows, {ohlcv['symbol'].nunique()} symbols")

    log("  Building base features (build_features_minimal)...")
    df = build_features_minimal(ohlcv, derivs)
    log(f"  After minimal: {len(df):,} rows, {len(df.columns)} cols")

    log("  Adding R19 features...")
    df = build_r19_features(df)
    log(f"  After R19: {len(df.columns)} cols")

    log("  Adding new features (macro, FNG, TA)...")
    df, added = add_new_features(df)
    log(f"  After add_new: {len(df.columns)} cols")

    log("  Adding R28 engineered features...")
    df = add_r28_features(df)
    log(f"  After R28: {len(df.columns)} cols")

    # Filter to SYM_35
    df = df[df["symbol"].isin(SYM_35)].copy()
    log(f"  Filtered to SYM_35: {len(df):,} rows, {df['symbol'].nunique()} symbols")

    # Check feature availability
    all_feats = set(df.columns)
    for name, flist in [("FEATURES_23", FEATURES_23), ("FEATURES_35", FEATURES_35),
                        ("FEATURES_50", FEATURES_50), ("FEATURES_65", FEATURES_65),
                        ("FEATURES_75", FEATURES_75)]:
        avail = [f for f in flist if f in all_feats]
        missing = [f for f in flist if f not in all_feats]
        log(f"  {name}: {len(avail)}/{len(flist)} available"
            + (f" — missing: {missing}" if missing else ""))

    # Regime
    regime_df = compute_regime(df)

    load_time = time.time() - t_start
    log(f"\n  Data loaded in {load_time:.0f}s")

    # ── Run experiments ──────────────────────────────────────────
    results = {}
    all_importances = {}

    # F first (baseline)
    if "f" in exps_to_run:
        r, imp = exp_f(df, regime_df)
        results["F"] = r
        all_importances["F"] = imp

    if "a" in exps_to_run:
        r, imp = exp_a(df, regime_df)
        results["A"] = r
        all_importances["A"] = imp

    if "b" in exps_to_run:
        r, imp = exp_b(df, regime_df)
        results["B"] = r
        all_importances["B"] = imp

    if "c" in exps_to_run:
        r, imp = exp_c(df, regime_df)
        results["C"] = r
        all_importances["C"] = imp

    if "d" in exps_to_run:
        r, imp = exp_d(df, regime_df)
        results["D"] = r
        all_importances["D"] = imp

    if "e" in exps_to_run:
        # Use best importance from C or D
        best_exp = "D" if "D" in all_importances else ("C" if "C" in all_importances else None)
        if best_exp:
            e_results = exp_e(df, regime_df, all_importances[best_exp])
            results["E"] = e_results
        else:
            log("  ⚠ Skipping E — need C or D importances first")

    # ── Summary ──────────────────────────────────────────────────
    log("\n" + "=" * 80)
    log("  R28 RESULTS SUMMARY")
    log("=" * 80)

    for name, r in results.items():
        if isinstance(r, dict) and "sharpe" in r:
            log(f"  {name:20s}  Sh={r['sharpe']:.2f}  Eq=${r['equity']:.0f}  "
                f"Worst={r.get('worst_month', 0)*100:.1f}%")
        elif isinstance(r, dict):
            # E returns dict of top_n → result
            for top_n, sub_r in r.items():
                if sub_r and "sharpe" in sub_r:
                    log(f"  {name}-top{top_n:>2d}          Sh={sub_r['sharpe']:.2f}  "
                        f"Eq=${sub_r['equity']:.0f}  "
                        f"Worst={sub_r.get('worst_month', 0)*100:.1f}%")
        else:
            log(f"  {name:20s}  FAILED")

    total_time = time.time() - t_start
    log(f"\n  Total R28 runtime: {total_time/60:.1f} min")
    log("  Done.")


if __name__ == "__main__":
    main()
