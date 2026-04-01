#!/usr/bin/env python3
"""R22 continuation — run only EXP-O (rank+ridge) and EXP-P that crashed."""
import numpy as np, pandas as pd, time, sys
import lightgbm as lgb, xgboost as xgb, catboost as cb
from sklearn.linear_model import Ridge
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from _research_round7 import SYM_35, WINDOWS, compute_regime, simulate, eval_config, show
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal
from _research_r22_models import (
    FEATURES_23, SEEDS, LEVERAGE, CAPITAL, CFG_BEST,
    build_r19_features, add_new_features, cs_rank_cols,
    train_lgb, train_xgb, train_catboost, ic_quick, run_eval, log,
)

def main():
    t0 = time.time()
    log("=" * 80)
    log("  R22-CONTINUE: O-rank, O-ridge, EXP-P")
    log("=" * 80)

    log("\n  Loading data...")
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df = build_features_minimal(ohlcv, derivs)
    regime_df = compute_regime(df)
    log(f"  {len(df):,} rows")

    log("  Building features...")
    df = build_r19_features(df)
    df, new_feats = add_new_features(df)
    avail_23 = [f for f in FEATURES_23 if f in df.columns]

    # --- Train all 3 models ---
    log("\n  Training LGB...")
    preds_lgb = train_lgb(df, avail_23)
    log("  Training XGB...")
    preds_xgb = train_xgb(df, avail_23)
    log("  Training CB...")
    preds_cb = train_catboost(df, avail_23)

    # --- EXP-O: Ensemble ---
    log("\n" + "=" * 80)
    log("  EXP-O: Stacked Ensemble (continued)")
    log("=" * 80)

    ens = preds_lgb[["timestamp", "symbol", "fwd_ret", "window"]].copy()
    ens = ens.merge(preds_lgb[["timestamp", "symbol", "pred"]].rename(
        columns={"pred": "pred_lgb"}), on=["timestamp", "symbol"])
    ens = ens.merge(preds_xgb[["timestamp", "symbol", "pred"]].rename(
        columns={"pred": "pred_xgb"}), on=["timestamp", "symbol"], how="inner")
    ens = ens.merge(preds_cb[["timestamp", "symbol", "pred"]].rename(
        columns={"pred": "pred_cb"}), on=["timestamp", "symbol"], how="inner")
    log(f"  Ensemble: {len(ens)} rows")

    # Rank-then-average
    ens_rank = ens.copy()
    for col in ["pred_lgb", "pred_xgb", "pred_cb"]:
        ens_rank[col] = ens_rank[col].astype(np.float64)
    for col in ["pred_lgb", "pred_xgb", "pred_cb"]:
        ens_rank[col] = ens_rank.groupby("timestamp")[col].rank(pct=True) - 0.5
    ens_rank["pred"] = (ens_rank["pred_lgb"] + ens_rank["pred_xgb"] + ens_rank["pred_cb"]) / 3
    r_rank = run_eval(ens_rank, regime_df, "O-rank-ensemble")

    # Ridge stacking
    log("\n  Ridge stacking (walk-forward)...")
    stacked = ens.copy()
    for col in ["pred_lgb", "pred_xgb", "pred_cb"]:
        stacked[col] = stacked[col].astype(np.float64)
    stacked["pred_meta"] = np.nan

    w1 = stacked[stacked["window"] == "W1"]
    w2 = stacked[stacked["window"] == "W2"]
    w3 = stacked[stacked["window"] == "W3"]

    if len(w1) > 100 and len(w2) > 100:
        ridge1 = Ridge(alpha=1.0)
        X1 = w1[["pred_lgb", "pred_xgb", "pred_cb"]].values
        y1 = w1["fwd_ret"].values
        ridge1.fit(X1, y1)
        stacked.loc[w2.index, "pred_meta"] = ridge1.predict(
            w2[["pred_lgb", "pred_xgb", "pred_cb"]].values)
        log(f"    Ridge W1→W2: coef={ridge1.coef_}")

    if len(w1) > 100 and len(w2) > 100 and len(w3) > 100:
        w12 = pd.concat([w1, w2])
        ridge2 = Ridge(alpha=1.0)
        X12 = w12[["pred_lgb", "pred_xgb", "pred_cb"]].values
        y12 = w12["fwd_ret"].values
        ridge2.fit(X12, y12)
        stacked.loc[w3.index, "pred_meta"] = ridge2.predict(
            w3[["pred_lgb", "pred_xgb", "pred_cb"]].values)
        log(f"    Ridge W1+W2→W3: coef={ridge2.coef_}")

    stacked_valid = stacked.dropna(subset=["pred_meta"]).copy()
    if len(stacked_valid) > 100:
        stacked_valid["pred"] = stacked_valid["pred_meta"]
        r_ridge = run_eval(stacked_valid, regime_df, "O-ridge-stack")
    else:
        log("  ⚠ Ridge: not enough OOS data")
        r_ridge = None

    # --- EXP-P: New features ---
    log("\n" + "=" * 80)
    log("  EXP-P: New Features")
    log("=" * 80)

    avail_new = [f for f in new_feats if f in df.columns]
    coverage = {f: df[f].notna().mean() for f in avail_new}
    good_new = [f for f in avail_new if coverage[f] > 0.5]
    log(f"  {len(good_new)} features with >50% coverage")

    # Control
    log("\n  P0: Control 23f...")
    preds0 = train_lgb(df, avail_23)
    r0 = run_eval(preds0, regime_df, "P-ctrl-23f")

    # All new
    if good_new:
        feats_all = avail_23 + good_new
        log(f"\n  P1: 23f + {len(good_new)} new = {len(feats_all)}f...")
        preds1 = train_lgb(df, feats_all)
        ic1 = ic_quick(preds1, f"LGB-{len(feats_all)}f")
        r1 = run_eval(preds1, regime_df, f"P-all-{len(feats_all)}f")

    # Feature groups
    groups = {
        "fng": [f for f in good_new if f in ["fng_value", "fng_zscore"]],
        "deriv": [f for f in good_new if f in ["premium_zscore_12h", "oi_velocity", "taker_imb_z"]],
        "vol": [f for f in good_new if f in ["vol_of_vol", "vol_ratio_24h"]],
        "ta": [f for f in good_new if f in ["rsi_14", "bb_pband_20", "adx", "mfi_14",
                                              "ret_skew_24h", "ret_kurt_24h",
                                              "vwap_dev_24h", "obv_ma_ratio_24"]],
        "mom": [f for f in good_new if f in ["dist_from_high_24h", "ret_168h"]],
    }
    for gname, gfeats in groups.items():
        if not gfeats:
            continue
        feats_g = avail_23 + gfeats
        log(f"\n  P-{gname}: 23f + {gfeats} = {len(feats_g)}f...")
        preds_g = train_lgb(df, feats_g)
        run_eval(preds_g, regime_df, f"P-{gname}-{len(feats_g)}f")

    log(f"\n  Done in {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")

if __name__ == "__main__":
    main()
