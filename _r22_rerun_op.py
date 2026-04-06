#!/usr/bin/env python3
"""Rerun EXP-O (rank+ridge) and EXP-P after fixing dtype crash."""
import numpy as np, pandas as pd, lightgbm as lgb, time, sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from _research_r22_models import (
    FEATURES_23, SEEDS, LEVERAGE, CAPITAL, CFG_BEST,
    SYM_35, WINDOWS, log, cs_rank_cols,
    build_r19_features, add_new_features,
    train_lgb, train_xgb, train_catboost,
    ic_quick, run_eval, exp_o, exp_p,
)
from _research_round7 import compute_regime, simulate, eval_config, show
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal

t0 = time.time()
log("Loading data...")
ohlcv = load_ohlcv()
ohlcv = ohlcv[ohlcv["symbol"].isin(SYM_35)]
derivs = load_derivatives()
df = build_features_minimal(ohlcv, derivs)
regime_df = compute_regime(df)
log(f"Base: {len(df):,} rows")

log("Building features...")
df = build_r19_features(df)
df, new_feats = add_new_features(df)

avail_23 = [f for f in FEATURES_23 if f in df.columns]

log("Training LGB/XGB/CB for ensemble...")
preds_lgb = train_lgb(df, avail_23)
preds_xgb = train_xgb(df, avail_23)
preds_cb = train_catboost(df, avail_23)

log("\n" + "=" * 80)
r_o = exp_o(preds_lgb, preds_xgb, preds_cb, regime_df)
log("\n" + "=" * 80)
results_p = exp_p(df, regime_df, new_feats)

log(f"\nTotal time: {time.time()-t0:.0f}s")
