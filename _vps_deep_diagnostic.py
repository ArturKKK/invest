#!/usr/bin/env python3
"""Deep VPS diagnostic: identify zero features and compare with model expectations."""
import json, os, sys
import pandas as pd
import numpy as np

root = '/home/trader/invest'
os.chdir(root)
sys.path.insert(0, root)

# 1. Load model features
v6_feats = json.load(open("results_v6_prod/feature_names.json"))
v7_feats = json.load(open("results_v7_prod/feature_names.json"))
cb_feats = json.load(open("results_catboost_prod/feature_names.json"))
all_feats = set(v6_feats) | set(v7_feats) | set(cb_feats)

# 2. Simulate inference pipeline
from run_trading import (
    SYMBOLS, EXCLUDE_COLS, build_features, cross_sectional_rank,
    add_12h_features, fetch_ohlcv, UNRANKED_COLS, generate_signal
)
from run_pipeline_v6 import (
    add_multi_horizon_targets, add_cross_asset_features,
    add_advanced_regime_features,
    add_derivatives_features, add_sentiment_features,
)

print("Fetching 800h OHLCV...")
df = fetch_ohlcv(SYMBOLS, 800)
print(f"Raw: {df.shape}")

print("Building features...")
df = build_features(df)

# Drop overlap
_overlap_prefixes = ('btc_close', 'eth_close',
    'btc_ret_', 'eth_ret_', 'btc_vol_24h', 'btc_ma', 'btc_rolling_high',
    'market_dispersion', 'ret_vs_btc', 'breadth_pct_positive',
    'regime_btc_above_ma720', 'regime_btc_dd_720', 'regime_btc_not_crashed',
    'fng_',
    'reversal_', 'vol_surge_', 'btc_beta_')
_overlap_cols = [c for c in df.columns if c.startswith(_overlap_prefixes)]
df.drop(columns=_overlap_cols, inplace=True, errors='ignore')

# Enrich
df = add_multi_horizon_targets(df)
df = add_cross_asset_features(df)
df = add_advanced_regime_features(df)
df = add_12h_features(df)
df = add_sentiment_features(df, root, news_mode='all')
df = add_derivatives_features(df, root)

feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS
             and not c.startswith('target_')
             and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

print(f"\nPRE-RANK features: {len(feat_cols)}")

# 3. Check pre-rank feature health (latest snapshot)
latest_pre = df.groupby('symbol').last().reset_index()
all_zero_pre = [f for f in feat_cols if (latest_pre[f] == 0).all()]
print(f"\n=== ALL-ZERO FEATURES (pre-rank, latest timestamp) ===")
print(f"  Count: {len(all_zero_pre)} of {len(feat_cols)}")
for f in sorted(all_zero_pre):
    in_models = []
    if f in v6_feats: in_models.append('v6')
    if f in v7_feats: in_models.append('v7')
    if f in cb_feats: in_models.append('cb')
    model_str = ', '.join(in_models) if in_models else 'NONE'
    print(f"  {f}: used in [{model_str}]")

# 4. Apply rank normalization (same as run_trading.py)
df = cross_sectional_rank(df, feat_cols)
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
df[feat_cols] = df[feat_cols].fillna(0)

# 5. Check POST-rank feature health
latest_post = df.groupby('symbol').last().reset_index()
all_zero_post = [f for f in feat_cols if (latest_post[f] == 0).all()]
print(f"\n=== ALL-ZERO FEATURES (post-rank, latest timestamp) ===")
print(f"  Count: {len(all_zero_post)} of {len(feat_cols)}")
for f in sorted(all_zero_post):
    in_models = []
    if f in v6_feats: in_models.append('v6')
    if f in v7_feats: in_models.append('v7')
    if f in cb_feats: in_models.append('cb')
    model_str = ', '.join(in_models) if in_models else 'NONE'
    print(f"  {f}: used in [{model_str}]")

# 6. Check model-specific missing features
print(f"\n=== PER-MODEL FEATURE CHECK (post-rank) ===")
for name, feats in [('v6', v6_feats), ('v7', v7_feats), ('cb', cb_feats)]:
    missing = [f for f in feats if f not in latest_post.columns]
    zero = [f for f in feats if f in latest_post.columns and (latest_post[f] == 0).all()]
    print(f"\n  --- {name} ({len(feats)} feats) ---")
    print(f"  Missing (0-padded): {missing}")
    print(f"  All-zero: {zero}")
    if zero:
        # Check if these were zero BEFORE rank too
        pre_zero = [f for f in zero if (latest_pre[f] == 0).all()]
        rank_killed = [f for f in zero if f not in pre_zero]
        print(f"    Pre-rank zero: {pre_zero}")
        print(f"    Rank killed (were non-zero, now zero): {rank_killed}")

# 7. Check score distribution from generate_signal
print(f"\n=== SIGNAL GENERATION ===")
signals = generate_signal(df, feat_cols, root)
if signals is not None:
    print(f"  Signals shape: {signals.shape}")
    print(f"  Score range: {signals['score'].min():.4f} → {signals['score'].max():.4f}")
    print(f"  Score spread: {signals['score'].max() - signals['score'].min():.4f}")
    print(f"  Score std: {signals['score'].std():.4f}")
    print(f"  Deriv scale: {signals['deriv_scale'].describe()}")
    print(f"\n  Top 5:")
    for _, r in signals.head(5).iterrows():
        print(f"    {r['symbol']}: {r['score']:.4f} deriv={r['deriv_scale']:.2f}")
    print(f"  Bottom 5:")
    for _, r in signals.tail(5).iterrows():
        print(f"    {r['symbol']}: {r['score']:.4f} deriv={r['deriv_scale']:.2f}")

# 8. Check if UNRANKED_COLS is applied correctly
print(f"\n=== UNRANKED_COLS CHECK ===")
for col in sorted(UNRANKED_COLS):
    if col in latest_post.columns:
        vals = latest_post[col]
        is_ranked = (vals.min() >= -0.6) and (vals.max() <= 0.6) and len(vals.unique()) > 2
        print(f"  {col}: min={vals.min():.3f} max={vals.max():.3f} → {'RANKED (BUG!)' if is_ranked else 'OK (unranked)'}")
