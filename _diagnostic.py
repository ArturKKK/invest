#!/usr/bin/env python3
"""Diagnostic: check feature health for production models."""
import pandas as pd, json, os, numpy as np, sys

root = os.path.dirname(os.path.abspath(__file__))

# Load model feature names
v6_feats = json.load(open(os.path.join(root, 'results/production/lgb_v6_no_news/feature_names.json')))
v7_feats = json.load(open(os.path.join(root, 'results/production/lgb_v7_no_news/feature_names.json')))
cb_feats = json.load(open(os.path.join(root, 'results/production/catboost_with_news/feature_names.json')))

all_model_feats = set(v6_feats) | set(v7_feats) | set(cb_feats)

print('=== MODEL FEATURE COUNTS ===')
print(f'v6: {len(v6_feats)} features')
print(f'v7: {len(v7_feats)} features')
print(f'cb: {len(cb_feats)} features')
print(f'Total unique: {len(all_model_feats)}')

# Features unique to each model
v6_only = set(v6_feats) - set(v7_feats) - set(cb_feats)
v7_only = set(v7_feats) - set(v6_feats) - set(cb_feats)
cb_only = set(cb_feats) - set(v6_feats) - set(v7_feats)
print(f'\nv6 only ({len(v6_only)}): {sorted(v6_only)}')
print(f'v7 only ({len(v7_only)}): {sorted(v7_only)}')
print(f'cb only ({len(cb_only)}): {sorted(cb_only)}')

# Check data freshness
print(f'\n=== DATA FRESHNESS ===')
for name, path in [
    ('funding_rates (OKX)', 'data/sentiment/funding_rates.parquet'),
    ('long_short_ratio', 'data/sentiment/long_short_ratio.parquet'),
    ('binance_futures_metrics', 'data/sentiment/binance_futures_metrics.parquet'),
    ('binance_funding_rates', 'data/sentiment/binance_funding_rates.parquet'),
    ('binance_premium_index', 'data/sentiment/binance_premium_index.parquet'),
    ('fear_greed', 'data/sentiment/fear_greed.parquet'),
    ('crypto_news', 'data/sentiment/crypto_news.parquet'),
]:
    full = os.path.join(root, path)
    if os.path.exists(full):
        df = pd.read_parquet(full)
        ts_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
        try:
            ts = pd.to_datetime(df[ts_col])
            print(f'  {name}: {len(df):,} rows, {ts.min().date()} → {ts.max().date()}, cols={list(df.columns)}')
        except:
            print(f'  {name}: {len(df):,} rows, cols={list(df.columns)}')
    else:
        print(f'  {name}: ❌ MISSING')

# Now build features from live data and check what's zero
print(f'\n=== SIMULATING LIVE INFERENCE ===')
sys.path.insert(0, root)

from run_trading import (
    SYMBOLS, EXCLUDE_COLS, build_features, cross_sectional_rank, 
    add_12h_features, fetch_ohlcv
)
from run_pipeline_v6 import (
    add_multi_horizon_targets, add_cross_asset_features,
    add_advanced_regime_features,
    add_derivatives_features, add_sentiment_features,
)

# Use frozen raw data instead of live fetch
frozen = os.path.join(root, 'trading_logs', 'frozen_raw.parquet')
if os.path.exists(frozen):
    print(f"Using frozen raw data: {frozen}")
    df = pd.read_parquet(frozen)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
else:
    print("Fetching live data (2 min)...")
    df = fetch_ohlcv(SYMBOLS, 800)

print(f"Raw shape: {df.shape}")

# Build features exactly as run_trading.py does
df = build_features(df)

# Drop overlap cols
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

# CS-rank normalize (same as run_trading.py)
df = cross_sectional_rank(df, feat_cols)

# Clean
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
df[feat_cols] = df[feat_cols].fillna(0)

# Latest snapshot
latest = df.groupby('symbol').last().reset_index()
print(f"\nLatest snapshot: {latest.shape}, {len(feat_cols)} features")

# Check which model features are available vs missing vs zero
print(f'\n=== FEATURE HEALTH PER MODEL ===')
for name, feats in [('v6', v6_feats), ('v7', v7_feats), ('cb', cb_feats)]:
    missing = [f for f in feats if f not in latest.columns]
    available = [f for f in feats if f in latest.columns]
    all_zero = [f for f in available if (latest[f] == 0).all()]
    all_nan = [f for f in available if latest[f].isna().all()]
    mostly_zero = [f for f in available if (latest[f] == 0).mean() > 0.9 and f not in all_zero]
    
    print(f'\n--- {name}: {len(feats)} features ---')
    print(f'  Missing (will be 0-filled): {len(missing)}')
    if missing:
        print(f'    {missing}')
    print(f'  All-zero in latest: {len(all_zero)}')
    if all_zero:
        print(f'    {all_zero}')
    print(f'  All-NaN in latest: {len(all_nan)}')
    if all_nan:
        print(f'    {all_nan}')
    print(f'  >90% zero: {len(mostly_zero)}')
    if mostly_zero:
        print(f'    {mostly_zero}')

# Feature importance check: are zero-filled features important?
print(f'\n=== ZERO-FILLED FEATURE IMPORTANCE ===')
for name, feats, imp_path in [
    ('v6', v6_feats, 'results/production/lgb_v6_no_news/feature_importance_v6.csv'),
    ('v7', v7_feats, 'results/production/lgb_v7_no_news/feature_importance_v7.csv'),
    ('cb', cb_feats, 'results/production/catboost_with_news/feature_importance_catboost.csv'),
]:
    imp_full = os.path.join(root, imp_path)
    if not os.path.exists(imp_full):
        print(f'  {name}: no importance file')
        continue
    imp = pd.read_csv(imp_full)
    # find the feature name and importance columns
    feat_col = [c for c in imp.columns if 'feat' in c.lower() or 'name' in c.lower()][0]
    imp_col = [c for c in imp.columns if 'import' in c.lower() or 'gain' in c.lower()][0]
    
    missing_or_zero = [f for f in feats if f not in latest.columns or (latest[f] == 0).all()]
    if not missing_or_zero:
        print(f'  {name}: no zero features ✅')
        continue
    
    imp_rows = imp[imp[feat_col].isin(missing_or_zero)].sort_values(imp_col, ascending=False)
    total_imp = imp[imp_col].sum()
    zero_imp = imp_rows[imp_col].sum()
    print(f'  {name}: {len(missing_or_zero)} zero feats, importance share = {zero_imp/total_imp*100:.1f}%')
    print(f'    Top zero-filled features by importance:')
    for _, row in imp_rows.head(10).iterrows():
        print(f'      {row[feat_col]}: {row[imp_col]:.0f} ({row[imp_col]/total_imp*100:.2f}%)')
