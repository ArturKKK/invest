#!/usr/bin/env python3
"""Quick test: verify lgb_minimal meta-model produces wider score spread."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_trading import (
    fetch_ohlcv, build_features, cross_sectional_rank,
    SYMBOLS, add_12h_features, generate_signal,
)
from run_pipeline_v6 import (
    add_multi_horizon_targets, add_cross_asset_features,
    add_advanced_regime_features,
    add_derivatives_features, add_sentiment_features,
)

EXCLUDE_COLS = {'symbol', 'timestamp', 'open_time', 'open', 'high', 'low',
                'close', 'volume', 'date', 'hour', 'coin'}
ROOT = os.path.dirname(os.path.abspath(__file__))

print('Fetching data...')
df = fetch_ohlcv(SYMBOLS, 800)
df = build_features(df)

_overlap_prefixes = ('btc_close', 'eth_close', 'btc_ret_', 'eth_ret_',
    'btc_vol_24h', 'btc_ma', 'btc_rolling_high', 'market_dispersion',
    'ret_vs_btc', 'breadth_pct_positive', 'regime_btc_above_ma720',
    'regime_btc_dd_720', 'regime_btc_not_crashed', 'fng_',
    'reversal_', 'vol_surge_', 'btc_beta_')
_overlap_cols = [c for c in df.columns if c.startswith(_overlap_prefixes)]
df.drop(columns=_overlap_cols, inplace=True, errors='ignore')

df = add_multi_horizon_targets(df)
df = add_cross_asset_features(df)
df = add_advanced_regime_features(df)
df = add_12h_features(df)
df = add_sentiment_features(df, ROOT, news_mode='all')
df = add_derivatives_features(df, ROOT)

feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS
             and not c.startswith('target_')
             and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
df = cross_sectional_rank(df, feat_cols)

for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
df[feat_cols] = df[feat_cols].fillna(0)

print('\nGenerating signal with lgb_minimal meta-model...')
signals = generate_signal(df, feat_cols, ROOT)

if signals is not None:
    print('\n' + '=' * 60)
    print('RESULTS (lgb_minimal)')
    print('=' * 60)
    print(f'Score range: [{signals["score"].min():.4f}, {signals["score"].max():.4f}]')
    print(f'Score spread: {signals["score"].max() - signals["score"].min():.4f}')
    print(f'Score std: {signals["score"].std():.4f}')
    print(f'\nTop 5:')
    print(signals.head(5).to_string(index=False))
    print(f'\nBottom 5:')
    print(signals.tail(5).to_string(index=False))
    
    # Compare with old (0.05-0.10 spread was broken)
    spread = signals["score"].max() - signals["score"].min()
    if spread > 0.5:
        print(f'\n✅ Score spread {spread:.2f} >> 0.10 — FIX WORKS!')
    else:
        print(f'\n⚠️  Score spread {spread:.2f} still narrow')
