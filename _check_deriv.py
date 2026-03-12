"""Check which derivative features are zero vs non-zero."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, sys, os
sys.path.insert(0, '.')
from run_trading import build_features, EXCLUDE_COLS, fetch_ohlcv, SYMBOLS, add_12h_features
from run_pipeline_v6 import (add_multi_horizon_targets, add_cross_asset_features,
                              add_advanced_regime_features, add_sentiment_features,
                              add_derivatives_features)

df = fetch_ohlcv(SYMBOLS, 800)
df = build_features(df)
_op = ('btc_close','eth_close','btc_ret_','eth_ret_','btc_vol_24h','btc_ma',
       'btc_rolling_high','market_dispersion','ret_vs_btc','breadth_pct_positive',
       'regime_btc_above_ma720','regime_btc_dd_720','regime_btc_not_crashed',
       'fng_','reversal_','vol_surge_','btc_beta_')
oc = [c for c in df.columns if c.startswith(_op)]
df.drop(columns=oc, inplace=True, errors='ignore')
df = add_multi_horizon_targets(df)
df = add_cross_asset_features(df)
df = add_advanced_regime_features(df)
df = add_12h_features(df)
df = add_sentiment_features(df, '.', news_mode='all')
df = add_derivatives_features(df, '.')

deriv_prefixes = ('oi_','taker_','ls_ratio','global_ls','basis_','premium_',
                  'liq_','funding_surprise','deriv_','mkt_oi','mkt_taker','mkt_ls')
fc = [c for c in df.columns if c not in EXCLUDE_COLS and any(c.startswith(p) for p in deriv_prefixes)]
latest = df.groupby('symbol').last()
print("\n=== DERIVATIVE FEATURES ===")
for c in sorted(fc):
    nz = (latest[c] != 0).sum()
    status = "OK" if nz > 0 else "ZERO"
    print(f"  [{status:4s}] {c:40s} nonzero={nz}/{len(latest)}")
print(f"\nTotal: {len(fc)} features, {sum(1 for c in fc if (latest[c]!=0).any())} non-zero, {sum(1 for c in fc if not (latest[c]!=0).any())} all-zero")
