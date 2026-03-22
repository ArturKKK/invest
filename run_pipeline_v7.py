#!/usr/bin/env python3
"""
Crypto Alpha Model v5 — Sentiment + Risk Overlay + Rolling Validation

Major improvements over v4:
1. Sentiment features:
   - Fear & Greed Index (daily → hourly forward-fill)
   - Funding rates from OKX (where available, ~3mo history)
   - Synthetic positioning proxies (computed from OHLCV)
   - Long/short ratio features
2. Risk overlay:
   - Volatility targeting (inverse realized vol position sizing)
   - Max drawdown circuit breaker (stop after -25% DD, restart after recovery)
   - Proper cost model: taker fee 0.05% + funding rate 0.01%/8h + slippage 0.02%
3. Rolling walk-forward validation (3 windows):
   - Window 1: train→2023-06, val→2024-06, test→2024-12
   - Window 2: train→2024-01, val→2024-12, test→2025-03+
   - Window 3: train→2024-06, val→2025-01, test→2025-03+
   Combined results across windows → robust estimate.
4. Enhanced features:
   - Momentum reversal scores
   - Crowding/positioning proxies
   - Cross-coin correlation regime
5. HIST+LGB ensemble only (GRU/MASTER dropped)

Usage:
  python run_pipeline_v5.py                          # Full run
  python run_pipeline_v5.py --skip-hpo               # Skip Optuna
  python run_pipeline_v5.py --single-window           # Quick single-window test
  python run_pipeline_v5.py --data /path/to/features

Requirements:
  pip install lightgbm pandas numpy scipy pyarrow tqdm
  Optional: pip install optuna (for HPO)
"""

import sys
import os
import argparse
import json
import warnings
from datetime import datetime
from copy import deepcopy

import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
HORIZON = 12       # <<< v7: predict 12h returns, aligned with rebalance interval
N_SEEDS = 5
SEEDS = [42, 123, 456, 789, 2024]
BLEND_24H = 0.25   # v7: blend 25% of 24h target for stability
PURGE_DAYS = 8     # gap between train_end and val_start to prevent target leakage

# Rolling walk-forward windows (RESEARCH mode — has held-out test set)
WALK_FORWARD_WINDOWS = [
    {
        'name': 'W1 (→2024-12)',
        'train_end': '2023-06-30',
        'val_start': '2023-07-08',
        'val_end': '2024-06-29',
        'test_start': '2024-07-01',
        'test_end': '2024-12-31',
    },
    {
        'name': 'W2 (→2025-06)',
        'train_end': '2024-01-01',
        'val_start': '2024-01-09',
        'val_end': '2024-12-30',
        'test_start': '2025-01-01',
        'test_end': '2025-06-30',
    },
    {
        'name': 'W3 (→2025-12)',
        'train_end': '2024-06-29',
        'val_start': '2024-07-07',
        'val_end': '2025-06-29',
        'test_start': '2025-07-01',
        'test_end': '2025-12-31',
    },
]

# RESEARCH windows — proper long-range OOS evaluation
# R1: train→2024, val 9mo, test 2025-Q4. R2: train→2025-H1, val 6mo, test 2026
# R2 train_end < R1 test_start → no meta-leakage between windows
RESEARCH_WINDOWS = [
    {
        'name': 'R1 (train→2024, test 2025-Q4)',
        'train_end': '2024-12-31',
        'val_start': '2025-01-08',
        'val_end': '2025-09-30',
        'test_start': '2025-10-01',
        'test_end': '2025-12-31',
    },
    {
        'name': 'R2 (train→2025-H1, test 2026)',
        'train_end': '2025-06-30',
        'val_start': '2025-07-08',
        'val_end': '2025-12-31',
        'test_start': '2026-01-01',
        'test_end': '2026-03-17',
    },
]

# PRODUCTION mode — maximum training data, no held-out test set
PRODUCTION_WINDOW = {
    'name': 'PROD (max data)',
    'train_end': '2025-12-31',
    'val_start': '2026-01-08',
    'val_end': '2026-03-17',
    'test_start': '2026-03-17',
    'test_end': '2026-12-31',
}

# Columns to exclude from features
EXCLUDE_COLS = {
    'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
    'target_ret_4h', 'target_ret_12h', 'target_ret_24h', 'target_ret_blended',
    'target_cls', 'target_ret', 'target_rank', 'target_excess',
    'hour', 'day_of_week',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
}

REGIME_COLS = {
    'btc_regime_24', 'btc_regime_72', 'btc_regime_168',
    'regime_btc_above_ma336', 'regime_btc_above_ma720',
    'regime_btc_ma720_slope', 'regime_btc_not_crashed',
    'regime_btc_dd_720', 'regime_low_vol',
    'regime_breadth_bullish', 'breadth_pct_positive',
    'regime_composite',
    # New sentiment columns that should NOT be ranked
    'fng_value', 'fng_extreme_fear', 'fng_extreme_greed',
    'fng_ma7', 'fng_ma30', 'fng_momentum',
    'market_avg_funding', 'market_funding_skew',
    # v6: binary features that should NOT be ranked
    'is_asian_session',
    # Calendar features (same for all coins at same timestamp)
    'cal_hour_sin', 'cal_hour_cos', 'cal_dow_sin', 'cal_dow_cos',
    'cal_is_us_session', 'cal_is_weekend',
    'cal_days_to_monthly_expiry', 'cal_month_sin', 'cal_month_cos',
    # DVOL features (market-level implied vol, same for all coins)
    'dvol_btc', 'dvol_btc_change_12h', 'dvol_btc_change_24h',
    'dvol_btc_z_30d', 'dvol_btc_z_60d',
    'dvol_eth', 'dvol_eth_change_12h', 'dvol_eth_change_24h',
    'dvol_eth_z_30d', 'dvol_eth_z_60d',
    'dvol_spread', 'dvol_term_ratio', 'dvol_vol_of_vol',
    # Macro / cross-market features (market-level, same for all coins)
    'vix_close', 'spx_close', 'dxy_close', 'gold_close',
    'yield_10y_close', 'hy_spread', 'breakeven_10y',
    'yield_curve_10y2y', 'fed_funds_rate',
    'vix_close_z20d', 'hy_spread_z20d', 'breakeven_10y_z20d',
    'yield_curve_10y2y_z20d', 'risk_aversion', 'real_rate',
    # Macro changes (1d/5d/20d) — market-level, not ranked
    'vix_close_chg_1d', 'vix_close_chg_5d', 'vix_close_chg_20d',
    'spx_close_chg_1d', 'spx_close_chg_5d', 'spx_close_chg_20d',
    'dxy_close_chg_1d', 'dxy_close_chg_5d', 'dxy_close_chg_20d',
    'gold_close_chg_1d', 'gold_close_chg_5d', 'gold_close_chg_20d',
    'hy_spread_chg_1d', 'hy_spread_chg_5d', 'hy_spread_chg_20d',
    'breakeven_10y_chg_1d', 'breakeven_10y_chg_5d', 'breakeven_10y_chg_20d',
    'yield_curve_10y2y_chg_1d', 'yield_curve_10y2y_chg_5d', 'yield_curve_10y2y_chg_20d',
    # Macro cross-interactions
    'risk_on_off_ratio', 'real_rate_chg_5d',
}

# Cost model for perpetual swaps — v7 (12h rebalance)
#   - 12h holding reduces turnover vs 4h (~35% vs ~62%)
#   - Mostly taker for now; maker would save 33%
#   - 1.5 funding periods per rebalance (12h/8h)
COST_MODEL = {
    'taker_fee': 0.0003,        # 0.03% blended
    'slippage': 0.0001,         # 0.01%
    'funding_per_8h': 0.00005,  # 0.005% net
    'turnover_pct': 0.35,       # 35% turnover at 12h (lower than 4h)
}


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_multi_horizon_targets(df):
    """Add forward return targets. v7: blended 12h+24h target."""
    print("   🎯 Adding targets...")
    for h in [4, 12, 24]:
        df[f'target_ret_{h}h'] = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(h).shift(-h)
        )
    # v7: blended target — 75% weight on 12h, 25% on 24h for stability
    df['target_ret_blended'] = (
        (1 - BLEND_24H) * df['target_ret_12h'] + BLEND_24H * df['target_ret_24h']
    )
    return df


def add_residual_targets(df, beta_window=168):
    """Add beta-residual targets: ret_coin - beta*ret_btc.

    Removes the common market factor (BTC) from returns so the model
    learns to predict *relative* outperformance rather than market
    direction.  Also creates residual blended target for v7.
    """
    print(f"   🎯 Adding residual targets (beta window={beta_window}h)...")
    if 'btc_close' not in df.columns:
        btc = df[df['symbol'] == 'BTC/USDT'][['timestamp', 'close']].copy()
        btc = btc.rename(columns={'close': 'btc_close'}).drop_duplicates('timestamp')
        df = df.merge(btc, on='timestamp', how='left')
        _drop_btc = True
    else:
        _drop_btc = False

    for h in [12, 24]:
        target_col = f'target_ret_{h}h'
        if target_col not in df.columns:
            continue
        btc_fwd = df.groupby('symbol')['btc_close'].transform(
            lambda x: x.pct_change(h).shift(-h)
        )
        coin_ret_past = df.groupby('symbol')['close'].transform(lambda x: x.pct_change(1))
        btc_ret_past = df.groupby('symbol')['btc_close'].transform(lambda x: x.pct_change(1))

        def _rolling_beta(group):
            cov = group['_coin_ret'].rolling(beta_window, min_periods=48).cov(group['_btc_ret'])
            var = group['_btc_ret'].rolling(beta_window, min_periods=48).var() + 1e-10
            return cov / var

        df['_coin_ret'] = coin_ret_past
        df['_btc_ret'] = btc_ret_past
        beta = df.groupby('symbol').apply(_rolling_beta).droplevel(0)
        beta = beta.clip(-3, 3).fillna(1.0)
        df[f'target_ret_{h}h_excess'] = df[target_col] - beta * btc_fwd
        df.drop(columns=['_coin_ret', '_btc_ret'], inplace=True)

    # v7: blended residual target
    if 'target_ret_12h_excess' in df.columns and 'target_ret_24h_excess' in df.columns:
        df['target_ret_blended_excess'] = (
            (1 - BLEND_24H) * df['target_ret_12h_excess']
            + BLEND_24H * df['target_ret_24h_excess']
        )
        print(f"      target_ret_blended_excess: mean={df['target_ret_blended_excess'].mean():.6f}")

    if _drop_btc:
        df.drop(columns=['btc_close'], inplace=True, errors='ignore')
    return df


def add_cross_asset_features(df):
    """BTC/ETH market factors."""
    print("   🌐 Adding cross-asset features...")
    btc = df[df['symbol'] == 'BTC/USDT'][['timestamp', 'close']].copy()
    btc = btc.rename(columns={'close': 'btc_close'}).drop_duplicates('timestamp')
    eth = df[df['symbol'] == 'ETH/USDT'][['timestamp', 'close']].copy()
    eth = eth.rename(columns={'close': 'eth_close'}).drop_duplicates('timestamp')

    df = df.merge(btc, on='timestamp', how='left')
    df = df.merge(eth, on='timestamp', how='left')

    for h in [1, 4, 12, 24, 48, 168]:
        df[f'btc_ret_{h}h'] = df.groupby('symbol')['btc_close'].transform(
            lambda x: x.pct_change(h))
    for h in [1, 4, 12, 24]:
        df[f'eth_ret_{h}h'] = df.groupby('symbol')['eth_close'].transform(
            lambda x: x.pct_change(h))

    for w in [24, 72, 168]:
        df[f'btc_ma{w}'] = df.groupby('symbol')['btc_close'].transform(
            lambda x: x.rolling(w).mean())

    df['btc_regime_24'] = (df['btc_close'] > df['btc_ma24']).astype(float)
    df['btc_regime_72'] = (df['btc_close'] > df['btc_ma72']).astype(float)
    df['btc_regime_168'] = (df['btc_close'] > df['btc_ma168']).astype(float)

    df['btc_vol_24h'] = df.groupby('symbol')['btc_close'].transform(
        lambda x: x.pct_change().rolling(24).std())

    df['eth_btc_ratio'] = df['eth_close'] / (df['btc_close'] + 1e-10)
    df['eth_btc_ret_24h'] = df.groupby('symbol')['eth_btc_ratio'].transform(
        lambda x: x.pct_change(24))

    cs_std = df.groupby('timestamp')['ret_1h'].transform('std')
    df['market_dispersion'] = cs_std
    df['ret_vs_btc_24h'] = df['ret_24h'] - df['btc_ret_24h']

    df.drop(columns=['btc_ma24', 'btc_ma72', 'btc_ma168', 'eth_btc_ratio'], inplace=True)
    return df


def add_advanced_regime_features(df):
    """Multi-factor regime filter (same as v4)."""
    print("   🔰 Adding advanced regime features...")

    btc = df[df['symbol'] == 'BTC/USDT'][['timestamp', 'btc_close']].drop_duplicates('timestamp')
    btc = btc.sort_values('timestamp').copy()

    btc['btc_ma336'] = btc['btc_close'].rolling(336, min_periods=100).mean()
    btc['btc_ma720'] = btc['btc_close'].rolling(720, min_periods=200).mean()

    btc['regime_btc_above_ma336'] = (btc['btc_close'] > btc['btc_ma336']).astype(float)
    btc['regime_btc_above_ma720'] = (btc['btc_close'] > btc['btc_ma720']).astype(float)
    btc['regime_btc_ma720_slope'] = (
        btc['btc_ma720'] > btc['btc_ma720'].shift(24)
    ).astype(float)

    btc['btc_rolling_high_720'] = btc['btc_close'].rolling(720, min_periods=100).max()
    btc['regime_btc_dd_720'] = btc['btc_close'] / btc['btc_rolling_high_720'] - 1
    btc['regime_btc_not_crashed'] = (btc['regime_btc_dd_720'] > -0.15).astype(float)

    btc['_btc_vol_24'] = btc['btc_close'].pct_change().rolling(24).std()
    btc['_btc_vol_720_med'] = btc['_btc_vol_24'].rolling(720, min_periods=100).median()
    btc['regime_low_vol'] = (btc['_btc_vol_24'] < btc['_btc_vol_720_med'] * 2.0).astype(float)

    btc_regime_cols = [
        'timestamp', 'regime_btc_above_ma336', 'regime_btc_above_ma720',
        'regime_btc_ma720_slope', 'regime_btc_not_crashed',
        'regime_btc_dd_720', 'regime_low_vol',
    ]
    df = df.merge(btc[btc_regime_cols], on='timestamp', how='left')

    breadth = df.groupby('timestamp')['ret_24h'].agg(
        breadth_pct_positive=lambda x: (x > 0).mean()
    ).reset_index()
    breadth['regime_breadth_bullish'] = (breadth['breadth_pct_positive'] > 0.5).astype(float)
    df = df.merge(breadth, on='timestamp', how='left')

    df['regime_composite'] = (
        0.25 * df['regime_btc_above_ma720'].fillna(0) +
        0.20 * df['regime_btc_ma720_slope'].fillna(0) +
        0.20 * df['regime_breadth_bullish'].fillna(0) +
        0.20 * df['regime_btc_not_crashed'].fillna(0) +
        0.15 * df['regime_low_vol'].fillna(0)
    )

    if 'btc_close' in df.columns:
        df.drop(columns=['btc_close'], inplace=True, errors='ignore')
    if 'eth_close' in df.columns:
        df.drop(columns=['eth_close'], inplace=True, errors='ignore')

    return df


def add_12h_features(df):
    """
    v7 features optimized for 12h holding period.
    Mean-reversion signals, multi-day momentum, overnight effects,
    enhanced funding rate features, exhaustion/breakout signals.
    """
    print("   🕐 Adding 12h-specific features (v7 enhanced)...")

    for sym, grp in df.groupby('symbol'):
        c = grp['close']
        v = grp['volume']
        h = grp['high']
        l = grp['low']
        idx = grp.index

        # 12h momentum z-score (is recent 12h return extreme vs history?)
        r12 = c.pct_change(12)
        r12_mean = r12.rolling(168).mean()
        r12_std = r12.rolling(168).std() + 1e-10
        df.loc[idx, 'mom_12h_zscore'] = (r12 - r12_mean) / r12_std

        # Mean-reversion: distance from 12h VWAP
        vwap_12 = (c * v).rolling(12).sum() / (v.rolling(12).sum() + 1e-10)
        df.loc[idx, 'vwap_12h_dist'] = c / vwap_12 - 1

        # Multi-day momentum (3d, 7d) — stronger for 12h holding
        df.loc[idx, 'mom_3d'] = c.pct_change(72)  # 72h = 3 days
        df.loc[idx, 'mom_7d'] = c.pct_change(168)

        # Momentum acceleration (is momentum accelerating or decelerating?)
        ret_12 = c.pct_change(12)
        ret_12_prev = c.shift(12).pct_change(12)
        df.loc[idx, 'mom_accel_12h'] = ret_12 - ret_12_prev

        # Volume trend (12h vs 48h avg)
        vol_12 = v.rolling(12).mean()
        vol_48 = v.rolling(48).mean() + 1e-10
        df.loc[idx, 'vol_trend_12_48'] = vol_12 / vol_48 - 1

        # Overnight/intraday pattern (are we in first or second 12h of day?)
        hours = grp['timestamp'].dt.hour
        df.loc[idx, 'is_asian_session'] = ((hours >= 0) & (hours < 12)).astype(float)

        # Range expansion: is 12h range wider than average?
        h12 = h.rolling(12).max()
        l12 = l.rolling(12).min()
        range12 = (h12 - l12) / (c + 1e-10)
        range_avg = range12.rolling(168).mean() + 1e-10
        df.loc[idx, 'range_expansion_12h'] = range12 / range_avg - 1

        # v7 NEW: Trend exhaustion — close position in range
        range_pos = (c - l12) / (h12 - l12 + 1e-10)
        df.loc[idx, 'range_position_12h'] = range_pos

        # v7 NEW: Volume-weighted price change (smarter momentum)
        vwap_change = (c * v).rolling(12).sum() / (v.rolling(12).sum() + 1e-10)
        vwap_prev = (c.shift(12) * v.shift(12)).rolling(12).sum() / (v.shift(12).rolling(12).sum() + 1e-10)
        df.loc[idx, 'vwpc_12h'] = vwap_change / (vwap_prev + 1e-10) - 1

        # v7 NEW: Higher-high / lower-low counts (trend detection)
        hh = (h > h.shift(1)).rolling(12).sum()
        ll = (l < l.shift(1)).rolling(12).sum()
        df.loc[idx, 'hh_count_12h'] = hh
        df.loc[idx, 'll_count_12h'] = ll
        df.loc[idx, 'trend_strength_12h'] = hh - ll  # positive = uptrend

        # v7 NEW: Volatility crush / expansion ratio
        vol_recent = c.pct_change().rolling(12).std()
        vol_base = c.pct_change().rolling(72).std() + 1e-10
        df.loc[idx, 'vol_crush_ratio'] = vol_recent / vol_base

        # v7 NEW: Close-to-close vs intraday range (directional quality)
        ret_abs = (c - c.shift(12)).abs()
        hi_lo_path = (h - l).rolling(12).sum()
        df.loc[idx, 'direction_quality_12h'] = ret_abs / (hi_lo_path + 1e-10)

    # Cross-sectional 12h momentum rank
    df['ret_12h_temp'] = df.groupby('symbol')['close'].transform(lambda x: x.pct_change(12))
    df['ret_12h_cs_rank'] = df.groupby('timestamp')['ret_12h_temp'].rank(pct=True)
    df.drop(columns=['ret_12h_temp'], inplace=True)

    # Cross-sectional volume surprise
    df['vol_12h_sum'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(12).sum())
    df['vol_12h_cs_rank'] = df.groupby('timestamp')['vol_12h_sum'].rank(pct=True)
    df.drop(columns=['vol_12h_sum'], inplace=True)

    # v7 NEW: Cross-sectional funding rate rank (if available)
    if 'funding_rate' in df.columns:
        df['funding_cs_rank'] = df.groupby('timestamp')['funding_rate'].rank(pct=True)
        # Cumulative funding (high cumulative = expensive to hold)
        df['cum_funding_24h'] = df.groupby('symbol')['funding_rate'].transform(
            lambda x: x.rolling(3, min_periods=1).sum()  # 3 x 8h = 24h
        )
        df['cum_funding_72h'] = df.groupby('symbol')['funding_rate'].transform(
            lambda x: x.rolling(9, min_periods=1).sum()  # 9 x 8h = 72h
        )

    n_new = 18  # approximate count of new features
    print(f"   ✅ Added ~{n_new} features for 12h holding period (v7)")
    return df


def add_sentiment_features(df, project_root):
    """
    Add sentiment/alternative data features:
    1. Fear & Greed Index (daily → hourly ffill) — market-level
    2. Funding rates from OKX (per-coin, where available)
    3. Synthetic positioning proxies (from OHLCV)
    """
    print("   📰 Adding sentiment features...")
    sent_dir = os.path.join(project_root, 'data', 'sentiment')

    n_feats_before = len([c for c in df.columns if c not in EXCLUDE_COLS])

    # ---- 1. Fear & Greed Index ----
    fng_path = os.path.join(sent_dir, 'fear_greed.parquet')
    if os.path.exists(fng_path):
        fng = pd.read_parquet(fng_path)
        fng['timestamp'] = pd.to_datetime(fng['timestamp'], utc=True)
        # Normalize to date for merge (daily data → forward-fill to hourly)
        fng['date'] = fng['timestamp'].dt.date
        fng_daily = fng[['date', 'fng_value']].drop_duplicates('date')

        df['date'] = df['timestamp'].dt.date
        df = df.merge(fng_daily, on='date', how='left')
        df['fng_value'] = df['fng_value'].ffill().fillna(50)  # neutral default

        # Derived FNG features
        # Extreme zones: fear < 25, greed > 75
        df['fng_extreme_fear'] = (df['fng_value'] < 25).astype(float)
        df['fng_extreme_greed'] = (df['fng_value'] > 75).astype(float)

        # FNG moving averages (compute per-symbol since same market-level value)
        # Actually same for all symbols at same timestamp, compute once then broadcast
        fng_ts = df.groupby('timestamp')['fng_value'].first().reset_index()
        fng_ts = fng_ts.sort_values('timestamp')
        fng_ts['fng_ma7'] = fng_ts['fng_value'].rolling(7 * 24, min_periods=24).mean()
        fng_ts['fng_ma30'] = fng_ts['fng_value'].rolling(30 * 24, min_periods=48).mean()
        fng_ts['fng_momentum'] = fng_ts['fng_value'] - fng_ts['fng_ma30']

        df = df.merge(
            fng_ts[['timestamp', 'fng_ma7', 'fng_ma30', 'fng_momentum']],
            on='timestamp', how='left'
        )
        df.drop(columns=['date'], inplace=True)

        print(f"      FNG: {fng['timestamp'].min().date()} → {fng['timestamp'].max().date()}, "
              f"mean={df['fng_value'].mean():.1f}")
    else:
        print(f"      ⚠️  No Fear & Greed data at {fng_path}")

    # ---- 2. Funding Rates (per-coin) ----
    fund_path = os.path.join(sent_dir, 'funding_rates.parquet')
    if os.path.exists(fund_path):
        fund = pd.read_parquet(fund_path)
        fund['timestamp'] = pd.to_datetime(fund['timestamp'], utc=True)

        # Funding rate is 8h — forward-fill to hourly
        # First, round to nearest 8h boundary for clean merge
        fund['ts_8h'] = fund['timestamp'].dt.floor('8h')
        df['ts_8h'] = df['timestamp'].dt.floor('8h')

        fund_pivot = fund.pivot_table(
            index='ts_8h', columns='symbol', values='funding_rate', aggfunc='last'
        )

        # Get funding for each coin
        fund_melt = fund[['ts_8h', 'symbol', 'funding_rate']].drop_duplicates(['ts_8h', 'symbol'])
        df = df.merge(fund_melt, on=['ts_8h', 'symbol'], how='left')
        df['funding_rate'] = df['funding_rate'].fillna(0)

        # Market average funding (across all coins at each timestamp)
        market_fund = fund.groupby('ts_8h')['funding_rate'].agg(['mean', 'std']).reset_index()
        market_fund.columns = ['ts_8h', 'market_avg_funding', 'market_funding_std']
        df = df.merge(market_fund, on='ts_8h', how='left')
        df['market_avg_funding'] = df['market_avg_funding'].fillna(0)
        df['market_funding_std'] = df['market_funding_std'].fillna(0)

        # Funding skew (positive = market imbalanced long)
        df['market_funding_skew'] = df['market_avg_funding'] / (df['market_funding_std'] + 1e-8)

        # Per-coin funding vs market average
        df['funding_vs_market'] = df['funding_rate'] - df['market_avg_funding']

        df.drop(columns=['ts_8h'], inplace=True)

        n_with_funding = (df['funding_rate'] != 0).sum()
        print(f"      Funding: {n_with_funding:,} non-zero rows "
              f"({n_with_funding / len(df) * 100:.1f}%)")
    else:
        print(f"      ⚠️  No funding rate data at {fund_path}")

    # ---- 3. Long/Short Ratio ----
    lsr_path = os.path.join(sent_dir, 'long_short_ratio.parquet')
    if os.path.exists(lsr_path):
        lsr = pd.read_parquet(lsr_path)
        lsr['timestamp'] = pd.to_datetime(lsr['timestamp'], utc=True)
        lsr_merge = lsr[['timestamp', 'symbol', 'long_short_ratio']].drop_duplicates(
            ['timestamp', 'symbol'])
        df = df.merge(lsr_merge, on=['timestamp', 'symbol'], how='left')
        df['long_short_ratio'] = df['long_short_ratio'].fillna(1.0)  # neutral
        print(f"      LS ratio: {(df['long_short_ratio'] != 1.0).sum():,} non-missing rows")
    else:
        print(f"      ⚠️  No LS ratio data")

    # ---- 4. Synthetic Positioning Proxies (from OHLCV) ----
    # These are available for ALL training data, unlike exchange-specific features
    print("      Synthetic positioning features...")

    # a) Momentum reversal score: recent return vs longer-term
    #    Strongly positive reversal score → recently rallied but overbought
    for short, long in [(4, 24), (12, 48), (24, 168)]:
        if f'ret_{short}h' in df.columns and f'ret_{long}h' in df.columns:
            df[f'reversal_{short}v{long}'] = df[f'ret_{short}h'] - df[f'ret_{long}h'] / (long / short)

    # b) Volume surge indicator: high volume relative to average → positioning change
    if 'volume' in df.columns:
        for w in [12, 24, 48]:
            vol_ma = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(w).mean())
            df[f'vol_surge_{w}h'] = df['volume'] / (vol_ma + 1e-10) - 1

    # c) Cross-coin momentum correlation (crowding proxy)
    #    When all coins move together → crowded trade → reversal risk
    if 'ret_1h' in df.columns:
        # Rolling cross-sectional correlation of 1h returns
        # Approximated by dispersion: low dispersion → high correlation → crowding
        cs_disp = df.groupby('timestamp')['ret_4h'].transform('std')
        df['cross_coin_dispersion'] = cs_disp
        # 24h rolling avg of dispersion
        df['cross_coin_disp_ma24'] = df.groupby('symbol')['cross_coin_dispersion'].transform(
            lambda x: x.rolling(24, min_periods=6).mean())
        df['dispersion_regime'] = df['cross_coin_dispersion'] / (df['cross_coin_disp_ma24'] + 1e-10)

    # d) Return asymmetry (captures put-like behavior)
    if 'ret_24h' in df.columns:
        for w in [48, 168]:
            df[f'ret_skew_{w}h_cs'] = df.groupby('timestamp')[f'ret_skew_{w}h'].transform(
                lambda x: x.rank(pct=True) - 0.5
            ) if f'ret_skew_{w}h' in df.columns else 0

    # e) BTC beta (sensitivity to BTC moves — high beta = leveraged position)
    if 'ret_1h' in df.columns and 'btc_ret_1h' in df.columns:
        for w in [48, 168]:
            cov = df.groupby('symbol').apply(
                lambda g: g['ret_1h'].rolling(w).cov(g['btc_ret_1h'])
            ).reset_index(level=0, drop=True)
            var = df.groupby('symbol')['btc_ret_1h'].transform(
                lambda x: x.rolling(w).var() + 1e-10)
            df[f'btc_beta_{w}h'] = cov / var

    n_feats_after = len([c for c in df.columns if c not in EXCLUDE_COLS
                         and not c.startswith('target_') and c not in ['date', 'ts_8h']])
    print(f"   ✅ Sentiment features added: {n_feats_before} → {n_feats_after} total")

    return df


# Import derivatives feature function from v6 to avoid code duplication
from run_pipeline_v6 import add_derivatives_features  # noqa: E402


# ============================================================
# NORMALIZATION & TARGET
# ============================================================

# Features that should be normalised per-symbol over time (TS-zscore)
TSZSCORE_COLS = {
    'vol_surge_12h', 'vol_surge_24h', 'vol_surge_48h',
    'range_expansion_12h',
    'funding_rate', 'funding_vs_market',
    'news_count_1h', 'news_count_24h', 'news_volume_zscore',
    'vol_trend_12_48',
    'mom_12h_zscore',
    # Derivatives features (spike-prone, need per-symbol zscore)
    'oi_change_1h', 'oi_change_4h', 'oi_change_12h', 'oi_change_24h',
    'oi_zscore_7d', 'oi_ret_interaction', 'oi_ret_interaction_12h',
    'taker_imbalance', 'taker_cvd_12h', 'taker_cvd_24h', 'taker_flow_zscore',
    'funding_surprise',
    'top_ls_change_12h', 'top_ls_change_24h', 'top_ls_zscore',
    'ls_divergence',
    'funding_rate_binance',
}


def cross_sectional_rank(df, feat_cols, hybrid=False):
    """Normalise features. Two modes:

    hybrid=True: CS-rank for most features, TS-zscore for spike features.
    hybrid=False: legacy CS-rank only.
    """
    print("   📐 Cross-sectional rank normalization...")

    regime_backup = {}
    for col in REGIME_COLS:
        if col in df.columns:
            regime_backup[col] = df[col].copy()

    if hybrid:
        ts_cols = [c for c in feat_cols if c in TSZSCORE_COLS and c in df.columns]
        rank_cols = [c for c in feat_cols if c not in REGIME_COLS and c not in TSZSCORE_COLS]

        if ts_cols:
            for col in ts_cols:
                zscored = df.groupby('symbol')[col].transform(
                    lambda x: (x - x.rolling(168, min_periods=24).mean())
                              / (x.rolling(168, min_periods=24).std() + 1e-10)
                )
                df[col] = zscored.clip(-3, 3)
            print(f"   📊 TS-zscore: {len(ts_cols)} features")

        if rank_cols:
            ranked = df.groupby('timestamp')[rank_cols].rank(pct=True)
            df[rank_cols] = ranked - 0.5
    else:
        rank_cols = [c for c in feat_cols if c not in REGIME_COLS]
        ranked = df.groupby('timestamp')[rank_cols].rank(pct=True)
        df[rank_cols] = ranked - 0.5

    for col, vals in regime_backup.items():
        df[col] = vals
    print(f"   ✅ Ranked {len(rank_cols)} CS features, preserved {len(regime_backup)} unranked")

    return df


def create_rank_target(df, horizon=4, use_excess=False):
    """Rank target per timestamp. v7: uses blended target by default."""
    if use_excess:
        excess_col = 'target_ret_blended_excess'
        if excess_col in df.columns:
            target_col = excess_col
            print(f"   🎯 Using residual blended target: {target_col}")
        else:
            target_col = 'target_ret_blended' if 'target_ret_blended' in df.columns else f'target_ret_{horizon}h'
            print(f"   ⚠️  Excess blended target not found, falling back to {target_col}")
    else:
        if 'target_ret_blended' in df.columns:
            target_col = 'target_ret_blended'
        else:
            target_col = f'target_ret_{horizon}h'
    df['target_rank'] = df.groupby('timestamp')[target_col].rank(pct=True)
    return df


# ============================================================
# HPO
# ============================================================

def run_optuna_hpo(X_train, y_train, X_val, y_val, val_dates, n_trials=50):
    """Optuna HPO with Rank ICIR objective."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print(f"   ⚠️  Optuna not installed: {sys.executable} -m pip install optuna")
        return None

    print(f"   🔍 Running Optuna HPO ({n_trials} trials)...")
    unique_dates = np.unique(val_dates)

    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'verbosity': -1,
            'n_estimators': 5000,
            'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.05, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 0.8),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 50, 500),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.01, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.01, 10.0, log=True),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.1),
            'path_smooth': trial.suggest_float('path_smooth', 0.0, 10.0),
            'random_state': 42,
            'n_jobs': -1,
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        preds = model.predict(X_val)

        daily_ics = []
        for d in unique_dates:
            mask = val_dates == d
            if mask.sum() < 10:
                continue
            p, a = preds[mask], y_val.values[mask]
            valid = ~(np.isnan(p) | np.isnan(a))
            if valid.sum() < 10:
                continue
            c, _ = spearmanr(p[valid], a[valid])
            if not np.isnan(c):
                daily_ics.append(c)

        if len(daily_ics) < 5:
            return 0
        daily_ics = np.array(daily_ics)
        return daily_ics.mean() / (daily_ics.std() + 1e-10)

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"   ✅ Best Rank ICIR: {study.best_value:.4f}")
    return study.best_params


# ============================================================
# TRAINING
# ============================================================

def train_lgbm(X_train, y_train, X_val, y_val, custom_params=None, seed=42,
               sample_weight=None):
    base_params = {
        'objective': 'regression',
        'metric': 'mse',
        'verbosity': -1,
        'n_estimators': 5000,
        'learning_rate': 0.01,
        'max_depth': 6,
        'num_leaves': 31,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'min_child_samples': 200,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'min_gain_to_split': 0.01,
        'random_state': seed,
        'n_jobs': -1,
    }
    if custom_params:
        base_params.update(custom_params)
    base_params['random_state'] = seed

    model = lgb.LGBMRegressor(**base_params)
    fit_kwargs = {
        'eval_set': [(X_val, y_val)],
        'callbacks': [lgb.early_stopping(100), lgb.log_evaluation(200)],
    }
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = sample_weight
    model.fit(X_train, y_train, **fit_kwargs)
    return model


LAMBDARANK_NBINS = 5   # quintiles 0..4, matches top-5/bottom-5 trading logic


def quantize_rank_target(y, n_bins=LAMBDARANK_NBINS):
    """Convert float target_rank (0-1) to integer relevance labels (0..n_bins-1).

    LGBMRanker requires integer labels.  We use equal-width bins:
      [0, 0.2) → 0,  [0.2, 0.4) → 1,  ... [0.8, 1.0] → 4
    Higher label = higher relevance (= higher expected return).
    """
    labels = np.clip((y * n_bins).astype(int), 0, n_bins - 1)
    return labels


def train_lgbm_ranker(X_train, y_train, X_val, y_val,
                      train_groups, val_groups,
                      custom_params=None, seed=42):
    """Train LGBMRanker with LambdaRank objective.

    y_train / y_val are float target_rank [0, 1] — automatically
    quantized to int relevance labels (0..4 quintiles).
    """
    # Quantize float rank → integer relevance (required by LGBMRanker)
    y_train_int = quantize_rank_target(y_train)
    y_val_int = quantize_rank_target(y_val)

    base_params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'lambdarank_truncation_level': 5,  # top/bottom 5 = what we trade
        'verbosity': -1,
        'n_estimators': 5000,
        'learning_rate': 0.01,
        'max_depth': 6,
        'num_leaves': 31,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'min_child_samples': 200,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'min_gain_to_split': 0.01,
        'random_state': seed,
        'n_jobs': -1,
    }
    if custom_params:
        base_params.update(custom_params)
    base_params['random_state'] = seed

    model = lgb.LGBMRanker(**base_params)
    model.fit(
        X_train, y_train_int,
        group=train_groups,
        eval_set=[(X_val, y_val_int)],
        eval_group=[val_groups],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    return model


def _compute_groups(df_subset):
    """Return group sizes for LGBMRanker: number of symbols per timestamp."""
    return df_subset.groupby('timestamp').size().values


def train_multi_seed(X_train, y_train, X_val, y_val, X_test,
                     params=None, seeds=None,
                     use_ranker=False, train_groups=None, val_groups=None,
                     sample_weight=None):
    seeds = seeds or SEEDS
    print(f"\n   🌱 Multi-seed ensemble ({len(seeds)} seeds)"
          f"{'[LambdaRank]' if use_ranker else ''}...")

    all_preds = []
    all_models = []
    for i, seed in enumerate(seeds):
        print(f"      Seed {seed} ({i+1}/{len(seeds)})...", end=" ")
        if use_ranker:
            model = train_lgbm_ranker(X_train, y_train, X_val, y_val,
                                      train_groups, val_groups,
                                      custom_params=params, seed=seed)
        else:
            model = train_lgbm(X_train, y_train, X_val, y_val,
                               custom_params=params, seed=seed,
                               sample_weight=sample_weight)
        preds = model.predict(X_test)
        all_preds.append(preds)
        all_models.append(model)
        print(f"iters={model.best_iteration_}")

    ensemble_pred = np.mean(all_preds, axis=0)
    return ensemble_pred, all_models


def feature_selection(model, feat_cols, threshold_pct=20):
    imp = pd.Series(model.feature_importances_, index=feat_cols)
    threshold = np.percentile(imp.values, threshold_pct)
    keep = imp[imp > threshold].index.tolist()
    print(f"   🔪 Feature selection: {len(feat_cols)} → {len(keep)}")
    return keep


def null_importance_filter(X_train, y_train, X_val, y_val, feat_cols,
                           n_shuffles=5, significance=0.90, seed=42):
    """Compare feature importance on real target vs shuffled targets."""
    print(f"   🎲 Null importance filter ({n_shuffles} shuffles, sig={significance:.0%})...")

    real_model = train_lgbm(X_train, y_train, X_val, y_val, seed=seed)
    real_imp = pd.Series(real_model.feature_importances_, index=feat_cols)

    null_imps = []
    for i in range(n_shuffles):
        y_shuffled = y_train.copy()
        rng = np.random.RandomState(seed + i + 1)
        y_shuffled[:] = rng.permutation(y_shuffled.values)
        null_model = train_lgbm(X_train, y_shuffled, X_val, y_val, seed=seed + i + 1)
        null_imps.append(pd.Series(null_model.feature_importances_, index=feat_cols))
        print(f"      shuffle {i+1}/{n_shuffles} done")

    null_df = pd.DataFrame(null_imps)
    null_threshold = null_df.quantile(significance, axis=0)
    keep = real_imp[real_imp > null_threshold].index.tolist()
    dropped = [f for f in feat_cols if f not in keep]
    print(f"   ✅ Null importance: {len(feat_cols)} → {len(keep)} "
          f"(dropped {len(dropped)} noise features)")
    return keep


# ============================================================
# EVALUATION (with risk overlay)
# ============================================================

def compute_ic(p, a):
    m = ~(np.isnan(p) | np.isnan(a))
    return np.corrcoef(p[m], a[m])[0, 1] if m.sum() >= 10 else np.nan

def compute_rank_ic(p, a):
    m = ~(np.isnan(p) | np.isnan(a))
    if m.sum() < 10:
        return np.nan
    c, _ = spearmanr(p[m], a[m])
    return c


def compute_costs_per_period(horizon_hours=4):
    """
    Realistic cost per rebalance period for perpetual swaps.
    Components:
    - Taker fee × turnover (buy + sell)
    - Slippage × turnover
    - Funding rate prorated per period
    """
    periods_per_8h = 8 / horizon_hours
    funding_per_period = COST_MODEL['funding_per_8h'] / periods_per_8h
    trade_cost = (COST_MODEL['taker_fee'] + COST_MODEL['slippage']) * 2  # round-trip
    cost_per_period = trade_cost * COST_MODEL['turnover_pct'] + funding_per_period
    return cost_per_period


def evaluate_model(df_test, pred_col, target_col, horizon_hours=4, label=""):
    """
    Comprehensive evaluation with risk overlay.
    Returns dict of metrics + per-period returns for equity curve.
    """
    periods_per_day = 24 // horizon_hours
    periods_per_year = periods_per_day * 365
    cost_per_period = compute_costs_per_period(horizon_hours)

    df_eval = df_test.copy()
    df_eval['date'] = df_eval['timestamp'].dt.date

    # Per-cross-section metrics
    daily_ics, daily_rank_ics = [], []
    for _, grp in df_eval.groupby('date'):
        if len(grp) >= 10:
            daily_ics.append(compute_ic(grp[pred_col].values, grp[target_col].values))
            daily_rank_ics.append(compute_rank_ic(grp[pred_col].values, grp[target_col].values))

    daily_ics = np.array([x for x in daily_ics if not np.isnan(x)])
    daily_rank_ics = np.array([x for x in daily_rank_ics if not np.isnan(x)])

    ic = compute_ic(df_eval[pred_col].values, df_eval[target_col].values)
    rank_ic = compute_rank_ic(df_eval[pred_col].values, df_eval[target_col].values)
    icir = daily_ics.mean() / (daily_ics.std() + 1e-10) if len(daily_ics) > 0 else 0
    rank_icir = daily_rank_ics.mean() / (daily_rank_ics.std() + 1e-10) if len(daily_rank_ics) > 0 else 0

    # --- Long-Short returns ---
    ls_rets_raw, ls_rets_net = [], []
    lo5_rets, lo10_rets = [], []
    ls_timestamps = []

    for ts, grp in df_eval.groupby('timestamp'):
        if len(grp) < 10:
            continue
        grp = grp.sort_values(pred_col, ascending=False)
        n = max(len(grp) // 5, 1)
        long_ret = grp.head(n)[target_col].mean()
        short_ret = grp.tail(n)[target_col].mean()
        ls_ret = long_ret - short_ret
        ls_rets_raw.append(ls_ret)
        ls_rets_net.append(ls_ret - cost_per_period * 2)  # cost on both sides
        lo5_rets.append(grp.head(5)[target_col].mean())
        lo10_rets.append(grp.head(10)[target_col].mean())
        ls_timestamps.append(ts)

    ls_rets_raw = np.array(ls_rets_raw)
    ls_rets_net = np.array(ls_rets_net)
    lo5 = np.array(lo5_rets) - cost_per_period
    lo10 = np.array(lo10_rets) - cost_per_period

    def sharpe(r, ppyr):
        return (r.mean() / (r.std() + 1e-10)) * np.sqrt(ppyr)

    def max_dd(r):
        if len(r) == 0:
            return 0
        cum = np.cumprod(1 + r)
        return np.min(cum / np.maximum.accumulate(cum) - 1)

    def total_ret(r):
        return np.prod(1 + r) - 1

    # --- Vol-targeted returns ---
    # Scale LS returns by inverse of trailing 48h realized vol
    ls_vol_target = vol_target_returns(ls_rets_raw, lookback=48, target_vol=0.02,
                                       cost_per_period=cost_per_period)

    # --- Drawdown-stopped returns ---
    ls_dd_stop = drawdown_stop_returns(ls_rets_net, max_dd_threshold=-0.25,
                                        recovery_threshold=-0.10)

    metrics = {
        'IC': round(float(ic), 4),
        'Rank_IC': round(float(rank_ic), 4),
        'ICIR': round(float(icir), 4),
        'Rank_ICIR': round(float(rank_icir), 4),
        # Raw LS (no costs)
        'LS_Sharpe_raw': round(float(sharpe(ls_rets_raw, periods_per_year)), 2),
        # LS with realistic costs
        'LS_Sharpe_net': round(float(sharpe(ls_rets_net, periods_per_year)), 2),
        'LS_Ann_Return_net_%': round(float(ls_rets_net.mean() * periods_per_year * 100), 1),
        'LS_MaxDD_net_%': round(float(max_dd(ls_rets_net) * 100), 1),
        'LS_Total_net_%': round(float(total_ret(ls_rets_net) * 100), 1),
        # Vol-targeted LS
        'LS_VolTarget_Sharpe': round(float(sharpe(ls_vol_target, periods_per_year)), 2),
        'LS_VolTarget_MaxDD_%': round(float(max_dd(ls_vol_target) * 100), 1),
        'LS_VolTarget_Total_%': round(float(total_ret(ls_vol_target) * 100), 1),
        # Drawdown-stopped LS
        'LS_DDStop_Sharpe': round(float(sharpe(ls_dd_stop, periods_per_year)), 2),
        'LS_DDStop_MaxDD_%': round(float(max_dd(ls_dd_stop) * 100), 1),
        'LS_DDStop_Total_%': round(float(total_ret(ls_dd_stop) * 100), 1),
        # Long-only
        'LO5_Sharpe': round(float(sharpe(lo5, periods_per_year)), 2),
        'LO10_Sharpe': round(float(sharpe(lo10, periods_per_year)), 2),
        'N_periods': len(ls_rets_raw),
        'Cost_per_period_bps': round(cost_per_period * 10000, 1),
    }

    return metrics, ls_rets_net, ls_vol_target, ls_dd_stop, ls_timestamps


def vol_target_returns(raw_rets, lookback=48, target_vol=0.02, cost_per_period=0.0):
    """
    Volatility targeting: scale position by target_vol / realized_vol.
    This reduces position during high vol and increases during low vol.
    Cap leverage at 2x, floor at 0.1x.
    """
    n = len(raw_rets)
    vt_rets = np.zeros(n)

    for i in range(n):
        if i < lookback:
            # Not enough history — use full position
            scale = 1.0
        else:
            realized_vol = np.std(raw_rets[max(0, i-lookback):i])
            if realized_vol > 1e-6:
                scale = target_vol / realized_vol
            else:
                scale = 1.0

        scale = np.clip(scale, 0.1, 2.0)
        vt_rets[i] = raw_rets[i] * scale - cost_per_period * 2 * scale

    return vt_rets


def drawdown_stop_returns(net_rets, max_dd_threshold=-0.25, recovery_threshold=-0.10):
    """
    Circuit breaker: stop trading when drawdown exceeds threshold.
    Resume when drawdown recovers above recovery_threshold.

    When stopped: equity is FLAT (positions closed). We track a separate
    'shadow_equity' from net_rets to know when the market has recovered
    enough to resume, but our actual trading equity stays flat.
    """
    n = len(net_rets)
    stopped_rets = np.zeros(n)
    equity = 1.0          # actual trading equity (flat when stopped)
    shadow_equity = 1.0   # tracks what would happen if we kept trading
    peak = 1.0
    is_stopped = False

    for i in range(n):
        if is_stopped:
            # Track shadow equity to decide when to resume
            shadow_equity *= (1 + net_rets[i])
            shadow_dd = shadow_equity / peak - 1
            if shadow_dd > recovery_threshold:
                is_stopped = False
                # Resume from NEXT bar (i+1), not this one — avoids lookahead
                # This bar we're still flat (positions not yet reopened)
            # else: equity stays flat (positions closed)
        else:
            equity *= (1 + net_rets[i])
            shadow_equity *= (1 + net_rets[i])
            if equity > peak:
                peak = equity
            dd = equity / peak - 1
            if dd < max_dd_threshold:
                is_stopped = True
                stopped_rets[i] = 0  # stop next period
            else:
                stopped_rets[i] = net_rets[i]

    return stopped_rets


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--results', type=str, default=None)
    parser.add_argument('--hpo-trials', type=int, default=50)
    parser.add_argument('--skip-hpo', action='store_true')
    parser.add_argument('--single-window', action='store_true',
                        help='Use only window 3 (same as v4) for quick test')
    parser.add_argument('--production', action='store_true',
                        help='Production mode: max training data, no test holdout')
    parser.add_argument('--train-end', type=str, default=None,
                        help='Override train cutoff date (YYYY-MM-DD) for --production')
    parser.add_argument('--val-end', type=str, default=None,
                        help='Override val end date (YYYY-MM-DD) for --production')
    parser.add_argument('--seeds', type=int, default=N_SEEDS)
    parser.add_argument('--residual-target', action='store_true',
                        help='Use beta-residual returns (remove BTC factor) for target')
    parser.add_argument('--hybrid-norm', action='store_true',
                        help='Hybrid normalization: CS-rank + TS-zscore for spike features')
    parser.add_argument('--lambdarank', action='store_true',
                        help='Use LambdaRank (LGBMRanker) instead of LGBMRegressor')
    parser.add_argument('--huber', action='store_true',
                        help='Use Huber loss instead of MSE (robust to outlier returns)')
    parser.add_argument('--huber-alpha', type=float, default=0.9,
                        help='Huber loss alpha parameter (0.5-0.99, higher = closer to MSE)')
    parser.add_argument('--deadzone-weight', type=float, default=0.0,
                        help='Dead-zone sample weighting: down-weight |ret| < threshold. '
                             'Value is threshold in %% (e.g. 0.3 = 0.3%%). 0=off.')
    parser.add_argument('--null-importance', action='store_true',
                        help='Use null-importance feature selection instead of gain-based')
    parser.add_argument('--no-news', action='store_true',
                        help='Skip loading crypto news features (for clean A/B tests)')
    parser.add_argument('--news-mode', type=str, default='all',
                        choices=['all', 'market-only', 'coin-only', 'none'],
                        help='News feature scope (v7 has no news by default, for consistency)')
    parser.add_argument('--no-derivatives', action='store_true',
                        help='Skip loading Binance derivatives features (for clean A/B tests)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for LightGBM training (requires lightgbm built with GPU support)')
    parser.add_argument('--research', action='store_true',
                        help='Research mode: use RESEARCH_WINDOWS for proper OOS evaluation')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data or os.path.join(project_root, 'data', 'features')
    if args.production:
        results_dir = args.results or os.path.join(project_root, 'results_v7_prod')
    elif args.research:
        results_dir = args.results or os.path.join(project_root, 'results_v7_research')
    else:
        results_dir = args.results or os.path.join(project_root, 'results_v7')
    os.makedirs(results_dir, exist_ok=True)

    feat_path = os.path.join(data_dir, 'crypto_features_1h.parquet')
    if not os.path.exists(feat_path):
        print(f"❌ Feature file not found: {feat_path}")
        sys.exit(1)

    print("=" * 70)
    print("  CRYPTO ALPHA MODEL v7")
    print("  12h Blended Target + Enhanced Features + Walk-Forward")
    print("=" * 70)

    # ========================================
    # 1. LOAD & ENRICH DATA
    # ========================================
    print(f"\n📊 Loading data...")
    df = pd.read_parquet(feat_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Shape: {df.shape}, Symbols: {df['symbol'].nunique()}")

    df = add_multi_horizon_targets(df)
    df = add_cross_asset_features(df)
    if args.residual_target:
        df = add_residual_targets(df, beta_window=168)
    from run_pipeline_v6 import add_market_mode_features, add_liquidity_features
    df = add_market_mode_features(df)
    df = add_liquidity_features(df)
    df = add_advanced_regime_features(df)
    df = add_12h_features(df)
    from run_pipeline_v6 import add_calendar_features, add_macro_features
    df = add_calendar_features(df)
    df = add_macro_features(df, project_root)
    # News support: v7's add_sentiment_features has NO news loading code.
    # v6's add_sentiment_features is a SUPERSET of v7's (same FNG/funding/LSR/synthetic
    # code, plus news). The v7-unique features (cum_funding, funding_cs_rank) are in
    # add_12h_features, already called above — NOT in add_sentiment_features.
    # IMPORTANT: cannot call BOTH — pd.merge creates _x/_y suffix columns on second call
    # (fng_value_x, funding_rate_x, etc.) causing KeyError downstream.
    _news_mode = 'none' if args.no_news else getattr(args, 'news_mode', 'none')
    if _news_mode != 'none':
        from run_pipeline_v6 import add_sentiment_features as add_sentiment_features_v6
        df = add_sentiment_features_v6(df, project_root, news_mode=_news_mode)
        print(f"   📰 v7+news: used v6 sentiment function (mode={_news_mode})")
    else:
        df = add_sentiment_features(df, project_root)
    if not args.no_derivatives:
        df = add_derivatives_features(df, project_root)
    else:
        print("   ⏭️  Skipping derivatives features (--no-derivatives)")

    # Clean infinities
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=['target_ret_blended'])   # v7: blended 12h+24h target

    # Feature columns
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS
                 and not c.startswith('target_')]
    feat_cols = [c for c in feat_cols if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    print(f"   Features: {len(feat_cols)}")

    df[feat_cols] = df[feat_cols].fillna(0)

    # Cross-sectional rank normalization
    df = cross_sectional_rank(df, feat_cols, hybrid=args.hybrid_norm)
    df = create_rank_target(df, HORIZON, use_excess=args.residual_target)

    print(f"   Final shape: {df.shape}")
    print(f"   Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # ========================================
    # 2. ROLLING WALK-FORWARD
    # ========================================
    if args.production:
        from copy import deepcopy
        prod_win = deepcopy(PRODUCTION_WINDOW)
        if args.train_end:
            prod_win['train_end'] = args.train_end
            te = pd.Timestamp(args.train_end)
            prod_win['val_start'] = (te + pd.Timedelta(days=PURGE_DAYS)).strftime('%Y-%m-%d')
        if args.val_end:
            prod_win['val_end'] = args.val_end
            prod_win['test_start'] = args.val_end
        windows = [prod_win]
        print(f"\n🔴 PRODUCTION MODE — max training data, models go to live trading")
        print(f"   Train: start → {prod_win['train_end']}")
        print(f"   Val:   {prod_win['val_start']} → {prod_win['val_end']}")
        print(f"   Test:  (none — live trading)")
    elif args.research:
        windows = RESEARCH_WINDOWS
        if not args.results:
            results_dir = os.path.join(project_root, 'results_v7_research')
            os.makedirs(results_dir, exist_ok=True)
        print(f"\n🔬 RESEARCH MODE — proper OOS evaluation on 2025 & 2026")
        for i, w in enumerate(windows):
            print(f"   R{i+1}: train→{w['train_end']}  val {w['val_start']}→{w['val_end']}  test {w['test_start']}→{w['test_end']}")
    else:
        windows = WALK_FORWARD_WINDOWS
        if args.single_window:
            windows = [windows[-1]]  # Window 3 = same as v4

    print(f"\n{'='*70}")
    print(f"  ROLLING WALK-FORWARD ({len(windows)} windows)")
    print(f"{'='*70}")

    target_col = f'target_ret_{HORIZON}h'
    all_window_metrics = []
    all_test_predictions = []
    combined_ls_rets = []
    combined_timestamps = []

    for w_idx, window in enumerate(windows):
        print(f"\n{'─'*70}")
        print(f"  Window {w_idx+1}/{len(windows)}: {window['name']}")
        print(f"  Train: → {window['train_end']}")
        print(f"  Val:   {window['val_start']} → {window['val_end']}")
        print(f"  Test:  {window['test_start']} → {window['test_end']}")
        print(f"{'─'*70}")

        train = df[df['timestamp'] < window['train_end']].copy()
        val = df[(df['timestamp'] >= window['val_start']) &
                 (df['timestamp'] < window['val_end'])].copy()
        test = df[(df['timestamp'] >= window['test_start']) &
                  (df['timestamp'] <= window['test_end'])].copy()

        has_test = len(test) > 0
        if not has_test and not args.production:
            print(f"   ⚠️  No test data for this window, skipping")
            continue

        print(f"   Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

        X_train, y_train = train[feat_cols], train['target_rank']
        X_val, y_val = val[feat_cols], val['target_rank']
        X_test = test[feat_cols]
        val_dates = val['timestamp'].dt.date.values

        # --- HPO (only for first window if not skipped) ---
        best_params = None
        if not args.skip_hpo and w_idx == 0:
            best_params = run_optuna_hpo(
                X_train, y_train, X_val, y_val, val_dates,
                n_trials=args.hpo_trials
            )
        # NOTE: LightGBM GPU requires OpenCL (not available on most CUDA clusters).
        # LGB trains fast enough on CPU. GPU flag only affects CatBoost/XGBoost.

        # --- Feature selection ---
        if args.null_importance:
            selected_feats = null_importance_filter(
                X_train, y_train, X_val, y_val, feat_cols,
                n_shuffles=5, significance=0.90,
            )
        else:
            model_base = train_lgbm(X_train, y_train, X_val, y_val,
                                    custom_params=best_params)
            selected_feats = feature_selection(model_base, feat_cols, threshold_pct=20)

        # --- Multi-seed ensemble ---
        X_pred = test[selected_feats] if has_test else val[selected_feats]

        # --- Huber loss override ---
        if args.huber:
            if best_params is None:
                best_params = {}
            best_params['objective'] = 'huber'
            best_params['alpha'] = args.huber_alpha  # LightGBM Huber param name is 'alpha'
            print(f"   🛡️  Huber loss enabled (alpha={args.huber_alpha})")

        # --- Dead-zone sample weighting ---
        sw = None
        if args.deadzone_weight > 0:
            tau = args.deadzone_weight / 100.0  # e.g. 0.3 → 0.003
            # v7 uses blended target, find raw return column
            raw_col = f'target_ret_{HORIZON}h'
            if raw_col in train.columns:
                raw_target = train[raw_col]
            else:
                print(f"   ⚠️  Dead-zone: '{raw_col}' not found, falling back to "
                      f"'target_ret_blended' (target_rank would be a no-op)")
                if 'target_ret_blended' in train.columns:
                    raw_target = train['target_ret_blended']
                else:
                    print(f"   ❌  Dead-zone: no raw return column found, SKIPPING deadzone")
                    raw_target = None
            if raw_target is not None:
                sw = np.where(np.abs(raw_target) > tau, 1.0, 0.2)
                pct_upweight = (sw > 0.5).mean() * 100
                print(f"   ⚖️  Dead-zone weighting: τ={args.deadzone_weight}%, "
                      f"{pct_upweight:.0f}% samples full-weight")

        ranker_kwargs = {}
        if args.lambdarank:
            ranker_kwargs['use_ranker'] = True
            ranker_kwargs['train_groups'] = _compute_groups(train)
            ranker_kwargs['val_groups'] = _compute_groups(val)

        ensemble_pred, all_models = train_multi_seed(
            train[selected_feats], y_train,
            val[selected_feats], y_val,
            X_pred,
            params=best_params,
            seeds=SEEDS[:args.seeds],
            sample_weight=sw,
            **ranker_kwargs,
        )
        if has_test:
            test['pred_v7'] = ensemble_pred
        last_model = all_models[-1]

        # --- Save trained models (for production inference) ---
        for i, mdl in enumerate(all_models):
            seed = SEEDS[:args.seeds][i]
            model_path = os.path.join(results_dir, f'lgb_model_seed_{seed}.txt')
            if hasattr(mdl, 'booster_'):
                mdl.booster_.save_model(model_path)
            else:
                mdl.save_model(model_path)
        # Save selected feature names
        with open(os.path.join(results_dir, 'feature_names.json'), 'w') as f:
            json.dump(selected_feats, f)
        print(f"   💾 Saved {len(all_models)} models + feature names")

        if not has_test:
            print(f"\n   ✅ Production models saved (no test evaluation)")
            if args.production:
                prod_meta = {
                    'mode': 'production',
                    'train_end': window['train_end'],
                    'val_end': window['val_end'],
                    'n_seeds': args.seeds,
                    'n_features': len(selected_feats),
                    'train_rows': len(train),
                    'val_rows': len(val),
                    'timestamp': datetime.now().isoformat(),
                }
                with open(os.path.join(results_dir, 'production_meta.json'), 'w') as f:
                    json.dump(prod_meta, f, indent=2)
            continue

        # --- Evaluate ---
        metrics, ls_net, ls_vt, ls_dd, timestamps = evaluate_model(
            test, 'pred_v7', target_col, HORIZON, label=window['name']
        )
        metrics['window'] = window['name']
        all_window_metrics.append(metrics)

        # Store test predictions and returns
        save_cols = ['timestamp', 'symbol', target_col, 'pred_v7']
        save_cols = [c for c in save_cols if c in test.columns]
        all_test_predictions.append(test[save_cols].copy())
        combined_ls_rets.extend(ls_net.tolist())
        combined_timestamps.extend(timestamps)

        # Print window results
        print(f"\n   📈 Window {w_idx+1} Results:")
        for k, v in metrics.items():
            if k == 'window':
                continue
            print(f"      {k:30s} {v}")

    # ========================================
    # 3. AGGREGATE ACROSS WINDOWS
    # ========================================
    print(f"\n{'='*70}")
    print(f"  AGGREGATE RESULTS ({len(all_window_metrics)} windows)")
    print(f"{'='*70}")

    if len(all_window_metrics) == 0:
        avg_metrics = {}
        print("   ⚠️  Production mode — no test windows, skipping metrics.")
    elif len(all_window_metrics) > 1:
        # Average metrics across windows
        metric_keys = [k for k in all_window_metrics[0].keys()
                       if k != 'window' and isinstance(all_window_metrics[0][k], (int, float))]

        avg_metrics = {}
        for k in metric_keys:
            vals = [m[k] for m in all_window_metrics]
            avg_metrics[k] = round(np.mean(vals), 4)
            avg_metrics[f'{k}_std'] = round(np.std(vals), 4)

        print(f"\n   📊 Per-Window Comparison:")
        header = f"   {'Metric':<35s}"
        for m in all_window_metrics:
            header += f" {m['window']:>12s}"
        header += f" {'AVG':>10s}"
        print(header)
        print(f"   {'─'*85}")

        key_metrics = ['Rank_IC', 'Rank_ICIR', 'LS_Sharpe_net', 'LS_MaxDD_net_%',
                        'LS_VolTarget_Sharpe', 'LS_VolTarget_MaxDD_%',
                        'LS_DDStop_Sharpe', 'LS_DDStop_MaxDD_%']
        for k in key_metrics:
            row = f"   {k:<35s}"
            for m in all_window_metrics:
                row += f" {m.get(k, 'N/A'):>12}"
            row += f" {avg_metrics.get(k, 'N/A'):>10}"
            print(row)
    else:
        avg_metrics = {k: v for k, v in all_window_metrics[0].items() if k != 'window'}

    # ========================================
    # 4. COMBINED EQUITY CURVE  
    # ========================================
    if len(combined_ls_rets) == 0:
        print("   ⚠️  No test returns — skipping equity curve & combined stats.")
        # Save minimal results for production mode
        all_results = {
            'per_window': all_window_metrics,
            'average': avg_metrics,
            'combined': {},
            'cost_model': COST_MODEL,
            'meta': {
                'timestamp': datetime.now().isoformat(),
                'horizon': HORIZON,
                'n_features': len(feat_cols),
                'n_selected': len(selected_feats) if 'selected_feats' in dir() else 0,
                'n_seeds': args.seeds,
                'n_windows': len(windows),
                'hpo_trials': args.hpo_trials if not args.skip_hpo else 0,
                'production_mode': True,
            },
        }
        with open(os.path.join(results_dir, 'all_results_v7.json'), 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n✅ Production models saved to {results_dir}/")
        return

    combined_ls = np.array(combined_ls_rets)
    periods_per_year = (24 // HORIZON) * 365

    def sharpe(r, ppyr):
        return (r.mean() / (r.std() + 1e-10)) * np.sqrt(ppyr)
    def max_dd(r):
        cum = np.cumprod(1 + r)
        return np.min(cum / np.maximum.accumulate(cum) - 1)
    def total_ret(r):
        return np.prod(1 + r) - 1

    # Apply vol targeting and DD stop to combined
    cost = compute_costs_per_period(HORIZON)
    combined_vt = vol_target_returns(combined_ls + cost * 2, lookback=48,
                                      target_vol=0.02, cost_per_period=cost)
    combined_dd = drawdown_stop_returns(combined_ls, max_dd_threshold=-0.25,
                                         recovery_threshold=-0.10)

    print(f"\n   📈 Combined Results (all windows):")
    print(f"      Periods: {len(combined_ls)}")
    print(f"      LS Net:       Sharpe={sharpe(combined_ls, periods_per_year):.2f}, "
          f"MaxDD={max_dd(combined_ls)*100:.1f}%, "
          f"Total={total_ret(combined_ls)*100:.1f}%")
    print(f"      LS VolTarget: Sharpe={sharpe(combined_vt, periods_per_year):.2f}, "
          f"MaxDD={max_dd(combined_vt)*100:.1f}%, "
          f"Total={total_ret(combined_vt)*100:.1f}%")
    print(f"      LS DDStop:    Sharpe={sharpe(combined_dd, periods_per_year):.2f}, "
          f"MaxDD={max_dd(combined_dd)*100:.1f}%, "
          f"Total={total_ret(combined_dd)*100:.1f}%")

    # ========================================
    # 5. FEATURE IMPORTANCE (from last window)
    # ========================================
    if 'last_model' in dir() and last_model is not None:
        importance = pd.DataFrame({
            'feature': selected_feats,
            'importance': last_model.feature_importances_,
        }).sort_values('importance', ascending=False)

        print(f"\n🏆 Top 30 Features (last window):")
        for _, row in importance.head(30).iterrows():
            marker = "📰" if any(s in row['feature'] for s in
                                  ['fng', 'funding', 'reversal', 'surge', 'dispersion',
                                   'long_short', 'beta']) else "  "
            print(f"   {marker} {row['feature']:40s} {row['importance']:.0f}")

        # Count sentiment features in top 30
        sent_keywords = ['fng', 'funding', 'reversal', 'surge', 'dispersion',
                          'long_short', 'beta', 'crowding']
        top30_sent = sum(1 for _, r in importance.head(30).iterrows()
                         if any(k in r['feature'] for k in sent_keywords))
        print(f"\n   📰 Sentiment features in top 30: {top30_sent}/30")

    # ========================================
    # 6. SAVE RESULTS
    # ========================================
    all_results = {
        'per_window': all_window_metrics,
        'average': avg_metrics,
        'combined': {
            'LS_Net_Sharpe': round(float(sharpe(combined_ls, periods_per_year)), 2),
            'LS_Net_MaxDD_%': round(float(max_dd(combined_ls) * 100), 1),
            'LS_Net_Total_%': round(float(total_ret(combined_ls) * 100), 1),
            'LS_VolTarget_Sharpe': round(float(sharpe(combined_vt, periods_per_year)), 2),
            'LS_VolTarget_MaxDD_%': round(float(max_dd(combined_vt) * 100), 1),
            'LS_DDStop_Sharpe': round(float(sharpe(combined_dd, periods_per_year)), 2),
            'LS_DDStop_MaxDD_%': round(float(max_dd(combined_dd) * 100), 1),
        },
        'cost_model': COST_MODEL,
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'horizon': HORIZON,
            'n_features': len(feat_cols),
            'n_selected': len(selected_feats) if 'selected_feats' in dir() else 0,
            'n_seeds': args.seeds,
            'n_windows': len(windows),
            'hpo_trials': args.hpo_trials if not args.skip_hpo else 0,
        },
    }

    with open(os.path.join(results_dir, 'all_results_v7.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    if 'importance' in dir():
        importance.to_csv(os.path.join(results_dir, 'feature_importance_v7.csv'), index=False)

    # Save test predictions from all windows
    if all_test_predictions:
        combined_preds = pd.concat(all_test_predictions, ignore_index=True)
        combined_preds.to_parquet(
            os.path.join(results_dir, 'test_predictions_v6.parquet'), index=False)

    # Save equity curves
    eq = pd.DataFrame({
        'ls_net': np.cumprod(1 + combined_ls) * 1000,
        'ls_vol_target': np.cumprod(1 + combined_vt) * 1000,
        'ls_dd_stop': np.cumprod(1 + combined_dd) * 1000,
    })
    eq.to_parquet(os.path.join(results_dir, 'equity_curves_v6.parquet'), index=False)

    # ========================================
    # FINAL VERDICT
    # ========================================
    best_sharpe = avg_metrics.get('LS_VolTarget_Sharpe', avg_metrics.get('LS_Sharpe_net', 0))
    best_dd = avg_metrics.get('LS_VolTarget_MaxDD_%', avg_metrics.get('LS_MaxDD_net_%', 0))

    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY v7 (12h blended target)")
    print(f"{'='*70}")
    print(f"   Rank IC (avg):            {avg_metrics.get('Rank_IC', 0):+.4f}")
    print(f"   Rank ICIR (avg):          {avg_metrics.get('Rank_ICIR', 0):+.4f}")
    print(f"   LS Sharpe net (avg):      {avg_metrics.get('LS_Sharpe_net', 0):+.2f}")
    print(f"   LS MaxDD net (avg):       {avg_metrics.get('LS_MaxDD_net_%', 0):.1f}%")
    print(f"   LS VolTarget Sharpe:      {avg_metrics.get('LS_VolTarget_Sharpe', 0):+.2f}")
    print(f"   LS VolTarget MaxDD:       {avg_metrics.get('LS_VolTarget_MaxDD_%', 0):.1f}%")
    print(f"   LS DDStop Sharpe:         {avg_metrics.get('LS_DDStop_Sharpe', 0):+.2f}")
    print(f"   LS DDStop MaxDD:          {avg_metrics.get('LS_DDStop_MaxDD_%', 0):.1f}%")
    print(f"   Cost model:               {COST_MODEL['taker_fee']*100:.2f}% taker + "
          f"{COST_MODEL['slippage']*100:.2f}% slip + "
          f"{COST_MODEL['funding_per_8h']*100:.3f}%/8h funding")
    print(f"{'='*70}")

    if best_sharpe > 3.0 and best_dd > -30:
        print("🟢 STRONG — Signal robust across windows with controlled risk.")
        print("   → Ready for HIST v2 with sentiment, then paper trading.")
    elif best_sharpe > 2.0:
        print("🟡 DECENT — Signal exists, risk partially controlled.")
        print("   → Consider tighter DD stop or reduce position sizing.")
    elif best_sharpe > 1.0:
        print("🟠 MARGINAL — Need transformer boost (HIST v2 with sentiment).")
    else:
        print("🔴 WEAK — Fundamental signal issues, revisit features.")

    print(f"\n✅ Results saved to {results_dir}/")


if __name__ == '__main__':
    main()
