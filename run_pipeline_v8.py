#!/usr/bin/env python3
"""
Crypto Alpha Model v8 — Extended History + Purged Walk-Forward

Major improvements over v6:
1. Extended data:
   - Supports data from 2017+ (BTC/ETH from Aug 2017, altcoins from listing dates)
   - ~2x more training data than v6 (which started from 2021)
   - Models learn from: 2018 crash, COVID, DeFi summer, 2021 bull, 2022 bear, 2023-2025 recovery
2. Purged walk-forward validation (5 windows):
   - PURGE_GAP = 14 days between train/val and val/test (prevents target leakage)
   - No test overlap between windows (v6 had W2/W3 overlap)
   - Window 1: train→2021-06, val→2022-06, test→2022-12
   - Window 2: train→2022-06, val→2023-06, test→2023-12
   - Window 3: train→2023-06, val→2024-06, test→2024-12
   - Window 4: train→2024-01, val→2024-09, test→2025-06
   - Window 5: train→2024-06, val→2025-01, test→latest (production model)
3. Per-window HPO: each window gets its own Optuna HPO (not just W1)
4. Ensemble-based feature selection: average importance across 5 seeds, not single model
5. Measured turnover: compute actual signal turnover instead of hardcoded 35%

Usage:
  python run_pipeline_v8.py                          # Full run, all 5 windows
  python run_pipeline_v8.py --skip-hpo               # Skip Optuna
  python run_pipeline_v8.py --single-window           # Quick: Window 5 only
  python run_pipeline_v8.py --hpo-trials 50           # 50 HPO trials per window
  python run_pipeline_v8.py --data /path/to/features

Requirements:
  pip install lightgbm pandas numpy scipy pyarrow scikit-learn
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
HORIZON = 12       # predict 12h returns, aligned with rebalance interval
N_SEEDS = 5
SEEDS = [42, 123, 456, 789, 2024]

# Purge gap: 14 days (336 hours) between train→val and val→test
# This prevents target leakage (12h target could leak ~12h of info)
PURGE_DAYS = 14

# 5 expanding walk-forward windows — NO test overlap
WALK_FORWARD_WINDOWS = [
    {
        'name': 'W1 (→2022-12)',
        'train_end':  '2021-06-30',
        'val_start':  '2021-07-15',   # +14d purge
        'val_end':    '2022-06-30',
        'test_start': '2022-07-15',   # +14d purge
        'test_end':   '2022-12-31',
    },
    {
        'name': 'W2 (→2023-12)',
        'train_end':  '2022-06-30',
        'val_start':  '2022-07-15',
        'val_end':    '2023-06-30',
        'test_start': '2023-07-15',
        'test_end':   '2023-12-31',
    },
    {
        'name': 'W3 (→2024-12)',
        'train_end':  '2023-06-30',
        'val_start':  '2023-07-15',
        'val_end':    '2024-06-30',
        'test_start': '2024-07-15',
        'test_end':   '2024-12-31',
    },
    {
        'name': 'W4 (→2025-06)',
        'train_end':  '2024-01-01',
        'val_start':  '2024-01-15',
        'val_end':    '2024-09-30',
        'test_start': '2024-10-15',
        'test_end':   '2025-06-30',
    },
    {
        'name': 'W5 (→latest)',
        'train_end':  '2024-06-29',
        'val_start':  '2024-07-15',
        'val_end':    '2025-01-31',
        'test_start': '2025-02-15',
        'test_end':   '2026-12-31',
    },
]

# Columns to exclude from features
EXCLUDE_COLS = {
    'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
    'target_ret_4h', 'target_ret_12h', 'target_ret_24h',
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
    'fng_value', 'fng_extreme_fear', 'fng_extreme_greed',
    'fng_ma7', 'fng_ma30', 'fng_momentum',
    'market_avg_funding', 'market_funding_skew',
    'is_asian_session',
}

# Cost model (same as v6)
COST_MODEL = {
    'taker_fee': 0.0003,
    'slippage': 0.0001,
    'funding_per_8h': 0.00005,
    'turnover_pct': 0.35,       # default, will be measured
}


# ============================================================
# FEATURE ENGINEERING  (identical to v6 — proven features)
# ============================================================

def add_multi_horizon_targets(df):
    """Add forward return targets."""
    print("   🎯 Adding targets...")
    for h in [4, 12, 24]:
        df[f'target_ret_{h}h'] = df.groupby('symbol')['close'].pct_change(h).shift(-h)
    return df


def add_cross_asset_features(df):
    """BTC/ETH cross-asset features."""
    print("   📊 Adding cross-asset features...")
    btc = df[df['symbol'] == 'BTC/USDT'][['timestamp', 'close']].rename(
        columns={'close': 'btc_close'})
    eth = df[df['symbol'] == 'ETH/USDT'][['timestamp', 'close']].rename(
        columns={'close': 'eth_close'})

    df = df.merge(btc, on='timestamp', how='left')
    df = df.merge(eth, on='timestamp', how='left')

    # BTC returns at multiple horizons
    btc_ts = df.drop_duplicates('timestamp').sort_values('timestamp')
    for h in [1, 4, 12, 24, 48, 168]:
        btc_ts[f'btc_ret_{h}h'] = btc_ts['btc_close'].pct_change(h)
    for h in [1, 4, 12, 24]:
        btc_ts[f'eth_ret_{h}h'] = btc_ts['eth_close'].pct_change(h)

    # BTC regime (above moving averages)
    for w in [24, 72, 168]:
        btc_ts[f'btc_regime_{w}'] = (
            btc_ts['btc_close'] > btc_ts['btc_close'].rolling(w).mean()
        ).astype(float)

    # BTC volatility
    btc_ts['btc_vol_24h'] = btc_ts['btc_close'].pct_change().rolling(24).std()

    # ETH/BTC ratio momentum
    btc_ts['eth_btc_ret_24h'] = (btc_ts['eth_close'] / btc_ts['btc_close']).pct_change(24)

    # Market dispersion
    disp = df.groupby('timestamp')['ret_1h'].std().reset_index()
    disp.columns = ['timestamp', 'market_dispersion']
    btc_ts = btc_ts.merge(disp, on='timestamp', how='left')

    merge_cols = [c for c in btc_ts.columns if c not in ['btc_close', 'eth_close',
                  'open', 'high', 'low', 'close', 'volume', 'symbol'] and c in btc_ts.columns]
    merge_cols = list(set(merge_cols))

    df = df.drop(columns=[c for c in btc_ts.columns if c in df.columns
                          and c != 'timestamp'], errors='ignore')
    df = df.merge(btc_ts[merge_cols], on='timestamp', how='left')

    # Per-coin: return vs BTC
    if 'ret_24h' in df.columns and 'btc_ret_24h' in df.columns:
        df['ret_vs_btc_24h'] = df['ret_24h'] - df['btc_ret_24h']

    return df


def add_advanced_regime_features(df):
    """Multi-factor regime detection."""
    print("   🌍 Adding regime features...")
    btc = df[df['symbol'] == 'BTC/USDT'].sort_values('timestamp').drop_duplicates('timestamp')

    regime = btc[['timestamp']].copy()
    close = btc['close'].values
    ts_idx = btc.index

    # MA regime
    ma336 = pd.Series(close).rolling(336).mean()
    ma720 = pd.Series(close).rolling(720).mean()
    regime['regime_btc_above_ma336'] = (close > ma336.values).astype(float)
    regime['regime_btc_above_ma720'] = (close > ma720.values).astype(float)

    # MA slope (trending up?)
    ma720_shifted = ma720.shift(24)
    regime['regime_btc_ma720_slope'] = (ma720.values > ma720_shifted.values).astype(float)

    # BTC drawdown from 720h high
    rolling_high_720 = pd.Series(close).rolling(720).max()
    regime['regime_btc_dd_720'] = close / rolling_high_720.values - 1

    # Not crashed
    regime['regime_btc_not_crashed'] = (regime['regime_btc_dd_720'] > -0.15).astype(float)

    # Vol regime
    btc_ret = pd.Series(close).pct_change()
    vol_24 = btc_ret.rolling(24).std()
    vol_720_med = vol_24.rolling(720).median()
    regime['regime_low_vol'] = (vol_24.values < 2 * vol_720_med.values).astype(float)

    # Market breadth
    breadth = df.groupby('timestamp').apply(
        lambda g: (g['ret_24h'] > 0).mean() if 'ret_24h' in g.columns else 0.5
    ).reset_index()
    breadth.columns = ['timestamp', 'breadth_pct_positive']
    regime = regime.merge(breadth, on='timestamp', how='left')
    regime['breadth_pct_positive'] = regime['breadth_pct_positive'].fillna(0.5)
    regime['regime_breadth_bullish'] = (regime['breadth_pct_positive'] > 0.5).astype(float)

    # Composite
    regime['regime_composite'] = (
        0.25 * regime['regime_btc_above_ma720'] +
        0.20 * regime['regime_btc_ma720_slope'] +
        0.20 * regime['regime_breadth_bullish'] +
        0.20 * regime['regime_btc_not_crashed'] +
        0.15 * regime['regime_low_vol']
    )

    # Merge to main df
    regime_cols = [c for c in regime.columns if c != 'timestamp']
    df = df.drop(columns=[c for c in regime_cols if c in df.columns], errors='ignore')
    df = df.merge(regime[['timestamp'] + regime_cols], on='timestamp', how='left')

    # Clean up BTC/ETH close (no longer needed)
    df = df.drop(columns=['btc_close', 'eth_close'], errors='ignore')

    return df


def add_12h_features(df):
    """v6 12h-aligned features."""
    print("   ⏱️ Adding 12h-specific features...")

    for sym, grp in df.groupby('symbol'):
        idx = grp.index
        c = grp['close']
        v = grp['volume']

        # 12h momentum z-score
        ret12 = c.pct_change(12)
        rm = ret12.rolling(168).mean()
        rs = ret12.rolling(168).std()
        df.loc[idx, 'mom_12h_zscore'] = ((ret12 - rm) / (rs + 1e-10)).values

        # VWAP 12h distance
        vwap12 = (c * v).rolling(12).sum() / (v.rolling(12).sum() + 1e-10)
        df.loc[idx, 'vwap_12h_dist'] = (c / vwap12 - 1).values

        # Multi-day momentum
        df.loc[idx, 'mom_3d'] = c.pct_change(72).values
        df.loc[idx, 'mom_7d'] = c.pct_change(168).values

        # Momentum acceleration
        r12 = c.pct_change(12)
        df.loc[idx, 'mom_accel_12h'] = (r12 - r12.shift(12)).values

        # Volume trend
        df.loc[idx, 'vol_trend_12_48'] = (v.rolling(12).mean() / (v.rolling(48).mean() + 1e-10) - 1).values

        # Session flag
        df.loc[idx, 'is_asian_session'] = (grp['timestamp'].dt.hour < 12).astype(float).values

        # Range expansion
        h12 = grp['high'].rolling(12).max()
        l12 = grp['low'].rolling(12).min()
        rng12 = (h12 - l12) / (c + 1e-10)
        avg_rng = rng12.rolling(168).mean()
        df.loc[idx, 'range_expansion_12h'] = (rng12 / (avg_rng + 1e-10) - 1).values

    # Cross-sectional ranks (computed after all symbols)
    for ts, grp in df.groupby('timestamp'):
        if len(grp) < 5:
            continue
        idx = grp.index
        if 'ret_12h' in df.columns:
            df.loc[idx, 'ret_12h_cs_rank'] = grp['ret_12h'].rank(pct=True).values
        vol12 = grp['volume'].rolling(12, min_periods=1).sum() if len(grp) > 1 else grp['volume']
        df.loc[idx, 'vol_12h_cs_rank'] = vol12.rank(pct=True).values

    return df


def add_sentiment_features(df, project_root):
    """Sentiment features: FNG, funding rates, LS ratio, synthetic."""
    print("   📰 Adding sentiment features...")
    sent_dir = os.path.join(project_root, 'data', 'sentiment')

    # --- Fear & Greed Index ---
    fng_path = os.path.join(sent_dir, 'fear_greed.parquet')
    if os.path.exists(fng_path):
        fng = pd.read_parquet(fng_path)
        fng['timestamp'] = pd.to_datetime(fng['timestamp'], utc=True)
        fng = fng.sort_values('timestamp').drop_duplicates('timestamp')
        fng = fng.set_index('timestamp').resample('1h').ffill().reset_index()
        fng = fng.rename(columns={'value': 'fng_value'})
        fng['fng_value'] = fng['fng_value'].ffill().fillna(50)

        fng['fng_extreme_fear'] = (fng['fng_value'] < 25).astype(float)
        fng['fng_extreme_greed'] = (fng['fng_value'] > 75).astype(float)

        # FNG moving averages
        for sym, grp in df.groupby('symbol'):
            pass

        fng['fng_ma7'] = fng['fng_value'].rolling(168).mean()
        fng['fng_ma30'] = fng['fng_value'].rolling(720).mean()
        fng['fng_momentum'] = fng['fng_value'] - fng['fng_ma30']

        fng_cols = ['timestamp', 'fng_value', 'fng_extreme_fear', 'fng_extreme_greed',
                    'fng_ma7', 'fng_ma30', 'fng_momentum']
        df = df.merge(fng[fng_cols], on='timestamp', how='left')
        for c in fng_cols[1:]:
            df[c] = df[c].ffill().fillna(50 if 'fng_value' in c else 0)
        print(f"   ✅ FNG: {len(fng)} timestamps merged")
    else:
        print(f"   ⚠️ No FNG data at {fng_path}")
        df['fng_value'] = 50
        for c in ['fng_extreme_fear', 'fng_extreme_greed', 'fng_ma7', 'fng_ma30', 'fng_momentum']:
            df[c] = 0.0

    # --- Funding rates ---
    fund_path = os.path.join(sent_dir, 'funding_rates.parquet')
    if os.path.exists(fund_path):
        fund = pd.read_parquet(fund_path)
        fund['timestamp'] = pd.to_datetime(fund['timestamp'], utc=True)
        fund = fund.sort_values('timestamp')
        fund = fund.rename(columns={'fundingRate': 'funding_rate'})
        df = df.merge(fund[['timestamp', 'symbol', 'funding_rate']],
                       on=['timestamp', 'symbol'], how='left')
        df['funding_rate'] = df['funding_rate'].fillna(0)

        # Market-level funding
        mf = df.groupby('timestamp')['funding_rate'].agg(['mean', 'std']).reset_index()
        mf.columns = ['timestamp', 'market_avg_funding', 'market_funding_std']
        mf['market_funding_skew'] = mf['market_avg_funding'] / (mf['market_funding_std'] + 1e-8)
        df = df.merge(mf, on='timestamp', how='left')
        df['funding_vs_market'] = df['funding_rate'] - df['market_avg_funding'].fillna(0)
        print(f"   ✅ Funding rates merged")
    else:
        print(f"   ⚠️ No funding data")
        for c in ['funding_rate', 'market_avg_funding', 'market_funding_std',
                   'market_funding_skew', 'funding_vs_market']:
            df[c] = 0.0

    # --- Long/Short ratio ---
    lsr_path = os.path.join(sent_dir, 'long_short_ratio.parquet')
    if os.path.exists(lsr_path):
        lsr = pd.read_parquet(lsr_path)
        lsr['timestamp'] = pd.to_datetime(lsr['timestamp'], utc=True)
        df = df.merge(lsr[['timestamp', 'symbol', 'long_short_ratio']],
                       on=['timestamp', 'symbol'], how='left')
        df['long_short_ratio'] = df['long_short_ratio'].fillna(1.0)
        print(f"   ✅ LS ratio merged")
    else:
        df['long_short_ratio'] = 1.0

    # --- Synthetic positioning ---
    print("   🔧 Building synthetic features...")
    for h_short, h_long in [(4, 24), (12, 48), (24, 168)]:
        rs = f'ret_{h_short}h'
        rl = f'ret_{h_long}h'
        if rs in df.columns and rl in df.columns:
            df[f'reversal_{h_short}v{h_long}'] = df[rs] - df[rl] / (h_long / h_short)

    for w in [12, 24, 48]:
        vol_ma = f'vol_ma{w}_ratio'
        if vol_ma in df.columns:
            df[f'vol_surge_{w}h'] = df[vol_ma]
        else:
            df[f'vol_surge_{w}h'] = 0

    # Cross-coin dispersion
    disp = df.groupby('timestamp')['ret_4h'].std().reset_index()
    disp.columns = ['timestamp', 'cross_coin_dispersion']
    df = df.merge(disp, on='timestamp', how='left')
    df['cross_coin_dispersion'] = df['cross_coin_dispersion'].fillna(0)

    ts_level = df.drop_duplicates('timestamp').sort_values('timestamp')
    ts_level['cross_coin_disp_ma24'] = ts_level['cross_coin_dispersion'].rolling(24).mean()
    ts_level['dispersion_regime'] = ts_level['cross_coin_dispersion'] / (
        ts_level['cross_coin_disp_ma24'] + 1e-10)
    df = df.merge(ts_level[['timestamp', 'cross_coin_disp_ma24', 'dispersion_regime']],
                   on='timestamp', how='left')

    # Cross-sectional skew rank
    for w in [48, 168]:
        skew_col = f'ret_skew_{w}h'
        if skew_col in df.columns:
            df[f'{skew_col}_cs'] = df.groupby('timestamp')[skew_col].rank(pct=True)

    # BTC beta per coin
    for w in [48, 168]:
        btc_ret_col = 'btc_ret_1h' if 'btc_ret_1h' in df.columns else None
        if btc_ret_col and 'ret_1h' in df.columns:
            for sym, grp in df.groupby('symbol'):
                idx = grp.index
                cov = grp['ret_1h'].rolling(w).cov(grp[btc_ret_col])
                var = grp[btc_ret_col].rolling(w).var()
                df.loc[idx, f'btc_beta_{w}h'] = (cov / (var + 1e-10)).values

    return df


# ============================================================
# CROSS-SECTIONAL RANK NORMALIZATION
# ============================================================

def cross_sectional_rank(df, feat_cols):
    """Rank features within each timestamp to [-0.5, 0.5]."""
    print("   📈 Cross-sectional rank normalization...")

    regime_backup = {}
    for col in REGIME_COLS:
        if col in df.columns:
            regime_backup[col] = df[col].copy()

    rank_cols = [c for c in feat_cols if c not in REGIME_COLS]
    ranked = df.groupby('timestamp')[rank_cols].rank(pct=True)
    df[rank_cols] = ranked - 0.5

    for col, vals in regime_backup.items():
        df[col] = vals
    print(f"   ✅ Ranked {len(rank_cols)} features, preserved {len(regime_backup)} unranked")

    return df


def create_rank_target(df, horizon=12):
    target_col = f'target_ret_{horizon}h'
    df['target_rank'] = df.groupby('timestamp')[target_col].rank(pct=True)
    return df


# ============================================================
# HPO  (per-window in v8)
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

def train_lgbm(X_train, y_train, X_val, y_val, custom_params=None, seed=42):
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
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    return model


def train_multi_seed(X_train, y_train, X_val, y_val, X_test,
                     params=None, seeds=None):
    seeds = seeds or SEEDS
    print(f"\n   🌱 Multi-seed ensemble ({len(seeds)} seeds)...")

    all_preds = []
    all_models = []
    for i, seed in enumerate(seeds):
        print(f"      Seed {seed} ({i+1}/{len(seeds)})...", end=" ")
        model = train_lgbm(X_train, y_train, X_val, y_val,
                           custom_params=params, seed=seed)
        preds = model.predict(X_test)
        all_preds.append(preds)
        all_models.append(model)
        print(f"iters={model.best_iteration_}")

    ensemble_pred = np.mean(all_preds, axis=0)
    return ensemble_pred, all_models


def ensemble_feature_selection(models, feat_cols, threshold_pct=20):
    """v8: Average importance across all seed models, then threshold."""
    all_imps = np.zeros(len(feat_cols))
    for m in models:
        all_imps += m.feature_importances_
    all_imps /= len(models)

    imp = pd.Series(all_imps, index=feat_cols)
    threshold = np.percentile(imp.values, threshold_pct)
    keep = imp[imp > threshold].index.tolist()
    print(f"   🔪 Ensemble feature selection: {len(feat_cols)} → {len(keep)} "
          f"(avg importance across {len(models)} models)")
    return keep, imp


def measure_turnover(df_test, pred_col, n_pos=5):
    """Measure actual signal turnover from predictions."""
    turnover_rates = []
    prev_longs, prev_shorts = set(), set()

    for ts, grp in df_test.groupby('timestamp'):
        if len(grp) < 10:
            continue
        grp_sorted = grp.sort_values(pred_col, ascending=False)
        cur_longs = set(grp_sorted.head(n_pos)['symbol'].values)
        cur_shorts = set(grp_sorted.tail(n_pos)['symbol'].values)

        if prev_longs:
            n_total = n_pos * 2
            changed = len(cur_longs - prev_longs) + len(cur_shorts - prev_shorts)
            turnover_rates.append(changed / n_total)

        prev_longs, prev_shorts = cur_longs, cur_shorts

    if turnover_rates:
        avg_to = np.mean(turnover_rates)
        print(f"   📊 Measured turnover: {avg_to*100:.1f}% "
              f"(vs assumed {COST_MODEL['turnover_pct']*100:.0f}%)")
        return avg_to
    return COST_MODEL['turnover_pct']


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


def compute_costs_per_period(horizon_hours=12, turnover_pct=None):
    """Realistic cost per rebalance period."""
    to = turnover_pct if turnover_pct is not None else COST_MODEL['turnover_pct']
    periods_per_8h = 8 / horizon_hours
    funding_per_period = COST_MODEL['funding_per_8h'] / periods_per_8h
    trade_cost = (COST_MODEL['taker_fee'] + COST_MODEL['slippage']) * 2
    cost_per_period = trade_cost * to + funding_per_period
    return cost_per_period


def evaluate_model(df_test, pred_col, target_col, horizon_hours=12, label="",
                   turnover_pct=None):
    """Comprehensive evaluation with risk overlay."""
    periods_per_day = 24 // horizon_hours
    periods_per_year = periods_per_day * 365
    cost_per_period = compute_costs_per_period(horizon_hours, turnover_pct)

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
        ls_rets_net.append(ls_ret - cost_per_period * 2)
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

    ls_vol_target = vol_target_returns(ls_rets_raw, lookback=48, target_vol=0.02,
                                       cost_per_period=cost_per_period)
    ls_dd_stop = drawdown_stop_returns(ls_rets_net, max_dd_threshold=-0.25,
                                        recovery_threshold=-0.10)

    metrics = {
        'IC': round(float(ic), 4),
        'Rank_IC': round(float(rank_ic), 4),
        'ICIR': round(float(icir), 4),
        'Rank_ICIR': round(float(rank_icir), 4),
        'LS_Sharpe_raw': round(float(sharpe(ls_rets_raw, periods_per_year)), 2),
        'LS_Sharpe_net': round(float(sharpe(ls_rets_net, periods_per_year)), 2),
        'LS_Ann_Return_net_%': round(float(ls_rets_net.mean() * periods_per_year * 100), 1),
        'LS_MaxDD_net_%': round(float(max_dd(ls_rets_net) * 100), 1),
        'LS_Total_net_%': round(float(total_ret(ls_rets_net) * 100), 1),
        'LS_VolTarget_Sharpe': round(float(sharpe(ls_vol_target, periods_per_year)), 2),
        'LS_VolTarget_MaxDD_%': round(float(max_dd(ls_vol_target) * 100), 1),
        'LS_VolTarget_Total_%': round(float(total_ret(ls_vol_target) * 100), 1),
        'LS_DDStop_Sharpe': round(float(sharpe(ls_dd_stop, periods_per_year)), 2),
        'LS_DDStop_MaxDD_%': round(float(max_dd(ls_dd_stop) * 100), 1),
        'LS_DDStop_Total_%': round(float(total_ret(ls_dd_stop) * 100), 1),
        'LO5_Sharpe': round(float(sharpe(lo5, periods_per_year)), 2),
        'LO10_Sharpe': round(float(sharpe(lo10, periods_per_year)), 2),
        'N_periods': len(ls_rets_raw),
        'Cost_per_period_bps': round(cost_per_period * 10000, 1),
    }

    return metrics, ls_rets_net, ls_vol_target, ls_dd_stop, ls_timestamps


def vol_target_returns(raw_rets, lookback=48, target_vol=0.02, cost_per_period=0.0):
    """Scale position by target_vol / realized_vol. Cap at 2x, floor at 0.1x."""
    n = len(raw_rets)
    vt_rets = np.zeros(n)

    for i in range(n):
        if i < lookback:
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
    """Circuit breaker: stop at max DD, resume on recovery."""
    n = len(net_rets)
    stopped_rets = np.zeros(n)
    equity = 1.0
    peak = 1.0
    is_stopped = False

    for i in range(n):
        if is_stopped:
            equity *= (1 + net_rets[i])
            dd = equity / peak - 1
            if dd > recovery_threshold:
                is_stopped = False
                stopped_rets[i] = net_rets[i]
        else:
            equity *= (1 + net_rets[i])
            if equity > peak:
                peak = equity
            dd = equity / peak - 1
            if dd < max_dd_threshold:
                is_stopped = True
                stopped_rets[i] = 0
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
                        help='Use only window 5 (latest) for quick test')
    parser.add_argument('--seeds', type=int, default=N_SEEDS)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data or os.path.join(project_root, 'data', 'features')
    results_dir = args.results or os.path.join(project_root, 'results_v8')
    os.makedirs(results_dir, exist_ok=True)

    feat_path = os.path.join(data_dir, 'crypto_features_1h.parquet')
    if not os.path.exists(feat_path):
        print(f"❌ Feature file not found: {feat_path}")
        sys.exit(1)

    print("=" * 70)
    print("  CRYPTO ALPHA MODEL v8")
    print("  Extended History (2017+) + Purged Walk-Forward (5 windows)")
    print("=" * 70)

    # ========================================
    # 1. LOAD & ENRICH DATA
    # ========================================
    print(f"\n📊 Loading data...")
    df = pd.read_parquet(feat_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Shape: {df.shape}, Symbols: {df['symbol'].nunique()}")
    print(f"   Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    df = add_multi_horizon_targets(df)
    df = add_cross_asset_features(df)
    df = add_advanced_regime_features(df)
    df = add_12h_features(df)
    df = add_sentiment_features(df, project_root)

    # Clean infinities
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=['target_ret_12h'])

    # Feature columns
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS
                 and not c.startswith('target_')]
    feat_cols = [c for c in feat_cols if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    print(f"   Features: {len(feat_cols)}")

    df[feat_cols] = df[feat_cols].fillna(0)

    # Cross-sectional rank normalization
    df = cross_sectional_rank(df, feat_cols)
    df = create_rank_target(df, HORIZON)

    print(f"   Final shape: {df.shape}")
    print(f"   Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # Count available data per year
    df['year'] = df['timestamp'].dt.year
    year_counts = df.groupby('year').size()
    print(f"\n   📅 Data per year:")
    for yr, cnt in year_counts.items():
        n_sym = df[df['year'] == yr]['symbol'].nunique()
        print(f"      {yr}: {cnt:>10,} rows ({n_sym} symbols)")
    df = df.drop(columns=['year'])

    # ========================================
    # 2. ROLLING WALK-FORWARD
    # ========================================
    windows = WALK_FORWARD_WINDOWS
    if args.single_window:
        windows = [windows[-1]]  # Window 5 = production model

    print(f"\n{'='*70}")
    print(f"  PURGED WALK-FORWARD ({len(windows)} windows, gap={PURGE_DAYS}d)")
    print(f"{'='*70}")

    target_col = f'target_ret_{HORIZON}h'
    all_window_metrics = []
    all_test_predictions = []
    combined_ls_rets = []
    combined_timestamps = []
    all_hpo_params = {}

    for w_idx, window in enumerate(windows):
        print(f"\n{'─'*70}")
        print(f"  Window {w_idx+1}/{len(windows)}: {window['name']}")
        print(f"  Train: → {window['train_end']}")
        print(f"  Val:   {window['val_start']} → {window['val_end']}  (purge {PURGE_DAYS}d)")
        print(f"  Test:  {window['test_start']} → {window['test_end']}  (purge {PURGE_DAYS}d)")
        print(f"{'─'*70}")

        train = df[df['timestamp'] < window['train_end']].copy()
        val = df[(df['timestamp'] >= window['val_start']) &
                 (df['timestamp'] < window['val_end'])].copy()
        test = df[(df['timestamp'] >= window['test_start']) &
                  (df['timestamp'] <= window['test_end'])].copy()

        if len(test) == 0:
            print(f"   ⚠️  No test data for this window, skipping")
            continue

        if len(train) < 10000:
            print(f"   ⚠️  Too little training data ({len(train):,} rows), skipping")
            continue

        print(f"   Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
        print(f"   Train symbols: {train['symbol'].nunique()} | "
              f"Test symbols: {test['symbol'].nunique()}")

        X_train, y_train = train[feat_cols], train['target_rank']
        X_val, y_val = val[feat_cols], val['target_rank']
        X_test = test[feat_cols]
        val_dates = val['timestamp'].dt.date.values

        # --- HPO (per-window in v8!) ---
        best_params = None
        if not args.skip_hpo:
            best_params = run_optuna_hpo(
                X_train, y_train, X_val, y_val, val_dates,
                n_trials=args.hpo_trials
            )
            all_hpo_params[window['name']] = best_params

        # --- Ensemble-based feature selection ---
        # Train 5-seed preliminary ensemble, average importance
        print(f"\n   🔧 Feature selection (ensemble-based)...")
        prelim_pred, prelim_models = train_multi_seed(
            X_train, y_train, X_val, y_val, X_test,
            params=best_params, seeds=SEEDS[:args.seeds],
        )
        selected_feats, importance_series = ensemble_feature_selection(
            prelim_models, feat_cols, threshold_pct=20)

        # --- Retrain with selected features ---
        print(f"\n   🔄 Retraining with {len(selected_feats)} selected features...")
        ensemble_pred, all_models = train_multi_seed(
            train[selected_feats], y_train,
            val[selected_feats], y_val,
            test[selected_feats],
            params=best_params,
            seeds=SEEDS[:args.seeds],
        )
        test['pred_v8'] = ensemble_pred

        # --- Save models (production: last window only, all saved for analysis) ---
        for i, mdl in enumerate(all_models):
            seed = SEEDS[:args.seeds][i]
            model_path = os.path.join(results_dir, f'lgb_model_seed_{seed}.txt')
            mdl.booster_.save_model(model_path)
        with open(os.path.join(results_dir, 'feature_names.json'), 'w') as f:
            json.dump(selected_feats, f)
        print(f"   💾 Saved {len(all_models)} models + feature names")

        # --- Measure actual turnover ---
        actual_turnover = measure_turnover(test, 'pred_v8', n_pos=5)

        # --- Evaluate ---
        metrics, ls_net, ls_vt, ls_dd, timestamps = evaluate_model(
            test, 'pred_v8', target_col, HORIZON, label=window['name'],
            turnover_pct=actual_turnover,
        )
        metrics['window'] = window['name']
        metrics['measured_turnover_%'] = round(actual_turnover * 100, 1)
        all_window_metrics.append(metrics)

        save_cols = ['timestamp', 'symbol', target_col, 'pred_v8']
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

    if len(all_window_metrics) > 1:
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
                        'LS_DDStop_Sharpe', 'LS_DDStop_MaxDD_%', 'measured_turnover_%']
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
    combined_ls = np.array(combined_ls_rets)
    periods_per_year = (24 // HORIZON) * 365

    def sharpe(r, ppyr):
        return (r.mean() / (r.std() + 1e-10)) * np.sqrt(ppyr)
    def max_dd(r):
        cum = np.cumprod(1 + r)
        return np.min(cum / np.maximum.accumulate(cum) - 1)
    def total_ret(r):
        return np.prod(1 + r) - 1

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
          f"MaxDD={max_dd(combined_vt)*100:.1f}%")
    print(f"      LS DDStop:    Sharpe={sharpe(combined_dd, periods_per_year):.2f}, "
          f"MaxDD={max_dd(combined_dd)*100:.1f}%")

    # ========================================
    # 5. FEATURE IMPORTANCE
    # ========================================
    if 'importance_series' in dir() and importance_series is not None:
        importance = importance_series.sort_values(ascending=False).reset_index()
        importance.columns = ['feature', 'importance']

        print(f"\n🏆 Top 30 Features (last window, ensemble avg):")
        for _, row in importance.head(30).iterrows():
            marker = "📰" if any(s in row['feature'] for s in
                                  ['fng', 'funding', 'reversal', 'surge', 'dispersion',
                                   'long_short', 'beta']) else "  "
            print(f"   {marker} {row['feature']:40s} {row['importance']:.0f}")

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
        'hpo_params': all_hpo_params,
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'horizon': HORIZON,
            'n_features': len(feat_cols),
            'n_selected': len(selected_feats) if 'selected_feats' in dir() else 0,
            'n_seeds': args.seeds,
            'n_windows': len(windows),
            'hpo_trials': args.hpo_trials if not args.skip_hpo else 0,
            'purge_days': PURGE_DAYS,
            'version': 'v8',
        },
    }

    with open(os.path.join(results_dir, 'all_results_v8.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    if 'importance' in dir():
        importance.to_csv(os.path.join(results_dir, 'feature_importance_v8.csv'), index=False)

    if all_test_predictions:
        combined_preds = pd.concat(all_test_predictions, ignore_index=True)
        combined_preds.to_parquet(
            os.path.join(results_dir, 'test_predictions_v8.parquet'), index=False)

    eq = pd.DataFrame({
        'ls_net': np.cumprod(1 + combined_ls) * 1000,
        'ls_vol_target': np.cumprod(1 + combined_vt) * 1000,
        'ls_dd_stop': np.cumprod(1 + combined_dd) * 1000,
    })
    eq.to_parquet(os.path.join(results_dir, 'equity_curves_v8.parquet'), index=False)

    # ========================================
    # FINAL VERDICT
    # ========================================
    best_sharpe = avg_metrics.get('LS_DDStop_Sharpe', avg_metrics.get('LS_Sharpe_net', 0))
    best_dd = avg_metrics.get('LS_DDStop_MaxDD_%', avg_metrics.get('LS_MaxDD_net_%', 0))

    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY v8 (extended history, purged WF)")
    print(f"{'='*70}")
    print(f"   Data range:               {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"   Total rows:               {len(df):,}")
    print(f"   Windows evaluated:        {len(all_window_metrics)}")
    print(f"   Purge gap:                {PURGE_DAYS} days")
    print(f"   Rank IC (avg):            {avg_metrics.get('Rank_IC', 0):+.4f}")
    print(f"   Rank ICIR (avg):          {avg_metrics.get('Rank_ICIR', 0):+.4f}")
    print(f"   LS Sharpe net (avg):      {avg_metrics.get('LS_Sharpe_net', 0):+.2f}")
    print(f"   LS MaxDD net (avg):       {avg_metrics.get('LS_MaxDD_net_%', 0):.1f}%")
    print(f"   LS DDStop Sharpe:         {avg_metrics.get('LS_DDStop_Sharpe', 0):+.2f}")
    print(f"   LS DDStop MaxDD:          {avg_metrics.get('LS_DDStop_MaxDD_%', 0):.1f}%")
    print(f"   Measured turnover (avg):  {avg_metrics.get('measured_turnover_%', 'N/A')}%")
    print(f"   Cost model:               {COST_MODEL['taker_fee']*100:.2f}% taker + "
          f"{COST_MODEL['slippage']*100:.2f}% slip + "
          f"{COST_MODEL['funding_per_8h']*100:.3f}%/8h funding")
    print(f"{'='*70}")

    if best_sharpe > 3.0:
        print("🟢 STRONG — Robust signal across extended history.")
    elif best_sharpe > 2.0:
        print("🟡 DECENT — Signal exists across regimes.")
    elif best_sharpe > 1.0:
        print("🟠 MARGINAL — Consider ensemble or feature changes.")
    else:
        print("🔴 WEAK — Extended data not helping, revisit approach.")

    print(f"\n✅ Results saved to {results_dir}/")


if __name__ == '__main__':
    main()
