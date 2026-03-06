#!/usr/bin/env python3
"""
Crypto Alpha Model v4 — Advanced Regime + HPO + Multi-Seed Ensemble

Key improvements over v3:
1. Optuna HPO with Rank ICIR objective (more robust than Rank IC)
2. Advanced multi-factor regime filter:
   - BTC long-term trend (336h / 720h MA)
   - BTC trend direction (MA slope)
   - Market breadth (% coins with positive 24h return)
   - BTC drawdown from rolling high
   - Volatility regime
3. Dynamic position sizing (composite regime score)
4. Multi-seed LightGBM ensemble (5 seeds)
5. Purged walk-forward (48h gap between train/val/test)
6. Feature selection (drop noise features after first pass)
7. Score-weighted portfolio (not equal-weight)

Usage:
  python run_pipeline_v4.py                          # Full run with HPO
  python run_pipeline_v4.py --hpo-trials 100         # More HPO trials
  python run_pipeline_v4.py --skip-hpo               # Skip HPO, use defaults
  python run_pipeline_v4.py --data /path/to/features # Custom data path
"""

import sys
import os
import argparse
import json
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
TRAIN_END = '2024-06-29'       # Train: everything before this
VAL_START = '2024-07-01'       # 48h purge gap
VAL_END = '2024-12-30'
TEST_START = '2025-01-01'      # 48h purge gap

HORIZON = 4                    # Best from v3
N_SEEDS = 5
SEEDS = [42, 123, 456, 789, 2024]

EXCLUDE_COLS = {
    'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
    'target_ret_4h', 'target_ret_12h', 'target_ret_24h',
    'target_cls', 'target_ret', 'target_rank', 'target_excess',
    'hour', 'day_of_week',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
}

# Regime columns: NOT ranked (binary or market-level)
REGIME_COLS = {
    'btc_regime_24', 'btc_regime_72', 'btc_regime_168',
    'regime_btc_above_ma336', 'regime_btc_above_ma720',
    'regime_btc_ma720_slope', 'regime_btc_not_crashed',
    'regime_btc_dd_720', 'regime_low_vol',
    'regime_breadth_bullish', 'breadth_pct_positive',
    'regime_composite',
}


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_multi_horizon_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add 4h forward return target."""
    print("   🎯 Adding 4h forward return target...")
    for h in [4, 12, 24]:
        df[f'target_ret_{h}h'] = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(h).shift(-h)
        )
    return df


def add_cross_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """BTC/ETH returns as market factors for ALL coins."""
    print("   🌐 Adding cross-asset features...")

    btc = df[df['symbol'] == 'BTC/USDT'][['timestamp', 'close']].copy()
    btc = btc.rename(columns={'close': 'btc_close'}).drop_duplicates('timestamp')

    eth = df[df['symbol'] == 'ETH/USDT'][['timestamp', 'close']].copy()
    eth = eth.rename(columns={'close': 'eth_close'}).drop_duplicates('timestamp')

    df = df.merge(btc, on='timestamp', how='left')
    df = df.merge(eth, on='timestamp', how='left')

    # BTC returns
    for h in [1, 4, 12, 24, 48, 168]:
        df[f'btc_ret_{h}h'] = df.groupby('symbol')['btc_close'].transform(
            lambda x: x.pct_change(h)
        )
    # ETH returns
    for h in [1, 4, 12, 24]:
        df[f'eth_ret_{h}h'] = df.groupby('symbol')['eth_close'].transform(
            lambda x: x.pct_change(h)
        )

    # BTC MAs (short-term, for model features)
    for w in [24, 72, 168]:
        df[f'btc_ma{w}'] = df.groupby('symbol')['btc_close'].transform(
            lambda x: x.rolling(w).mean()
        )

    # BTC regime (short-term, binary)
    df['btc_regime_24'] = (df['btc_close'] > df['btc_ma24']).astype(float)
    df['btc_regime_72'] = (df['btc_close'] > df['btc_ma72']).astype(float)
    df['btc_regime_168'] = (df['btc_close'] > df['btc_ma168']).astype(float)

    # BTC volatility
    df['btc_vol_24h'] = df.groupby('symbol')['btc_close'].transform(
        lambda x: x.pct_change().rolling(24).std()
    )

    # ETH/BTC ratio momentum
    df['eth_btc_ratio'] = df['eth_close'] / (df['btc_close'] + 1e-10)
    df['eth_btc_ret_24h'] = df.groupby('symbol')['eth_btc_ratio'].transform(
        lambda x: x.pct_change(24)
    )

    # Market breadth
    cs_std = df.groupby('timestamp')['ret_1h'].transform('std')
    df['market_dispersion'] = cs_std

    # Relative strength vs BTC
    df['ret_vs_btc_24h'] = df['ret_24h'] - df['btc_ret_24h']

    df.drop(columns=['btc_ma24', 'btc_ma72', 'btc_ma168', 'eth_btc_ratio'], inplace=True)

    return df


def add_advanced_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-factor regime filter (computed BEFORE ranking).

    Signals:
    1. BTC vs 336h (14d) MA — medium-term trend
    2. BTC vs 720h (30d) MA — long-term trend
    3. 720h MA slope — is the trend direction positive?
    4. BTC drawdown from 720h rolling high — not in crash
    5. BTC volatility regime — not in extreme vol
    6. Market breadth — majority of coins positive

    Composite score: weighted average → used for position sizing.
    """
    print("   🔰 Adding advanced regime features...")

    # === BTC-based signals ===
    btc = df[df['symbol'] == 'BTC/USDT'][['timestamp', 'btc_close']].drop_duplicates('timestamp')
    btc = btc.sort_values('timestamp').copy()

    # Long-term MAs
    btc['btc_ma336'] = btc['btc_close'].rolling(336, min_periods=100).mean()
    btc['btc_ma720'] = btc['btc_close'].rolling(720, min_periods=200).mean()

    # Signal 1: BTC above 14d MA
    btc['regime_btc_above_ma336'] = (btc['btc_close'] > btc['btc_ma336']).astype(float)

    # Signal 2: BTC above 30d MA
    btc['regime_btc_above_ma720'] = (btc['btc_close'] > btc['btc_ma720']).astype(float)

    # Signal 3: 30d MA is rising (compared to 24h ago)
    btc['regime_btc_ma720_slope'] = (
        btc['btc_ma720'] > btc['btc_ma720'].shift(24)
    ).astype(float)

    # Signal 4: BTC drawdown from 30d rolling high
    btc['btc_rolling_high_720'] = btc['btc_close'].rolling(720, min_periods=100).max()
    btc['regime_btc_dd_720'] = btc['btc_close'] / btc['btc_rolling_high_720'] - 1
    btc['regime_btc_not_crashed'] = (btc['regime_btc_dd_720'] > -0.15).astype(float)

    # Signal 5: BTC vol not extreme (< 2× 30d median)
    btc['_btc_vol_24'] = btc['btc_close'].pct_change().rolling(24).std()
    btc['_btc_vol_720_med'] = btc['_btc_vol_24'].rolling(720, min_periods=100).median()
    btc['regime_low_vol'] = (
        btc['_btc_vol_24'] < btc['_btc_vol_720_med'] * 2.0
    ).astype(float)

    btc_regime_cols = [
        'timestamp', 'regime_btc_above_ma336', 'regime_btc_above_ma720',
        'regime_btc_ma720_slope', 'regime_btc_not_crashed',
        'regime_btc_dd_720', 'regime_low_vol',
    ]
    df = df.merge(btc[btc_regime_cols], on='timestamp', how='left')

    # === Market breadth ===
    breadth = df.groupby('timestamp')['ret_24h'].agg(
        breadth_pct_positive=lambda x: (x > 0).mean()
    ).reset_index()
    breadth['regime_breadth_bullish'] = (breadth['breadth_pct_positive'] > 0.5).astype(float)
    df = df.merge(breadth, on='timestamp', how='left')

    # === Composite regime score [0, 1] ===
    df['regime_composite'] = (
        0.25 * df['regime_btc_above_ma720'].fillna(0) +
        0.20 * df['regime_btc_ma720_slope'].fillna(0) +
        0.20 * df['regime_breadth_bullish'].fillna(0) +
        0.20 * df['regime_btc_not_crashed'].fillna(0) +
        0.15 * df['regime_low_vol'].fillna(0)
    )

    # Stats
    test_mask = df['timestamp'] >= TEST_START
    if test_mask.any():
        test_regime = df.loc[test_mask, 'regime_composite']
        ts_scores = test_regime.groupby(df.loc[test_mask, 'timestamp']).first()
        pct_high = (ts_scores >= 0.6).mean() * 100
        pct_zero = (ts_scores < 0.4).mean() * 100
        print(f"   Regime in test: {pct_high:.0f}% high (≥0.6), {pct_zero:.0f}% low (<0.4)")

    # Cleanup
    if 'btc_close' in df.columns:
        df.drop(columns=['btc_close'], inplace=True, errors='ignore')
    if 'eth_close' in df.columns:
        df.drop(columns=['eth_close'], inplace=True, errors='ignore')

    return df


def cross_sectional_rank(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """Rank-normalize features within each timestamp. Preserves regime columns."""
    print("   📐 Cross-sectional rank normalization...")

    # Backup regime cols
    regime_backup = {}
    for col in REGIME_COLS:
        if col in df.columns:
            regime_backup[col] = df[col].copy()

    # Rank features (exclude regime cols from ranking)
    rank_cols = [c for c in feat_cols if c not in REGIME_COLS]
    ranked = df.groupby('timestamp')[rank_cols].rank(pct=True)
    df[rank_cols] = ranked - 0.5  # Center around 0

    # Restore regime cols
    for col, vals in regime_backup.items():
        df[col] = vals
    print(f"   ✅ Restored {len(regime_backup)} regime columns (not ranked)")

    return df


def create_rank_target(df: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
    """Cross-sectional rank target."""
    target_col = f'target_ret_{horizon}h'
    df['target_rank'] = df.groupby('timestamp')[target_col].rank(pct=True)
    return df


# ============================================================
# OPTUNA HPO  (Rank ICIR objective)
# ============================================================

def run_optuna_hpo(X_train, y_train, X_val, y_val, val_dates, n_trials=50):
    """
    Auto-tune LightGBM with Optuna.
    Objective: maximize Rank ICIR on validation (stability of signal).
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("   ⚠️  Optuna not installed, using default params")
        return None

    print(f"   🔍 Running Optuna HPO ({n_trials} trials, objective=Rank_ICIR)...")

    # Precompute unique dates for ICIR calculation
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

        # Compute Rank ICIR (daily Rank IC mean / std)
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
        icir = daily_ics.mean() / (daily_ics.std() + 1e-10)

        # Report intermediate for pruning
        trial.report(icir, 0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return icir

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"   ✅ Best Rank ICIR on val: {study.best_value:.4f}")
    print(f"   Best params: {json.dumps(study.best_params, indent=4)}")

    return study.best_params


# ============================================================
# TRAINING
# ============================================================

def train_lgbm(X_train, y_train, X_val, y_val, custom_params=None, seed=42):
    """Train one LightGBM model."""
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


def train_multi_seed_ensemble(X_train, y_train, X_val, y_val, X_test,
                               params=None, seeds=None):
    """Train N models with different seeds, return averaged predictions."""
    seeds = seeds or SEEDS
    print(f"\n   🌱 Multi-seed ensemble ({len(seeds)} seeds)...")

    all_preds = []
    for i, seed in enumerate(seeds):
        print(f"      Seed {seed} ({i+1}/{len(seeds)})...", end=" ")
        model = train_lgbm(X_train, y_train, X_val, y_val,
                           custom_params=params, seed=seed)
        preds = model.predict(X_test)
        all_preds.append(preds)
        print(f"iters={model.best_iteration_}")

    # Average predictions
    ensemble_pred = np.mean(all_preds, axis=0)
    print(f"   ✅ Ensemble done, averaging {len(seeds)} models")
    return ensemble_pred, model  # Return last model for feature importance


def feature_selection(model, feat_cols, threshold_pct=20):
    """Drop bottom N% features by importance."""
    imp = pd.Series(model.feature_importances_, index=feat_cols)
    threshold = np.percentile(imp.values, threshold_pct)
    keep = imp[imp > threshold].index.tolist()
    dropped = len(feat_cols) - len(keep)
    print(f"   🔪 Feature selection: {len(feat_cols)} → {len(keep)} (dropped {dropped})")
    return keep


# ============================================================
# EVALUATION
# ============================================================

def compute_ic(p, a):
    m = ~(np.isnan(p) | np.isnan(a))
    return np.corrcoef(p[m], a[m])[0, 1] if m.sum() >= 10 else np.nan

def compute_rank_ic(p, a):
    m = ~(np.isnan(p) | np.isnan(a))
    if m.sum() < 10: return np.nan
    c, _ = spearmanr(p[m], a[m])
    return c


def evaluate_model(df_test, pred_col, target_col, horizon_hours=4, label=""):
    """Standard evaluation: IC, ICIR, LS Sharpe, LO."""
    ic = compute_ic(df_test[pred_col].values, df_test[target_col].values)
    rank_ic = compute_rank_ic(df_test[pred_col].values, df_test[target_col].values)

    df_eval = df_test.copy()
    df_eval['date'] = df_eval['timestamp'].dt.date

    daily_ics, daily_rank_ics = [], []
    for _, grp in df_eval.groupby('date'):
        if len(grp) >= 10:
            daily_ics.append(compute_ic(grp[pred_col].values, grp[target_col].values))
            daily_rank_ics.append(compute_rank_ic(grp[pred_col].values, grp[target_col].values))

    daily_ics = np.array([x for x in daily_ics if not np.isnan(x)])
    daily_rank_ics = np.array([x for x in daily_rank_ics if not np.isnan(x)])

    icir = daily_ics.mean() / (daily_ics.std() + 1e-10) if len(daily_ics) > 0 else 0
    rank_icir = daily_rank_ics.mean() / (daily_rank_ics.std() + 1e-10) if len(daily_rank_ics) > 0 else 0

    periods_per_day = 24 // horizon_hours
    periods_per_year = periods_per_day * 365

    ls_rets, lo5_rets, lo10_rets = [], [], []
    for ts, grp in df_eval.groupby('timestamp'):
        if len(grp) < 10:
            continue
        grp = grp.sort_values(pred_col, ascending=False)
        n = max(len(grp) // 5, 1)
        long_ret = grp.head(n)[target_col].mean()
        short_ret = grp.tail(n)[target_col].mean()
        ls_rets.append(long_ret - short_ret)
        lo5_rets.append(grp.head(5)[target_col].mean())
        lo10_rets.append(grp.head(10)[target_col].mean())

    ls_rets = np.array(ls_rets)
    lo5 = np.array(lo5_rets)
    lo10 = np.array(lo10_rets)

    def sharpe(rets, ppyr):
        return (rets.mean() / (rets.std() + 1e-10)) * np.sqrt(ppyr)
    def max_dd(rets):
        cum = np.cumprod(1 + rets)
        return np.min(cum / np.maximum.accumulate(cum) - 1)
    def total_ret(rets):
        return np.prod(1 + rets) - 1

    comm = 0.0008  # 0.08% per period (0.2% round-trip × 40% turnover)
    lo5_net = lo5 - comm
    lo10_net = lo10 - comm

    metrics = {
        'IC': round(float(ic), 4),
        'Rank_IC': round(float(rank_ic), 4),
        'Daily_IC_mean': round(float(daily_ics.mean()), 4) if len(daily_ics) > 0 else 0,
        'ICIR': round(float(icir), 4),
        'Rank_ICIR': round(float(rank_icir), 4),
        'LS_Sharpe': round(float(sharpe(ls_rets, periods_per_year)), 2),
        'LS_Ann_Return_%': round(float(ls_rets.mean() * periods_per_year * 100), 1),
        'LS_MaxDD_%': round(float(max_dd(ls_rets) * 100), 1),
        'LO5_Sharpe': round(float(sharpe(lo5_net, periods_per_year)), 2),
        'LO5_Total_%': round(float(total_ret(lo5_net) * 100), 1),
        'LO10_Sharpe': round(float(sharpe(lo10_net, periods_per_year)), 2),
        'LO10_Total_%': round(float(total_ret(lo10_net) * 100), 1),
        'N_periods': len(ls_rets),
        'Horizon_h': horizon_hours,
    }
    return metrics


def evaluate_advanced_regime(df_test, pred_col, target_col, horizon_hours=4):
    """
    Long-only backtest with advanced multi-factor regime + dynamic sizing.

    Position sizing:
    - regime_composite >= 0.8 → 100% allocation
    - regime_composite >= 0.6 → 60% allocation
    - regime_composite >= 0.4 → 25% allocation
    - regime_composite  < 0.4 → 0% (sit out)

    Score-weighted portfolio: higher predicted coins get more weight (softmax).
    """
    df_eval = df_test.copy()
    periods_per_day = 24 // horizon_hours
    periods_per_year = periods_per_day * 365
    comm = 0.0008

    # Results containers
    results = {}

    # --- Strategy 1: Simple Top-5 Long-Only (no filter) ---
    lo5_rets = []
    for ts, grp in df_eval.groupby('timestamp'):
        if len(grp) < 10:
            continue
        grp = grp.sort_values(pred_col, ascending=False)
        ret = grp.head(5)[target_col].mean() - comm
        lo5_rets.append(ret)
    lo5_rets = np.array(lo5_rets)

    # --- Strategy 2: BTC 72h MA regime (v3 baseline) ---
    lo5_v3regime = []
    for ts, grp in df_eval.groupby('timestamp'):
        if len(grp) < 10:
            continue
        grp = grp.sort_values(pred_col, ascending=False)
        regime = grp['btc_regime_72'].iloc[0] if 'btc_regime_72' in grp.columns else 1
        ret = grp.head(5)[target_col].mean() - comm
        lo5_v3regime.append(ret * (1.0 if regime > 0.5 else 0.0))
    lo5_v3regime = np.array(lo5_v3regime)

    # --- Strategy 3: Advanced composite regime + dynamic sizing ---
    lo5_advanced = []
    regime_counts = {'full': 0, 'partial_60': 0, 'partial_25': 0, 'out': 0}
    for ts, grp in df_eval.groupby('timestamp'):
        if len(grp) < 10:
            continue
        grp = grp.sort_values(pred_col, ascending=False)
        regime_score = grp['regime_composite'].iloc[0] if 'regime_composite' in grp.columns else 0.5

        if regime_score >= 0.8:
            alloc = 1.0
            regime_counts['full'] += 1
        elif regime_score >= 0.6:
            alloc = 0.6
            regime_counts['partial_60'] += 1
        elif regime_score >= 0.4:
            alloc = 0.25
            regime_counts['partial_25'] += 1
        else:
            alloc = 0.0
            regime_counts['out'] += 1

        # Score-weighted top-5
        top5 = grp.head(5)
        scores = top5[pred_col].values
        # Softmax weights
        exp_s = np.exp(scores - scores.max())
        weights = exp_s / (exp_s.sum() + 1e-10)
        weighted_ret = (top5[target_col].values * weights).sum()
        lo5_advanced.append((weighted_ret - comm) * alloc)

    lo5_advanced = np.array(lo5_advanced)

    # --- Strategy 4: Advanced regime + score-weighted Top-10 ---
    lo10_advanced = []
    for ts, grp in df_eval.groupby('timestamp'):
        if len(grp) < 10:
            continue
        grp = grp.sort_values(pred_col, ascending=False)
        regime_score = grp['regime_composite'].iloc[0] if 'regime_composite' in grp.columns else 0.5

        if regime_score >= 0.8:
            alloc = 1.0
        elif regime_score >= 0.6:
            alloc = 0.6
        elif regime_score >= 0.4:
            alloc = 0.25
        else:
            alloc = 0.0

        top10 = grp.head(10)
        scores = top10[pred_col].values
        exp_s = np.exp(scores - scores.max())
        weights = exp_s / (exp_s.sum() + 1e-10)
        weighted_ret = (top10[target_col].values * weights).sum()
        lo10_advanced.append((weighted_ret - comm) * alloc)

    lo10_advanced = np.array(lo10_advanced)

    def sharpe(r, ppyr):
        if len(r) == 0 or r.std() == 0:
            return 0.0
        return (r.mean() / (r.std() + 1e-10)) * np.sqrt(ppyr)

    def max_dd(r):
        if len(r) == 0:
            return 0.0
        cum = np.cumprod(1 + r)
        return np.min(cum / np.maximum.accumulate(cum) - 1)

    def final_equity(r, init=1000):
        return init * np.prod(1 + r)

    n_total = sum(regime_counts.values())

    results = {
        # No filter
        'NoFilter_LO5_Sharpe': round(float(sharpe(lo5_rets, periods_per_year)), 2),
        'NoFilter_LO5_Final': round(float(final_equity(lo5_rets)), 2),
        'NoFilter_LO5_MaxDD_%': round(float(max_dd(lo5_rets) * 100), 1),
        # v3 regime (BTC 72h MA)
        'v3Regime_LO5_Sharpe': round(float(sharpe(lo5_v3regime, periods_per_year)), 2),
        'v3Regime_LO5_Final': round(float(final_equity(lo5_v3regime)), 2),
        # v4 advanced regime + dynamic sizing (Top-5)
        'v4Regime_LO5_Sharpe': round(float(sharpe(lo5_advanced, periods_per_year)), 2),
        'v4Regime_LO5_Final': round(float(final_equity(lo5_advanced)), 2),
        'v4Regime_LO5_MaxDD_%': round(float(max_dd(lo5_advanced) * 100), 1),
        # v4 advanced regime + dynamic sizing (Top-10)
        'v4Regime_LO10_Sharpe': round(float(sharpe(lo10_advanced, periods_per_year)), 2),
        'v4Regime_LO10_Final': round(float(final_equity(lo10_advanced)), 2),
        'v4Regime_LO10_MaxDD_%': round(float(max_dd(lo10_advanced) * 100), 1),
        # Regime breakdown
        'Regime_Full_%': round(regime_counts['full'] / (n_total + 1e-10) * 100, 1),
        'Regime_60_%': round(regime_counts['partial_60'] / (n_total + 1e-10) * 100, 1),
        'Regime_25_%': round(regime_counts['partial_25'] / (n_total + 1e-10) * 100, 1),
        'Regime_Out_%': round(regime_counts['out'] / (n_total + 1e-10) * 100, 1),
    }

    # Equity curves for plotting
    eq_curves = pd.DataFrame({
        'no_filter': 1000 * np.cumprod(1 + lo5_rets),
        'v3_regime': 1000 * np.cumprod(1 + lo5_v3regime),
        'v4_regime_top5': 1000 * np.cumprod(1 + lo5_advanced),
        'v4_regime_top10': 1000 * np.cumprod(1 + lo10_advanced),
    })

    return results, eq_curves


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--results', type=str, default=None)
    parser.add_argument('--hpo-trials', type=int, default=50)
    parser.add_argument('--skip-hpo', action='store_true')
    parser.add_argument('--skip-ensemble', action='store_true')
    parser.add_argument('--seeds', type=int, default=N_SEEDS, help='Number of ensemble seeds')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data or os.path.join(project_root, 'data', 'features')
    results_dir = args.results or os.path.join(project_root, 'results_v4')
    os.makedirs(results_dir, exist_ok=True)

    feat_path = os.path.join(data_dir, 'crypto_features_1h.parquet')
    if not os.path.exists(feat_path):
        print(f"❌ Feature file not found: {feat_path}")
        sys.exit(1)

    print("=" * 70)
    print("  CRYPTO ALPHA MODEL v4")
    print("  Advanced Regime + HPO (ICIR) + Multi-Seed Ensemble")
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
    df = add_advanced_regime_features(df)

    # Clean infinities
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=['target_ret_4h'])

    # Feature columns (excluding targets, metadata, time features)
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS
                 and not c.startswith('target_')]
    feat_cols = [c for c in feat_cols if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    print(f"   Features: {len(feat_cols)}")

    df[feat_cols] = df[feat_cols].fillna(0)

    # Cross-sectional rank normalization (preserves regime cols)
    df = cross_sectional_rank(df, feat_cols)

    # Create rank target
    df = create_rank_target(df, HORIZON)

    print(f"   Final shape: {df.shape}")

    # ========================================
    # 2. SPLIT DATA (purged walk-forward)
    # ========================================
    train = df[df['timestamp'] < TRAIN_END].copy()
    val = df[(df['timestamp'] >= VAL_START) & (df['timestamp'] < VAL_END)].copy()
    test = df[df['timestamp'] >= TEST_START].copy()

    target_col = f'target_ret_{HORIZON}h'

    print(f"\n📋 Split (with 48h purge gap):")
    print(f"   Train: {len(train):,} rows | {train['timestamp'].min()} → {train['timestamp'].max()}")
    print(f"   Val:   {len(val):,} rows | {val['timestamp'].min()} → {val['timestamp'].max()}")
    print(f"   Test:  {len(test):,} rows | {test['timestamp'].min()} → {test['timestamp'].max()}")

    X_train, y_train = train[feat_cols], train['target_rank']
    X_val, y_val = val[feat_cols], val['target_rank']
    X_test = test[feat_cols]

    val_dates = val['timestamp'].dt.date.values

    # ========================================
    # 3. BASELINE (v3 default params)
    # ========================================
    print(f"\n{'='*70}")
    print(f"  STEP 1: Baseline (v3 default params)")
    print(f"{'='*70}")

    model_base = train_lgbm(X_train, y_train, X_val, y_val)
    test['pred_baseline'] = model_base.predict(X_test)

    metrics_base = evaluate_model(test, 'pred_baseline', target_col, HORIZON)

    print(f"\n   📈 Baseline Results:")
    for k, v in metrics_base.items():
        flag = ""
        if k == 'Rank_IC' and abs(v) > 0.02: flag = " ✓"
        if k == 'LS_Sharpe' and v > 1.0: flag = " ✓"
        if k == 'ICIR' and abs(v) > 0.3: flag = " ✓"
        print(f"      {k:25s} {v}{flag}")

    # ========================================
    # 4. HPO
    # ========================================
    best_params = None
    if not args.skip_hpo:
        print(f"\n{'='*70}")
        print(f"  STEP 2: Optuna HPO ({args.hpo_trials} trials)")
        print(f"{'='*70}")

        best_params = run_optuna_hpo(
            X_train, y_train, X_val, y_val, val_dates,
            n_trials=args.hpo_trials
        )

        if best_params:
            model_hpo = train_lgbm(X_train, y_train, X_val, y_val,
                                    custom_params=best_params)
            test['pred_hpo'] = model_hpo.predict(X_test)

            metrics_hpo = evaluate_model(test, 'pred_hpo', target_col, HORIZON)

            print(f"\n   📈 HPO Results:")
            for k, v in metrics_hpo.items():
                flag = ""
                if k == 'Rank_IC' and abs(v) > 0.02: flag = " ✓"
                if k == 'LS_Sharpe' and v > 1.0: flag = " ✓"
                if k == 'ICIR' and abs(v) > 0.3: flag = " ✓"
                print(f"      {k:25s} {v}{flag}")

            delta_sharpe = metrics_hpo['LS_Sharpe'] - metrics_base['LS_Sharpe']
            print(f"\n   HPO ΔSharpe: {delta_sharpe:+.2f}")

    # ========================================
    # 5. FEATURE SELECTION + RETRAIN
    # ========================================
    print(f"\n{'='*70}")
    print(f"  STEP 3: Feature Selection")
    print(f"{'='*70}")

    best_model_so_far = model_hpo if best_params and 'model_hpo' in dir() else model_base
    selected_feats = feature_selection(best_model_so_far, feat_cols, threshold_pct=20)

    X_train_sel = train[selected_feats]
    X_val_sel = val[selected_feats]
    X_test_sel = test[selected_feats]

    model_sel = train_lgbm(X_train_sel, y_train, X_val_sel, y_val,
                            custom_params=best_params)
    test['pred_selected'] = model_sel.predict(X_test_sel)

    metrics_sel = evaluate_model(test, 'pred_selected', target_col, HORIZON)

    print(f"\n   📈 Feature-selected Results:")
    for k, v in metrics_sel.items():
        flag = ""
        if k == 'Rank_IC' and abs(v) > 0.02: flag = " ✓"
        if k == 'LS_Sharpe' and v > 1.0: flag = " ✓"
        if k == 'ICIR' and abs(v) > 0.3: flag = " ✓"
        print(f"      {k:25s} {v}{flag}")

    # Decide which features to use for ensemble
    if metrics_sel['LS_Sharpe'] >= metrics_base['LS_Sharpe']:
        print(f"   ✅ Feature selection improved results, using {len(selected_feats)} features")
        ens_feats = selected_feats
    else:
        print(f"   ⚠️  Feature selection didn't help, using all {len(feat_cols)} features")
        ens_feats = feat_cols

    # ========================================
    # 6. MULTI-SEED ENSEMBLE
    # ========================================
    if not args.skip_ensemble:
        print(f"\n{'='*70}")
        print(f"  STEP 4: Multi-Seed Ensemble ({args.seeds} seeds)")
        print(f"{'='*70}")

        ensemble_pred, last_model = train_multi_seed_ensemble(
            train[ens_feats], y_train,
            val[ens_feats], y_val,
            test[ens_feats],
            params=best_params,
            seeds=SEEDS[:args.seeds],
        )
        test['pred_ensemble'] = ensemble_pred

        metrics_ens = evaluate_model(test, 'pred_ensemble', target_col, HORIZON)

        print(f"\n   📈 Ensemble Results:")
        for k, v in metrics_ens.items():
            flag = ""
            if k == 'Rank_IC' and abs(v) > 0.02: flag = " ✓"
            if k == 'LS_Sharpe' and v > 1.0: flag = " ✓"
            if k == 'ICIR' and abs(v) > 0.3: flag = " ✓"
            print(f"      {k:25s} {v}{flag}")

    # ========================================
    # 7. ADVANCED REGIME EVALUATION
    # ========================================
    # Pick best prediction column
    best_pred_col = 'pred_ensemble' if 'pred_ensemble' in test.columns else \
                    'pred_hpo' if 'pred_hpo' in test.columns else 'pred_baseline'

    print(f"\n{'='*70}")
    print(f"  STEP 5: Advanced Regime Evaluation (using {best_pred_col})")
    print(f"{'='*70}")

    regime_results, eq_curves = evaluate_advanced_regime(
        test, best_pred_col, target_col, HORIZON
    )

    print(f"\n   📊 Long-Only Strategy Comparison:")
    print(f"      {'Strategy':<35s} {'Sharpe':>8s} {'Final $':>10s} {'MaxDD':>8s}")
    print(f"      {'-'*65}")
    print(f"      {'No filter (Top-5 equal-wt)':<35s} {regime_results['NoFilter_LO5_Sharpe']:>8.2f} {regime_results['NoFilter_LO5_Final']:>10.2f} {regime_results['NoFilter_LO5_MaxDD_%']:>7.1f}%")
    print(f"      {'v3 regime (BTC 72h MA)':<35s} {regime_results['v3Regime_LO5_Sharpe']:>8.2f} {regime_results['v3Regime_LO5_Final']:>10.2f}")
    print(f"      {'v4 regime (composite, Top-5)':<35s} {regime_results['v4Regime_LO5_Sharpe']:>8.2f} {regime_results['v4Regime_LO5_Final']:>10.2f} {regime_results['v4Regime_LO5_MaxDD_%']:>7.1f}%")
    print(f"      {'v4 regime (composite, Top-10)':<35s} {regime_results['v4Regime_LO10_Sharpe']:>8.2f} {regime_results['v4Regime_LO10_Final']:>10.2f} {regime_results['v4Regime_LO10_MaxDD_%']:>7.1f}%")

    print(f"\n   🔰 Regime Breakdown:")
    print(f"      Full (≥0.8):     {regime_results['Regime_Full_%']:.1f}%")
    print(f"      Partial 60%:     {regime_results['Regime_60_%']:.1f}%")
    print(f"      Partial 25%:     {regime_results['Regime_25_%']:.1f}%")
    print(f"      Sit out (<0.4):  {regime_results['Regime_Out_%']:.1f}%")

    # ========================================
    # 8. FEATURE IMPORTANCE
    # ========================================
    imp_model = last_model if 'last_model' in dir() else best_model_so_far
    importance = pd.DataFrame({
        'feature': ens_feats,
        'importance': imp_model.feature_importances_,
    }).sort_values('importance', ascending=False)

    print(f"\n🏆 Top 25 Features:")
    for _, row in importance.head(25).iterrows():
        print(f"   {row['feature']:35s} {row['importance']:.0f}")

    # ========================================
    # 9. COMPARISON TABLE
    # ========================================
    print(f"\n{'='*70}")
    print(f"  📊 COMPARISON TABLE")
    print(f"{'='*70}")

    all_metrics = {'baseline': metrics_base}
    if best_params and 'metrics_hpo' in dir():
        all_metrics['hpo'] = metrics_hpo
    all_metrics['feat_selected'] = metrics_sel
    if 'metrics_ens' in dir():
        all_metrics['ensemble'] = metrics_ens

    header = f"   {'Model':<18s} {'Rank_IC':>8s} {'ICIR':>8s} {'R_ICIR':>8s} {'LS_Shrp':>8s} {'LS_Ret%':>8s} {'LS_DD%':>8s}"
    print(header)
    print(f"   {'-'*66}")
    for name, m in all_metrics.items():
        print(f"   {name:<18s} {m['Rank_IC']:>8.4f} {m['ICIR']:>8.4f} {m['Rank_ICIR']:>8.4f} {m['LS_Sharpe']:>8.2f} {m['LS_Ann_Return_%']:>7.1f}% {m['LS_MaxDD_%']:>7.1f}%")

    # ========================================
    # 10. SAVE
    # ========================================
    all_results = {
        'metrics': all_metrics,
        'regime_results': regime_results,
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'horizon': HORIZON,
            'n_features_orig': len(feat_cols),
            'n_features_selected': len(ens_feats),
            'n_seeds': args.seeds,
            'hpo_trials': args.hpo_trials if not args.skip_hpo else 0,
            'train_end': TRAIN_END,
            'val_start': VAL_START,
            'val_end': VAL_END,
            'test_start': TEST_START,
            'feature_list': ens_feats,
        },
    }
    if best_params:
        all_results['hpo_best_params'] = best_params

    with open(os.path.join(results_dir, 'all_results_v4.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    importance.to_csv(os.path.join(results_dir, 'feature_importance_v4.csv'), index=False)
    eq_curves.to_parquet(os.path.join(results_dir, 'equity_curves_v4.parquet'), index=False)

    # Save test predictions
    pred_cols = [c for c in test.columns if c.startswith('pred_')]
    target_cols = [c for c in test.columns if c.startswith('target_ret_')]
    regime_save = [c for c in test.columns if c.startswith('regime_') or c.startswith('btc_regime_')]
    save_cols = ['timestamp', 'symbol'] + target_cols + pred_cols + regime_save
    save_cols = [c for c in save_cols if c in test.columns]
    test[save_cols].to_parquet(os.path.join(results_dir, 'test_predictions_v4.parquet'), index=False)

    # ========================================
    # FINAL VERDICT
    # ========================================
    best = all_metrics.get('ensemble', all_metrics.get('hpo', all_metrics['baseline']))

    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"   Rank IC:              {best['Rank_IC']:+.4f}")
    print(f"   ICIR:                 {best['ICIR']:+.4f}")
    print(f"   Rank ICIR:            {best['Rank_ICIR']:+.4f}")
    print(f"   LS Sharpe:            {best['LS_Sharpe']:+.2f}")
    print(f"   LS Ann Return:        {best['LS_Ann_Return_%']:+.1f}%")
    print(f"   LS Max Drawdown:      {best['LS_MaxDD_%']:.1f}%")
    print(f"   ---")
    print(f"   LO Top-5 no filter:     $1000 → ${regime_results['NoFilter_LO5_Final']:,.2f}")
    print(f"   LO Top-5 v3 regime:     $1000 → ${regime_results['v3Regime_LO5_Final']:,.2f}")
    print(f"   LO Top-5 v4 regime:     $1000 → ${regime_results['v4Regime_LO5_Final']:,.2f}")
    print(f"   LO Top-10 v4 regime:    $1000 → ${regime_results['v4Regime_LO10_Final']:,.2f}")
    print(f"{'='*70}")

    if best['LS_Sharpe'] > 2.0 and regime_results['v4Regime_LO5_Sharpe'] > 0:
        print("🟢 STRONG — Both LS and regime-filtered LO show positive signal.")
    elif best['LS_Sharpe'] > 2.0:
        print("🟡 LS strong but LO still negative. Consider futures/margin for short side.")
    elif best['LS_Sharpe'] > 1.0:
        print("🟠 DECENT — Signal exists but needs HIST/MASTER transformer boost.")
    else:
        print("🔴 WEAK — Need transformer models.")

    print(f"\n✅ Results saved to {results_dir}/")


if __name__ == '__main__':
    main()
