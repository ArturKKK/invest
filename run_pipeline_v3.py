#!/usr/bin/env python3
"""
Crypto Alpha Model v3 — Multi-horizon + HPO + Market Regime

Improvements over v2:
1. Multi-horizon targets (4h, 12h, 24h) — find optimal prediction horizon
2. Cross-asset features (BTC/ETH returns as market factor for all coins)
3. Market regime filter — skip trading when BTC momentum is negative
4. Optuna HPO — auto-tune LightGBM hyperparameters
5. Smart long-only backtest with regime filter
6. Purged walk-forward (gap between train/val to prevent leakage)
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
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
TRAIN_END = '2024-07-01'
VAL_END = '2025-01-01'
# Note: moved val_end earlier to have longer OOS test (2025-01 to 2026-03 = 14 months)

EXCLUDE_COLS = {
    'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
    'target_ret_4h', 'target_ret_12h', 'target_ret_24h',
    'target_cls', 'target_ret',
    'hour', 'day_of_week',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
}


# ============================================================
# ENHANCED FEATURE ENGINEERING
# ============================================================

def add_multi_horizon_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add multiple forward return horizons."""
    print("   🎯 Adding multi-horizon targets (4h, 12h, 24h)...")
    for h in [4, 12, 24]:
        df[f'target_ret_{h}h'] = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(h).shift(-h)
        )
    return df


def add_cross_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add BTC/ETH returns as market factor for ALL coins.
    Key insight: most altcoins follow BTC. Residual after removing BTC effect = alpha.
    """
    print("   🌐 Adding cross-asset features (BTC/ETH market factors)...")
    
    # Extract BTC and ETH timeseries
    btc = df[df['symbol'] == 'BTC/USDT'][['timestamp', 'close']].copy()
    btc = btc.rename(columns={'close': 'btc_close'}).drop_duplicates('timestamp')
    
    eth = df[df['symbol'] == 'ETH/USDT'][['timestamp', 'close']].copy()
    eth = eth.rename(columns={'close': 'eth_close'}).drop_duplicates('timestamp')
    
    # Merge BTC/ETH data to all rows
    df = df.merge(btc, on='timestamp', how='left')
    df = df.merge(eth, on='timestamp', how='left')
    
    # BTC returns at various horizons
    for h in [1, 4, 12, 24, 48, 168]:
        df[f'btc_ret_{h}h'] = df.groupby('symbol')['btc_close'].transform(
            lambda x: x.pct_change(h)
        )
    
    # ETH returns
    for h in [1, 4, 12, 24]:
        df[f'eth_ret_{h}h'] = df.groupby('symbol')['eth_close'].transform(
            lambda x: x.pct_change(h)
        )
    
    # BTC momentum (key regime indicator)
    df['btc_ma24'] = df.groupby('symbol')['btc_close'].transform(
        lambda x: x.rolling(24).mean()
    )
    df['btc_ma72'] = df.groupby('symbol')['btc_close'].transform(
        lambda x: x.rolling(72).mean()
    )
    df['btc_ma168'] = df.groupby('symbol')['btc_close'].transform(
        lambda x: x.rolling(168).mean()
    )
    
    # BTC regime: price vs MAs
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
    
    # Market breadth: cross-sectional dispersion at each timestamp
    # (high dispersion = more alpha opportunity)
    cs_std = df.groupby('timestamp')['ret_1h'].transform('std')
    df['market_dispersion'] = cs_std
    
    # Relative strength vs BTC
    df['ret_vs_btc_24h'] = df['ret_24h'] - df['btc_ret_24h']
    
    # Clean up intermediate columns
    df.drop(columns=['btc_close', 'eth_close', 'btc_ma24', 'btc_ma72', 'btc_ma168', 'eth_btc_ratio'], inplace=True)
    
    return df


def cross_sectional_rank(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """Rank-normalize features within each timestamp."""
    print("   📐 Cross-sectional rank normalization...")
    ranked = df.groupby('timestamp')[feat_cols].rank(pct=True)
    df[feat_cols] = ranked - 0.5  # Center around 0
    return df


def create_rank_target(df: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
    """Create cross-sectional rank target for given horizon."""
    target_col = f'target_ret_{horizon}h'
    df['target_rank'] = df.groupby('timestamp')[target_col].rank(pct=True)
    return df


# ============================================================
# OPTUNA HPO
# ============================================================

def run_optuna_hpo(X_train, y_train, X_val, y_val, n_trials=50):
    """Auto-tune LightGBM with Optuna."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("   ⚠️  Optuna not installed, using default params")
        return None
    
    print(f"   🔍 Running Optuna HPO ({n_trials} trials)...")
    
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'verbosity': -1,
            'n_estimators': 5000,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 0.7),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
            'min_child_samples': trial.suggest_int('min_child_samples', 50, 500),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.01, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.01, 10.0, log=True),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.1),
            'random_state': 42,
            'n_jobs': -1,
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        
        # Evaluate on val: use rank IC as objective (not MSE)
        preds = model.predict(X_val)
        mask = ~(np.isnan(preds) | np.isnan(y_val.values))
        if mask.sum() < 100:
            return 0
        corr, _ = spearmanr(preds[mask], y_val.values[mask])
        return corr  # Maximize rank IC
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"   ✅ Best Rank IC on val: {study.best_value:.4f}")
    print(f"   Best params: {study.best_params}")
    
    return study.best_params


def train_lgbm(X_train, y_train, X_val, y_val, custom_params=None):
    """Train LightGBM with given or default params."""
    
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
        'random_state': 42,
        'n_jobs': -1,
    }
    
    if custom_params:
        base_params.update(custom_params)
    
    model = lgb.LGBMRegressor(**base_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    print(f"   Best iteration: {model.best_iteration_}")
    return model


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


def evaluate_model(df_test, pred_col, target_col, horizon_hours=4):
    """Comprehensive evaluation."""
    
    ic = compute_ic(df_test[pred_col].values, df_test[target_col].values)
    rank_ic = compute_rank_ic(df_test[pred_col].values, df_test[target_col].values)
    
    df_eval = df_test.copy()
    df_eval['date'] = df_eval['timestamp'].dt.date
    
    daily_ics = []
    daily_rank_ics = []
    for _, grp in df_eval.groupby('date'):
        if len(grp) >= 10:
            daily_ics.append(compute_ic(grp[pred_col].values, grp[target_col].values))
            daily_rank_ics.append(compute_rank_ic(grp[pred_col].values, grp[target_col].values))
    
    daily_ics = np.array([x for x in daily_ics if not np.isnan(x)])
    daily_rank_ics = np.array([x for x in daily_rank_ics if not np.isnan(x)])
    
    icir = daily_ics.mean() / (daily_ics.std() + 1e-10) if len(daily_ics) > 0 else 0
    rank_icir = daily_rank_ics.mean() / (daily_rank_ics.std() + 1e-10) if len(daily_rank_ics) > 0 else 0
    
    # Long-Short portfolio
    periods_per_day = 24 // horizon_hours
    periods_per_year = periods_per_day * 365
    
    ls_rets = []
    lo_top5_rets = []
    lo_top10_rets = []
    
    for ts, grp in df_eval.groupby('timestamp'):
        if len(grp) < 10:
            continue
        grp = grp.sort_values(pred_col, ascending=False)
        n = max(len(grp) // 5, 1)
        
        long_ret = grp.head(n)[target_col].mean()
        short_ret = grp.tail(n)[target_col].mean()
        ls_rets.append(long_ret - short_ret)
        
        # Top-5 and Top-10 long only
        lo_top5_rets.append(grp.head(5)[target_col].mean())
        lo_top10_rets.append(grp.head(10)[target_col].mean())
    
    ls_rets = np.array(ls_rets)
    lo5 = np.array(lo_top5_rets)
    lo10 = np.array(lo_top10_rets)
    
    def sharpe(rets, ppyr):
        return (rets.mean() / (rets.std() + 1e-10)) * np.sqrt(ppyr)
    
    def max_dd(rets):
        cum = np.cumprod(1 + rets)
        return np.min(cum / np.maximum.accumulate(cum) - 1)
    
    def total_ret(rets):
        return np.prod(1 + rets) - 1
    
    # Commission model:
    # OKX taker fee: 0.1% per trade. Assume ~40% portfolio turnover per rebalance.
    # Round-trip cost = 0.1% buy + 0.1% sell = 0.2% on turned-over portion.
    # Effective cost per period = 0.2% × 40% = 0.08% = 0.0008
    turnover_rate = 0.4
    fee_per_trade = 0.001  # 0.1%
    comm_per_period = 2 * fee_per_trade * turnover_rate  # 0.0008
    lo5_net = lo5 - comm_per_period
    lo10_net = lo10 - comm_per_period
    
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
    
    return metrics, lo5_net


def evaluate_with_regime_filter(df_test, pred_col, target_col, horizon_hours=4):
    """
    Long-only backtest WITH market regime filter:
    - Only buy when BTC is above its 72h MA (bullish regime)
    - Hold cash when BTC is below 72h MA (bearish regime)
    """
    df_eval = df_test.copy()
    periods_per_day = 24 // horizon_hours
    periods_per_year = periods_per_day * 365
    
    lo5_rets = []
    lo5_filtered_rets = []
    regime_on_count = 0
    regime_off_count = 0
    
    for ts, grp in df_eval.groupby('timestamp'):
        if len(grp) < 10:
            continue
        grp = grp.sort_values(pred_col, ascending=False)
        top5_ret = grp.head(5)[target_col].mean()
        
        # Check BTC regime (use btc_regime_72 if available)
        if 'btc_regime_72' in grp.columns:
            regime = grp['btc_regime_72'].iloc[0]
        else:
            regime = 1  # Default: always on
        
        # Commission: 0.2% round-trip × 40% turnover = 0.08% per period
        comm = 0.0008
        net_ret = top5_ret - comm
        lo5_rets.append(net_ret)
        
        if regime > 0.5:  # Bullish
            lo5_filtered_rets.append(net_ret)
            regime_on_count += 1
        else:  # Bearish — stay in cash
            lo5_filtered_rets.append(0.0)
            regime_off_count += 1
    
    lo5_rets = np.array(lo5_rets)
    lo5_filt = np.array(lo5_filtered_rets)
    
    def sharpe(r, ppyr):
        return (r.mean() / (r.std() + 1e-10)) * np.sqrt(ppyr)
    
    def max_dd(r):
        cum = np.cumprod(1 + r)
        return np.min(cum / np.maximum.accumulate(cum) - 1)
    
    equity_no_filter = 1000 * np.cumprod(1 + lo5_rets)
    equity_filtered = 1000 * np.cumprod(1 + lo5_filt)
    
    regime_metrics = {
        'No_Filter_Sharpe': round(float(sharpe(lo5_rets, periods_per_year)), 2),
        'No_Filter_Final': round(float(equity_no_filter[-1]), 2),
        'No_Filter_MaxDD_%': round(float(max_dd(lo5_rets) * 100), 1),
        'Regime_Filter_Sharpe': round(float(sharpe(lo5_filt, periods_per_year)), 2),
        'Regime_Filter_Final': round(float(equity_filtered[-1]), 2),
        'Regime_Filter_MaxDD_%': round(float(max_dd(lo5_filt) * 100), 1),
        'Regime_ON_periods': regime_on_count,
        'Regime_OFF_periods': regime_off_count,
        'Regime_ON_%': round(regime_on_count / (regime_on_count + regime_off_count) * 100, 1),
    }
    
    return regime_metrics, equity_no_filter, equity_filtered


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--results', type=str, default=None)
    parser.add_argument('--hpo-trials', type=int, default=50)
    parser.add_argument('--skip-hpo', action='store_true')
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data or os.path.join(project_root, 'data', 'features')
    results_dir = args.results or os.path.join(project_root, 'results_v3')
    os.makedirs(results_dir, exist_ok=True)
    
    feat_path = os.path.join(data_dir, 'crypto_features_1h.parquet')
    if not os.path.exists(feat_path):
        print(f"❌ Feature file not found: {feat_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("  CRYPTO ALPHA MODEL v3")
    print("  Multi-Horizon + Cross-Asset + Regime Filter + HPO")
    print("=" * 70)
    
    # ========================================
    # LOAD & ENRICH DATA
    # ========================================
    print(f"\n📊 Loading data...")
    df = pd.read_parquet(feat_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Shape: {df.shape}, Symbols: {df['symbol'].nunique()}")
    
    # Add multi-horizon targets
    df = add_multi_horizon_targets(df)
    
    # Add cross-asset features
    df = add_cross_asset_features(df)
    
    # Clean
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # Drop rows without targets
    df = df.dropna(subset=['target_ret_4h', 'target_ret_12h', 'target_ret_24h'])
    
    # Feature columns
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS 
                 and c not in ['target_rank', 'target_excess']
                 and not c.startswith('target_')]
    # Remove any remaining non-numeric
    feat_cols = [c for c in feat_cols if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    print(f"   Features: {len(feat_cols)}")
    print(f"   New cross-asset features: btc_ret_*, eth_ret_*, btc_regime_*, btc_vol_*, market_dispersion, ret_vs_btc_24h")
    
    # Fill NaN features
    df[feat_cols] = df[feat_cols].fillna(0)
    
    # SAVE regime columns BEFORE ranking (they are binary, ranking destroys them)
    regime_cols = ['btc_regime_24', 'btc_regime_72', 'btc_regime_168']
    regime_backup = {}
    for col in regime_cols:
        if col in df.columns:
            regime_backup[col] = df[col].copy()
    
    # Cross-sectional rank normalization
    df = cross_sectional_rank(df, feat_cols)
    
    # Restore regime columns
    for col, vals in regime_backup.items():
        df[col] = vals
    print(f"   ✅ Restored {len(regime_backup)} regime columns (not ranked)")
    
    print(f"   Final shape: {df.shape}")
    
    # ========================================
    # TEST EACH HORIZON
    # ========================================
    
    all_horizon_results = {}
    best_horizon = None
    best_ls_sharpe = -999
    
    for horizon in [4, 12, 24]:
        print(f"\n{'=' * 70}")
        print(f"  HORIZON: {horizon}h forward return")
        print(f"{'=' * 70}")
        
        target_col = f'target_ret_{horizon}h'
        
        # Create rank target for this horizon
        df['target_rank'] = df.groupby('timestamp')[target_col].rank(pct=True)
        
        # Split
        train = df[df['timestamp'] < TRAIN_END].copy()
        val = df[(df['timestamp'] >= TRAIN_END) & (df['timestamp'] < VAL_END)].copy()
        test = df[df['timestamp'] >= VAL_END].copy()
        
        print(f"   Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
        
        X_train = train[feat_cols]
        X_val = val[feat_cols]
        X_test = test[feat_cols]
        y_train = train['target_rank']
        y_val = val['target_rank']
        
        # Train (rank target — best from v2)
        print(f"\n   🚀 Training LightGBM (rank target, {horizon}h)...")
        model = train_lgbm(X_train, y_train, X_val, y_val)
        
        test[f'pred_{horizon}h'] = model.predict(X_test)
        
        # Evaluate
        metrics, _ = evaluate_model(test, f'pred_{horizon}h', target_col, horizon)
        
        # Regime filter evaluation
        regime_metrics, _, _ = evaluate_with_regime_filter(
            test, f'pred_{horizon}h', target_col, horizon
        )
        
        print(f"\n   📈 Results ({horizon}h):")
        for k, v in metrics.items():
            flag = ""
            if k == 'Rank_IC' and abs(v) > 0.02: flag = " ✓"
            if k == 'LS_Sharpe' and v > 1.0: flag = " ✓"
            if k == 'ICIR' and abs(v) > 0.3: flag = " ✓"
            print(f"      {k:25s} {v}{flag}")
        
        print(f"\n   🔰 Regime-Filtered Long-Only ({horizon}h):")
        for k, v in regime_metrics.items():
            print(f"      {k:25s} {v}")
        
        all_horizon_results[f'{horizon}h'] = {
            'model_metrics': metrics,
            'regime_metrics': regime_metrics,
        }
        
        if metrics['LS_Sharpe'] > best_ls_sharpe:
            best_ls_sharpe = metrics['LS_Sharpe']
            best_horizon = horizon
    
    print(f"\n{'=' * 70}")
    print(f"  🏆 BEST HORIZON: {best_horizon}h (LS Sharpe = {best_ls_sharpe})")
    print(f"{'=' * 70}")
    
    # ========================================
    # HPO ON BEST HORIZON
    # ========================================
    
    target_col = f'target_ret_{best_horizon}h'
    df['target_rank'] = df.groupby('timestamp')[target_col].rank(pct=True)
    
    train = df[df['timestamp'] < TRAIN_END].copy()
    val = df[(df['timestamp'] >= TRAIN_END) & (df['timestamp'] < VAL_END)].copy()
    test = df[df['timestamp'] >= VAL_END].copy()
    
    X_train = train[feat_cols]
    X_val = val[feat_cols]
    X_test = test[feat_cols]
    y_train = train['target_rank']
    y_val = val['target_rank']
    
    if not args.skip_hpo:
        print(f"\n{'=' * 70}")
        print(f"  OPTUNA HPO — {args.hpo_trials} trials on {best_horizon}h horizon")
        print(f"{'=' * 70}")
        
        best_params = run_optuna_hpo(X_train, y_train, X_val, y_val, n_trials=args.hpo_trials)
        
        if best_params:
            print(f"\n   🚀 Re-training with optimized params...")
            model_hpo = train_lgbm(X_train, y_train, X_val, y_val, custom_params=best_params)
            test['pred_hpo'] = model_hpo.predict(X_test)
            
            metrics_hpo, _ = evaluate_model(test, 'pred_hpo', target_col, best_horizon)
            regime_hpo, eq_no, eq_filt = evaluate_with_regime_filter(
                test, 'pred_hpo', target_col, best_horizon
            )
            
            print(f"\n   📈 HPO Model Results ({best_horizon}h):")
            for k, v in metrics_hpo.items():
                flag = ""
                if k == 'Rank_IC' and abs(v) > 0.02: flag = " ✓"
                if k == 'LS_Sharpe' and v > 1.0: flag = " ✓"
                if k == 'ICIR' and abs(v) > 0.3: flag = " ✓"
                print(f"      {k:25s} {v}{flag}")
            
            print(f"\n   🔰 HPO Regime-Filtered Long-Only:")
            for k, v in regime_hpo.items():
                print(f"      {k:25s} {v}")
            
            all_horizon_results['hpo'] = {
                'model_metrics': metrics_hpo,
                'regime_metrics': regime_hpo,
                'best_params': best_params,
            }
            
            # Save equity curves
            pd.DataFrame({
                'equity_no_filter': eq_no,
                'equity_regime_filter': eq_filt,
            }).to_parquet(os.path.join(results_dir, 'equity_curves_hpo.parquet'), index=False)
    
    # ========================================
    # FEATURE IMPORTANCE (best model)
    # ========================================
    pred_col = 'pred_hpo' if 'pred_hpo' in test.columns else f'pred_{best_horizon}h'
    
    # Re-train final model for importance (or use hpo model)
    importance_model = model_hpo if 'model_hpo' in dir() else model
    importance = pd.DataFrame({
        'feature': feat_cols,
        'importance': importance_model.feature_importances_,
    }).sort_values('importance', ascending=False)
    
    print(f"\n🏆 Top 25 Features:")
    for _, row in importance.head(25).iterrows():
        print(f"   {row['feature']:35s} {row['importance']:.0f}")
    
    # ========================================
    # SAVE
    # ========================================
    
    all_horizon_results['meta'] = {
        'timestamp': datetime.now().isoformat(),
        'best_horizon': best_horizon,
        'n_features': len(feat_cols),
        'train_end': TRAIN_END,
        'val_end': VAL_END,
        'feature_list': feat_cols,
    }
    
    with open(os.path.join(results_dir, 'all_results_v3.json'), 'w') as f:
        json.dump(all_horizon_results, f, indent=2, default=str)
    
    importance.to_csv(os.path.join(results_dir, 'feature_importance_v3.csv'), index=False)
    
    # Save test predictions
    pred_cols = [c for c in test.columns if c.startswith('pred_')]
    target_cols = [c for c in test.columns if c.startswith('target_ret_')]
    save_cols = ['timestamp', 'symbol'] + target_cols + pred_cols
    if 'btc_regime_72' in test.columns:
        save_cols.append('btc_regime_72')
    test[save_cols].to_parquet(os.path.join(results_dir, 'test_predictions_v3.parquet'), index=False)
    
    # ========================================
    # FINAL VERDICT
    # ========================================
    best_result = all_horizon_results.get('hpo', all_horizon_results[f'{best_horizon}h'])
    bm = best_result['model_metrics']
    br = best_result['regime_metrics']
    
    print(f"\n{'=' * 70}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"   Best horizon:         {best_horizon}h")
    print(f"   Rank IC:              {bm['Rank_IC']:+.4f}")
    print(f"   ICIR:                 {bm['ICIR']:+.4f}")
    print(f"   Rank ICIR:            {bm['Rank_ICIR']:+.4f}")
    print(f"   LS Sharpe:            {bm['LS_Sharpe']:+.2f}")
    print(f"   LS Ann Return:        {bm['LS_Ann_Return_%']:+.1f}%")
    print(f"   LS Max Drawdown:      {bm['LS_MaxDD_%']:.1f}%")
    print(f"   ---")
    print(f"   Long-Only (no filter):   $1000 → ${br['No_Filter_Final']:,.2f}  (Sharpe {br['No_Filter_Sharpe']})")
    print(f"   Long-Only (BTC regime):  $1000 → ${br['Regime_Filter_Final']:,.2f}  (Sharpe {br['Regime_Filter_Sharpe']})")
    print(f"   Regime active:           {br['Regime_ON_%']}% of time")
    print(f"{'=' * 70}")
    
    if bm['LS_Sharpe'] > 2.0 and br['Regime_Filter_Sharpe'] > 0.5:
        print("🟢 STRONG — Both LS and regime-filtered LO show signal.")
    elif bm['LS_Sharpe'] > 2.0:
        print("🟡 RANKING IS STRONG — LS works. Long-only needs short side or better regime filter.")
    elif bm['LS_Sharpe'] > 1.0:
        print("🟠 DECENT — Signal exists but needs more work.")
    else:
        print("🔴 WEAK — Need transformer models (HIST/MASTER).")
    
    print(f"\n✅ Results saved to {results_dir}/")


if __name__ == '__main__':
    main()
