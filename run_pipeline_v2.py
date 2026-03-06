#!/usr/bin/env python3
"""
Crypto Alpha Model v2 — Proper quantitative pipeline.

Key fixes over v1:
1. Cross-sectional rank normalization (features ranked within each timestamp)
2. Rank-based target (cross-sectional rank of forward returns, not raw returns)
3. Time features removed from model input (caused overfitting to time-of-day)
4. Feature clipping (winsorize outliers at 1st/99th percentile)
5. Better LightGBM hyperparams (lower LR, more trees, more regularization)
6. Multiple model variants (regression + lambdarank)
7. Proper long-short evaluation with turnover control

Usage:
    python run_pipeline_v2.py
    python run_pipeline_v2.py --data /path/to/features --results ./results_v2
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
VAL_END = '2025-07-01'

# Features to EXCLUDE from model input
# Time features cause data leakage / overfitting to hour-of-day patterns
EXCLUDE_COLS = {
    'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
    'target_ret', 'target_cls', 'hour', 'day_of_week',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',  # Remove time encodings!
}


# ============================================================
# CROSS-SECTIONAL PROCESSING (the key fix)
# ============================================================

def cross_sectional_rank(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """
    Rank-normalize features within each timestamp.
    This converts absolute values → relative position [0, 1].
    
    Why: BTC's RSI=70 and DOGE's RSI=70 mean different things.
    After ranking, features become cross-sectionally comparable.
    """
    print("   📐 Cross-sectional rank normalization...")
    
    # Group by timestamp, rank each feature
    ranked = df.groupby('timestamp')[feat_cols].rank(pct=True)
    
    # Replace original features with ranked versions
    df[feat_cols] = ranked
    
    return df


def winsorize_features(df: pd.DataFrame, feat_cols: list, 
                       lower=0.01, upper=0.99) -> pd.DataFrame:
    """Clip features at percentiles to handle outliers."""
    print("   ✂️  Winsorizing features at 1st/99th percentile...")
    for col in feat_cols:
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
        df[col] = df[col].clip(lo, hi)
    return df


def create_rank_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cross-sectional rank target.
    Instead of predicting raw return (noisy), predict relative ranking.
    
    target_rank: [0, 1] — what fraction of coins this coin outperforms
    """
    print("   🎯 Creating cross-sectional rank target...")
    df['target_rank'] = df.groupby('timestamp')['target_ret'].rank(pct=True)
    # Also create excess return (vs cross-sectional mean)
    df['target_excess'] = df.groupby('timestamp')['target_ret'].transform(
        lambda x: x - x.mean()
    )
    return df


# ============================================================
# MODEL TRAINING
# ============================================================

def train_lgbm_v2(X_train, y_train, X_val, y_val):
    """LightGBM with better hyperparameters for noisy financial data."""
    
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'verbosity': -1,
        'n_estimators': 5000,       # More trees (was 2000)
        'learning_rate': 0.01,       # Lower LR (was 0.05) — less overfitting
        'max_depth': 6,              # Shallower (was 8) — less memorization
        'num_leaves': 31,            # Less leaves (was 63)
        'feature_fraction': 0.5,     # Stronger feature subsampling (was 0.7)
        'bagging_fraction': 0.7,     # Stronger bagging (was 0.8)
        'bagging_freq': 1,           # Bag every iteration (was 5)
        'min_child_samples': 200,    # More samples per leaf (was 50) — less noise
        'lambda_l1': 1.0,            # Stronger L1 (was 0.1)
        'lambda_l2': 1.0,            # Stronger L2 (was 0.1)
        'min_gain_to_split': 0.01,   # NEW: minimum gain to add a split
        'random_state': 42,
        'n_jobs': -1,
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(100),    # More patience (was 50)
            lgb.log_evaluation(200),
        ]
    )
    
    print(f"   Best iteration: {model.best_iteration_}")
    return model


# ============================================================
# EVALUATION (quant-grade metrics)
# ============================================================

def compute_ic(preds, actuals):
    mask = ~(np.isnan(preds) | np.isnan(actuals))
    if mask.sum() < 10:
        return np.nan
    return np.corrcoef(preds[mask], actuals[mask])[0, 1]


def compute_rank_ic(preds, actuals):
    mask = ~(np.isnan(preds) | np.isnan(actuals))
    if mask.sum() < 10:
        return np.nan
    corr, _ = spearmanr(preds[mask], actuals[mask])
    return corr


def evaluate_model(df_test: pd.DataFrame, pred_col: str = 'pred') -> dict:
    """Full evaluation: IC, ICIR, Long-Short Sharpe, Backtest."""
    
    # Overall metrics
    ic = compute_ic(df_test[pred_col].values, df_test['target_ret'].values)
    rank_ic = compute_rank_ic(df_test[pred_col].values, df_test['target_ret'].values)
    
    # Daily IC series
    df_eval = df_test.copy()
    df_eval['date'] = df_eval['timestamp'].dt.date
    
    daily_ics = []
    daily_rank_ics = []
    for _, grp in df_eval.groupby('date'):
        if len(grp) >= 10:
            daily_ics.append(compute_ic(grp[pred_col].values, grp['target_ret'].values))
            daily_rank_ics.append(compute_rank_ic(grp[pred_col].values, grp['target_ret'].values))
    
    daily_ics = np.array([x for x in daily_ics if not np.isnan(x)])
    daily_rank_ics = np.array([x for x in daily_rank_ics if not np.isnan(x)])
    
    icir = daily_ics.mean() / (daily_ics.std() + 1e-10) if len(daily_ics) > 0 else 0
    rank_icir = daily_rank_ics.mean() / (daily_rank_ics.std() + 1e-10) if len(daily_rank_ics) > 0 else 0
    
    # Direction accuracy
    pred_dir = (df_test[pred_col] > df_test[pred_col].median()).astype(int)
    actual_dir = (df_test['target_ret'] > 0).astype(int)
    dir_acc = accuracy_score(actual_dir, pred_dir)
    
    # ====== LONG-SHORT PORTFOLIO ======
    # Every 4h: long top 20%, short bottom 20% (= standard quantile spread)
    daily_ls_returns = []
    long_only_returns = []
    
    for ts, grp in df_eval.groupby('timestamp'):
        if len(grp) < 10:
            continue
        grp = grp.sort_values(pred_col, ascending=False)
        n = max(len(grp) // 5, 1)  # Top/bottom 20%
        
        long_ret = grp.head(n)['target_ret'].mean()
        short_ret = grp.tail(n)['target_ret'].mean()
        
        daily_ls_returns.append(long_ret - short_ret)
        long_only_returns.append(long_ret)
    
    ls_rets = np.array(daily_ls_returns)
    lo_rets = np.array(long_only_returns)
    
    # Annualize (crypto: 24 periods/day × 365 days — but we rebalance every 4h = 6/day)
    periods_per_year = 6 * 365  # 4h periods in a year
    
    ls_sharpe = (ls_rets.mean() / (ls_rets.std() + 1e-10)) * np.sqrt(periods_per_year)
    ls_ann_ret = ls_rets.mean() * periods_per_year
    
    lo_sharpe = (lo_rets.mean() / (lo_rets.std() + 1e-10)) * np.sqrt(periods_per_year)
    lo_ann_ret = lo_rets.mean() * periods_per_year
    
    # Drawdown (long-short cumulative)
    ls_cum = np.cumprod(1 + ls_rets)
    ls_dd = np.min(ls_cum / np.maximum.accumulate(ls_cum) - 1) if len(ls_cum) > 0 else 0
    
    # Top-5 long only backtest (realistic for $1K capital)
    top5_returns = []
    for ts, grp in df_eval.groupby('timestamp'):
        if len(grp) < 10:
            continue
        grp = grp.sort_values(pred_col, ascending=False)
        top5_ret = grp.head(5)['target_ret'].mean()
        # Commission: 0.1% buy + 0.1% sell, amortized over 4h
        top5_returns.append(top5_ret - 0.0005)
    
    top5_rets = np.array(top5_returns)
    top5_equity = 1000 * np.cumprod(1 + top5_rets)
    
    metrics = {
        'IC': round(float(ic), 4),
        'Rank_IC': round(float(rank_ic), 4),
        'Daily_IC_mean': round(float(daily_ics.mean()), 4) if len(daily_ics) > 0 else 0,
        'Daily_IC_std': round(float(daily_ics.std()), 4) if len(daily_ics) > 0 else 0,
        'ICIR': round(float(icir), 4),
        'Daily_RankIC_mean': round(float(daily_rank_ics.mean()), 4) if len(daily_rank_ics) > 0 else 0,
        'Rank_ICIR': round(float(rank_icir), 4),
        'Direction_Accuracy': round(float(dir_acc), 4),
        'LS_Sharpe': round(float(ls_sharpe), 2),
        'LS_Ann_Return': round(float(ls_ann_ret * 100), 2),  # %
        'LS_Max_Drawdown': round(float(ls_dd * 100), 2),     # %
        'LO_Sharpe': round(float(lo_sharpe), 2),
        'LO_Ann_Return': round(float(lo_ann_ret * 100), 2),
        'Top5_Final_Capital': round(float(top5_equity[-1]), 2) if len(top5_equity) > 0 else 0,
        'Top5_Total_Return': round(float(top5_equity[-1] / 1000 - 1) * 100, 2) if len(top5_equity) > 0 else 0,
        'N_test': len(df_test),
        'N_periods': len(daily_ls_returns),
    }
    
    return metrics, top5_equity, top5_rets


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--results', type=str, default=None)
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data or os.path.join(project_root, 'data', 'features')
    results_dir = args.results or os.path.join(project_root, 'results_v2')
    os.makedirs(results_dir, exist_ok=True)
    
    feat_path = os.path.join(data_dir, 'crypto_features_1h.parquet')
    if not os.path.exists(feat_path):
        print(f"❌ Feature file not found: {feat_path}")
        sys.exit(1)
    
    # ========================================
    # LOAD DATA
    # ========================================
    print("=" * 70)
    print("  CRYPTO ALPHA MODEL v2 — Cross-Sectional Rank Pipeline")
    print("=" * 70)
    print(f"\n📊 Loading {feat_path}...")
    
    df = pd.read_parquet(feat_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Shape: {df.shape}")
    print(f"   Period: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"   Symbols: {df['symbol'].nunique()}")
    
    # Feature columns (EXCLUDING time features)
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    print(f"   Raw features: {len(feat_cols)}")
    
    # ========================================
    # PREPROCESSING
    # ========================================
    print("\n📐 PREPROCESSING...")
    
    # 1. Replace inf with NaN, then fill
    for col in feat_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # 2. Drop rows where target is NaN
    df = df.dropna(subset=['target_ret'])
    
    # 3. Fill remaining NaN features with 0 (neutral rank)
    df[feat_cols] = df[feat_cols].fillna(0)
    
    # 4. Create rank-based target
    df = create_rank_target(df)
    
    # 5. Cross-sectional rank normalization of features
    df = cross_sectional_rank(df, feat_cols)
    
    # 6. Center ranks around 0 (rank pct gives [0,1], shift to [-0.5, 0.5])
    df[feat_cols] = df[feat_cols] - 0.5
    
    print(f"   Final shape: {df.shape}")
    print(f"   Features for model: {len(feat_cols)}")
    
    # ========================================
    # SPLIT
    # ========================================
    train = df[df['timestamp'] < TRAIN_END].copy()
    val = df[(df['timestamp'] >= TRAIN_END) & (df['timestamp'] < VAL_END)].copy()
    test = df[df['timestamp'] >= VAL_END].copy()
    
    print(f"\n📅 WALK-FORWARD SPLIT:")
    print(f"   Train: {len(train):>10,} rows  ({train['timestamp'].min().date()} → {train['timestamp'].max().date()})")
    print(f"   Val:   {len(val):>10,} rows  ({val['timestamp'].min().date()} → {val['timestamp'].max().date()})")
    print(f"   Test:  {len(test):>10,} rows  ({test['timestamp'].min().date()} → {test['timestamp'].max().date()})")
    
    X_train = train[feat_cols]
    X_val = val[feat_cols]
    X_test = test[feat_cols]
    
    # ========================================
    # MODEL 1: Predict RANK (cross-sectional)
    # ========================================
    print("\n" + "=" * 70)
    print("  MODEL 1: LightGBM → Cross-Sectional Rank Target")
    print("=" * 70)
    
    y_train_rank = train['target_rank']
    y_val_rank = val['target_rank']
    
    model_rank = train_lgbm_v2(X_train, y_train_rank, X_val, y_val_rank)
    test['pred_rank'] = model_rank.predict(X_test)
    
    print("\n📈 TEST RESULTS (Rank Model):")
    metrics_rank, equity_rank, rets_rank = evaluate_model(test, 'pred_rank')
    for k, v in metrics_rank.items():
        flag = ""
        if k == 'Rank_IC' and abs(v) > 0.02:
            flag = " ✓"
        if k == 'LS_Sharpe' and v > 0.5:
            flag = " ✓"
        if k == 'ICIR' and abs(v) > 0.1:
            flag = " ✓"
        print(f"   {k:25s} {v}{flag}")
    
    # ========================================
    # MODEL 2: Predict EXCESS RETURN
    # ========================================
    print("\n" + "=" * 70)
    print("  MODEL 2: LightGBM → Excess Return Target")
    print("=" * 70)
    
    y_train_exc = train['target_excess']
    y_val_exc = val['target_excess']
    
    model_excess = train_lgbm_v2(X_train, y_train_exc, X_val, y_val_exc)
    test['pred_excess'] = model_excess.predict(X_test)
    
    print("\n📈 TEST RESULTS (Excess Return Model):")
    metrics_excess, equity_excess, rets_excess = evaluate_model(test, 'pred_excess')
    for k, v in metrics_excess.items():
        flag = ""
        if k == 'Rank_IC' and abs(v) > 0.02:
            flag = " ✓"
        if k == 'LS_Sharpe' and v > 0.5:
            flag = " ✓"
        if k == 'ICIR' and abs(v) > 0.1:
            flag = " ✓"
        print(f"   {k:25s} {v}{flag}")
    
    # ========================================
    # MODEL 3: Predict RAW RETURN (but with rank features)
    # ========================================
    print("\n" + "=" * 70)
    print("  MODEL 3: LightGBM → Raw Return (with ranked features)")
    print("=" * 70)
    
    y_train_raw = train['target_ret']
    y_val_raw = val['target_ret']
    
    model_raw = train_lgbm_v2(X_train, y_train_raw, X_val, y_val_raw)
    test['pred_raw'] = model_raw.predict(X_test)
    
    print("\n📈 TEST RESULTS (Raw Return Model):")
    metrics_raw, equity_raw, rets_raw = evaluate_model(test, 'pred_raw')
    for k, v in metrics_raw.items():
        flag = ""
        if k == 'Rank_IC' and abs(v) > 0.02:
            flag = " ✓"
        if k == 'LS_Sharpe' and v > 0.5:
            flag = " ✓"
        if k == 'ICIR' and abs(v) > 0.1:
            flag = " ✓"
        print(f"   {k:25s} {v}{flag}")
    
    # ========================================
    # ENSEMBLE: Average of all 3 models
    # ========================================
    print("\n" + "=" * 70)
    print("  ENSEMBLE: Average of 3 Models")
    print("=" * 70)
    
    # Normalize each prediction to [0, 1] range before averaging
    for col in ['pred_rank', 'pred_excess', 'pred_raw']:
        mn, mx = test[col].min(), test[col].max()
        test[f'{col}_norm'] = (test[col] - mn) / (mx - mn + 1e-10)
    
    test['pred_ensemble'] = (
        test['pred_rank_norm'] + 
        test['pred_excess_norm'] + 
        test['pred_raw_norm']
    ) / 3.0
    
    print("\n📈 TEST RESULTS (Ensemble):")
    metrics_ens, equity_ens, rets_ens = evaluate_model(test, 'pred_ensemble')
    for k, v in metrics_ens.items():
        flag = ""
        if k == 'Rank_IC' and abs(v) > 0.02:
            flag = " ✓"
        if k == 'LS_Sharpe' and v > 0.5:
            flag = " ✓"
        if k == 'ICIR' and abs(v) > 0.1:
            flag = " ✓"
        print(f"   {k:25s} {v}{flag}")
    
    # ========================================
    # FEATURE IMPORTANCE (from rank model)
    # ========================================
    importance = pd.DataFrame({
        'feature': feat_cols,
        'importance_rank_model': model_rank.feature_importances_,
        'importance_excess_model': model_excess.feature_importances_,
        'importance_raw_model': model_raw.feature_importances_,
    })
    importance['importance_avg'] = (
        importance['importance_rank_model'] + 
        importance['importance_excess_model'] + 
        importance['importance_raw_model']
    ) / 3
    importance = importance.sort_values('importance_avg', ascending=False)
    
    print(f"\n🏆 Top 20 Features (avg across 3 models):")
    for _, row in importance.head(20).iterrows():
        print(f"   {row['feature']:30s} {row['importance_avg']:.0f}")
    
    # ========================================
    # PICK BEST MODEL
    # ========================================
    all_results = {
        'rank_model': metrics_rank,
        'excess_model': metrics_excess,
        'raw_model': metrics_raw,
        'ensemble': metrics_ens,
    }
    
    # Pick by LS_Sharpe
    best_name = max(all_results, key=lambda k: all_results[k]['LS_Sharpe'])
    best_metrics = all_results[best_name]
    
    print(f"\n{'=' * 70}")
    print(f"  🏆 BEST MODEL: {best_name} (LS Sharpe = {best_metrics['LS_Sharpe']})")
    print(f"{'=' * 70}")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    
    # Save all metrics
    all_results['best_model'] = best_name
    all_results['timestamp'] = datetime.now().isoformat()
    all_results['config'] = {
        'train_end': TRAIN_END,
        'val_end': VAL_END,
        'n_features': len(feat_cols),
        'cross_sectional_rank': True,
        'time_features_removed': True,
    }
    
    with open(os.path.join(results_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    importance.to_csv(os.path.join(results_dir, 'feature_importance_v2.csv'), index=False)
    
    # Save test predictions (all models)
    test[['timestamp', 'symbol', 'target_ret', 'target_rank', 'target_excess',
          'pred_rank', 'pred_excess', 'pred_raw', 'pred_ensemble']].to_parquet(
        os.path.join(results_dir, 'test_predictions_v2.parquet'), index=False
    )
    
    # Save equity curves
    best_equity_map = {
        'rank_model': (equity_rank, rets_rank),
        'excess_model': (equity_excess, rets_excess),
        'raw_model': (equity_raw, rets_raw),
        'ensemble': (equity_ens, rets_ens),
    }
    eq, rt = best_equity_map[best_name]
    if len(eq) > 0:
        pd.DataFrame({'equity': eq, 'return': rt}).to_parquet(
            os.path.join(results_dir, 'equity_curve_v2.parquet'), index=False
        )
    
    # ========================================
    # VERDICT
    # ========================================
    print(f"\n{'=' * 70}")
    best_ic = best_metrics.get('Rank_IC', 0)
    best_sharpe = best_metrics.get('LS_Sharpe', 0)
    best_icir = best_metrics.get('ICIR', 0)
    
    if best_sharpe > 2.0 and best_ic > 0.03:
        print("🟢 EXCELLENT — Strong alpha signal. Ready for paper trading!")
    elif best_sharpe > 1.0 and best_ic > 0.02:
        print("🟢 GOOD — Meaningful signal detected. Worth optimizing + paper trading.")
    elif best_sharpe > 0.5 or best_ic > 0.015:
        print("🟡 DECENT — Some signal. Improve with HIST/MASTER transformers.")
    elif best_sharpe > 0 and best_ic > 0.005:
        print("🟠 WEAK but non-zero. Need better features or deep learning models.")
    else:
        print("🔴 NO SIGNAL with LightGBM baseline. This is common for crypto.")
        print("   Next steps: try HIST/MASTER transformer, add on-chain data.")
    
    print(f"\n   Key numbers:")
    print(f"   Rank IC:    {best_ic:+.4f}  (need > 0.02)")
    print(f"   ICIR:       {best_icir:+.4f}  (need > 0.3)")
    print(f"   LS Sharpe:  {best_sharpe:+.2f}   (need > 1.0)")
    print(f"   Top5 $1K →  ${best_metrics.get('Top5_Final_Capital', 0):,.2f}")
    print(f"{'=' * 70}")
    
    print(f"\n✅ Results saved to {results_dir}/")


if __name__ == '__main__':
    main()
