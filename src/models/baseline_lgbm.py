"""
LightGBM baseline model for crypto return prediction.
Walk-forward validation with proper time-series split.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
import os
import json
from datetime import datetime

DATA_FEAT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'features')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
TIMEFRAME = '1h'


# ============================================================
# Walk-Forward Split (NO random split — this is time series!)
# ============================================================
# Train:  2021-01 to 2024-06  (3.5 years)
# Val:    2024-07 to 2025-06  (1 year)
# Test:   2025-07 to 2026-03  (8 months — TRUE out-of-sample)
# ============================================================

TRAIN_END = '2024-07-01'
VAL_END = '2025-07-01'


def get_feature_cols(df: pd.DataFrame) -> list:
    """Get feature column names (everything except meta/target columns)."""
    exclude = {'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
               'target_ret', 'target_cls', 'hour', 'day_of_week'}
    return [c for c in df.columns if c not in exclude]


def train_lightgbm(X_train, y_train, X_val, y_val, task='regression'):
    """Train LightGBM with early stopping."""

    if task == 'regression':
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'verbosity': -1,
            'n_estimators': 2000,
            'learning_rate': 0.05,
            'max_depth': 8,
            'num_leaves': 63,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 50,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'random_state': 42,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(100),
            ]
        )
    else:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'n_estimators': 2000,
            'learning_rate': 0.05,
            'max_depth': 8,
            'num_leaves': 63,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 50,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'random_state': 42,
            'is_unbalance': True,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(100),
            ]
        )

    return model


def compute_ic(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Information Coefficient = Pearson correlation between prediction and actual."""
    mask = ~(np.isnan(predictions) | np.isnan(actuals))
    if mask.sum() < 10:
        return 0.0
    return np.corrcoef(predictions[mask], actuals[mask])[0, 1]


def compute_rank_ic(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Rank IC = Spearman correlation."""
    from scipy.stats import spearmanr
    mask = ~(np.isnan(predictions) | np.isnan(actuals))
    if mask.sum() < 10:
        return 0.0
    corr, _ = spearmanr(predictions[mask], actuals[mask])
    return corr


def evaluate_regression(df_test: pd.DataFrame, pred_col='pred_ret') -> dict:
    """Evaluate regression predictions with quant-standard metrics."""
    # Overall IC
    ic = compute_ic(df_test[pred_col].values, df_test['target_ret'].values)
    rank_ic = compute_rank_ic(df_test[pred_col].values, df_test['target_ret'].values)

    # Daily IC (group by date, compute IC per day, then mean/std)
    df_test = df_test.copy()
    df_test['date'] = df_test['timestamp'].dt.date
    daily_ics = []
    for _, group in df_test.groupby('date'):
        if len(group) >= 5:
            d_ic = compute_ic(group[pred_col].values, group['target_ret'].values)
            daily_ics.append(d_ic)

    daily_ics = np.array(daily_ics)
    daily_ics = daily_ics[~np.isnan(daily_ics)]

    icir = daily_ics.mean() / (daily_ics.std() + 1e-10) if len(daily_ics) > 0 else 0

    # Direction accuracy
    pred_direction = (df_test[pred_col] > 0).astype(int)
    actual_direction = (df_test['target_ret'] > 0).astype(int)
    direction_acc = accuracy_score(actual_direction, pred_direction)

    # Long-short return simulation
    # Buy top quantile, sell bottom quantile each "day"
    daily_returns = []
    for _, group in df_test.groupby('date'):
        if len(group) < 10:
            continue
        group = group.sort_values(pred_col, ascending=False)
        n = max(len(group) // 5, 1)  # Top/bottom 20%
        long_ret = group.head(n)['target_ret'].mean()
        short_ret = group.tail(n)['target_ret'].mean()
        ls_ret = long_ret - short_ret
        daily_returns.append(ls_ret)

    daily_returns = np.array(daily_returns)
    ann_factor = np.sqrt(365)  # Crypto 365 days/year

    sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-10)) * ann_factor if len(daily_returns) > 0 else 0
    ann_return = daily_returns.mean() * 365 if len(daily_returns) > 0 else 0

    cumulative = np.cumprod(1 + daily_returns)
    max_dd = np.min(cumulative / np.maximum.accumulate(cumulative) - 1) if len(cumulative) > 0 else 0

    metrics = {
        'IC': round(ic, 4),
        'Rank_IC': round(rank_ic, 4),
        'ICIR': round(icir, 4),
        'Daily_IC_mean': round(daily_ics.mean(), 4) if len(daily_ics) > 0 else 0,
        'Daily_IC_std': round(daily_ics.std(), 4) if len(daily_ics) > 0 else 0,
        'Direction_Accuracy': round(direction_acc, 4),
        'LS_Sharpe': round(sharpe, 4),
        'LS_Ann_Return': round(ann_return, 4),
        'LS_Max_Drawdown': round(max_dd, 4),
        'LS_Daily_Count': len(daily_returns),
        'N_test_samples': len(df_test),
    }
    return metrics


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load features
    feat_path = os.path.join(DATA_FEAT_DIR, f'crypto_features_{TIMEFRAME}.parquet')
    if not os.path.exists(feat_path):
        print(f"❌ Feature file not found: {feat_path}")
        print("   Run build_features.py first!")
        return

    print("📊 Loading features...")
    df = pd.read_parquet(feat_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    feat_cols = get_feature_cols(df)
    print(f"   Features: {len(feat_cols)}")

    # === Walk-forward split ===
    train = df[df['timestamp'] < TRAIN_END].copy()
    val = df[(df['timestamp'] >= TRAIN_END) & (df['timestamp'] < VAL_END)].copy()
    test = df[df['timestamp'] >= VAL_END].copy()

    print(f"\n📅 Split sizes:")
    print(f"   Train: {len(train):,} rows ({train['timestamp'].min()} → {train['timestamp'].max()})")
    print(f"   Val:   {len(val):,} rows ({val['timestamp'].min()} → {val['timestamp'].max()})")
    print(f"   Test:  {len(test):,} rows ({test['timestamp'].min()} → {test['timestamp'].max()})")

    if len(test) == 0:
        print("⚠️  Test set is empty! Adjusting splits...")
        # Fallback: use last 20% as test
        n = len(df)
        train = df.iloc[:int(n*0.6)].copy()
        val = df.iloc[int(n*0.6):int(n*0.8)].copy()
        test = df.iloc[int(n*0.8):].copy()
        print(f"   Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    X_train, y_train = train[feat_cols], train['target_ret']
    X_val, y_val = val[feat_cols], val['target_ret']
    X_test, y_test = test[feat_cols], test['target_ret']

    # === Train Regression Model ===
    print("\n🚀 Training LightGBM (regression on forward return)...")
    model = train_lightgbm(X_train, y_train, X_val, y_val, task='regression')

    # === Predict & Evaluate ===
    test['pred_ret'] = model.predict(X_test)

    print("\n📈 === TEST SET RESULTS (Out-of-Sample) ===")
    metrics = evaluate_regression(test)
    for k, v in metrics.items():
        print(f"   {k}: {v}")

    # === Feature Importance ===
    importance = pd.DataFrame({
        'feature': feat_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n🏆 Top 20 Features:")
    for _, row in importance.head(20).iterrows():
        print(f"   {row['feature']:30s} {row['importance']:.0f}")

    # === Save Results ===
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['model'] = 'LightGBM_baseline'
    metrics['features'] = len(feat_cols)

    results_path = os.path.join(RESULTS_DIR, 'baseline_results.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    importance.to_csv(os.path.join(RESULTS_DIR, 'feature_importance.csv'), index=False)
    test[['timestamp', 'symbol', 'target_ret', 'pred_ret']].to_parquet(
        os.path.join(RESULTS_DIR, 'test_predictions.parquet'), index=False
    )

    print(f"\n✅ Results saved to {RESULTS_DIR}/")

    # === Quick verdict ===
    print("\n" + "=" * 60)
    if metrics['IC'] > 0.03 and metrics['LS_Sharpe'] > 1.0:
        print("🟢 PROMISING! IC and Sharpe look good for a baseline.")
        print("   Next: try HIST/MASTER transformer models for improvement.")
    elif metrics['IC'] > 0.02:
        print("🟡 DECENT signal detected. Room for improvement.")
        print("   Next: add more features, try ensemble, HPO.")
    else:
        print("🟠 Weak signal. Normal for a first attempt on crypto.")
        print("   Next: try different targets, add alt-data, check data quality.")
    print("=" * 60)


if __name__ == '__main__':
    main()
