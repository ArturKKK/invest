#!/usr/bin/env python3
"""
Cluster runner: train LightGBM + backtest in one go.
Usage:
    python run_pipeline.py                          # default paths
    python run_pipeline.py --data /path/to/data     # custom data dir
"""

import sys
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run full training + backtest pipeline')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to directory containing crypto_features_1h.parquet')
    parser.add_argument('--results', type=str, default=None,
                        help='Directory to save results')
    args = parser.parse_args()

    # Set up paths
    project_root = os.path.dirname(os.path.abspath(__file__))

    if args.data:
        data_dir = args.data
    else:
        data_dir = os.path.join(project_root, 'data', 'features')

    if args.results:
        results_dir = args.results
    else:
        results_dir = os.path.join(project_root, 'results')

    os.makedirs(results_dir, exist_ok=True)

    feat_path = os.path.join(data_dir, 'crypto_features_1h.parquet')
    if not os.path.exists(feat_path):
        print(f"❌ Feature file not found: {feat_path}")
        print(f"   Place crypto_features_1h.parquet in: {data_dir}")
        sys.exit(1)

    print(f"📂 Data dir:    {data_dir}")
    print(f"📂 Results dir: {results_dir}")
    print(f"📊 Feature file: {feat_path} ({os.path.getsize(feat_path) / 1024 / 1024:.0f} MB)")

    # ========================================
    # Step 1: Train LightGBM
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 1: Training LightGBM baseline")
    print("=" * 60)

    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score
    from scipy.stats import spearmanr
    import json
    from datetime import datetime

    TRAIN_END = '2024-07-01'
    VAL_END = '2025-07-01'

    # Load data
    print("📊 Loading features...")
    df = pd.read_parquet(feat_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # Get feature columns
    exclude = {'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
               'target_ret', 'target_cls', 'hour', 'day_of_week'}
    feat_cols = [c for c in df.columns if c not in exclude]
    print(f"   Features: {len(feat_cols)}")

    # Walk-forward split
    train = df[df['timestamp'] < TRAIN_END].copy()
    val = df[(df['timestamp'] >= TRAIN_END) & (df['timestamp'] < VAL_END)].copy()
    test = df[df['timestamp'] >= VAL_END].copy()

    print(f"\n📅 Split sizes:")
    print(f"   Train: {len(train):,} rows ({train['timestamp'].min()} → {train['timestamp'].max()})")
    print(f"   Val:   {len(val):,} rows ({val['timestamp'].min()} → {val['timestamp'].max()})")
    print(f"   Test:  {len(test):,} rows ({test['timestamp'].min()} → {test['timestamp'].max()})")

    if len(test) == 0:
        print("⚠️  Test set is empty! Adjusting splits...")
        n = len(df)
        train = df.iloc[:int(n * 0.6)].copy()
        val = df.iloc[int(n * 0.6):int(n * 0.8)].copy()
        test = df.iloc[int(n * 0.8):].copy()
        print(f"   Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    X_train, y_train = train[feat_cols], train['target_ret']
    X_val, y_val = val[feat_cols], val['target_ret']
    X_test = test[feat_cols]

    # Train
    print("\n🚀 Training LightGBM (regression on forward return)...")
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

    # Predict
    test['pred_ret'] = model.predict(X_test)

    # ========================================
    # Evaluate
    # ========================================
    def compute_ic(preds, actuals):
        mask = ~(np.isnan(preds) | np.isnan(actuals))
        if mask.sum() < 10:
            return 0.0
        return np.corrcoef(preds[mask], actuals[mask])[0, 1]

    def compute_rank_ic(preds, actuals):
        mask = ~(np.isnan(preds) | np.isnan(actuals))
        if mask.sum() < 10:
            return 0.0
        corr, _ = spearmanr(preds[mask], actuals[mask])
        return corr

    ic = compute_ic(test['pred_ret'].values, test['target_ret'].values)
    rank_ic = compute_rank_ic(test['pred_ret'].values, test['target_ret'].values)

    test_eval = test.copy()
    test_eval['date'] = test_eval['timestamp'].dt.date
    daily_ics = []
    for _, group in test_eval.groupby('date'):
        if len(group) >= 5:
            d_ic = compute_ic(group['pred_ret'].values, group['target_ret'].values)
            daily_ics.append(d_ic)
    daily_ics = np.array(daily_ics)
    daily_ics = daily_ics[~np.isnan(daily_ics)]
    icir = daily_ics.mean() / (daily_ics.std() + 1e-10) if len(daily_ics) > 0 else 0

    pred_dir = (test['pred_ret'] > 0).astype(int)
    actual_dir = (test['target_ret'] > 0).astype(int)
    direction_acc = accuracy_score(actual_dir, pred_dir)

    # Long-short returns
    daily_returns = []
    for _, group in test_eval.groupby('date'):
        if len(group) < 10:
            continue
        group = group.sort_values('pred_ret', ascending=False)
        n = max(len(group) // 5, 1)
        long_ret = group.head(n)['target_ret'].mean()
        short_ret = group.tail(n)['target_ret'].mean()
        daily_returns.append(long_ret - short_ret)

    daily_returns = np.array(daily_returns)
    ann_factor = np.sqrt(365)
    sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-10)) * ann_factor if len(daily_returns) > 0 else 0
    ann_return = daily_returns.mean() * 365 if len(daily_returns) > 0 else 0
    cumulative = np.cumprod(1 + daily_returns)
    max_dd = np.min(cumulative / np.maximum.accumulate(cumulative) - 1) if len(cumulative) > 0 else 0

    metrics = {
        'IC': round(ic, 4),
        'Rank_IC': round(rank_ic, 4),
        'ICIR': round(icir, 4),
        'Daily_IC_mean': round(float(daily_ics.mean()), 4) if len(daily_ics) > 0 else 0,
        'Daily_IC_std': round(float(daily_ics.std()), 4) if len(daily_ics) > 0 else 0,
        'Direction_Accuracy': round(direction_acc, 4),
        'LS_Sharpe': round(sharpe, 4),
        'LS_Ann_Return': round(ann_return, 4),
        'LS_Max_Drawdown': round(max_dd, 4),
        'N_test_samples': len(test),
    }

    print("\n📈 === TEST SET RESULTS (Out-of-Sample) ===")
    for k, v in metrics.items():
        print(f"   {k}: {v}")

    # Feature importance
    importance = pd.DataFrame({
        'feature': feat_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n🏆 Top 20 Features:")
    for _, row in importance.head(20).iterrows():
        print(f"   {row['feature']:30s} {row['importance']:.0f}")

    # Save
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['model'] = 'LightGBM_baseline'
    metrics['features'] = len(feat_cols)

    with open(os.path.join(results_dir, 'baseline_results.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    importance.to_csv(os.path.join(results_dir, 'feature_importance.csv'), index=False)
    test[['timestamp', 'symbol', 'target_ret', 'pred_ret']].to_parquet(
        os.path.join(results_dir, 'test_predictions.parquet'), index=False
    )

    # ========================================
    # Step 2: Backtest
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 2: Running backtest")
    print("=" * 60)

    pred_df = test[['timestamp', 'symbol', 'target_ret', 'pred_ret']].copy()
    pred_df = pred_df.sort_values('timestamp')

    top_k = 5
    commission = 0.001  # 0.1% OKX taker fee
    initial_capital = 1000.0

    portfolio_returns = []
    dates = []
    trades_count = 0

    for ts, group in pred_df.groupby('timestamp'):
        if len(group) < top_k * 2:
            continue
        group = group.sort_values('pred_ret', ascending=False)
        long_coins = group.head(top_k)
        long_ret = long_coins['target_ret'].mean()
        net_ret = long_ret - 2 * commission / 4
        portfolio_returns.append(net_ret)
        dates.append(ts)
        trades_count += top_k

    if portfolio_returns:
        returns = np.array(portfolio_returns)
        dates_arr = pd.to_datetime(dates)
        equity = initial_capital * np.cumprod(1 + returns)

        total_return = equity[-1] / initial_capital - 1
        n_days = (dates_arr[-1] - dates_arr[0]).total_seconds() / 86400
        bt_ann_return = (1 + total_return) ** (365 / max(n_days, 1)) - 1

        daily_df = pd.DataFrame({'date': dates_arr, 'ret': returns})
        daily_df['date'] = daily_df['date'].dt.date
        daily_rets = daily_df.groupby('date')['ret'].sum()
        bt_sharpe = (daily_rets.mean() / (daily_rets.std() + 1e-10)) * np.sqrt(365)

        cum_max = np.maximum.accumulate(equity)
        drawdown = equity / cum_max - 1
        bt_max_dd = drawdown.min()
        win_rate = (returns > 0).mean()
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / (losses + 1e-10)

        print(f"   Period:         {dates_arr[0].date()} → {dates_arr[-1].date()} ({n_days:.0f} days)")
        print(f"   Initial:        ${initial_capital:,.2f}")
        print(f"   Final:          ${equity[-1]:,.2f}")
        print(f"   Total Return:   {total_return*100:+.2f}%")
        print(f"   Ann. Return:    {bt_ann_return*100:+.2f}%")
        print(f"   Sharpe Ratio:   {bt_sharpe:.2f}")
        print(f"   Max Drawdown:   {bt_max_dd*100:.2f}%")
        print(f"   Win Rate:       {win_rate*100:.1f}%")
        print(f"   Profit Factor:  {profit_factor:.2f}")
        print(f"   Total Trades:   {trades_count:,}")

        bt_results = {
            'period_start': str(dates_arr[0].date()),
            'period_end': str(dates_arr[-1].date()),
            'initial_capital': initial_capital,
            'final_capital': round(float(equity[-1]), 2),
            'total_return_pct': round(total_return * 100, 2),
            'ann_return_pct': round(bt_ann_return * 100, 2),
            'sharpe_ratio': round(float(bt_sharpe), 2),
            'max_drawdown_pct': round(float(bt_max_dd) * 100, 2),
            'win_rate_pct': round(float(win_rate) * 100, 1),
            'profit_factor': round(float(profit_factor), 2),
            'total_trades': trades_count,
            'top_k': top_k,
            'commission_pct': commission * 100,
        }
        with open(os.path.join(results_dir, 'backtest_results.json'), 'w') as f:
            json.dump(bt_results, f, indent=2)

        equity_df = pd.DataFrame({'timestamp': dates_arr, 'equity': equity, 'return': returns})
        equity_df.to_parquet(os.path.join(results_dir, 'equity_curve.parquet'), index=False)
    else:
        print("❌ No trades generated!")

    # ========================================
    # Verdict
    # ========================================
    print("\n" + "=" * 60)
    if metrics['IC'] > 0.03 and metrics['LS_Sharpe'] > 1.0:
        print("🟢 PROMISING! IC and Sharpe look good for baseline.")
    elif metrics['IC'] > 0.02:
        print("🟡 DECENT signal. Room for improvement with HIST/MASTER.")
    else:
        print("🟠 Weak signal. Normal for first attempt on crypto.")
    print("=" * 60)

    print(f"\n✅ All results saved to {results_dir}/")
    print("   - baseline_results.json")
    print("   - feature_importance.csv")
    print("   - test_predictions.parquet")
    print("   - backtest_results.json")
    print("   - equity_curve.parquet")


if __name__ == '__main__':
    main()
