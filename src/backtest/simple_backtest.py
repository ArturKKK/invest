"""
Simple backtest: simulate trading based on model predictions.
Long-short strategy on top/bottom quantiles.
"""

import pandas as pd
import numpy as np
import os
import json

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')


def backtest_long_short(
    predictions_path: str,
    top_k: int = 5,
    commission: float = 0.001,  # 0.1% OKX taker fee
    initial_capital: float = 1000.0,
):
    """
    Simple long-only backtest:
    - Each period, buy top_k predicted coins
    - Equal weight allocation
    - Rebalance every 4 hours (prediction horizon)
    """
    df = pd.read_parquet(predictions_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp')

    # Group by timestamp (each timestamp = one decision point)
    portfolio_returns = []
    dates = []
    trades_count = 0

    for ts, group in df.groupby('timestamp'):
        if len(group) < top_k * 2:
            continue

        # Rank by predicted return
        group = group.sort_values('pred_ret', ascending=False)

        # Long top K
        long_coins = group.head(top_k)
        long_ret = long_coins['target_ret'].mean()

        # Account for commission (buy + sell = 2 * commission)
        net_ret = long_ret - 2 * commission / 4  # Amortize over 4h holding

        portfolio_returns.append(net_ret)
        dates.append(ts)
        trades_count += top_k

    if not portfolio_returns:
        print("❌ No trades generated!")
        return

    returns = np.array(portfolio_returns)
    dates = pd.to_datetime(dates)

    # Compute equity curve
    equity = initial_capital * np.cumprod(1 + returns)

    # Metrics
    total_return = equity[-1] / initial_capital - 1
    n_days = (dates[-1] - dates[0]).total_seconds() / 86400
    ann_return = (1 + total_return) ** (365 / max(n_days, 1)) - 1

    # Daily aggregation for Sharpe
    daily_df = pd.DataFrame({'date': dates, 'ret': returns})
    daily_df['date'] = daily_df['date'].dt.date
    daily_returns = daily_df.groupby('date')['ret'].sum()

    sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-10)) * np.sqrt(365)

    # Drawdown
    cum_max = np.maximum.accumulate(equity)
    drawdown = equity / cum_max - 1
    max_dd = drawdown.min()

    # Win rate
    win_rate = (returns > 0).mean()

    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = gains / (losses + 1e-10)

    print("=" * 60)
    print("📊 BACKTEST RESULTS (Long-Only, Top-K)")
    print("=" * 60)
    print(f"   Period: {dates[0].date()} → {dates[-1].date()} ({n_days:.0f} days)")
    print(f"   Initial Capital: ${initial_capital:,.2f}")
    print(f"   Final Capital:   ${equity[-1]:,.2f}")
    print(f"   Total Return:    {total_return*100:+.2f}%")
    print(f"   Ann. Return:     {ann_return*100:+.2f}%")
    print(f"   Sharpe Ratio:    {sharpe:.2f}")
    print(f"   Max Drawdown:    {max_dd*100:.2f}%")
    print(f"   Win Rate:        {win_rate*100:.1f}%")
    print(f"   Profit Factor:   {profit_factor:.2f}")
    print(f"   Total Trades:    {trades_count:,}")
    print(f"   Commission:      {commission*100:.2f}% per trade")
    print(f"   Top K:           {top_k}")
    print("=" * 60)

    # Verdict
    print()
    if sharpe > 2.0 and max_dd > -0.15:
        print("🟢 EXCELLENT — Ready for paper trading!")
    elif sharpe > 1.0 and max_dd > -0.25:
        print("🟡 GOOD — Worth optimizing further, then paper trade.")
    elif sharpe > 0.5:
        print("🟠 MEDIOCRE — Signal exists but weak. Need better model/features.")
    else:
        print("🔴 POOR — Model doesn't beat random. Rethink approach.")

    # Save backtest results
    results = {
        'period_start': str(dates[0].date()),
        'period_end': str(dates[-1].date()),
        'initial_capital': initial_capital,
        'final_capital': round(equity[-1], 2),
        'total_return_pct': round(total_return * 100, 2),
        'ann_return_pct': round(ann_return * 100, 2),
        'sharpe_ratio': round(sharpe, 2),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'win_rate_pct': round(win_rate * 100, 1),
        'profit_factor': round(profit_factor, 2),
        'total_trades': trades_count,
        'top_k': top_k,
        'commission_pct': commission * 100,
    }

    bt_path = os.path.join(RESULTS_DIR, 'backtest_results.json')
    with open(bt_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save equity curve
    equity_df = pd.DataFrame({'timestamp': dates, 'equity': equity, 'return': returns})
    equity_df.to_parquet(os.path.join(RESULTS_DIR, 'equity_curve.parquet'), index=False)

    print(f"\n💾 Results saved to {RESULTS_DIR}/")
    return results


def main():
    pred_path = os.path.join(RESULTS_DIR, 'test_predictions.parquet')
    if not os.path.exists(pred_path):
        print(f"❌ Predictions not found: {pred_path}")
        print("   Run baseline_lgbm.py first!")
        return

    backtest_long_short(pred_path, top_k=5, commission=0.001, initial_capital=1000)


if __name__ == '__main__':
    main()
