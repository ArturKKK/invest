#!/usr/bin/env python3
"""
Quick comparison of SKIP vs CLOSE risk-off modes without full training.
Uses the simulate logic directly to verify correctness.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Mock data: create synthetic merged predictions and regime data
np.random.seed(42)

# Create 2000 random rebalance periods with symbols
n_periods = 2000
n_symbols = 35
rebal_timestamps = pd.date_range('2024-06-01', periods=n_periods, freq='12h')

data = []
for ts in rebal_timestamps:
    for sym in [f'SYM{i:02d}' for i in range(n_symbols)]:
        data.append({
            'timestamp': ts,
            'symbol': sym,
            'pred': np.random.uniform(-0.5, 0.5),
            'fwd_ret': np.random.normal(0.002, 0.02)
        })

merged = pd.DataFrame(data)

# Create regime data with varying trend strength
regime_data = []
for ts in rebal_timestamps:
    trend_str = np.random.uniform(0.5, 1.2)
    regime_data.append({'timestamp': ts, 'trend_strength': trend_str})

regime_df = pd.DataFrame(regime_data).set_index('timestamp')

print("=" * 80)
print("  RISK-OFF MODE COMPARISON: SKIP vs CLOSE")
print("=" * 80)

def simulate_close(merged, regime_df, trend_cutoff=0.95):
    """CLOSE mode: pay commission when closing positions on risk-off"""
    all_rets = []
    prev_longs = set()
    prev_shorts = set()
    rebal_timestamps = sorted(merged['timestamp'].unique())
    grouped = {ts: grp for ts, grp in merged.groupby('timestamp')}
    
    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        
        trend_str = regime_df.loc[ts, 'trend_strength']
        
        # CLOSE mode: if risk-off, close positions and pay commission
        if trend_str > trend_cutoff:
            if prev_longs or prev_shorts:
                close_cost = 0.0003 * (len(prev_longs) + len(prev_shorts))
                all_rets.append({
                    'timestamp': ts, 'gross_ret': 0.0, 'net_ret': -close_cost,
                    'cost': close_cost, 'n_long': 0, 'n_short': 0, 'turnover': len(prev_longs) + len(prev_shorts)
                })
            else:
                all_rets.append({
                    'timestamp': ts, 'gross_ret': 0.0, 'net_ret': 0.0,
                    'cost': 0.0, 'n_long': 0, 'n_short': 0, 'turnover': 0
                })
            prev_longs = set()
            prev_shorts = set()
            continue
        
        grp = grouped[ts].copy()
        grp = grp.sort_values('pred', ascending=False)
        
        longs = set(grp.head(6)['symbol'].values)
        shorts = set(grp.tail(3)['symbol'].values)
        
        long_rets = grp.loc[grp['symbol'].isin(longs), 'fwd_ret'].mean() if len(longs) > 0 else 0
        short_rets = -grp.loc[grp['symbol'].isin(shorts), 'fwd_ret'].mean() if len(shorts) > 0 else 0
        gross_ret = 0.5 * long_rets + 0.5 * short_rets
        
        turnover = len((longs | shorts) ^ (prev_longs | prev_shorts))
        cost = turnover * 0.0003
        net_ret = gross_ret - cost
        
        all_rets.append({
            'timestamp': ts, 'gross_ret': gross_ret, 'net_ret': net_ret,
            'cost': cost, 'n_long': len(longs), 'n_short': len(shorts), 'turnover': turnover
        })
        
        prev_longs = longs
        prev_shorts = shorts
    
    return pd.DataFrame(all_rets)

def simulate_skip(merged, regime_df, trend_cutoff=0.95):
    """SKIP mode: just skip trading during risk-off, don't close positions"""
    all_rets = []
    prev_longs = set()
    prev_shorts = set()
    rebal_timestamps = sorted(merged['timestamp'].unique())
    grouped = {ts: grp for ts, grp in merged.groupby('timestamp')}
    
    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        
        trend_str = regime_df.loc[ts, 'trend_strength']
        
        # SKIP mode: if risk-off, just skip (don't open new positions)
        if trend_str > trend_cutoff:
            continue
        
        grp = grouped[ts].copy()
        grp = grp.sort_values('pred', ascending=False)
        
        longs = set(grp.head(6)['symbol'].values)
        shorts = set(grp.tail(3)['symbol'].values)
        
        long_rets = grp.loc[grp['symbol'].isin(longs), 'fwd_ret'].mean() if len(longs) > 0 else 0
        short_rets = -grp.loc[grp['symbol'].isin(shorts), 'fwd_ret'].mean() if len(shorts) > 0 else 0
        gross_ret = 0.5 * long_rets + 0.5 * short_rets
        
        turnover = len((longs | shorts) ^ (prev_longs | prev_shorts))
        cost = turnover * 0.0003
        net_ret = gross_ret - cost
        
        all_rets.append({
            'timestamp': ts, 'gross_ret': gross_ret, 'net_ret': net_ret,
            'cost': cost, 'n_long': len(longs), 'n_short': len(shorts), 'turnover': turnover
        })
        
        prev_longs = longs
        prev_shorts = shorts
    
    return pd.DataFrame(all_rets)

def compute_sharpe(rets):
    if len(rets) == 0 or rets.std() == 0:
        return 0
    return (rets.sum() / rets.std()) / np.sqrt(len(rets)) * np.sqrt(730)

results = []

for cutoff in [0.8, 0.85, 0.9, 0.95, 1.0]:
    print(f"\ntestcutoff={cutoff:.2f}")
    
    # CLOSE mode
    port_close = simulate_close(merged, regime_df, cutoff)
    sh_close = compute_sharpe(port_close['net_ret'].values)
    ret_close = port_close['net_ret'].sum() * 100
    results.append({'mode': 'CLOSE', 'cutoff': cutoff, 'sharpe': sh_close, 'return': ret_close, 'periods': len(port_close)})
    print(f"  CLOSE: Sharpe={sh_close:.4f}, return={ret_close:.2f}%, periods={len(port_close)}")
    
    # SKIP mode
    port_skip = simulate_skip(merged, regime_df, cutoff)
    sh_skip = compute_sharpe(port_skip['net_ret'].values)
    ret_skip = port_skip['net_ret'].sum() * 100
    results.append({'mode': 'SKIP', 'cutoff': cutoff, 'sharpe': sh_skip, 'return': ret_skip, 'periods': len(port_skip)})
    print(f"  SKIP:  Sharpe={sh_skip:.4f}, return={ret_skip:.2f}%, periods={len(port_skip)}")
    print(f"  Δ: {sh_skip - sh_close:+.4f}")

print("\n" + "=" * 80)
df_res = pd.DataFrame(results).sort_values('sharpe', ascending=False)
print(df_res.to_string(index=False))
print("=" * 80)
print("\n✅ Comparison complete (using synthetic data for logic verification)")
