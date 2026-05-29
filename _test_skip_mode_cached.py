#!/usr/bin/env python3
"""
Test SKIP vs CLOSE modes using cached r128 canonical predictions
"""
import pandas as pd
import numpy as np

print("Loading cached predictions...")
merged = pd.read_parquet('cache/r128_canonical_preds.parquet')
regime_df = pd.read_parquet('cache/r128_canonical_regime.parquet')
regime_df = regime_df.set_index('timestamp')

print(f"  Merged: {len(merged):,} rows, {merged['symbol'].nunique()} symbols")
print(f"  Regime: {len(regime_df):,} rows")

def simulate_close(merged, regime_df, n_long, n_short, trend_cutoff):
    all_rets = []
    prev_longs = set()
    prev_shorts = set()
    
    timestamps_sorted = sorted(merged['timestamp'].unique())
    grouped = {ts: grp for ts, grp in merged.groupby('timestamp')}
    rebal_timestamps = timestamps_sorted[::12]
    
    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        
        trend_str = regime_df.loc[ts, 'trend_strength']
        
        if trend_str > trend_cutoff:
            if prev_longs or prev_shorts:
                close_cost = 0.0003 * (len(prev_longs) + len(prev_shorts))
                all_rets.append(-close_cost)
            else:
                all_rets.append(0.0)
            prev_longs, prev_shorts = set(), set()
            continue
        
        grp = grouped[ts].copy()
        n = len(grp)
        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl == 0 and ns == 0:
            continue
        
        grp = grp.sort_values('pred', ascending=False).reset_index(drop=True)
        longs = set(grp.head(int(nl))['symbol'].values)
        shorts = set(grp.tail(int(ns))['symbol'].values)
        
        long_rets = grp.loc[grp['symbol'].isin(longs), 'fwd_ret'].mean() if len(longs) > 0 else 0
        short_rets = -grp.loc[grp['symbol'].isin(shorts), 'fwd_ret'].mean() if len(shorts) > 0 else 0
        gross_ret = 0.5 * long_rets + 0.5 * short_rets
        
        turnover = len((longs | shorts) ^ (prev_longs | prev_shorts))
        cost = turnover * 0.0003
        net_ret = gross_ret - cost
        
        all_rets.append(net_ret)
        prev_longs, prev_shorts = longs, shorts
    
    return np.array(all_rets) if all_rets else np.array([])

def simulate_skip(merged, regime_df, n_long, n_short, trend_cutoff):
    all_rets = []
    prev_longs = set()
    prev_shorts = set()
    
    timestamps_sorted = sorted(merged['timestamp'].unique())
    grouped = {ts: grp for ts, grp in merged.groupby('timestamp')}
    rebal_timestamps = timestamps_sorted[::12]
    
    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        
        trend_str = regime_df.loc[ts, 'trend_strength']
        
        if trend_str > trend_cutoff:
            continue
        
        grp = grouped[ts].copy()
        n = len(grp)
        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl == 0 and ns == 0:
            continue
        
        grp = grp.sort_values('pred', ascending=False).reset_index(drop=True)
        longs = set(grp.head(int(nl))['symbol'].values)
        shorts = set(grp.tail(int(ns))['symbol'].values)
        
        long_rets = grp.loc[grp['symbol'].isin(longs), 'fwd_ret'].mean() if len(longs) > 0 else 0
        short_rets = -grp.loc[grp['symbol'].isin(shorts), 'fwd_ret'].mean() if len(shorts) > 0 else 0
        gross_ret = 0.5 * long_rets + 0.5 * short_rets
        
        turnover = len((longs | shorts) ^ (prev_longs | prev_shorts))
        cost = turnover * 0.0003
        net_ret = gross_ret - cost
        
        all_rets.append(net_ret)
        prev_longs, prev_shorts = longs, shorts
    
    return np.array(all_rets) if all_rets else np.array([])

def compute_sharpe(rets):
    if len(rets) == 0 or np.std(rets) == 0:
        return 0
    return (np.sum(rets) / np.std(rets)) / np.sqrt(len(rets)) * np.sqrt(730)

print("\n" + "=" * 90)
print("  SKIP vs CLOSE MODE COMPARISON (cached r128 predictions)")
print("=" * 90)

results = []

for cutoff in [0.8, 0.85, 0.9, 0.95, 1.0]:
    print(f"\ntrend_cutoff={cutoff:.2f}")
    
    rets_close = simulate_close(merged, regime_df, 6, 3, cutoff)
    sh_close = compute_sharpe(rets_close)
    ret_close = np.sum(rets_close) * 100
    results.append({'mode': 'CLOSE', 'cutoff': cutoff, 'sharpe': sh_close, 'return': ret_close, 'periods': len(rets_close)})
    print(f"  CLOSE: Sharpe={sh_close:.3f}, return={ret_close:.1f}%, periods={len(rets_close)}")
    
    rets_skip = simulate_skip(merged, regime_df, 6, 3, cutoff)
    sh_skip = compute_sharpe(rets_skip)
    ret_skip = np.sum(rets_skip) * 100
    results.append({'mode': 'SKIP', 'cutoff': cutoff, 'sharpe': sh_skip, 'return': ret_skip, 'periods': len(rets_skip)})
    print(f"  SKIP:  Sharpe={sh_skip:.3f}, return={ret_skip:.1f}%, periods={len(rets_skip)}")
    
    delta = sh_skip - sh_close
    sign = "✅" if delta > 0 else "❌"
    print(f"  {sign} SKIP advantage: {delta:+.3f}")

df_res = pd.DataFrame(results).sort_values('sharpe', ascending=False)
print("\n" + "=" * 90)
print("RESULTS (sorted by Sharpe)")
print("=" * 90)
print(df_res[['mode', 'cutoff', 'sharpe', 'return', 'periods']].to_string(index=False))

best_overall = df_res.iloc[0]
skip_res = df_res[df_res['mode'] == 'SKIP'].copy()
close_res = df_res[df_res['mode'] == 'CLOSE'].copy()

if len(skip_res) > 0 and len(close_res) > 0:
    skip_best = skip_res.iloc[0]
    close_best = close_res.iloc[0]
    
    print(f"\n{'='*90}")
    print(f"✅ BEST OVERALL: {best_overall['mode']} cutoff={best_overall['cutoff']:.2f} → Sharpe {best_overall['sharpe']:.3f}")
    print(f"✅ BEST SKIP:    cutoff={skip_best['cutoff']:.2f} → Sharpe {skip_best['sharpe']:.3f}")
    print(f"   BEST CLOSE:   cutoff={close_best['cutoff']:.2f} → Sharpe {close_best['sharpe']:.3f}")
    print(f"   SKIP advantage: {skip_best['sharpe'] - close_best['sharpe']:+.3f}")

