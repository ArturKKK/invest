#!/usr/bin/env python3
"""
Risk Optimization Study — Find optimal risk parameters for live trading.

Uses ensemble backtest predictions (HIST v1 + LGB v5) to find the best
combination of risk controls that maximizes risk-adjusted returns.

Sweeps:
  - Vol target: 0.3% to 3% per period
  - DD stop threshold: -10% to -40%
  - DD resume threshold: -3% to -15%
  - Quintile vs Top/Bot-K construction
  - Rebalance skip filter (confidence threshold)
  - Kelly fraction (position sizing)

Usage:
  python run_risk_study.py
  python run_risk_study.py --predictions results_ensemble_v2/ensemble_v2_predictions.parquet
"""

import os
import sys
import json
import argparse
import warnings
from itertools import product

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

HORIZON = 4
PERIODS_PER_DAY = 24 // HORIZON
PERIODS_PER_YEAR = PERIODS_PER_DAY * 365

# Cost model
COST_1SIDE = 0.0003 * 0.25 + 0.0001 * 0.25 + 0.00005 * (HORIZON / 8)  # ~0.000125
COST_LS = COST_1SIDE * 2  # both sides


def load_ensemble(path):
    """Load ensemble predictions with actual returns."""
    df = pd.read_parquet(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    target_col = f'target_ret_{HORIZON}h'
    if target_col not in df.columns:
        print(f"❌ No {target_col} in predictions")
        sys.exit(1)

    # Find prediction columns
    pred_cols = [c for c in df.columns if c.startswith('pred_')]
    print(f"   Predictions: {pred_cols}")
    print(f"   Rows: {len(df):,}, Timestamps: {df['timestamp'].nunique()}")

    return df, target_col, pred_cols


def build_ls_returns(df, pred_col, target_col, n_long=10, n_short=10, confidence=0.0):
    """
    Build Long-Short return series from predictions.

    confidence: min absolute z-score to trade. 0 = always trade.
    n_long/n_short: number of coins to go long/short.
    """
    timestamps = sorted(df['timestamp'].unique())
    ls_rets = []

    for ts in timestamps:
        grp = df[df['timestamp'] == ts].copy()
        if len(grp) < 20:
            ls_rets.append(0.0)
            continue

        p = grp[pred_col].values
        a = grp[target_col].values
        valid = ~(np.isnan(p) | np.isnan(a))
        if valid.sum() < 20:
            ls_rets.append(0.0)
            continue

        grp_valid = grp[valid].copy()
        pv = grp_valid[pred_col].values
        av = grp_valid[target_col].values

        # Confidence filter: signal strength
        signal_spread = pv.max() - pv.min()
        if signal_spread < 1e-10:
            ls_rets.append(0.0)
            continue

        signal_z = (pv - pv.mean()) / (pv.std() + 1e-10)
        max_z = np.max(np.abs(signal_z))
        if max_z < confidence:
            ls_rets.append(0.0)  # Skip weak signals
            continue

        order = np.argsort(-pv)
        sorted_a = av[order]
        nl = min(n_long, len(sorted_a) // 3)
        ns = min(n_short, len(sorted_a) // 3)

        long_ret = sorted_a[:nl].mean()
        short_ret = sorted_a[-ns:].mean()
        ls_rets.append(long_ret - short_ret)

    return np.array(ls_rets)


def apply_risk_overlay(ls_raw, vol_target, dd_stop, dd_resume,
                       kelly_frac, vol_lookback=48):
    """
    Apply risk overlay to raw LS returns.

    Returns net returns with:
    - Cost deduction
    - Vol targeting (with kelly scaling)
    - DD circuit breaker
    """
    n = len(ls_raw)
    net_rets = np.zeros(n)
    cum_eq = 1.0
    peak = 1.0
    active = True

    for i in range(n):
        # Vol targeting
        lookback = ls_raw[max(0, i - vol_lookback):i]
        if len(lookback) >= 6:
            vol = np.std(lookback) + 1e-10
            scale = np.clip(vol_target / vol, 0.1, 3.0) * kelly_frac
        else:
            scale = kelly_frac

        # DD circuit breaker
        if not active:
            net_rets[i] = 0.0
            # Check if we should resume
            # Resume based on peak tracking (even while off)
            # Simulate: equity stays flat while stopped
            dd = cum_eq / peak - 1
            if dd > dd_resume:
                active = True
            continue

        # Apply scaled return minus costs
        raw = ls_raw[i]
        cost = COST_LS * scale  # costs scale with position size
        net_rets[i] = raw * scale - cost

        cum_eq *= (1 + net_rets[i])
        peak = max(peak, cum_eq)
        dd = cum_eq / peak - 1

        if dd < dd_stop:
            active = False

    return net_rets


def compute_metrics(rets, label=""):
    """Compute risk-adjusted metrics."""
    if len(rets) == 0 or np.std(rets) < 1e-12:
        return None

    rets_clipped = np.clip(rets, -0.99, None)
    cum = np.cumprod(1 + rets_clipped)
    running_max = np.maximum.accumulate(cum)
    dd = cum / running_max - 1

    total = float(cum[-1] - 1)
    max_dd = float(np.min(dd))
    sharpe = float((np.mean(rets) / (np.std(rets) + 1e-10)) * np.sqrt(PERIODS_PER_YEAR))

    # Calmar = annual return / max drawdown
    ann_ret = np.mean(rets) * PERIODS_PER_YEAR
    calmar = abs(ann_ret / (max_dd + 1e-10)) if max_dd < -0.01 else ann_ret * 100

    # Sortino (downside deviation)
    downside = rets[rets < 0]
    downside_std = np.std(downside) if len(downside) > 10 else np.std(rets)
    sortino = float((np.mean(rets) / (downside_std + 1e-10)) * np.sqrt(PERIODS_PER_YEAR))

    # Win rate
    trading_rets = rets[rets != 0]
    win_rate = float(np.mean(trading_rets > 0)) if len(trading_rets) > 0 else 0

    # Profit factor
    gains = trading_rets[trading_rets > 0].sum() if len(trading_rets) > 0 else 0
    losses = abs(trading_rets[trading_rets < 0].sum()) if len(trading_rets) > 0 else 1e-10
    profit_factor = float(gains / (losses + 1e-10))

    # Time in market
    pct_active = float(np.mean(rets != 0))

    return {
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'calmar': round(calmar, 2),
        'ann_ret_%': round(ann_ret * 100, 1),
        'total_%': round(total * 100, 1),
        'max_dd_%': round(max_dd * 100, 1),
        'win_rate_%': round(win_rate * 100, 1),
        'profit_factor': round(profit_factor, 2),
        'pct_active_%': round(pct_active * 100, 1),
        'n_periods': len(rets),
    }


def main():
    parser = argparse.ArgumentParser(description='Risk Optimization Study')
    parser.add_argument('--predictions', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))

    # Find predictions
    pred_path = args.predictions
    if pred_path is None:
        candidates = [
            'results_ensemble_v2/ensemble_v2_predictions.parquet',
            'results_v5/test_predictions_v5.parquet',
        ]
        for c in candidates:
            p = os.path.join(root, c)
            if os.path.exists(p):
                pred_path = p
                break

    if not pred_path or not os.path.exists(pred_path):
        print("❌ No prediction file found")
        sys.exit(1)

    outdir = args.output or os.path.join(root, 'results_risk_study')
    os.makedirs(outdir, exist_ok=True)

    print("=" * 70)
    print("  RISK OPTIMIZATION STUDY")
    print("  Find best risk params for live trading")
    print("=" * 70)

    print(f"\n📊 Loading predictions from {os.path.basename(pred_path)}...")
    df, target_col, pred_cols = load_ensemble(pred_path)

    # Choose best ensemble signal
    # If ensemble has combo columns, use those; otherwise combine
    best_pred = None
    if 'pred_hist_v1' in df.columns and 'pred_lgb_v5' in df.columns:
        # Best ensemble from previous analysis
        df['pred_best'] = (df['pred_hist_v1'] + df['pred_lgb_v5']) / 2
        best_pred = 'pred_best'
        print(f"   Using ensemble: HIST v1 + LGB v5")
    elif 'pred_opt_primary' in df.columns:
        best_pred = 'pred_opt_primary'
    else:
        # Just use first pred column
        best_pred = pred_cols[0]

    print(f"   Prediction column: {best_pred}")

    # ================================================================
    # PHASE 1: Portfolio construction sweep
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"  PHASE 1: Portfolio Construction")
    print(f"{'=' * 70}")

    construction_results = []
    for n_long, n_short in [(5, 5), (8, 8), (10, 10), (3, 3)]:
        ls_raw = build_ls_returns(df, best_pred, target_col, n_long, n_short)
        ls_net = ls_raw - COST_LS
        m = compute_metrics(ls_net)
        if m:
            m['n_long'] = n_long
            m['n_short'] = n_short
            construction_results.append(m)
            print(f"   Top/Bot {n_long}: Sharpe={m['sharpe']:+.2f}, "
                  f"Ann={m['ann_ret_%']:+.1f}%, MaxDD={m['max_dd_%']:.1f}%")

    best_construction = max(construction_results, key=lambda x: x['sharpe'])
    n_long_best = best_construction['n_long']
    n_short_best = best_construction['n_short']
    print(f"   ✅ Best: Top/Bot {n_long_best}")

    # Build raw returns with best construction
    ls_raw = build_ls_returns(df, best_pred, target_col, n_long_best, n_short_best)

    # ================================================================
    # PHASE 2: Vol target + Kelly fraction sweep
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"  PHASE 2: Vol Target × Kelly Fraction")
    print(f"{'=' * 70}")

    vol_targets = [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03]
    kelly_fracs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

    phase2_results = []
    print(f"\n   {'Vol%':>5} {'Kelly':>6} | {'Sharpe':>7} {'Sortino':>8} {'Ann%':>6} {'MaxDD%':>7} {'Calmar':>7} {'WinR%':>6}")
    print(f"   {'─' * 70}")

    for vt, kf in product(vol_targets, kelly_fracs):
        rets = apply_risk_overlay(ls_raw, vt, dd_stop=-1.0, dd_resume=-1.0,
                                  kelly_frac=kf)  # No DD stop in this phase
        m = compute_metrics(rets)
        if m:
            m['vol_target'] = vt
            m['kelly_frac'] = kf
            phase2_results.append(m)

    # Sort by Calmar (return/risk balance)
    phase2_results.sort(key=lambda x: -x['calmar'])
    for r in phase2_results[:15]:
        print(f"   {r['vol_target']*100:>5.1f} {r['kelly_frac']:>6.1f} | "
              f"{r['sharpe']:>+7.2f} {r['sortino']:>+8.2f} {r['ann_ret_%']:>+6.1f} "
              f"{r['max_dd_%']:>7.1f} {r['calmar']:>7.2f} {r['win_rate_%']:>6.1f}")

    best_vt_kf = phase2_results[0]
    print(f"\n   ✅ Best Calmar: vol_target={best_vt_kf['vol_target']*100:.1f}%, "
          f"kelly={best_vt_kf['kelly_frac']:.1f}")

    # ================================================================
    # PHASE 3: DD Stop sweep (with best vol target + kelly)
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"  PHASE 3: Drawdown Stop Sweep")
    print(f"{'=' * 70}")

    vt_use = best_vt_kf['vol_target']
    kf_use = best_vt_kf['kelly_frac']

    dd_stops = [-0.08, -0.10, -0.12, -0.15, -0.18, -0.20, -0.25, -0.30, -0.40, -1.0]
    dd_resumes = [-0.02, -0.04, -0.06, -0.08, -0.10]

    phase3_results = []
    print(f"\n   {'DDStop%':>7} {'Resume%':>8} | {'Sharpe':>7} {'MaxDD%':>7} {'Total%':>8} {'Active%':>8} {'Calmar':>7}")
    print(f"   {'─' * 65}")

    for dd_stop, dd_resume in product(dd_stops, dd_resumes):
        if dd_resume <= dd_stop:  # Resume must be above stop
            continue
        rets = apply_risk_overlay(ls_raw, vt_use, dd_stop, dd_resume, kf_use)
        m = compute_metrics(rets)
        if m:
            m['dd_stop'] = dd_stop
            m['dd_resume'] = dd_resume
            phase3_results.append(m)

    phase3_results.sort(key=lambda x: -x['calmar'])
    for r in phase3_results[:15]:
        print(f"   {r['dd_stop']*100:>7.1f} {r['dd_resume']*100:>8.1f} | "
              f"{r['sharpe']:>+7.2f} {r['max_dd_%']:>7.1f} {r['total_%']:>+8.1f} "
              f"{r['pct_active_%']:>8.1f} {r['calmar']:>7.2f}")

    best_dd = phase3_results[0]

    # ================================================================
    # PHASE 4: Confidence filter sweep
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"  PHASE 4: Confidence Filter (min signal strength to trade)")
    print(f"{'=' * 70}")

    confidence_levels = [0.0, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5]
    phase4_results = []

    for conf in confidence_levels:
        ls_raw_filt = build_ls_returns(df, best_pred, target_col,
                                       n_long_best, n_short_best, confidence=conf)
        rets = apply_risk_overlay(ls_raw_filt,
                                  vt_use,
                                  best_dd['dd_stop'],
                                  best_dd['dd_resume'],
                                  kf_use)
        m = compute_metrics(rets)
        if m:
            m['confidence'] = conf
            phase4_results.append(m)
            print(f"   confidence={conf:.1f}: Sharpe={m['sharpe']:+.2f}, "
                  f"MaxDD={m['max_dd_%']:.1f}%, Active={m['pct_active_%']:.0f}%, "
                  f"Calmar={m['calmar']:.2f}")

    best_conf = max(phase4_results, key=lambda x: x['calmar'])

    # ================================================================
    # FINAL: Optimal configuration
    # ================================================================
    optimal_config = {
        'n_long': n_long_best,
        'n_short': n_short_best,
        'vol_target': vt_use,
        'vol_lookback': 48,
        'kelly_frac': kf_use,
        'dd_stop': best_dd['dd_stop'],
        'dd_resume': best_dd['dd_resume'],
        'confidence_threshold': best_conf['confidence'],
        'cost_per_ls_period': round(COST_LS, 6),
        'horizon_hours': HORIZON,
        'ensemble': 'hist_v1 + lgb_v5 (equal weight)',
    }

    # Run final eval with optimal config
    ls_raw_final = build_ls_returns(df, best_pred, target_col,
                                     optimal_config['n_long'],
                                     optimal_config['n_short'],
                                     confidence=optimal_config['confidence_threshold'])
    final_rets = apply_risk_overlay(
        ls_raw_final,
        optimal_config['vol_target'],
        optimal_config['dd_stop'],
        optimal_config['dd_resume'],
        optimal_config['kelly_frac'],
    )
    final_metrics = compute_metrics(final_rets)

    # Also compute no-overlay metrics for comparison
    raw_no_overlay = ls_raw - COST_LS
    raw_metrics = compute_metrics(raw_no_overlay)

    print(f"\n{'=' * 70}")
    print(f"  🏆 OPTIMAL CONFIGURATION FOR LIVE TRADING")
    print(f"{'=' * 70}")
    print(f"\n   📋 Risk Parameters:")
    for k, v in optimal_config.items():
        if isinstance(v, float):
            if 'pct' in k or 'frac' in k or 'target' in k:
                print(f"      {k:30s} {v*100:.1f}%")
            else:
                print(f"      {k:30s} {v}")
        else:
            print(f"      {k:30s} {v}")

    print(f"\n   📊 Backtest Results (optimal):")
    if final_metrics:
        for k, v in final_metrics.items():
            print(f"      {k:30s} {v}")

    print(f"\n   📊 Comparison — no risk overlay:")
    if raw_metrics:
        print(f"      Sharpe:    {raw_metrics['sharpe']:+.2f}  →  {final_metrics['sharpe']:+.2f}")
        print(f"      MaxDD:     {raw_metrics['max_dd_%']:.1f}%  →  {final_metrics['max_dd_%']:.1f}%")
        print(f"      Ann Ret:   {raw_metrics['ann_ret_%']:+.1f}%  →  {final_metrics['ann_ret_%']:+.1f}%")
        print(f"      Calmar:    {raw_metrics['calmar']:.2f}  →  {final_metrics['calmar']:.2f}")

    # ========================================
    # For $1000 capital: what does this mean?
    # ========================================
    capital = 1000
    ann_ret = final_metrics['ann_ret_%'] / 100
    max_dd = final_metrics['max_dd_%'] / 100

    print(f"\n   💰 For ${capital:,} capital:")
    print(f"      Expected annual return:  ${capital * ann_ret:+,.0f}")
    print(f"      Worst drawdown:          ${capital * max_dd:,.0f}")
    print(f"      Expected after 1 year:   ${capital * (1 + ann_ret):,.0f}")

    # Per-trade allocation
    trade_size_long = capital * 0.5 * optimal_config['kelly_frac'] / optimal_config['n_long']
    trade_size_short = capital * 0.5 * optimal_config['kelly_frac'] / optimal_config['n_short']
    print(f"      Per long position:       ${trade_size_long:.0f}")
    print(f"      Per short position:      ${trade_size_short:.0f}")

    # ========================================
    # SAVE
    # ========================================
    results = {
        'optimal_config': optimal_config,
        'final_metrics': final_metrics,
        'raw_metrics': raw_metrics,
        'phase1_construction': construction_results,
        'phase2_top5': phase2_results[:5],
        'phase3_top5': phase3_results[:5],
        'phase4_confidence': phase4_results,
        'capital_projection': {
            'initial': capital,
            'ann_return': round(ann_ret, 4),
            'max_drawdown': round(max_dd, 4),
            'per_long_usd': round(trade_size_long, 2),
            'per_short_usd': round(trade_size_short, 2),
        },
    }

    # Save optimal config as standalone file for live trading to load
    with open(os.path.join(outdir, 'optimal_config.json'), 'w') as f:
        json.dump(optimal_config, f, indent=2)

    with open(os.path.join(outdir, 'risk_study_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save equity curve
    eq = pd.DataFrame({
        'ret_optimal': final_rets,
        'ret_raw_net': raw_no_overlay,
        'equity_optimal': np.cumprod(1 + final_rets) * capital,
        'equity_raw': np.cumprod(1 + raw_no_overlay) * capital,
    })
    eq.to_parquet(os.path.join(outdir, 'equity_curves.parquet'), index=False)

    print(f"\n✅ Results saved to {outdir}/")
    print(f"   optimal_config.json — load this in live trading")
    print(f"   risk_study_results.json — full analysis")


if __name__ == '__main__':
    main()
