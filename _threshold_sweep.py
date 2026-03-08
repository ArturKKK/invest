#!/usr/bin/env python3
"""
Threshold sweep: run fast sim with different min_score thresholds.
Tests: how does win rate / Sharpe change when we only bet on strong signals?
Also tracks top-3 most confident picks' performance.
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Import from run_fast_sim / run_trading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_trading import (
    SYMBOLS, EXCLUDE_COLS, UNRANKED_COLS,
    fetch_ohlcv, build_features, cross_sectional_rank,
    load_lgb_models, load_catboost_models,
    DEFAULT_RISK,
)

COST_SIDE = 0.0006  # 6bp per side
FUNDING_PER_8H = 0.0001

def main():
    root = os.path.dirname(os.path.abspath(__file__))

    DAYS = 30
    WARMUP = 720
    CAPITAL = 1000.0
    REBAL_H = 12
    LEVERAGE = 3.0
    KELLY = 1.0

    total_h = WARMUP + DAYS * 24
    print("=" * 70)
    print(f"  THRESHOLD SWEEP — {DAYS}d, ${CAPITAL}, lev={LEVERAGE}x, rebal={REBAL_H}h")
    print("=" * 70)

    # ── 1. Load data ──
    print(f"\n📊 Fetching {total_h}h data...")
    raw = fetch_ohlcv(SYMBOLS, total_h)
    if raw is None or len(raw) == 0:
        print("❌ fetch failed"); return
    print(f"   {raw.shape}, {raw['symbol'].nunique()} symbols")

    # ── 2. Features ──
    print("🔧 Features...")
    df = build_features(raw)
    fc = [c for c in df.columns if c not in EXCLUDE_COLS
          and not c.startswith("target_")
          and df[c].dtype in ("float64","float32","int64","int32")]
    df = cross_sectional_rank(df, fc)

    # ── 3. Load models ──
    print("🤖 Loading models...")
    model_groups = []
    for ver in ['results_v6', 'results_v7']:
        p = os.path.join(root, ver)
        if os.path.isdir(p):
            ms = load_lgb_models(p)
            if ms:
                model_groups.append((ver, ms))
                print(f"   ✅ {ver}: {len(ms)} models")

    cb_dir = os.path.join(root, 'results_catboost')
    if os.path.isdir(cb_dir):
        ms = load_catboost_models(cb_dir)
        if ms:
            model_groups.append(('catboost', ms))
            print(f"   ✅ catboost: {len(ms)} models")

    if not model_groups:
        print("❌ No models found"); return

    def predict_ensemble(snap):
        snap = snap.copy()
        all_preds = []
        for name, models in model_groups:
            # Align features to what the model expects
            if name == 'catboost':
                model_features = models[0].feature_names_
            else:
                model_features = models[0].feature_name()
            # Pad missing features with 0
            for col in model_features:
                if col not in snap.columns:
                    snap[col] = 0.0
            X = snap[model_features].values

            preds = []
            for m in models:
                try:
                    p = m.predict(X)
                    preds.append(p)
                except Exception:
                    pass
            if preds:
                avg = np.mean(preds, axis=0)
                z = (avg - avg.mean()) / (avg.std() + 1e-10)
                all_preds.append(z)
        if not all_preds:
            return np.zeros(len(snap))
        return np.mean(all_preds, axis=0)

    # ── 4. Build prediction timeline ──
    all_ts = sorted(df['timestamp'].unique())
    sim_start = WARMUP
    steps = all_ts[sim_start::REBAL_H]

    print(f"\n📐 Generating predictions for {len(steps)} rebalance steps...")
    predictions = []
    for i, ts in enumerate(steps[:-1]):
        snap = df[df['timestamp'] == ts]
        if len(snap) < 20:
            continue
        scores = predict_ensemble(snap)
        syms = snap['symbol'].values
        # Get forward returns (to next rebalance)
        ts_next = steps[i+1] if i+1 < len(steps) else steps[-1]
        snap_next = df[df['timestamp'] == ts_next]
        px0 = dict(zip(snap['symbol'], snap['close']))
        px1 = dict(zip(snap_next['symbol'], snap_next['close']))

        for sym, score in zip(syms, scores):
            p0 = px0.get(sym, 0)
            p1 = px1.get(sym, 0)
            if p0 > 0 and p1 > 0:
                fwd_ret = (p1 - p0) / p0
                predictions.append({
                    'timestamp': ts,
                    'symbol': sym,
                    'score': score,
                    'fwd_ret': fwd_ret,
                })
        if (i+1) % 10 == 0:
            print(f"   {i+1}/{len(steps)-1}")

    pred_df = pd.DataFrame(predictions)
    print(f"   Total predictions: {len(pred_df):,}")

    # ── 5. Sweep thresholds ──
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
    n_pos_options = [5, 3]  # positions per side

    print(f"\n{'='*70}")
    print(f"  THRESHOLD SWEEP RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Thresh':>7} {'N':>3} {'Sharpe':>8} {'WinRate':>8} {'Return':>8} {'MaxDD':>8} {'Trades':>7} {'Avg|Score|':>10}")
    print(f"{'─'*70}")

    results = []
    for n_pos in n_pos_options:
        for thresh in thresholds:
            equity = CAPITAL
            peak = CAPITAL
            pnls = []
            n_trades = 0
            avg_scores = []

            for ts in sorted(pred_df['timestamp'].unique()):
                step = pred_df[pred_df['timestamp'] == ts].copy()
                # Apply threshold
                step_filtered = step[step['score'].abs() >= thresh]
                if len(step_filtered) == 0:
                    continue

                # Split by sign
                longs = step_filtered[step_filtered['score'] > 0].nlargest(n_pos, 'score')
                shorts = step_filtered[step_filtered['score'] < 0].nsmallest(n_pos, 'score')

                if len(longs) == 0 and len(shorts) == 0:
                    continue

                total_alloc = equity * KELLY * LEVERAGE
                half = total_alloc / 2

                step_pnl = 0.0
                step_trades = len(longs) + len(shorts)

                if len(longs) > 0:
                    per_long = half / len(longs)
                    for _, r in longs.iterrows():
                        step_pnl += per_long * r['fwd_ret']
                        avg_scores.append(abs(r['score']))

                if len(shorts) > 0:
                    per_short = half / len(shorts)
                    for _, r in shorts.iterrows():
                        step_pnl += per_short * (-r['fwd_ret'])
                        avg_scores.append(abs(r['score']))

                # Costs
                cost = step_trades * (total_alloc / max(step_trades, 1)) * COST_SIDE
                funding = total_alloc * FUNDING_PER_8H * (REBAL_H / 8.0) if LEVERAGE > 1 else 0
                step_pnl -= cost + funding

                pnls.append(step_pnl)
                equity += step_pnl
                peak = max(peak, equity)
                n_trades += step_trades

            if not pnls:
                continue

            a = np.array(pnls)
            tot_ret = equity / CAPITAL - 1
            max_dd = min(
                (sum(pnls[:i+1]) + CAPITAL) / max(max(sum(pnls[:j+1]) + CAPITAL for j in range(i+1)), CAPITAL) - 1
                for i in range(len(pnls))
            ) if len(pnls) > 1 else 0
            # Simpler DD
            eq_curve = np.cumsum(a) + CAPITAL
            peak_curve = np.maximum.accumulate(eq_curve)
            drawdowns = eq_curve / peak_curve - 1
            max_dd = drawdowns.min()

            sharpe = np.mean(a) / (np.std(a) + 1e-10) * np.sqrt(365 * 24 / REBAL_H)
            wr = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
            avg_sc = np.mean(avg_scores) if avg_scores else 0

            flag = " ⭐" if sharpe > 5 and wr > 0.55 else ""
            print(f"  {thresh:>5.1f}  {n_pos:>3}  {sharpe:>+7.2f}  {wr:>7.0%}  "
                  f"{tot_ret:>+7.1%}  {max_dd:>7.1%}  {n_trades:>6}  {avg_sc:>9.2f}{flag}")

            results.append({
                'threshold': thresh,
                'n_pos': n_pos,
                'sharpe': round(sharpe, 3),
                'win_rate': round(wr, 4),
                'total_return': round(tot_ret, 4),
                'max_dd': round(max_dd, 4),
                'n_trades': n_trades,
                'avg_score': round(avg_sc, 3),
                'final_equity': round(equity, 2),
            })

        print(f"{'─'*70}")

    # ── 6. Top-3 confident analysis ──
    print(f"\n{'='*70}")
    print(f"  TOP-K MOST CONFIDENT SIGNALS — PERFORMANCE")
    print(f"{'='*70}")
    print(f"\n{'TopK':>5} {'Side':>6} {'Sharpe':>8} {'WinRate':>8} {'Return':>8} {'AvgRet':>8} {'N':>5}")
    print(f"{'─'*55}")

    for topk in [1, 2, 3, 5]:
        for side_name, side_filter in [('long', 'long'), ('short', 'short'), ('both', 'both')]:
            wins = 0
            losses = 0
            rets = []

            for ts in sorted(pred_df['timestamp'].unique()):
                step = pred_df[pred_df['timestamp'] == ts].copy()

                if side_filter == 'long':
                    candidates = step[step['score'] > 0].nlargest(topk, 'score')
                    for _, r in candidates.iterrows():
                        rets.append(r['fwd_ret'])
                        if r['fwd_ret'] > 0: wins += 1
                        else: losses += 1
                elif side_filter == 'short':
                    candidates = step[step['score'] < 0].nsmallest(topk, 'score')
                    for _, r in candidates.iterrows():
                        rets.append(-r['fwd_ret'])
                        if -r['fwd_ret'] > 0: wins += 1
                        else: losses += 1
                else:
                    # Both sides by confidence
                    step['abs_score'] = step['score'].abs()
                    top = step.nlargest(topk, 'abs_score')
                    for _, r in top.iterrows():
                        if r['score'] > 0:
                            rets.append(r['fwd_ret'])
                            if r['fwd_ret'] > 0: wins += 1
                            else: losses += 1
                        else:
                            rets.append(-r['fwd_ret'])
                            if -r['fwd_ret'] > 0: wins += 1
                            else: losses += 1

            if not rets:
                continue
            a = np.array(rets)
            wr = wins / (wins + losses) if (wins + losses) > 0 else 0
            tot = np.sum(a)
            sh = np.mean(a) / (np.std(a) + 1e-10) * np.sqrt(365 * 24 / REBAL_H)
            avg_r = np.mean(a) * 100
            flag = " ⭐" if sh > 2 and wr > 0.55 else ""
            print(f"  {topk:>3}  {side_name:>6}  {sh:>+7.2f}  {wr:>7.0%}  "
                  f"{tot:>+7.1%}  {avg_r:>+7.3f}%  {wins+losses:>4}{flag}")

    print(f"\n{'='*70}\n")

    # Save results
    out = os.path.join(root, 'trading_logs', 'threshold_sweep.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Saved: {out}")


if __name__ == '__main__':
    main()
