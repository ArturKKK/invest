#!/usr/bin/env python3
"""
Ensemble v2 — HIST v2 + LGB v5 (with cost-aware evaluation)

Combines sentiment-aware HIST v2 and LGB v5 predictions.
Evaluates with actual returns, costs, vol targeting, DD stop.

Usage:
  python run_ensemble_v2.py
  python run_ensemble_v2.py --hist results_hist_v2/test_predictions_hist_v2.parquet \
                            --lgb results_v5/test_predictions_v5.parquet
  python run_ensemble_v2.py --hist results_hist_v2/test_predictions_hist_v2.parquet \
                            --lgb results_v5/test_predictions_v5.parquet \
                            --lgb-old results_v4/test_predictions_v4.parquet \
                            --hist-old results_hist/test_predictions_hist.parquet
"""

import os
import sys
import argparse
import json
import warnings
from datetime import datetime
from itertools import combinations

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────────────────────
HORIZON = 4
PERIODS_PER_DAY = 24 // HORIZON   # 6
PERIODS_PER_YEAR = PERIODS_PER_DAY * 365  # 2190

# Cost model (same as v5 pipeline)
COST_CFG = {
    'taker_fee': 0.0003,     # 3 bps blended (maker+taker)
    'slippage': 0.0001,      # 1 bp avg slippage
    'funding_per_8h': 0.00005,  # 0.5 bp net per 8h
    'turnover_frac': 0.25,   # 25% turnover per rebalance
}

# Risk overlay
VOL_TARGET = 0.02            # 2% per period
VOL_LOOKBACK = 48            # 48 periods = 8 days
DD_STOP_THRESHOLD = -0.25    # -25% drawdown → stop
DD_RESUME_THRESHOLD = -0.10  # -10% → resume


def compute_cost_per_period():
    """Per-period cost for one side of LS portfolio."""
    c = COST_CFG
    one_way = (c['taker_fee'] + c['slippage']) * c['turnover_frac']
    funding = c['funding_per_8h'] * (HORIZON / 8)
    return one_way + funding


COST_PER_PERIOD = compute_cost_per_period()


# ──────────────────────────────────────────────────────────────
#  DATA LOADING
# ──────────────────────────────────────────────────────────────
def load_predictions(path, pred_col_guess, rename_to):
    """Load prediction parquet, find pred column, rename."""
    if not path or not os.path.exists(path):
        return None

    df = pd.read_parquet(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Find prediction column
    if pred_col_guess in df.columns:
        pred_col = pred_col_guess
    else:
        pred_cols = [c for c in df.columns if c.startswith('pred')]
        if pred_cols:
            pred_col = pred_cols[0]
        else:
            print(f"   ⚠️  No prediction column in {path}: {df.columns.tolist()}")
            return None

    target_col = f'target_ret_{HORIZON}h'
    cols = ['timestamp', 'symbol', pred_col]
    if target_col in df.columns:
        cols.append(target_col)

    result = df[cols].copy()
    if pred_col != rename_to:
        result = result.rename(columns={pred_col: rename_to})

    print(f"   ✅ {rename_to}: {len(result):,} rows, "
          f"{result['timestamp'].min().date()} → {result['timestamp'].max().date()}")
    return result


# ──────────────────────────────────────────────────────────────
#  EVALUATION (cost-aware, with risk overlay)
# ──────────────────────────────────────────────────────────────
def evaluate(merged, pred_col, actual_col, label=""):
    """Full evaluation: IC, LS returns, costs, vol target, DD stop."""
    rank_ics, ics = [], []
    ls_rets_raw = []
    lo5_rets, lo10_rets = [], []

    for ts, grp in merged.groupby('timestamp'):
        if len(grp) < 10:
            continue

        p = grp[pred_col].values
        a = grp[actual_col].values
        valid = ~(np.isnan(p) | np.isnan(a))
        if valid.sum() < 10:
            continue

        pv, av = p[valid], a[valid]
        ic = np.corrcoef(pv, av)[0, 1]
        ric, _ = spearmanr(pv, av)
        ics.append(ic if np.isfinite(ic) else 0)
        rank_ics.append(ric if np.isfinite(ric) else 0)

        order = np.argsort(-pv)
        sorted_a = av[order]
        n_q = max(len(pv) // 5, 1)

        ls_rets_raw.append(sorted_a[:n_q].mean() - sorted_a[-n_q:].mean())
        lo5_rets.append(sorted_a[:min(5, len(sorted_a))].mean())
        lo10_rets.append(sorted_a[:min(10, len(sorted_a))].mean())

    if not rank_ics:
        return None

    rank_ics = np.array(rank_ics)
    ics = np.array(ics)
    ls_raw = np.array(ls_rets_raw)

    # Net returns
    ls_net = ls_raw - COST_PER_PERIOD * 2  # both sides
    lo5 = np.array(lo5_rets) - COST_PER_PERIOD
    lo10 = np.array(lo10_rets) - COST_PER_PERIOD

    # Vol targeting
    vt_rets = np.zeros_like(ls_net)
    for i in range(len(ls_net)):
        lookback = ls_raw[max(0, i - VOL_LOOKBACK):i]
        if len(lookback) >= 6:
            vol = np.std(lookback) + 1e-10
            scale = np.clip(VOL_TARGET / vol, 0.1, 2.0)
        else:
            scale = 1.0
        vt_rets[i] = ls_raw[i] * scale - COST_PER_PERIOD * 2 * scale

    # DD stop
    dd_rets = np.zeros_like(ls_net)
    cum_eq = 1.0
    peak = 1.0
    active = True
    for i in range(len(ls_net)):
        if active:
            dd_rets[i] = ls_net[i]
            cum_eq *= (1 + ls_net[i])
        else:
            dd_rets[i] = 0.0

        peak = max(peak, cum_eq)
        dd = cum_eq / peak - 1
        if active and dd < DD_STOP_THRESHOLD:
            active = False
        elif not active and dd > DD_RESUME_THRESHOLD:
            active = True

    # Helper functions
    def sharpe(r):
        if len(r) == 0 or np.std(r) < 1e-12:
            return 0.0
        return (np.mean(r) / (np.std(r) + 1e-10)) * np.sqrt(PERIODS_PER_YEAR)

    def max_dd(r):
        if len(r) == 0:
            return 0.0
        cum = np.cumprod(1 + np.clip(r, -0.99, None))
        running_max = np.maximum.accumulate(cum)
        return float(np.min(cum / running_max - 1))

    def total_ret(r):
        return float(np.prod(1 + np.clip(r, -0.99, None)) - 1)

    # Daily Rank ICIR
    daily_rics = []
    n_per_day = PERIODS_PER_DAY
    for i in range(0, max(1, len(rank_ics) - n_per_day + 1), n_per_day):
        daily_rics.append(rank_ics[i:i + n_per_day].mean())
    daily_rics = np.array(daily_rics) if daily_rics else np.array([0.0])
    rank_icir = (np.mean(daily_rics) / (np.std(daily_rics) + 1e-10)) if len(daily_rics) > 1 else 0

    return {
        'IC': round(float(np.mean(ics)), 4),
        'Rank_IC': round(float(np.mean(rank_ics)), 4),
        'ICIR': round(float(np.mean(ics) / (np.std(ics) + 1e-10)), 4),
        'Rank_ICIR': round(float(rank_icir), 4),
        'LS_Sharpe_raw': round(float(sharpe(ls_raw)), 2),
        'LS_Sharpe_net': round(float(sharpe(ls_net)), 2),
        'LS_Ann_Return_net_%': round(float(np.mean(ls_net) * PERIODS_PER_YEAR * 100), 1),
        'LS_MaxDD_net_%': round(float(max_dd(ls_net) * 100), 1),
        'LS_Total_net_%': round(float(total_ret(ls_net) * 100), 1),
        'LS_VolTarget_Sharpe': round(float(sharpe(vt_rets)), 2),
        'LS_VolTarget_MaxDD_%': round(float(max_dd(vt_rets) * 100), 1),
        'LS_DDStop_Sharpe': round(float(sharpe(dd_rets)), 2),
        'LS_DDStop_MaxDD_%': round(float(max_dd(dd_rets) * 100), 1),
        'LS_DDStop_Total_%': round(float(total_ret(dd_rets) * 100), 1),
        'LO5_Sharpe': round(float(sharpe(lo5)), 2),
        'LO10_Sharpe': round(float(sharpe(lo10)), 2),
        'N_periods': len(ls_raw),
        'Cost_per_period_bps': round(COST_PER_PERIOD * 2 * 10000, 1),
    }


def optimize_weights_grid(merged, pred_cols, actual_col, step=0.05):
    """Grid search weights on first half of test, eval on second half."""
    timestamps = sorted(merged['timestamp'].unique())
    mid = len(timestamps) // 2
    train_ts = set(timestamps[:mid])
    train_df = merged[merged['timestamp'].isin(train_ts)].copy()

    n = len(pred_cols)
    best_sharpe = -999
    best_w = None

    steps = np.arange(0, 1.001, step)

    if n == 2:
        for w0 in steps:
            w1 = round(1.0 - w0, 3)
            if w1 < -0.001:
                continue
            train_df['_ens'] = w0 * train_df[pred_cols[0]] + w1 * train_df[pred_cols[1]]
            m = evaluate(train_df, '_ens', actual_col)
            if m and m['LS_Sharpe_net'] > best_sharpe:
                best_sharpe = m['LS_Sharpe_net']
                best_w = {pred_cols[0]: w0, pred_cols[1]: w1}
    elif n == 3:
        for w0 in steps:
            for w1 in np.arange(0, 1.001 - w0, step):
                w2 = round(1.0 - w0 - w1, 3)
                if w2 < -0.001:
                    continue
                train_df['_ens'] = (w0 * train_df[pred_cols[0]]
                                    + w1 * train_df[pred_cols[1]]
                                    + w2 * train_df[pred_cols[2]])
                m = evaluate(train_df, '_ens', actual_col)
                if m and m['LS_Sharpe_net'] > best_sharpe:
                    best_sharpe = m['LS_Sharpe_net']
                    best_w = {pred_cols[0]: w0, pred_cols[1]: w1, pred_cols[2]: w2}
    elif n == 4:
        for w0 in np.arange(0, 1.001, step):
            for w1 in np.arange(0, 1.001 - w0, step):
                for w2 in np.arange(0, 1.001 - w0 - w1, step):
                    w3 = round(1.0 - w0 - w1 - w2, 3)
                    if w3 < -0.001:
                        continue
                    train_df['_ens'] = (w0 * train_df[pred_cols[0]]
                                        + w1 * train_df[pred_cols[1]]
                                        + w2 * train_df[pred_cols[2]]
                                        + w3 * train_df[pred_cols[3]])
                    m = evaluate(train_df, '_ens', actual_col)
                    if m and m['LS_Sharpe_net'] > best_sharpe:
                        best_sharpe = m['LS_Sharpe_net']
                        best_w = dict(zip(pred_cols, [w0, w1, w2, w3]))
    else:
        best_w = {c: 1.0 / n for c in pred_cols}

    if best_w is None:
        best_w = {c: 1.0 / n for c in pred_cols}

    return best_w, best_sharpe


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Ensemble v2: HIST v2 + LGB v5')
    parser.add_argument('--hist', type=str, default=None,
                        help='HIST v2 predictions parquet')
    parser.add_argument('--lgb', type=str, default=None,
                        help='LGB v5 predictions parquet')
    parser.add_argument('--hist-old', type=str, default=None,
                        help='HIST v1 predictions parquet (optional)')
    parser.add_argument('--lgb-old', type=str, default=None,
                        help='LGB v4 predictions parquet (optional)')
    parser.add_argument('--results', type=str, default=None)
    parser.add_argument('--weight-step', type=float, default=0.05,
                        help='Grid search step for weights')
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    results_dir = args.results or os.path.join(root, 'results_ensemble_v2')
    os.makedirs(results_dir, exist_ok=True)
    target_col = f'target_ret_{HORIZON}h'

    print("=" * 70)
    print("  ENSEMBLE v2 — Cost-Aware Multi-Model Combiner")
    print("  HIST v2 + LGB v5 (+ optional v1 models)")
    print("=" * 70)

    # ── Load predictions ──────────────────────────────────────
    print(f"\n📊 Loading predictions...")

    auto = {
        'hist_v2': (args.hist, [
            'results_hist_v2/test_predictions_hist_v2.parquet',
        ], 'pred_hist_v2'),
        'lgb_v5': (args.lgb, [
            'results_v5/test_predictions_v5.parquet',
        ], 'pred_v5'),
        'hist_v1': (args.hist_old, [
            'results_hist/test_predictions_hist.parquet',
        ], 'pred_hist'),
        'lgb_v4': (args.lgb_old, [
            'results_v4/test_predictions_v4.parquet',
        ], 'pred_ensemble'),
    }

    dfs = {}
    for name, (explicit, defaults, guess_col) in auto.items():
        path = explicit
        if path is None:
            for d in defaults:
                full = os.path.join(root, d)
                if os.path.exists(full):
                    path = full
                    break
        if path:
            rename = f'pred_{name}'
            df = load_predictions(path, guess_col, rename)
            if df is not None:
                dfs[name] = df

    if len(dfs) < 2:
        print(f"❌ Need at least 2 models, found: {list(dfs.keys())}")
        if not dfs:
            print("   No prediction files found! Run models first.")
        sys.exit(1)

    names = sorted(dfs.keys())
    print(f"\n   Models: {names}")

    # ── Merge ─────────────────────────────────────────────────
    print(f"\n🔗 Merging predictions...")

    merged = dfs[names[0]].copy()
    for name in names[1:]:
        other = dfs[name]
        pred_c = f'pred_{name}'
        merge_cols = ['timestamp', 'symbol', pred_c]
        if target_col in other.columns and target_col not in merged.columns:
            merge_cols.append(target_col)
        merged = merged.merge(other[merge_cols], on=['timestamp', 'symbol'], how='inner')

    merged = merged.dropna(subset=[target_col])
    pred_cols = [f'pred_{n}' for n in names]

    print(f"   Merged: {len(merged):,} rows, {merged['timestamp'].nunique()} timestamps")
    print(f"   Period: {merged['timestamp'].min().date()} → {merged['timestamp'].max().date()}")

    # ── Cross-sectional z-score ───────────────────────────────
    print(f"\n📐 Cross-sectional z-score normalization...")
    for col in pred_cols:
        merged[col] = merged.groupby('timestamp')[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-10)
        )

    # ── Individual models ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  INDIVIDUAL MODELS (cost-aware evaluation)")
    print(f"{'=' * 70}")

    all_metrics = {}
    for col in pred_cols:
        m = evaluate(merged, col, target_col)
        if m:
            all_metrics[col] = m
            print(f"\n   {col}:")
            print(f"      Rank IC: {m['Rank_IC']:+.4f}  |  ICIR: {m['Rank_ICIR']:.3f}")
            print(f"      LS raw:  {m['LS_Sharpe_raw']:+.2f}  |  LS net: {m['LS_Sharpe_net']:+.2f}  |  MaxDD: {m['LS_MaxDD_net_%']:.1f}%")
            print(f"      VT:      {m['LS_VolTarget_Sharpe']:+.2f}  |  DDStop: {m['LS_DDStop_Sharpe']:+.2f}  (MaxDD {m['LS_DDStop_MaxDD_%']:.1f}%)")

    # ── Pairwise ensembles (equal weight) ─────────────────────
    print(f"\n{'=' * 70}")
    print(f"  PAIRWISE ENSEMBLES (equal weight)")
    print(f"{'=' * 70}")

    for c1, c2 in combinations(pred_cols, 2):
        ens_name = f'{c1}+{c2}'
        merged[ens_name] = (merged[c1] + merged[c2]) / 2
        m = evaluate(merged, ens_name, target_col)
        if m:
            all_metrics[ens_name] = m
            print(f"\n   {ens_name}:")
            print(f"      Rank IC: {m['Rank_IC']:+.4f}  |  LS net: {m['LS_Sharpe_net']:+.2f}  |  "
                  f"DDStop: {m['LS_DDStop_Sharpe']:+.2f}  (MaxDD {m['LS_DDStop_MaxDD_%']:.1f}%)")

    # ── Equal-weight all models ───────────────────────────────
    if len(pred_cols) >= 3:
        merged['pred_equal_all'] = sum(merged[c] for c in pred_cols) / len(pred_cols)
        m = evaluate(merged, 'pred_equal_all', target_col)
        if m:
            all_metrics['pred_equal_all'] = m
            print(f"\n   Equal-weight ALL ({len(pred_cols)} models):")
            print(f"      Rank IC: {m['Rank_IC']:+.4f}  |  LS net: {m['LS_Sharpe_net']:+.2f}  |  "
                  f"DDStop: {m['LS_DDStop_Sharpe']:+.2f}  (MaxDD {m['LS_DDStop_MaxDD_%']:.1f}%)")

    # ── Optimized weights ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  WEIGHT OPTIMIZATION (grid search on 1st half of test)")
    print(f"{'=' * 70}")

    # Optimize on primary models (hist_v2 + lgb_v5)
    primary = [c for c in pred_cols if 'v2' in c or 'v5' in c]
    if len(primary) >= 2:
        print(f"\n   🔍 Primary: {primary}")
        w, s = optimize_weights_grid(merged, primary, target_col, args.weight_step)
        print(f"   Weights: {', '.join(f'{k}={v:.2f}' for k, v in w.items())}  (train Sharpe: {s:.2f})")
        merged['pred_opt_primary'] = sum(w[c] * merged[c] for c in primary)
        m = evaluate(merged, 'pred_opt_primary', target_col)
        if m:
            all_metrics['pred_opt_primary'] = m
            print(f"   → LS net: {m['LS_Sharpe_net']:+.2f}  |  DDStop: {m['LS_DDStop_Sharpe']:+.2f}  (MaxDD {m['LS_DDStop_MaxDD_%']:.1f}%)")

    # Optimize all models
    if len(pred_cols) >= 3:
        print(f"\n   🔍 All models: {pred_cols}")
        w_all, s_all = optimize_weights_grid(merged, pred_cols, target_col, args.weight_step)
        print(f"   Weights: {', '.join(f'{k}={v:.2f}' for k, v in w_all.items())}  (train Sharpe: {s_all:.2f})")
        merged['pred_opt_all'] = sum(w_all[c] * merged[c] for c in pred_cols)
        m = evaluate(merged, 'pred_opt_all', target_col)
        if m:
            all_metrics['pred_opt_all'] = m
            print(f"   → LS net: {m['LS_Sharpe_net']:+.2f}  |  DDStop: {m['LS_DDStop_Sharpe']:+.2f}  (MaxDD {m['LS_DDStop_MaxDD_%']:.1f}%)")

    # ── Best model ────────────────────────────────────────────
    best_name = max(all_metrics, key=lambda k: all_metrics[k]['LS_Sharpe_net'])
    best = all_metrics[best_name]

    print(f"\n{'=' * 70}")
    print(f"  🏆 BEST: {best_name}")
    print(f"{'=' * 70}")
    for k, v in best.items():
        print(f"      {k:30s} {v}")

    # ── Comparison table ──────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON TABLE (sorted by LS Sharpe net)")
    print(f"{'=' * 70}")
    header = f"  {'Model':<40} {'Rank IC':>8} {'LS raw':>7} {'LS net':>7} {'MaxDD%':>7} {'DDStop':>7} {'DD MaxDD':>8}"
    print(header)
    print(f"  {'─' * 87}")
    for name, m in sorted(all_metrics.items(), key=lambda x: -x[1]['LS_Sharpe_net']):
        marker = " 🏆" if name == best_name else ""
        print(f"  {name:<40} {m['Rank_IC']:>+8.4f} {m['LS_Sharpe_raw']:>+7.2f} "
              f"{m['LS_Sharpe_net']:>+7.2f} {m['LS_MaxDD_net_%']:>7.1f} "
              f"{m['LS_DDStop_Sharpe']:>+7.2f} {m['LS_DDStop_MaxDD_%']:>8.1f}{marker}")
    print(f"{'=' * 70}")

    # ── Save ──────────────────────────────────────────────────
    save_cols = ['timestamp', 'symbol', target_col] + pred_cols
    for extra in ['pred_equal_all', 'pred_opt_primary', 'pred_opt_all']:
        if extra in merged.columns:
            save_cols.append(extra)

    merged[save_cols].to_parquet(
        os.path.join(results_dir, 'ensemble_v2_predictions.parquet'), index=False
    )

    results = {
        'all_metrics': all_metrics,
        'best': {'name': best_name, 'metrics': best},
        'cost_model': COST_CFG,
        'cost_per_ls_period_bps': round(COST_PER_PERIOD * 2 * 10000, 1),
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'models': list(dfs.keys()),
            'n_merged_rows': len(merged),
            'horizon': HORIZON,
        },
    }

    with open(os.path.join(results_dir, 'ensemble_v2_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✅ Saved to {results_dir}/")


if __name__ == '__main__':
    main()
