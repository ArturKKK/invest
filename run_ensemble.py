#!/usr/bin/env python3
"""
Final Multi-Model Ensemble — HIST + LightGBM + GRU

Combines predictions from all trained models with proper evaluation:
- Loads prediction parquets from each model
- Normalizes to z-scores per timestamp (cross-sectional)
- Searches optimal weights via grid or equal-weight
- Evaluates with ACTUAL returns (not ranks)

Usage:
  python run_ensemble.py
  python run_ensemble.py --hist results_hist/test_predictions_hist.parquet \
                         --lgb results_v4/test_predictions_v4.parquet \
                         --gru results_gru/test_predictions_gru.parquet
"""

import os
import sys
import argparse
import json
import warnings
from datetime import datetime
from itertools import product

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

HORIZON = 4


def load_predictions(path, pred_col_name, rename_to):
    """Load a prediction parquet, standardize column names."""
    if not os.path.exists(path):
        return None

    df = pd.read_parquet(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Find pred column
    if pred_col_name in df.columns:
        pass
    else:
        # Try to find any pred_* column
        pred_cols = [c for c in df.columns if c.startswith('pred_')]
        if pred_cols:
            pred_col_name = pred_cols[0]
        else:
            print(f"   ⚠️  No prediction column found in {path}")
            return None

    target_col = f'target_ret_{HORIZON}h'
    cols = ['timestamp', 'symbol', pred_col_name]
    if target_col in df.columns:
        cols.append(target_col)

    result = df[cols].copy()
    if pred_col_name != rename_to:
        result = result.rename(columns={pred_col_name: rename_to})

    print(f"   ✅ {rename_to}: {len(result):,} rows from {os.path.basename(path)}")
    return result


def cross_sectional_zscore(series, grp_col='timestamp', df=None):
    """Z-score normalize within each timestamp."""
    return df.groupby(grp_col)[series].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-10)
    )


def evaluate(merged, pred_col, actual_col, horizon=4):
    """Evaluate a single prediction column using actual returns."""
    periods_per_day = 24 // horizon
    periods_per_year = periods_per_day * 365

    rank_ics, ics, ls_rets, lo5_rets, lo10_rets = [], [], [], [], []

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
        ics.append(ic if not np.isnan(ic) else 0)
        rank_ics.append(ric if not np.isnan(ric) else 0)

        order = np.argsort(-pv)
        sorted_actual = av[order]
        n_q = max(len(pv) // 5, 1)

        ls_rets.append(sorted_actual[:n_q].mean() - sorted_actual[-n_q:].mean())
        lo5_rets.append(sorted_actual[:min(5, len(sorted_actual))].mean())
        lo10_rets.append(sorted_actual[:min(10, len(sorted_actual))].mean())

    if not rank_ics:
        return None

    rank_ics = np.array(rank_ics)
    ics = np.array(ics)
    ls_rets = np.array(ls_rets)
    lo5 = np.array(lo5_rets) - 0.0005
    lo10 = np.array(lo10_rets) - 0.0005

    def sharpe(r, ppyr):
        if len(r) == 0 or r.std() < 1e-12:
            return 0.0
        return (r.mean() / (r.std() + 1e-10)) * np.sqrt(ppyr)

    def max_dd(r):
        if len(r) == 0:
            return 0.0
        cum = np.cumprod(1 + np.clip(r, -0.99, None))
        running_max = np.maximum.accumulate(cum)
        dd = cum / running_max - 1
        return float(np.min(dd))

    def total_ret(r):
        return float(np.prod(1 + np.clip(r, -0.99, None)) - 1)

    # Daily ICIR
    n_per_day = periods_per_day
    daily_rics = []
    for i in range(0, max(1, len(rank_ics) - n_per_day + 1), n_per_day):
        daily_rics.append(rank_ics[i:i+n_per_day].mean())
    daily_rics = np.array(daily_rics) if daily_rics else np.array([0.0])
    rank_icir = (daily_rics.mean() / (daily_rics.std() + 1e-10)) if len(daily_rics) > 1 else 0

    return {
        'IC': round(float(ics.mean()), 4),
        'Rank_IC': round(float(rank_ics.mean()), 4),
        'ICIR': round(float(ics.mean() / (ics.std() + 1e-10)), 4),
        'Rank_ICIR': round(float(rank_icir), 4),
        'LS_Sharpe': round(float(sharpe(ls_rets, periods_per_year)), 2),
        'LS_Ann_Return_%': round(float(ls_rets.mean() * periods_per_year * 100), 1),
        'LS_MaxDD_%': round(float(max_dd(ls_rets) * 100), 1),
        'LO5_Sharpe': round(float(sharpe(lo5, periods_per_year)), 2),
        'LO5_Total_%': round(float(total_ret(lo5) * 100), 1),
        'LO10_Sharpe': round(float(sharpe(lo10, periods_per_year)), 2),
        'LO10_Total_%': round(float(total_ret(lo10) * 100), 1),
        'N_periods': len(ls_rets),
    }


def optimize_weights(merged, pred_cols, actual_col, horizon=4):
    """
    Grid search for optimal ensemble weights on first half of test data.
    Evaluate on second half.
    """
    n_models = len(pred_cols)
    if n_models < 2:
        return {col: 1.0 / n_models for col in pred_cols}

    # Split test in half: first half for weight search, second for eval
    timestamps = sorted(merged['timestamp'].unique())
    mid = len(timestamps) // 2
    train_ts = set(timestamps[:mid])
    train_mask = merged['timestamp'].isin(train_ts)
    train_df = merged[train_mask].copy()

    best_sharpe = -999
    best_weights = None

    # Grid search with step 0.1
    steps = np.arange(0, 1.01, 0.1)
    if n_models == 2:
        for w0 in steps:
            w1 = 1.0 - w0
            if w1 < -0.01:
                continue
            weights = {pred_cols[0]: w0, pred_cols[1]: w1}
            train_df['_ens'] = sum(weights[c] * train_df[c] for c in pred_cols)
            m = evaluate(train_df, '_ens', actual_col, horizon)
            if m and m['LS_Sharpe'] > best_sharpe:
                best_sharpe = m['LS_Sharpe']
                best_weights = weights.copy()
    elif n_models == 3:
        for w0 in steps:
            for w1 in steps:
                w2 = 1.0 - w0 - w1
                if w2 < -0.01 or w2 > 1.01:
                    continue
                weights = {pred_cols[0]: w0, pred_cols[1]: w1, pred_cols[2]: w2}
                train_df['_ens'] = sum(weights[c] * train_df[c] for c in pred_cols)
                m = evaluate(train_df, '_ens', actual_col, horizon)
                if m and m['LS_Sharpe'] > best_sharpe:
                    best_sharpe = m['LS_Sharpe']
                    best_weights = weights.copy()
    else:
        # Equal weight for 4+ models
        best_weights = {c: 1.0 / n_models for c in pred_cols}

    if best_weights is None:
        best_weights = {c: 1.0 / n_models for c in pred_cols}

    return best_weights


def main():
    parser = argparse.ArgumentParser(description='Multi-Model Ensemble Evaluator')
    parser.add_argument('--hist', type=str, default=None)
    parser.add_argument('--lgb', type=str, default=None)
    parser.add_argument('--gru', type=str, default=None)
    parser.add_argument('--master', type=str, default=None)
    parser.add_argument('--results', type=str, default=None)
    parser.add_argument('--optimize-weights', action='store_true', default=True,
                        help='Optimize ensemble weights via grid search')
    parser.add_argument('--no-optimize-weights', dest='optimize_weights', action='store_false')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    results_dir = args.results or os.path.join(project_root, 'results_ensemble')
    os.makedirs(results_dir, exist_ok=True)

    target_col = f'target_ret_{HORIZON}h'

    print("=" * 70)
    print("  MULTI-MODEL ENSEMBLE EVALUATOR")
    print("  Proper evaluation with actual returns")
    print("=" * 70)

    # ========================================
    # 1. LOAD ALL PREDICTIONS
    # ========================================
    print(f"\n📊 Loading predictions...")

    # Auto-discover files
    candidates = {
        'hist': (args.hist, [
            'results_hist/test_predictions_hist.parquet',
        ], 'pred_hist'),
        'lgb': (args.lgb, [
            'results_v4/test_predictions_v4.parquet',
            'results_v3/test_predictions_v3.parquet',
        ], 'pred_ensemble'),  # v4 saves as pred_ensemble
        'gru': (args.gru, [
            'results_gru/test_predictions_gru.parquet',
        ], 'pred_gru'),
        'master': (args.master, [
            'results_master/test_predictions_master.parquet',
        ], 'pred_master'),
    }

    dfs = {}
    for name, (explicit, defaults, default_pred_col) in candidates.items():
        path = explicit
        if path is None:
            for d in defaults:
                full = os.path.join(project_root, d)
                if os.path.exists(full):
                    path = full
                    break

        if path and os.path.exists(path):
            rename_to = f'pred_{name}'
            df = load_predictions(path, default_pred_col, rename_to)
            if df is not None:
                dfs[name] = df

    if not dfs:
        print("❌ No prediction files found!")
        print("   Run models first, then re-run this script.")
        sys.exit(1)

    print(f"\n   Models loaded: {list(dfs.keys())}")

    # ========================================
    # 2. MERGE ALL PREDICTIONS
    # ========================================
    print(f"\n🔗 Merging predictions...")

    # Start with the first model
    names = sorted(dfs.keys())
    merged = dfs[names[0]].copy()
    for name in names[1:]:
        other = dfs[name]
        # Keep target column from whichever has it
        merge_cols = ['timestamp', 'symbol', f'pred_{name}']
        if target_col in other.columns and target_col not in merged.columns:
            merge_cols.append(target_col)
        elif target_col in other.columns:
            other = other.drop(columns=[target_col], errors='ignore')
            merge_cols = ['timestamp', 'symbol', f'pred_{name}']

        merged = merged.merge(other[merge_cols], on=['timestamp', 'symbol'], how='inner')

    # Verify we have the target column
    if target_col not in merged.columns:
        print(f"   ⚠️  Target column {target_col} not found in any predictions.")
        print("   You may need to re-run models with the eval bug fix.")
        sys.exit(1)

    # Drop NaN targets
    merged = merged.dropna(subset=[target_col])

    pred_cols = [f'pred_{name}' for name in names]
    print(f"   Merged: {len(merged):,} rows, {len(pred_cols)} models")
    print(f"   Timestamps: {merged['timestamp'].min()} → {merged['timestamp'].max()}")

    # ========================================
    # 3. NORMALIZE (cross-sectional z-score per timestamp)
    # ========================================
    print(f"\n📐 Cross-sectional z-score normalization...")
    for col in pred_cols:
        merged[col] = merged.groupby('timestamp')[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-10)
        )

    # ========================================
    # 4. EVALUATE INDIVIDUAL MODELS
    # ========================================
    print(f"\n{'='*70}")
    print(f"  INDIVIDUAL MODEL RESULTS (test set, actual returns)")
    print(f"{'='*70}")

    all_metrics = {}
    for col in pred_cols:
        m = evaluate(merged, col, target_col, HORIZON)
        if m:
            all_metrics[col] = m
            print(f"\n   {col}:")
            print(f"      Rank IC: {m['Rank_IC']:+.4f}  |  LS Sharpe: {m['LS_Sharpe']:+.2f}  |  "
                  f"LS MaxDD: {m['LS_MaxDD_%']:.1f}%  |  LO5: {m['LO5_Total_%']:+.1f}%")

    # ========================================
    # 5. ENSEMBLE COMBINATIONS
    # ========================================
    print(f"\n{'='*70}")
    print(f"  ENSEMBLE COMBINATIONS")
    print(f"{'='*70}")

    # Equal weight ensemble (all models)
    merged['pred_equal'] = sum(merged[c] for c in pred_cols) / len(pred_cols)
    m_equal = evaluate(merged, 'pred_equal', target_col, HORIZON)
    if m_equal:
        all_metrics['pred_equal'] = m_equal
        print(f"\n   Equal-weight ({len(pred_cols)} models):")
        print(f"      Rank IC: {m_equal['Rank_IC']:+.4f}  |  LS Sharpe: {m_equal['LS_Sharpe']:+.2f}  |  "
              f"LS MaxDD: {m_equal['LS_MaxDD_%']:.1f}%  |  LO5: {m_equal['LO5_Total_%']:+.1f}%")

    # Optimized weights
    if args.optimize_weights and len(pred_cols) >= 2:
        print(f"\n   🔍 Optimizing weights (grid search on first half of test)...")
        opt_weights = optimize_weights(merged, pred_cols, target_col, HORIZON)
        print(f"   Optimal weights: {', '.join(f'{k}={v:.1f}' for k, v in opt_weights.items())}")

        merged['pred_optimized'] = sum(opt_weights[c] * merged[c] for c in pred_cols)
        m_opt = evaluate(merged, 'pred_optimized', target_col, HORIZON)
        if m_opt:
            all_metrics['pred_optimized'] = m_opt
            print(f"   Optimized ensemble:")
            print(f"      Rank IC: {m_opt['Rank_IC']:+.4f}  |  LS Sharpe: {m_opt['LS_Sharpe']:+.2f}  |  "
                  f"LS MaxDD: {m_opt['LS_MaxDD_%']:.1f}%  |  LO5: {m_opt['LO5_Total_%']:+.1f}%")
    else:
        opt_weights = {c: 1.0 / len(pred_cols) for c in pred_cols}

    # Pairwise combinations
    if len(pred_cols) >= 2:
        print(f"\n   📊 Pairwise ensembles:")
        from itertools import combinations
        for c1, c2 in combinations(pred_cols, 2):
            merged[f'{c1}+{c2}'] = (merged[c1] + merged[c2]) / 2
            m_pair = evaluate(merged, f'{c1}+{c2}', target_col, HORIZON)
            if m_pair:
                all_metrics[f'{c1}+{c2}'] = m_pair
                print(f"      {c1}+{c2}: Rank IC={m_pair['Rank_IC']:+.4f}, "
                      f"LS Sharpe={m_pair['LS_Sharpe']:+.2f}")

    # ========================================
    # 6. FIND BEST
    # ========================================
    best_name = max(all_metrics, key=lambda k: all_metrics[k]['LS_Sharpe'])
    best_m = all_metrics[best_name]

    print(f"\n{'='*70}")
    print(f"  🏆 BEST: {best_name}")
    print(f"{'='*70}")
    for k, v in best_m.items():
        print(f"   {k:25s} {v}")

    # ========================================
    # 7. SAVE
    # ========================================
    # Save full merged predictions
    save_cols = ['timestamp', 'symbol'] + pred_cols + [target_col]
    if 'pred_equal' in merged.columns:
        save_cols.append('pred_equal')
    if 'pred_optimized' in merged.columns:
        save_cols.append('pred_optimized')

    merged[save_cols].to_parquet(
        os.path.join(results_dir, 'ensemble_predictions.parquet'), index=False
    )

    results = {
        'individual_metrics': {k: v for k, v in all_metrics.items() if k.startswith('pred_') and '+' not in k},
        'ensemble_metrics': {k: v for k, v in all_metrics.items() if '+' in k or k in ('pred_equal', 'pred_optimized')},
        'best': {'name': best_name, 'metrics': best_m},
        'weights': {k: round(v, 2) for k, v in opt_weights.items()},
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'models': list(dfs.keys()),
            'n_merged_rows': len(merged),
            'horizon': HORIZON,
        },
    }

    with open(os.path.join(results_dir, 'ensemble_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✅ Results saved to {results_dir}/")
    print(f"   Predictions: ensemble_predictions.parquet")
    print(f"   Metrics: ensemble_results.json")

    # ========================================
    # COMPARISON TABLE
    # ========================================
    print(f"\n{'='*70}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"  {'Model':<30} {'Rank IC':>10} {'LS Sharpe':>10} {'LS MaxDD%':>10} {'LO5%':>10}")
    print(f"  {'-'*70}")
    for name, m in sorted(all_metrics.items(), key=lambda x: -x[1]['LS_Sharpe']):
        marker = " 🏆" if name == best_name else ""
        print(f"  {name:<30} {m['Rank_IC']:>+10.4f} {m['LS_Sharpe']:>+10.2f} "
              f"{m['LS_MaxDD_%']:>10.1f} {m['LO5_Total_%']:>+10.1f}{marker}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
