#!/usr/bin/env python3
"""
Post-training research analysis.

Run after train_research.sh to get:
  1. Per-model OOS metrics summary (Sharpe, ICIR, DD, win rate)
  2. Model prediction correlation matrix on test periods
  3. Ensemble simulation on OOS data

Usage:
  python analyze_research.py
  python analyze_research.py --leverage 3 --capital 5000
"""

import os, sys, json, glob, argparse, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

RESEARCH_DIRS = {
    'v6':  'results_v6_research',
    'v7':  'results_v7_research',
    'cb':  'results_catboost_research',
    'xgb': 'results_xgboost_research',
}


def load_results_json(results_dir):
    """Load all_results_*.json from a research results directory."""
    for pattern in ['all_results_*.json', 'results_*.json']:
        files = glob.glob(os.path.join(results_dir, pattern))
        if files:
            with open(sorted(files)[-1]) as f:
                return json.load(f)
    return None


def print_oos_metrics():
    """Print OOS test metrics from each model's results JSON."""
    print("=" * 70)
    print("  OOS TEST METRICS (from research windows)")
    print("=" * 70)

    all_metrics = {}
    for label, dirname in RESEARCH_DIRS.items():
        dirpath = os.path.join(PROJECT_ROOT, dirname)
        if not os.path.isdir(dirpath):
            print(f"\n  {label}: ⚠️  {dirname}/ not found — skipping")
            continue

        results = load_results_json(dirpath)
        if not results:
            print(f"\n  {label}: ⚠️  no results JSON found in {dirname}/")
            continue

        print(f"\n  ── {label} ({dirname}/) ──")

        # Actual format: { per_window: [{IC, Rank_IC, ICIR, Rank_ICIR,
        #   LS_Sharpe_net, LS_MaxDD_net_%, LS_Total_net_%, window, ...}], average: {...} }
        windows = results.get('per_window', results.get('windows', []))
        if isinstance(windows, dict):
            windows = list(windows.values())

        for i, w in enumerate(windows):
            if not isinstance(w, dict):
                continue
            name = w.get('window', w.get('name', f'W{i+1}'))
            sharpe = w.get('LS_Sharpe_net', w.get('sharpe', '?'))
            icir = w.get('Rank_ICIR', w.get('ICIR', '?'))
            ic = w.get('Rank_IC', w.get('IC', '?'))
            dd = w.get('LS_MaxDD_net_%', w.get('max_dd', '?'))
            ret = w.get('LS_Total_net_%', w.get('total_ret', '?'))
            n_periods = w.get('N_periods', '?')

            print(f"    {name}:")
            print(f"      Sharpe={_fmt(sharpe)}  ICIR={_fmt(icir)}  "
                  f"IC={_fmt(ic)}  DD={_fmt(dd)}%  Ret={_fmt(ret)}%  "
                  f"N={n_periods}")

            all_metrics.setdefault(label, []).append({
                'window': name, 'sharpe': sharpe, 'icir': icir,
                'ic': ic, 'dd': dd, 'ret': ret
            })

        # Also print average if available
        avg = results.get('average', {})
        if avg:
            print(f"    AVG: Sharpe={_fmt(avg.get('LS_Sharpe_net', '?'))}  "
                  f"ICIR={_fmt(avg.get('Rank_ICIR', '?'))}  "
                  f"DD={_fmt(avg.get('LS_MaxDD_net_%', '?'))}%")

    return all_metrics


def compute_prediction_correlations():
    """Load test predictions from each model and compute cross-correlations."""
    print("\n" + "=" * 70)
    print("  MODEL PREDICTION CORRELATIONS (on OOS test data)")
    print("=" * 70)

    # Try to load test predictions from each research dir
    preds = {}
    for label, dirname in RESEARCH_DIRS.items():
        dirpath = os.path.join(PROJECT_ROOT, dirname)
        if not os.path.isdir(dirpath):
            continue

        # Look for test predictions saved during training
        pred_files = glob.glob(os.path.join(dirpath, 'test_predictions*.csv'))
        if not pred_files:
            pred_files = glob.glob(os.path.join(dirpath, 'predictions*.csv'))
        if not pred_files:
            # Try to load from results JSON
            results = load_results_json(dirpath)
            if results and 'test_predictions' in results:
                preds[label] = np.array(results['test_predictions'])
                continue
            continue

        df_pred = pd.read_csv(sorted(pred_files)[-1])
        if 'pred' in df_pred.columns:
            preds[label] = df_pred['pred'].values
        elif 'prediction' in df_pred.columns:
            preds[label] = df_pred['prediction'].values

    if len(preds) < 2:
        print("\n  ⚠️  Need predictions from at least 2 models for correlation analysis.")
        print("     Predictions not saved during training — consider adding pred export.")
        print("\n  💡 Alternative: run inference check to compute correlations live:")
        print("     python _inference_check2.py")
        return None

    # Align lengths (take minimum)
    min_len = min(len(v) for v in preds.values())
    for k in preds:
        preds[k] = preds[k][:min_len]

    labels = sorted(preds.keys())
    n = len(labels)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr_matrix[i, j] = np.corrcoef(preds[labels[i]], preds[labels[j]])[0, 1]

    print(f"\n  Correlation matrix ({min_len} test samples):\n")
    header = "         " + "  ".join(f"{l:>6}" for l in labels)
    print(header)
    for i, l in enumerate(labels):
        row = f"  {l:>5}  " + "  ".join(f"{corr_matrix[i, j]:6.3f}" for j in range(n))
        print(row)

    # Flag high correlations
    print()
    for i in range(n):
        for j in range(i + 1, n):
            c = corr_matrix[i, j]
            if c > 0.9:
                print(f"  🔴 {labels[i]} ↔ {labels[j]} = {c:.3f} — essentially same model, "
                      f"no diversification benefit")
            elif c > 0.7:
                print(f"  🟡 {labels[i]} ↔ {labels[j]} = {c:.3f} — high correlation, "
                      f"limited diversification")
            else:
                print(f"  🟢 {labels[i]} ↔ {labels[j]} = {c:.3f} — good diversification")

    return corr_matrix


def _fmt(v):
    """Format a metric value for display."""
    if isinstance(v, (int, float)):
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"
    return str(v)


def main():
    parser = argparse.ArgumentParser(description="Analyze research training results")
    parser.add_argument('--leverage', type=float, default=3.0,
                        help='Leverage for ensemble simulation (default: 3)')
    parser.add_argument('--capital', type=float, default=5000.0,
                        help='Starting capital for simulation (default: 5000)')
    args = parser.parse_args()

    # 1. Per-model OOS metrics
    metrics = print_oos_metrics()

    # 2. Prediction correlations
    corr = compute_prediction_correlations()

    # 3. Summary recommendations
    print("\n" + "=" * 70)
    print("  RECOMMENDATIONS")
    print("=" * 70)

    if not metrics:
        print("\n  ⚠️  No OOS metrics found. Run train_research.sh first.")
        return

    # Analyze per-model performance
    good_models = []
    weak_models = []
    for label, wins in metrics.items():
        sharpes = [w['sharpe'] for w in wins if isinstance(w.get('sharpe'), (int, float))]
        if sharpes:
            avg_sharpe = np.mean(sharpes)
            if avg_sharpe > 0.5:
                good_models.append((label, avg_sharpe))
            elif avg_sharpe < 0:
                weak_models.append((label, avg_sharpe))

    if good_models:
        print(f"\n  ✅ Models with positive OOS edge:")
        for label, s in sorted(good_models, key=lambda x: -x[1]):
            print(f"     {label}: avg Sharpe = {s:.2f}")

    if weak_models:
        print(f"\n  ❌ Models with negative OOS edge (consider removing from ensemble):")
        for label, s in sorted(weak_models, key=lambda x: x[1]):
            print(f"     {label}: avg Sharpe = {s:.2f}")

    print("\n  Next steps:")
    print("  1. If any model has Sharpe < 0 on OOS → remove from ensemble")
    print("  2. If v6 ↔ v7 corr > 0.9 → differentiate targets/horizons/features")
    print("  3. Run ensemble sim: python run_fast_sim.py --ensemble --leverage 3 --days 90")
    print("  4. If OOS metrics are good → retrain production: ./train_production.sh")
    print()


if __name__ == '__main__':
    main()
