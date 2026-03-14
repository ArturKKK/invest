#!/usr/bin/env python3
"""
Analyze overnight v2 results: IC, correlation, and model comparison.

Loads all model variants (baseline, lambdarank, residual) and computes:
  1. Per-model cross-sectional IC (Spearman corr with forward returns)
  2. Inter-model correlation matrix (how diverse are predictions)
  3. Ensemble IC for different combinations
  4. Auto-verdict: which models to keep

Usage:
    python _analyze_overnight_v2.py [--output overnight_v2_results/analysis_report.json]
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = Path(__file__).resolve().parent


def rank_ic(pred, target):
    """Cross-sectional Spearman rank IC."""
    mask = np.isfinite(pred) & np.isfinite(target)
    if mask.sum() < 10:
        return np.nan
    return scipy_stats.spearmanr(pred[mask], target[mask]).statistic


def load_model_group(directory, model_type='lgb'):
    """Load a model group from a directory."""
    p = ROOT / directory
    if not p.is_dir():
        return None, None

    if model_type == 'lgb':
        files = sorted(p.glob('lgb_model_seed_*.txt'))
        if not files:
            return None, None
        models = [lgb.Booster(model_file=str(f)) for f in files]
        feats = models[0].feature_name()
        return models, feats

    elif model_type == 'cb':
        files = sorted(p.glob('cb_model_seed_*.cbm'))
        if not files:
            return None, None
        from catboost import CatBoostRegressor
        models = [CatBoostRegressor().load_model(str(f)) for f in files]
        fn_path = p / 'feature_names.json'
        if fn_path.exists():
            with open(fn_path) as f:
                feats = json.load(f)
        else:
            feats = models[0].feature_names_
        return models, feats

    elif model_type == 'xgb':
        files = sorted(p.glob('xgb_model_seed_*.json'))
        if not files:
            return None, None
        import xgboost as xgb_lib
        models = [xgb_lib.Booster(model_file=str(f)) for f in files]
        fn_path = p / 'feature_names.json'
        if fn_path.exists():
            with open(fn_path) as f:
                feats = json.load(f)
        else:
            feats = models[0].feature_names
        return models, feats

    return None, None


def predict_group(models, feats, df, model_type='lgb'):
    """Mean prediction across seeds."""
    for c in feats:
        if c not in df.columns:
            df[c] = 0.0
    X = df[feats].values

    if model_type == 'xgb':
        import xgboost as xgb_lib
        dm = xgb_lib.DMatrix(X, feature_names=feats)
        return np.mean([m.predict(dm) for m in models], axis=0)
    elif model_type == 'cb':
        return np.mean([m.predict(X) for m in models], axis=0)
    else:
        return np.mean([m.predict(X) for m in models], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='overnight_v2_results/analysis_report.json')
    parser.add_argument('--start-date', type=str, default='2026-02-09')
    parser.add_argument('--end-date', type=str, default='2026-03-07')
    args = parser.parse_args()

    print("=" * 70)
    print("  MODEL COMPARISON — Overnight v2 Analysis")
    print("=" * 70)

    # ── 1. Load data ──
    feat_path = ROOT / 'data' / 'features' / 'crypto_features_1h.parquet'
    if not feat_path.exists():
        print(f"❌ {feat_path} not found"); sys.exit(1)

    print(f"\n📦 Loading features...")
    df = pd.read_parquet(feat_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    from run_pipeline_v6 import (
        add_multi_horizon_targets, add_cross_asset_features,
        add_advanced_regime_features, add_derivatives_features,
        add_sentiment_features,
    )
    from run_trading import (
        add_12h_features, cross_sectional_rank, EXCLUDE_COLS,
    )

    print("🔧 Enriching features...")
    df = add_multi_horizon_targets(df)
    df = add_cross_asset_features(df)
    df = add_advanced_regime_features(df)
    df = add_12h_features(df)
    df = add_sentiment_features(df, str(ROOT), news_mode='all')
    df = add_derivatives_features(df, str(ROOT))

    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS
                 and not c.startswith('target_')
                 and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]
    df = cross_sectional_rank(df, feat_cols)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[feat_cols] = df[feat_cols].fillna(0)

    # Filter to OOS
    sd = pd.Timestamp(args.start_date, tz='UTC')
    ed = pd.Timestamp(args.end_date, tz='UTC')
    df = df[(df['timestamp'] >= sd) & (df['timestamp'] <= ed)].copy()
    print(f"   OOS period: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"   Rows: {df.shape[0]:,}, Timestamps: {df['timestamp'].nunique()}")

    if 'target_ret_12h' not in df.columns:
        print("❌ target_ret_12h not found"); sys.exit(1)

    # ── 2. Load all model variants ──
    print(f"\n📡 Loading model variants...")
    model_variants = {}

    # Baseline (no-calendar Gen#3)
    variant_dirs = {
        'v6_base':       ('results_v6_prod', 'lgb'),
        'v7_base':       ('results_v7_prod', 'lgb'),
        'cb_base':       ('results_catboost_prod', 'cb'),
        'xgb_base':      ('results_xgboost_prod', 'xgb'),
        'v6_24h':        ('results_v6_24h_prod', 'lgb'),
        # LambdaRank experiments
        'v6_rank':       ('results_v6_rank_prod', 'lgb'),
        'v7_rank':       ('results_v7_rank_prod', 'lgb'),
        # Residual experiments
        'v6_resid':      ('results_v6_resid_prod', 'lgb'),
        'v7_resid':      ('results_v7_resid_prod', 'lgb'),
    }

    for name, (dir_name, mtype) in variant_dirs.items():
        models, feats = load_model_group(dir_name, model_type=mtype)
        if models is not None:
            model_variants[name] = (models, feats, mtype)
            print(f"   ✓ {name:15s}: {len(models)} models, {len(feats)} feats from {dir_name}")
        else:
            print(f"   ✗ {name:15s}: not found ({dir_name})")

    if len(model_variants) < 3:
        print(f"❌ Need at least 3 model variants, got {len(model_variants)}")
        sys.exit(1)

    # ── 3. Generate predictions on all OOS timestamps ──
    print(f"\n🔮 Generating predictions...")
    timestamps = sorted(df['timestamp'].unique())
    # Sample every 12h for IC (same as sim rebalance)
    timestamps = timestamps[::12]
    print(f"   {len(timestamps)} rebalance timestamps")

    all_preds = {name: [] for name in model_variants}
    all_targets = []
    all_ts_labels = []

    for ti, ts in enumerate(timestamps):
        snap = df[df['timestamp'] == ts].copy()
        if len(snap) < 10:
            continue

        target = snap['target_ret_12h'].values
        if np.isnan(target).all():
            continue

        for name, (models, feats, mtype) in model_variants.items():
            preds = predict_group(models, feats, snap, model_type=mtype)
            all_preds[name].append(preds)

        all_targets.append(target)
        all_ts_labels.append(ts)

        if (ti + 1) % 50 == 0:
            print(f"   ... {ti+1}/{len(timestamps)}")

    n_steps = len(all_targets)
    print(f"   Done: {n_steps} valid steps")

    # ── 4. Per-model IC (cross-sectional, per step) ──
    print(f"\n{'='*70}")
    print(f"  CROSS-SECTIONAL IC (Spearman, per 12h step)")
    print(f"{'='*70}")

    ic_per_model = {}
    for name in model_variants:
        ics = []
        for si in range(n_steps):
            ic = rank_ic(all_preds[name][si], all_targets[si])
            if not np.isnan(ic):
                ics.append(ic)
        mean_ic = np.mean(ics) if ics else 0.0
        std_ic = np.std(ics) if ics else 0.0
        icir = mean_ic / (std_ic + 1e-10)
        ic_per_model[name] = {
            'mean': float(mean_ic),
            'std': float(std_ic),
            'icir': float(icir),
            'n_valid': len(ics),
        }

    # Sort by mean IC
    sorted_models = sorted(ic_per_model.items(), key=lambda x: -x[1]['mean'])
    print(f"\n  {'Model':>15s} {'Mean IC':>10s} {'Std IC':>8s} {'ICIR':>8s} {'Verdict':>10s}")
    print(f"  {'-'*15} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")
    for name, stats in sorted_models:
        verdict = "✓ keep" if stats['mean'] > 0.05 else ("~ maybe" if stats['mean'] > 0.02 else "✗ drop")
        print(f"  {name:>15s} {stats['mean']:>10.4f} {stats['std']:>8.4f} "
              f"{stats['icir']:>8.2f} {verdict:>10s}")

    # ── 5. Inter-model correlation (average cross-sectional pred correlation) ──
    print(f"\n{'='*70}")
    print(f"  INTER-MODEL PREDICTION CORRELATION")
    print(f"{'='*70}")

    model_names = list(model_variants.keys())
    n_models = len(model_names)
    corr_matrix = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                corr_matrix[i, j] = 1.0
                continue
            corrs = []
            for si in range(n_steps):
                pi = all_preds[model_names[i]][si]
                pj = all_preds[model_names[j]][si]
                mask = np.isfinite(pi) & np.isfinite(pj)
                if mask.sum() >= 10:
                    c = scipy_stats.spearmanr(pi[mask], pj[mask]).statistic
                    if not np.isnan(c):
                        corrs.append(c)
            corr_matrix[i, j] = np.mean(corrs) if corrs else 0.0

    # Print correlation matrix
    header = f"  {'':>15s}" + "".join(f"{n[:7]:>8s}" for n in model_names)
    print(f"\n{header}")
    for i, name in enumerate(model_names):
        row = f"  {name:>15s}"
        for j in range(n_models):
            c = corr_matrix[i, j]
            row += f"{c:>8.3f}"
        print(row)

    # ── 6. Ensemble IC for key combinations ──
    print(f"\n{'='*70}")
    print(f"  ENSEMBLE IC — Key Combinations")
    print(f"{'='*70}")

    ensembles = {}

    # Define ensemble combos to test
    combos = {}

    # Baseline 4-group
    base4 = [n for n in ['v6_base', 'v7_base', 'cb_base', 'xgb_base'] if n in model_variants]
    if len(base4) >= 3:
        combos['Base 4-grp'] = base4

    # Baseline + 24h
    if 'v6_24h' in model_variants:
        combos['Base 4+24h'] = base4 + ['v6_24h']

    # LambdaRank (replace v6+v7 with ranked)
    rank_models = [n for n in ['v6_rank', 'v7_rank'] if n in model_variants]
    if rank_models:
        rank_combo = rank_models + [n for n in ['cb_base', 'xgb_base'] if n in model_variants]
        combos['Rank 4-grp'] = rank_combo
        if 'v6_24h' in model_variants:
            combos['Rank 4+24h'] = rank_combo + ['v6_24h']

    # Residual (replace v6+v7 with residual)
    resid_models = [n for n in ['v6_resid', 'v7_resid'] if n in model_variants]
    if resid_models:
        resid_combo = resid_models + [n for n in ['cb_base', 'xgb_base'] if n in model_variants]
        combos['Resid 4-grp'] = resid_combo
        if 'v6_24h' in model_variants:
            combos['Resid 4+24h'] = resid_combo + ['v6_24h']

    # Mixed: rank v6 + resid v7 + cb + xgb
    if 'v6_rank' in model_variants and 'v7_resid' in model_variants:
        mixed = ['v6_rank', 'v7_resid'] + [n for n in ['cb_base', 'xgb_base'] if n in model_variants]
        combos['Mixed 4-grp'] = mixed
        if 'v6_24h' in model_variants:
            combos['Mixed 4+24h'] = mixed + ['v6_24h']

    # All models combined
    combos['ALL models'] = list(model_variants.keys())

    print(f"\n  {'Ensemble':>15s} {'Mean IC':>10s} {'ICIR':>8s} {'Models':>8s} {'Members'}")
    print(f"  {'-'*15} {'-'*10} {'-'*8} {'-'*8} {'-'*30}")

    for label, members in combos.items():
        ics = []
        for si in range(n_steps):
            # Simple mean of z-scored predictions
            zscores = []
            for name in members:
                p = all_preds[name][si]
                mu, s = p.mean(), p.std() + 1e-10
                zscores.append((p - mu) / s)
            ens = np.mean(zscores, axis=0)
            ic = rank_ic(ens, all_targets[si])
            if not np.isnan(ic):
                ics.append(ic)

        mean_ic = np.mean(ics) if ics else 0.0
        std_ic = np.std(ics) if ics else 0.0
        icir = mean_ic / (std_ic + 1e-10)
        ensembles[label] = {
            'mean_ic': float(mean_ic),
            'std_ic': float(std_ic),
            'icir': float(icir),
            'members': members,
        }
        print(f"  {label:>15s} {mean_ic:>10.4f} {icir:>8.2f} {len(members):>8d}  "
              f"{', '.join(members)}")

    # Find best ensemble
    if ensembles:
        best_label = max(ensembles, key=lambda x: ensembles[x]['icir'])
        best = ensembles[best_label]
        print(f"\n  🏆 Best ensemble: {best_label} (ICIR={best['icir']:.2f}, IC={best['mean_ic']:.4f})")

    # ── 7. Diversity analysis — unique contribution ──
    print(f"\n{'='*70}")
    print(f"  UNIQUE CONTRIBUTION — IC drop when removing each model")
    print(f"{'='*70}")

    if len(base4) >= 3:
        all_names = list(model_variants.keys())
        full_ics = []
        for si in range(n_steps):
            zscores = []
            for name in all_names:
                p = all_preds[name][si]
                mu, s = p.mean(), p.std() + 1e-10
                zscores.append((p - mu) / s)
            ens = np.mean(zscores, axis=0)
            ic = rank_ic(ens, all_targets[si])
            if not np.isnan(ic):
                full_ics.append(ic)
        full_mean_ic = np.mean(full_ics)

        print(f"\n  Full ensemble IC: {full_mean_ic:.4f} ({len(all_names)} models)")
        print(f"\n  {'Remove':>15s} {'New IC':>10s} {'IC Drop':>10s} {'Unique?':>10s}")
        print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")

        for drop_name in all_names:
            remaining = [n for n in all_names if n != drop_name]
            ics = []
            for si in range(n_steps):
                zscores = []
                for name in remaining:
                    p = all_preds[name][si]
                    mu, s = p.mean(), p.std() + 1e-10
                    zscores.append((p - mu) / s)
                ens = np.mean(zscores, axis=0)
                ic = rank_ic(ens, all_targets[si])
                if not np.isnan(ic):
                    ics.append(ic)
            new_ic = np.mean(ics)
            drop = full_mean_ic - new_ic
            unique = "✓ unique" if drop > 0.002 else ("~ marginal" if drop > 0 else "✗ redundant")
            print(f"  {drop_name:>15s} {new_ic:>10.4f} {drop:>+10.4f} {unique:>10s}")

    # ── 8. Save report ──
    report = {
        'oos_period': f"{args.start_date} → {args.end_date}",
        'n_steps': n_steps,
        'n_models': len(model_variants),
        'per_model_ic': ic_per_model,
        'correlation_matrix': {
            'names': model_names,
            'values': corr_matrix.tolist(),
        },
        'ensemble_ic': ensembles,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n💾 Report saved → {output_path}")

    # ── 9. Final verdict ──
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")

    # Check if LambdaRank helps
    if 'v6_rank' in ic_per_model and 'v6_base' in ic_per_model:
        rank_delta = ic_per_model['v6_rank']['mean'] - ic_per_model['v6_base']['mean']
        verdict = "✅ HELPS" if rank_delta > 0.005 else ("⚠️  MARGINAL" if rank_delta > 0 else "❌ HURTS")
        print(f"\n  LambdaRank IC delta vs baseline: {rank_delta:+.4f} → {verdict}")

    if 'v6_resid' in ic_per_model and 'v6_base' in ic_per_model:
        resid_delta = ic_per_model['v6_resid']['mean'] - ic_per_model['v6_base']['mean']
        verdict = "✅ HELPS" if resid_delta > 0.005 else ("⚠️  MARGINAL" if resid_delta > 0 else "❌ HURTS")
        print(f"  Residual IC delta vs baseline:   {resid_delta:+.4f} → {verdict}")

    # Check diversity
    if 'v6_rank' in model_variants and 'v6_base' in model_variants:
        i = model_names.index('v6_rank') if 'v6_rank' in model_names else -1
        j = model_names.index('v6_base') if 'v6_base' in model_names else -1
        if i >= 0 and j >= 0:
            corr_rank_base = corr_matrix[i, j]
            print(f"  LambdaRank ↔ Baseline correlation: {corr_rank_base:.3f} "
                  f"({'✅ diverse' if corr_rank_base < 0.8 else '⚠️  similar'})")

    if 'v6_resid' in model_variants and 'v6_base' in model_variants:
        i = model_names.index('v6_resid') if 'v6_resid' in model_names else -1
        j = model_names.index('v6_base') if 'v6_base' in model_names else -1
        if i >= 0 and j >= 0:
            corr_resid_base = corr_matrix[i, j]
            print(f"  Residual ↔ Baseline correlation:   {corr_resid_base:.3f} "
                  f"({'✅ diverse' if corr_resid_base < 0.8 else '⚠️  similar'})")

    print()


if __name__ == '__main__':
    main()
