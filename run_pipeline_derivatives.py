#!/usr/bin/env python3
"""
Derivatives-Only Mini-Model — a focused expert using ONLY Binance Futures data.

Trains a small, heavily-regularized LightGBM that uses only:
  - OI features (8): oi_change_1h/4h/12h/24h, oi_zscore_7d,
    oi_ret_interaction/12h, oi_change_12h_cs
  - Taker features (6): taker_buy_sell_ratio, taker_imbalance,
    taker_cvd_12h/24h, taker_flow_zscore, taker_imbalance_cs
  - L/S features (8): top_ls_ratio, top_long_pct, top_ls_change_12h/24h,
    top_ls_zscore, global_ls_ratio, global_long_pct, ls_divergence
  - Funding (2): funding_rate_binance, funding_surprise

Plus a TINY set of "context" features:
  - btc_ret_12h, btc_vol_24h (market context)
  - regime_composite, breadth_pct_positive (market regime)
  - ret_12h, vol_12h_cs_rank (coin-level momentum, no overlap with derivatives)

Total: ~30 features.  Heavily regularized to avoid overfitting.
The idea: this model is DECORRELATED from the main LGB (which is FNG/price-driven)
and CatBoost, so adding it to the ensemble improves diversification.

Usage:
  python run_pipeline_derivatives.py --skip-hpo
  python run_pipeline_derivatives.py --skip-hpo --results results/exp12_deriv_only
  python run_pipeline_derivatives.py --production
"""

import sys
import os
import argparse
import json
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# Import shared functions from v6
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_pipeline_v6 import (
    HORIZON, N_SEEDS, SEEDS, PURGE_DAYS,
    WALK_FORWARD_WINDOWS, PRODUCTION_WINDOW,
    EXCLUDE_COLS, REGIME_COLS, COST_MODEL,
    add_multi_horizon_targets, add_cross_asset_features,
    add_advanced_regime_features, add_12h_features,
    add_derivatives_features,
    create_rank_target, evaluate_model,
    _compute_groups,
)

# ─── Derivatives-specific config ──────────────────────────────

# These are the ONLY feature columns allowed (derivatives + minimal context)
DERIV_FEATURES = [
    # OI (8)
    'oi_change_1h', 'oi_change_4h', 'oi_change_12h', 'oi_change_24h',
    'oi_zscore_7d', 'oi_ret_interaction', 'oi_ret_interaction_12h',
    'oi_change_12h_cs',
    # Taker (6)
    'taker_buy_sell_ratio', 'taker_imbalance',
    'taker_cvd_12h', 'taker_cvd_24h',
    'taker_flow_zscore', 'taker_imbalance_cs',
    # Top Trader L/S (5)
    'top_ls_ratio', 'top_long_pct',
    'top_ls_change_12h', 'top_ls_change_24h', 'top_ls_zscore',
    # Global L/S (3)
    'global_ls_ratio', 'global_long_pct', 'ls_divergence',
    # Funding (2)
    'funding_rate_binance', 'funding_surprise',
    # Market-wide aggregates (4) — NEW
    'agg_oi_change_12h', 'agg_taker_imbalance',
    'funding_dispersion', 'agg_oi_total_change_12h',
    # Basis / Premium (6) — NEW
    'basis_pct', 'basis_zscore_7d', 'basis_change_12h', 'basis_change_24h',
    'basis_cs_rank', 'basis_funding_divergence',
    # Liquidations (10) — NEW
    'liq_long_usd', 'liq_short_usd', 'liq_total_usd',
    'liq_imbalance', 'liq_cascade_12h', 'liq_cascade_24h',
    'liq_imbalance_12h', 'liq_total_zscore', 'liq_ret_interaction',
    'agg_liq_zscore',
    # Context — market (4)
    'btc_ret_12h', 'btc_vol_24h',
    'regime_composite', 'breadth_pct_positive',
    # Context — coin-level (2)
    'ret_12h', 'vol_12h_cs_rank',
]


def train_deriv_lgbm(X_train, y_train, X_val, y_val, seed=42):
    """Train a heavily-regularized small LGB for derivatives-only signal.

    Key differences from main pipeline:
    - Higher min_child_samples (500 vs 200) — derivatives are noisy
    - Lower num_leaves (15 vs 31) — simpler model
    - Higher L1/L2 (3.0 vs 1.0) — stronger regularization
    - Lower feature_fraction (0.7 vs 0.5) — we have fewer features
    """
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'verbosity': -1,
        'n_estimators': 3000,          # fewer trees than main model
        'learning_rate': 0.01,
        'max_depth': 5,                # shallower
        'num_leaves': 15,              # simpler splits
        'feature_fraction': 0.7,       # see more features per tree (we have few)
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'min_child_samples': 500,      # stronger regularization (noisy data)
        'lambda_l1': 3.0,              # 3x stronger than main model
        'lambda_l2': 3.0,
        'min_gain_to_split': 0.02,     # 2x stronger
        'random_state': seed,
        'n_jobs': -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    return model


def train_multi_seed(X_train, y_train, X_val, y_val, X_test, seeds=None):
    """Train N seeds and return ensemble prediction."""
    seeds = seeds or SEEDS
    print(f"\n   🌱 Multi-seed ensemble ({len(seeds)} seeds) [Derivatives-Only]...")

    all_preds = []
    all_models = []
    for i, seed in enumerate(seeds):
        print(f"      Seed {seed} ({i+1}/{len(seeds)})...", end=" ")
        model = train_deriv_lgbm(X_train, y_train, X_val, y_val, seed=seed)
        preds = model.predict(X_test)
        all_preds.append(preds)
        all_models.append(model)
        print(f"iters={model.best_iteration_}")

    ensemble_pred = np.mean(all_preds, axis=0)
    return ensemble_pred, all_models


def main():
    parser = argparse.ArgumentParser(
        description='Derivatives-Only Mini-Model Pipeline')
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--results', type=str, default=None)
    parser.add_argument('--production', action='store_true')
    parser.add_argument('--seeds', type=int, default=N_SEEDS,
                        help=f'Number of seeds (default: {N_SEEDS})')
    parser.add_argument('--skip-hpo', action='store_true',
                        help='Skip HPO (recommended for speed)')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data or os.path.join(project_root, 'data', 'features')
    if args.production:
        results_dir = args.results or os.path.join(
            project_root, 'results', 'production', 'deriv_only')
    else:
        results_dir = args.results or os.path.join(project_root, 'results_deriv')
    os.makedirs(results_dir, exist_ok=True)

    feat_path = os.path.join(data_dir, 'crypto_features_1h.parquet')
    if not os.path.exists(feat_path):
        print(f"❌ Feature file not found: {feat_path}")
        sys.exit(1)

    print("=" * 70)
    print("  DERIVATIVES-ONLY MINI-MODEL")
    print("  Focused expert: OI + Taker + L/S + Funding")
    print("=" * 70)

    # ─── 1. LOAD & ENRICH DATA ──────────────────────────
    print(f"\n📊 Loading data...")
    df = pd.read_parquet(feat_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Shape: {df.shape}, Symbols: {df['symbol'].nunique()}")

    df = add_multi_horizon_targets(df)
    df = add_cross_asset_features(df)
    df = add_advanced_regime_features(df)
    df = add_12h_features(df)
    df = add_derivatives_features(df, project_root)

    # Clean
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['target_ret_12h'])

    # ─── 2. SELECT ONLY DERIVATIVES FEATURES ────────────
    available_feats = [f for f in DERIV_FEATURES if f in df.columns]
    missing_feats = [f for f in DERIV_FEATURES if f not in df.columns]
    if missing_feats:
        print(f"   ⚠️  Missing features ({len(missing_feats)}): {missing_feats[:5]}...")
    feat_cols = available_feats
    print(f"   📋 Using {len(feat_cols)} derivatives + context features")

    # ─── 3. NORMALIZATION ───────────────────────────────
    # Cross-sectional rank for most features (same as main pipeline)
    rank_cols = [c for c in feat_cols if c not in REGIME_COLS]
    for col in rank_cols:
        df[col] = df.groupby('timestamp')[col].transform(
            lambda x: x.rank(pct=True) - 0.5)

    # Create target
    df = create_rank_target(df, HORIZON)
    print(f"   Target: target_rank (rank of {HORIZON}h forward return)")

    # ─── 4. WALK-FORWARD ────────────────────────────────
    windows = [PRODUCTION_WINDOW] if args.production else WALK_FORWARD_WINDOWS
    all_window_results = []
    all_equity_curves = []

    for w_idx, window in enumerate(windows):
        print(f"\n{'='*70}")
        print(f"  Window {w_idx+1}/{len(windows)}: {window['name']}")
        print(f"  Train: → {window['train_end']}  |  "
              f"Test: {window['test_start']} → {window['test_end']}")
        print(f"{'─'*70}")

        train = df[df['timestamp'] < window['train_end']].copy()
        val = df[(df['timestamp'] >= window['val_start']) &
                 (df['timestamp'] < window['val_end'])].copy()
        test = df[(df['timestamp'] >= window['test_start']) &
                  (df['timestamp'] <= window['test_end'])].copy()

        has_test = len(test) > 0
        if not has_test and not args.production:
            print(f"   ⚠️  No test data for this window, skip")
            continue

        print(f"   Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

        # Filter to rows that have derivatives data (OI != 0)
        if 'oi_change_12h' in train.columns:
            train_has_deriv = (train['oi_change_12h'] != 0).sum()
            print(f"   Rows with derivatives data: "
                  f"train={train_has_deriv:,}/{len(train):,} "
                  f"({train_has_deriv/len(train):.1%})")

        X_train, y_train = train[feat_cols], train['target_rank']
        X_val, y_val = val[feat_cols], val['target_rank']
        X_test = test[feat_cols] if has_test else val[feat_cols]

        # ─── Train ──────────────────────────
        ensemble_pred, all_models = train_multi_seed(
            X_train, y_train, X_val, y_val, X_test,
            seeds=SEEDS[:args.seeds],
        )

        if has_test:
            test['pred_deriv'] = ensemble_pred
        last_model = all_models[-1]

        # ─── Feature importance ──────────────
        imp = pd.Series(
            last_model.feature_importances_, index=feat_cols
        ).sort_values(ascending=False)
        print(f"\n   📊 Top 10 features (gain):")
        for fname, fval in imp.head(10).items():
            print(f"      {fname:35s} {fval:8.0f}")

        # ─── Evaluate ───────────────────────
        eval_df = test if has_test else val
        eval_df_copy = eval_df.copy()
        eval_df_copy['pred'] = ensemble_pred

        metrics, ls_rets, ls_vol, ls_dd, ts_vals = evaluate_model(
            eval_df_copy, 'pred', f'target_ret_{HORIZON}h', HORIZON)

        print(f"\n   📊 Results:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"      {k:30s} {v:+.4f}")
            else:
                print(f"      {k:30s} {v}")

        all_window_results.append({
            'window': window['name'],
            'metrics': metrics,
        })

        # Save equity curve
        if ls_dd is not None:
            eq_df = pd.DataFrame({
                'timestamp': ts_vals,
                'ls_ret': ls_rets,
                'dd_stop': ls_dd,
            })
            all_equity_curves.append(eq_df)

        # ─── Save models ────────────────────
        for i, mdl in enumerate(all_models):
            seed = SEEDS[:args.seeds][i]
            model_path = os.path.join(
                results_dir, f'deriv_model_seed_{seed}.txt')
            if hasattr(mdl, 'booster_'):
                mdl.booster_.save_model(model_path)
            else:
                mdl.save_model(model_path)

        # Save feature names
        with open(os.path.join(results_dir, 'feature_names.json'), 'w') as f:
            json.dump(feat_cols, f)

    # ─── Summary ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY — Derivatives-Only Mini-Model")
    print(f"{'='*70}")
    print(f"  Features: {len(feat_cols)}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Windows: {len(all_window_results)}")

    # Aggregate across windows
    if all_window_results:
        result_dict = {
            'pipeline': 'derivatives_only',
            'features_count': len(feat_cols),
            'features': feat_cols,
            'seeds': args.seeds,
            'windows': all_window_results,
        }

        # Compute average metrics
        metric_keys = all_window_results[0]['metrics'].keys()
        avg_metrics = {}
        for k in metric_keys:
            vals = [w['metrics'][k] for w in all_window_results
                    if isinstance(w['metrics'].get(k), (int, float))]
            if vals:
                avg_metrics[k] = np.mean(vals)
        result_dict['avg_metrics'] = avg_metrics

        print(f"\n  Average across {len(all_window_results)} windows:")
        for k, v in avg_metrics.items():
            if isinstance(v, float):
                print(f"    {k:35s} {v:+.4f}")

        # Save results
        results_path = os.path.join(results_dir, 'all_results_deriv.json')
        with open(results_path, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        print(f"\n  💾 Results saved → {results_path}")

        # Save equity curves
        if all_equity_curves:
            eq_combined = pd.concat(all_equity_curves, ignore_index=True)
            eq_path = os.path.join(
                results_dir, 'equity_curves_deriv.parquet')
            eq_combined.to_parquet(eq_path, index=False)

        # Save feature importance
        if last_model is not None:
            imp_df = pd.DataFrame({
                'feature': feat_cols,
                'importance': last_model.feature_importances_,
            }).sort_values('importance', ascending=False)
            imp_path = os.path.join(
                results_dir, 'feature_importance_deriv.csv')
            imp_df.to_csv(imp_path, index=False)

    print(f"\n✅ Done!")


if __name__ == '__main__':
    main()
