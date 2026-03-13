#!/usr/bin/env python3
"""
CatBoost Model — Ensemble diversity via ordered boosting.

Uses the same data pipeline and features as LGB v6 (12h target),
but trains CatBoost models instead of LightGBM.
CatBoost adds genuine ensemble diversity through:
  - Ordered boosting (reduces overfitting vs leaf-wise)
  - Symmetric tree structure (different from LGB asymmetric)
  - Different regularization strategy

Saved models are loaded alongside LGB v6/v7 in run_fast_sim.py --ensemble.

Usage:
  python run_pipeline_catboost.py                       # Full run
  python run_pipeline_catboost.py --skip-hpo            # Skip Optuna HPO
  python run_pipeline_catboost.py --single-window       # Quick single-window test
  python run_pipeline_catboost.py --seeds 3             # Fewer seeds (faster)

Requirements:
  pip install catboost pandas numpy scipy pyarrow
  Optional: pip install optuna (for HPO)
"""

import sys
import os
import argparse
import json
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# Import shared feature engineering from v6
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_pipeline_v6 import (
    add_multi_horizon_targets, add_cross_asset_features,
    add_advanced_regime_features, add_12h_features, add_calendar_features, add_sentiment_features,
    add_derivatives_features,
    cross_sectional_rank, create_rank_target, add_residual_targets,
    evaluate_model, vol_target_returns, drawdown_stop_returns,
    compute_costs_per_period,
    EXCLUDE_COLS, REGIME_COLS, WALK_FORWARD_WINDOWS, PRODUCTION_WINDOW, HORIZON, SEEDS, COST_MODEL,
    PURGE_DAYS,
)

N_SEEDS = 5
_task_type = 'CPU'  # overridden by --gpu flag at runtime


# ============================================================
# HPO (CatBoost-specific parameter space)
# ============================================================

def run_optuna_hpo(X_train, y_train, X_val, y_val, val_dates, n_trials=50):
    """Optuna HPO with Rank ICIR objective for CatBoost."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print(f"   ⚠️  Optuna not installed: pip install optuna")
        return None

    print(f"   🔍 Running CatBoost HPO ({n_trials} trials)...")
    unique_dates = np.unique(val_dates)

    def objective(trial):
        params = {
            'iterations': 5000,
            'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.05, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 0.1, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 500),
            'grow_policy': trial.suggest_categorical('grow_policy',
                                                      ['SymmetricTree', 'Depthwise']),
            'random_seed': 42,
            'verbose': 0,
            'task_type': _task_type,
            'loss_function': 'RMSE',
        }

        model = CatBoostRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=0,
        )
        preds = model.predict(X_val)

        daily_ics = []
        for d in unique_dates:
            mask = val_dates == d
            if mask.sum() < 10:
                continue
            p, a = preds[mask], y_val.values[mask]
            valid = ~(np.isnan(p) | np.isnan(a))
            if valid.sum() < 10:
                continue
            c, _ = spearmanr(p[valid], a[valid])
            if not np.isnan(c):
                daily_ics.append(c)

        if len(daily_ics) < 5:
            return 0
        daily_ics = np.array(daily_ics)
        return daily_ics.mean() / (daily_ics.std() + 1e-10)

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"   ✅ Best Rank ICIR: {study.best_value:.4f}")
    return study.best_params


# ============================================================
# TRAINING
# ============================================================

def train_catboost(X_train, y_train, X_val, y_val,
                   feat_names=None, custom_params=None, seed=42):
    """Train a single CatBoost model."""
    base_params = {
        'iterations': 5000,
        'learning_rate': 0.01,
        'depth': 6,
        'l2_leaf_reg': 3.0,
        'random_strength': 1.0,
        'bagging_temperature': 0.5,
        'border_count': 128,
        'min_data_in_leaf': 200,
        'grow_policy': 'SymmetricTree',
        'random_seed': seed,
        'verbose': 0,
        'task_type': _task_type,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
    }
    if custom_params:
        # Map HPO params, skip non-catboost keys
        for k, v in custom_params.items():
            if k in base_params or k in ('learning_rate', 'depth', 'l2_leaf_reg',
                                          'random_strength', 'bagging_temperature',
                                          'border_count', 'min_data_in_leaf',
                                          'grow_policy'):
                base_params[k] = v
    base_params['random_seed'] = seed

    model = CatBoostRegressor(**base_params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=200,
    )
    return model


def train_multi_seed(X_train, y_train, X_val, y_val, X_test,
                     feat_names=None, params=None, seeds=None):
    """Train CatBoost ensemble with multiple seeds."""
    seeds = seeds or SEEDS
    print(f"\n   🌱 CatBoost multi-seed ensemble ({len(seeds)} seeds)...")

    all_preds = []
    all_models = []
    for i, seed in enumerate(seeds):
        print(f"      Seed {seed} ({i+1}/{len(seeds)})...", end=" ")
        model = train_catboost(X_train, y_train, X_val, y_val,
                               feat_names=feat_names,
                               custom_params=params, seed=seed)
        preds = model.predict(X_test)
        all_preds.append(preds)
        all_models.append(model)
        print(f"iters={model.best_iteration_}")

    ensemble_pred = np.mean(all_preds, axis=0)
    return ensemble_pred, all_models


def feature_selection_catboost(model, feat_cols, threshold_pct=20):
    """Feature selection based on CatBoost feature importance."""
    imp = pd.Series(
        model.get_feature_importance(),
        index=feat_cols
    )
    threshold = np.percentile(imp.values, threshold_pct)
    keep = imp[imp > threshold].index.tolist()
    print(f"   🔪 Feature selection: {len(feat_cols)} → {len(keep)}")
    return keep


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CatBoost model for crypto alpha")
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--results', type=str, default=None)
    parser.add_argument('--hpo-trials', type=int, default=50)
    parser.add_argument('--skip-hpo', action='store_true')
    parser.add_argument('--single-window', action='store_true',
                        help='Use only window 3 for quick test')
    parser.add_argument('--production', action='store_true',
                        help='Production mode: max training data, no test holdout')
    parser.add_argument('--train-end', type=str, default=None,
                        help='Override train cutoff date (YYYY-MM-DD) for --production')
    parser.add_argument('--val-end', type=str, default=None,
                        help='Override val end date (YYYY-MM-DD) for --production')
    parser.add_argument('--seeds', type=int, default=N_SEEDS)
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for CatBoost training (requires CUDA)')
    parser.add_argument('--residual-target', action='store_true',
                        help='Use beta-residual returns (remove BTC factor) for target')
    parser.add_argument('--hybrid-norm', action='store_true',
                        help='Hybrid normalization: CS-rank + TS-zscore for spike features')
    parser.add_argument('--no-news', action='store_true',
                        help='Skip loading crypto news features (for clean A/B tests)')
    parser.add_argument('--news-mode', type=str, default='all',
                        choices=['all', 'market-only', 'coin-only', 'none'],
                        help='News feature scope: all, market-only, coin-only, none')
    parser.add_argument('--no-derivatives', action='store_true',
                        help='Skip loading Binance derivatives features (for clean A/B tests)')
    args = parser.parse_args()

    if args.no_news:
        args.news_mode = 'none'

    global _task_type
    _task_type = 'GPU' if args.gpu else 'CPU'

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data or os.path.join(project_root, 'data', 'features')
    if args.production:
        results_dir = args.results or os.path.join(project_root, 'results_catboost_prod')
    else:
        results_dir = args.results or os.path.join(project_root, 'results_catboost')
    os.makedirs(results_dir, exist_ok=True)

    feat_path = os.path.join(data_dir, 'crypto_features_1h.parquet')
    if not os.path.exists(feat_path):
        print(f"❌ Feature file not found: {feat_path}")
        sys.exit(1)

    print("=" * 70)
    print("  CATBOOST CRYPTO ALPHA MODEL")
    print("  12h Target + Ordered Boosting + Walk-Forward")
    print("=" * 70)

    # ========================================
    # 1. LOAD & ENRICH DATA (same pipeline as v6)
    # ========================================
    print(f"\n📊 Loading data...")
    df = pd.read_parquet(feat_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Shape: {df.shape}, Symbols: {df['symbol'].nunique()}")

    df = add_multi_horizon_targets(df)
    df = add_cross_asset_features(df)
    if args.residual_target:
        df = add_residual_targets(df, beta_window=168)
    df = add_advanced_regime_features(df)
    df = add_12h_features(df)
    df = add_calendar_features(df)
    df = add_sentiment_features(df, project_root, news_mode=args.news_mode)
    if not args.no_derivatives:
        df = add_derivatives_features(df, project_root)
    else:
        print("   ⏭️  Skipping derivatives features (--no-derivatives)")

    # Clean infinities
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=['target_ret_12h'])

    # Feature columns
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS
                 and not c.startswith('target_')]
    feat_cols = [c for c in feat_cols if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    print(f"   Features: {len(feat_cols)}")

    df[feat_cols] = df[feat_cols].fillna(0)

    # Cross-sectional rank normalization
    df = cross_sectional_rank(df, feat_cols, hybrid=args.hybrid_norm)
    df = create_rank_target(df, HORIZON, use_excess=args.residual_target)

    print(f"   Final shape: {df.shape}")
    print(f"   Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # ========================================
    # 2. ROLLING WALK-FORWARD
    # ========================================
    if args.production:
        from copy import deepcopy
        prod_win = deepcopy(PRODUCTION_WINDOW)
        if args.train_end:
            prod_win['train_end'] = args.train_end
            te = pd.Timestamp(args.train_end)
            prod_win['val_start'] = (te + pd.Timedelta(days=PURGE_DAYS)).strftime('%Y-%m-%d')
        if args.val_end:
            prod_win['val_end'] = args.val_end
            prod_win['test_start'] = args.val_end
        windows = [prod_win]
        print(f"\n🔴 PRODUCTION MODE — max training data, models go to live trading")
        print(f"   Train: start → {prod_win['train_end']}")
        print(f"   Val:   {prod_win['val_start']} → {prod_win['val_end']}")
        print(f"   Test:  (none — live trading)")
    else:
        windows = WALK_FORWARD_WINDOWS
        if args.single_window:
            windows = [windows[-1]]

    print(f"\n{'='*70}")
    print(f"  ROLLING WALK-FORWARD ({len(windows)} windows)")
    print(f"{'='*70}")

    target_col = f'target_ret_{HORIZON}h'
    all_window_metrics = []
    all_test_predictions = []
    combined_ls_rets = []
    combined_timestamps = []

    for w_idx, window in enumerate(windows):
        print(f"\n{'─'*70}")
        print(f"  Window {w_idx+1}/{len(windows)}: {window['name']}")
        print(f"  Train: → {window['train_end']}")
        print(f"  Val:   {window['val_start']} → {window['val_end']}")
        print(f"  Test:  {window['test_start']} → {window['test_end']}")
        print(f"{'─'*70}")

        train = df[df['timestamp'] < window['train_end']].copy()
        val = df[(df['timestamp'] >= window['val_start']) &
                 (df['timestamp'] < window['val_end'])].copy()
        test = df[(df['timestamp'] >= window['test_start']) &
                  (df['timestamp'] <= window['test_end'])].copy()

        has_test = len(test) > 0
        if not has_test and not args.production:
            print(f"   ⚠️  No test data for this window, skipping")
            continue

        print(f"   Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

        X_train, y_train = train[feat_cols], train['target_rank']
        X_val, y_val = val[feat_cols], val['target_rank']
        X_test = test[feat_cols]
        val_dates = val['timestamp'].dt.date.values

        # --- HPO (only for first window if not skipped) ---
        best_params = None
        if not args.skip_hpo and w_idx == 0:
            best_params = run_optuna_hpo(
                X_train, y_train, X_val, y_val, val_dates,
                n_trials=args.hpo_trials
            )

        # --- Feature selection (train a base model first) ---
        model_base = train_catboost(X_train, y_train, X_val, y_val,
                                     feat_names=feat_cols,
                                     custom_params=best_params)
        selected_feats = feature_selection_catboost(model_base, feat_cols,
                                                     threshold_pct=20)

        # --- Multi-seed ensemble ---
        X_pred = test[selected_feats] if has_test else val[selected_feats]
        ensemble_pred, all_models = train_multi_seed(
            train[selected_feats], y_train,
            val[selected_feats], y_val,
            X_pred,
            feat_names=selected_feats,
            params=best_params,
            seeds=SEEDS[:args.seeds],
        )
        if has_test:
            test['pred_cb'] = ensemble_pred
        last_model = all_models[-1]

        # --- Save trained models ---
        for i, mdl in enumerate(all_models):
            seed = SEEDS[:args.seeds][i]
            model_path = os.path.join(results_dir, f'cb_model_seed_{seed}.cbm')
            mdl.save_model(model_path)

        # Save selected feature names (for loading in fast_sim)
        with open(os.path.join(results_dir, 'feature_names.json'), 'w') as f:
            json.dump(selected_feats, f)
        print(f"   💾 Saved {len(all_models)} CatBoost models + feature names")

        if not has_test:
            print(f"\n   ✅ Production models saved (no test evaluation)")
            if args.production:
                prod_meta = {
                    'mode': 'production',
                    'model_type': 'CatBoost',
                    'train_end': window['train_end'],
                    'val_end': window['val_end'],
                    'n_seeds': args.seeds,
                    'n_features': len(selected_feats),
                    'train_rows': len(train),
                    'val_rows': len(val),
                    'timestamp': datetime.now().isoformat(),
                }
                with open(os.path.join(results_dir, 'production_meta.json'), 'w') as f:
                    json.dump(prod_meta, f, indent=2)
            continue

        # --- Evaluate ---
        metrics, ls_net, ls_vt, ls_dd, timestamps = evaluate_model(
            test, 'pred_cb', target_col, HORIZON, label=window['name']
        )
        metrics['window'] = window['name']
        all_window_metrics.append(metrics)

        save_cols = ['timestamp', 'symbol', target_col, 'pred_cb']
        save_cols = [c for c in save_cols if c in test.columns]
        all_test_predictions.append(test[save_cols].copy())
        combined_ls_rets.extend(ls_net.tolist())
        combined_timestamps.extend(timestamps)

        # Print window results
        print(f"\n   📈 Window {w_idx+1} Results:")
        for k, v in metrics.items():
            if k == 'window':
                continue
            print(f"      {k:30s} {v}")

    # ========================================
    # 3. AGGREGATE ACROSS WINDOWS
    # ========================================
    print(f"\n{'='*70}")
    print(f"  AGGREGATE RESULTS ({len(all_window_metrics)} windows)")
    print(f"{'='*70}")

    if len(all_window_metrics) == 0:
        avg_metrics = {}
        print("   ⚠️  Production mode — no test windows, skipping metrics.")
    elif len(all_window_metrics) > 1:
        metric_keys = [k for k in all_window_metrics[0].keys()
                       if k != 'window' and isinstance(all_window_metrics[0][k], (int, float))]

        avg_metrics = {}
        for k in metric_keys:
            vals = [m[k] for m in all_window_metrics]
            avg_metrics[k] = round(np.mean(vals), 4)
            avg_metrics[f'{k}_std'] = round(np.std(vals), 4)

        print(f"\n   📊 Per-Window Comparison:")
        header = f"   {'Metric':<35s}"
        for m in all_window_metrics:
            header += f" {m['window']:>12s}"
        header += f" {'AVG':>10s}"
        print(header)
        print(f"   {'─'*85}")

        key_metrics = ['Rank_IC', 'Rank_ICIR', 'LS_Sharpe_net', 'LS_MaxDD_net_%',
                        'LS_VolTarget_Sharpe', 'LS_VolTarget_MaxDD_%',
                        'LS_DDStop_Sharpe', 'LS_DDStop_MaxDD_%']
        for k in key_metrics:
            row = f"   {k:<35s}"
            for m in all_window_metrics:
                row += f" {m.get(k, 'N/A'):>12}"
            row += f" {avg_metrics.get(k, 'N/A'):>10}"
            print(row)
    else:
        avg_metrics = {k: v for k, v in all_window_metrics[0].items() if k != 'window'}

    # ========================================
    # 4. COMBINED EQUITY CURVE
    # ========================================
    if len(combined_ls_rets) == 0:
        print("   ⚠️  No test returns — skipping equity curve & combined stats.")
        all_results = {
            'per_window': all_window_metrics,
            'average': avg_metrics,
            'combined': {},
            'cost_model': COST_MODEL,
            'meta': {
                'timestamp': datetime.now().isoformat(),
                'model_type': 'CatBoost',
                'horizon': HORIZON,
                'n_features': len(feat_cols),
                'n_selected': len(selected_feats) if 'selected_feats' in dir() else 0,
                'n_seeds': args.seeds,
                'n_windows': len(windows),
                'hpo_trials': args.hpo_trials if not args.skip_hpo else 0,
                'production_mode': True,
            },
        }
        with open(os.path.join(results_dir, 'all_results_catboost.json'), 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n✅ Production models saved to {results_dir}/")
        return

    combined_ls = np.array(combined_ls_rets)
    periods_per_year = (24 // HORIZON) * 365

    def sharpe(r, ppyr):
        return (r.mean() / (r.std() + 1e-10)) * np.sqrt(ppyr)
    def max_dd(r):
        cum = np.cumprod(1 + r)
        return np.min(cum / np.maximum.accumulate(cum) - 1)
    def total_ret(r):
        return np.prod(1 + r) - 1

    cost = compute_costs_per_period(HORIZON)
    combined_vt = vol_target_returns(combined_ls + cost * 2, lookback=48,
                                      target_vol=0.02, cost_per_period=cost)
    combined_dd = drawdown_stop_returns(combined_ls, max_dd_threshold=-0.25,
                                         recovery_threshold=-0.10)

    print(f"\n   📈 Combined Results (all windows):")
    print(f"      Periods: {len(combined_ls)}")
    print(f"      LS Net:       Sharpe={sharpe(combined_ls, periods_per_year):.2f}, "
          f"MaxDD={max_dd(combined_ls)*100:.1f}%, "
          f"Total={total_ret(combined_ls)*100:.1f}%")
    print(f"      LS VolTarget: Sharpe={sharpe(combined_vt, periods_per_year):.2f}, "
          f"MaxDD={max_dd(combined_vt)*100:.1f}%, "
          f"Total={total_ret(combined_vt)*100:.1f}%")
    print(f"      LS DDStop:    Sharpe={sharpe(combined_dd, periods_per_year):.2f}, "
          f"MaxDD={max_dd(combined_dd)*100:.1f}%, "
          f"Total={total_ret(combined_dd)*100:.1f}%")

    # ========================================
    # 5. FEATURE IMPORTANCE
    # ========================================
    if 'last_model' in dir() and last_model is not None:
        importance = pd.DataFrame({
            'feature': selected_feats,
            'importance': last_model.get_feature_importance(),
        }).sort_values('importance', ascending=False)

        print(f"\n🏆 Top 30 Features (CatBoost, last window):")
        for _, row in importance.head(30).iterrows():
            marker = "📰" if any(s in row['feature'] for s in
                                  ['fng', 'funding', 'reversal', 'surge', 'dispersion',
                                   'long_short', 'beta']) else "  "
            print(f"   {marker} {row['feature']:40s} {row['importance']:.1f}")

        top30_sent = sum(1 for _, r in importance.head(30).iterrows()
                         if any(k in r['feature'] for k in
                                ['fng', 'funding', 'reversal', 'surge', 'dispersion',
                                 'long_short', 'beta', 'crowding']))
        print(f"\n   📰 Sentiment features in top 30: {top30_sent}/30")

    # ========================================
    # 6. SAVE RESULTS
    # ========================================
    all_results = {
        'per_window': all_window_metrics,
        'average': avg_metrics,
        'combined': {
            'LS_Net_Sharpe': round(float(sharpe(combined_ls, periods_per_year)), 2),
            'LS_Net_MaxDD_%': round(float(max_dd(combined_ls) * 100), 1),
            'LS_Net_Total_%': round(float(total_ret(combined_ls) * 100), 1),
            'LS_VolTarget_Sharpe': round(float(sharpe(combined_vt, periods_per_year)), 2),
            'LS_VolTarget_MaxDD_%': round(float(max_dd(combined_vt) * 100), 1),
            'LS_DDStop_Sharpe': round(float(sharpe(combined_dd, periods_per_year)), 2),
            'LS_DDStop_MaxDD_%': round(float(max_dd(combined_dd) * 100), 1),
        },
        'cost_model': COST_MODEL,
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'CatBoost',
            'horizon': HORIZON,
            'n_features': len(feat_cols),
            'n_selected': len(selected_feats) if 'selected_feats' in dir() else 0,
            'n_seeds': args.seeds,
            'n_windows': len(windows),
            'hpo_trials': args.hpo_trials if not args.skip_hpo else 0,
        },
    }

    with open(os.path.join(results_dir, 'all_results_catboost.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    if 'importance' in dir():
        importance.to_csv(os.path.join(results_dir, 'feature_importance_catboost.csv'),
                          index=False)

    if all_test_predictions:
        combined_preds = pd.concat(all_test_predictions, ignore_index=True)
        combined_preds.to_parquet(
            os.path.join(results_dir, 'test_predictions_catboost.parquet'), index=False)

    eq = pd.DataFrame({
        'ls_net': np.cumprod(1 + combined_ls) * 1000,
        'ls_vol_target': np.cumprod(1 + combined_vt) * 1000,
        'ls_dd_stop': np.cumprod(1 + combined_dd) * 1000,
    })
    eq.to_parquet(os.path.join(results_dir, 'equity_curves_catboost.parquet'), index=False)

    # ========================================
    # FINAL VERDICT
    # ========================================
    best_sharpe = avg_metrics.get('LS_VolTarget_Sharpe', avg_metrics.get('LS_Sharpe_net', 0))
    best_dd = avg_metrics.get('LS_VolTarget_MaxDD_%', avg_metrics.get('LS_MaxDD_net_%', 0))

    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY — CatBoost (12h target)")
    print(f"{'='*70}")
    print(f"   Rank IC (avg):            {avg_metrics.get('Rank_IC', 0):+.4f}")
    print(f"   Rank ICIR (avg):          {avg_metrics.get('Rank_ICIR', 0):+.4f}")
    print(f"   LS Sharpe net (avg):      {avg_metrics.get('LS_Sharpe_net', 0):+.2f}")
    print(f"   LS MaxDD net (avg):       {avg_metrics.get('LS_MaxDD_net_%', 0):.1f}%")
    print(f"   LS VolTarget Sharpe:      {avg_metrics.get('LS_VolTarget_Sharpe', 0):+.2f}")
    print(f"   LS DDStop Sharpe:         {avg_metrics.get('LS_DDStop_Sharpe', 0):+.2f}")
    print(f"   Cost model:               {COST_MODEL['taker_fee']*100:.2f}% taker + "
          f"{COST_MODEL['slippage']*100:.2f}% slip + "
          f"{COST_MODEL['funding_per_8h']*100:.3f}%/8h funding")
    print(f"{'='*70}")

    if best_sharpe > 3.0 and best_dd > -30:
        print("🟢 STRONG — CatBoost signal robust. Ready for ensemble with LGB.")
    elif best_sharpe > 2.0:
        print("🟡 DECENT — CatBoost adds diversity, useful as ensemble member.")
    elif best_sharpe > 1.0:
        print("🟠 MARGINAL — CatBoost may still help via decorrelation in ensemble.")
    else:
        print("🔴 WEAK — CatBoost underperforms, skip in ensemble.")

    print(f"\n✅ Results saved to {results_dir}/")
    print(f"   Models: cb_model_seed_*.cbm")
    print(f"   Run ensemble: python run_fast_sim.py --ensemble --leverage 3 --rebal 24 --edge-boost")


if __name__ == '__main__':
    main()
