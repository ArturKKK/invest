#!/usr/bin/env python3
"""
XGBoost News-Interaction Model — 4th ensemble member.

Why XGBoost + news interactions:
  1. Level-wise tree building (different from LGB leaf-wise & CatBoost symmetric)
     → adds decorrelation to ensemble
  2. Explicit news × price interaction features that GBDT cannot discover alone:
     - sentiment × volume (strong sentiment + many news = real signal)
     - news vs price divergence (positive news but price drops = contrarian)
     - news burst detection (sudden cluster = breaking event)
     - sentiment × momentum alignment  
  3. Same walk-forward evaluation as other models

Uses same base features from v6 + news from CryptoCompare + hand-crafted interactions.

Usage:
  python run_pipeline_xgboost.py                       # Full run
  python run_pipeline_xgboost.py --skip-hpo            # Skip Optuna HPO
  python run_pipeline_xgboost.py --single-window       # Quick single-window test
  python run_pipeline_xgboost.py --seeds 3             # Fewer seeds (faster)
  python run_pipeline_xgboost.py --production          # Max data for live trading
  python run_pipeline_xgboost.py --gpu                 # Use GPU (requires xgboost[gpu])

Requirements:
  pip install xgboost pandas numpy scipy pyarrow
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
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# Import shared feature engineering from v6
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_pipeline_v6 import (
    add_multi_horizon_targets, add_cross_asset_features,
    add_advanced_regime_features, add_12h_features, add_sentiment_features,
    add_derivatives_features,
    cross_sectional_rank, create_rank_target, add_residual_targets,
    evaluate_model, vol_target_returns, drawdown_stop_returns,
    compute_costs_per_period,
    EXCLUDE_COLS, REGIME_COLS, WALK_FORWARD_WINDOWS, PRODUCTION_WINDOW, HORIZON, SEEDS, COST_MODEL,
    PURGE_DAYS,
)

N_SEEDS = 5
_tree_method = 'hist'       # overridden by --gpu → 'gpu_hist'
_device = 'cpu'             # overridden by --gpu → 'cuda'

NEWS_FEATURES = [
    'news_count_1h', 'news_count_24h', 'news_count_7d',
    'news_sentiment_1h', 'news_sentiment_24h', 'news_sentiment_7d',
    'news_sentiment_momentum', 'news_volume_zscore',
    'market_news_count_24h', 'market_news_sentiment_24h',
]


# ============================================================
# NEWS INTERACTION FEATURES (the key innovation)
# ============================================================

def add_news_interaction_features(df):
    """
    Create explicit interaction features between news and price action.
    These capture patterns that tree-based models struggle to learn implicitly.
    """
    print("   🧬 Adding news interaction features...")
    n_before = len([c for c in df.columns if c not in EXCLUDE_COLS])

    # Ensure base news features exist (filled with 0 if missing)
    for col in NEWS_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    # --- 1. Sentiment × Volume (strong sentiment + many news = real signal) ---
    df['nx_sent_x_count_1h'] = df['news_sentiment_1h'] * df['news_count_1h']
    df['nx_sent_x_count_24h'] = df['news_sentiment_24h'] * df['news_count_24h']
    df['nx_sent_x_count_7d'] = df['news_sentiment_7d'] * df['news_count_7d']

    # --- 2. News burst (sudden spike in 1h count vs 24h baseline) ---
    df['nx_burst_ratio'] = df['news_count_1h'] / (df['news_count_24h'] / 24 + 1e-6)
    df['nx_is_burst'] = (df['nx_burst_ratio'] > 3).astype(float)
    df['nx_burst_x_sent'] = df['nx_is_burst'] * df['news_sentiment_1h']

    # --- 3. Sentiment vs Price divergence (contrarian signal) ---
    if 'ret_12h' in df.columns:
        # Positive news but price drops → potentially bullish reversal (buying opportunity)
        # Negative news but price rises → potentially bearish (market ignoring bad news temporarily)
        df['nx_sent_price_div'] = df['news_sentiment_24h'] * np.sign(-df['ret_12h'])
        df['nx_sent_ret_product'] = df['news_sentiment_24h'] * df['ret_12h']

    if 'ret_24h' in df.columns:
        df['nx_sent_price_div_24h'] = df['news_sentiment_24h'] * np.sign(-df['ret_24h'])

    # --- 4. Sentiment momentum × price momentum alignment ---
    if 'mom_12h_zscore' in df.columns:
        df['nx_sent_mom_align'] = df['news_sentiment_momentum'] * df['mom_12h_zscore']

    if 'mom_3d' in df.columns:
        df['nx_sent_mom_3d'] = df['news_sentiment_momentum'] * np.sign(df['mom_3d'])

    # --- 5. Coin vs Market sentiment (relative strength) ---
    df['nx_sent_vs_market'] = df['news_sentiment_24h'] - df['market_news_sentiment_24h']
    df['nx_count_vs_market'] = df['news_count_24h'] / (df['market_news_count_24h'] + 1e-6)

    # --- 6. Sentiment × Volatility (news during calm vs panic) ---
    if 'gk_vol_24h' in df.columns:
        df['nx_sent_x_vol'] = df['news_sentiment_24h'] * df['gk_vol_24h']

    if 'fng_value' in df.columns:
        # News sentiment in context of market fear/greed
        df['nx_sent_x_fear'] = df['news_sentiment_24h'] * (50 - df['fng_value']) / 50
        # Positive news during fear → strong buy signal
        # Negative news during greed → strong sell signal

    # --- 7. News cluster features (1h window analysis) ---
    df['nx_high_volume'] = (df['news_count_1h'] >= 3).astype(float)
    df['nx_high_vol_positive'] = df['nx_high_volume'] * (df['news_sentiment_1h'] > 0.2).astype(float)
    df['nx_high_vol_negative'] = df['nx_high_volume'] * (df['news_sentiment_1h'] < -0.2).astype(float)

    # --- 8. Sentiment acceleration (2nd derivative) ---
    df['nx_sent_accel'] = df['news_sentiment_1h'] - df['news_sentiment_24h']
    df['nx_sent_accel_7d'] = df['news_sentiment_24h'] - df['news_sentiment_7d']

    # --- 9. Funding rate × News (positioning + news = informed flow) ---
    if 'funding_rate' in df.columns:
        df['nx_funding_x_sent'] = df['funding_rate'] * df['news_sentiment_24h']
        # Positive funding (longs paying) + negative news → squeeze risk
        df['nx_funding_sent_div'] = df['funding_rate'] * np.sign(-df['news_sentiment_24h'])

    # --- 10. Cross-coin news asymmetry ---
    #     When one coin gets lots of news but others don't → specific catalyst
    if 'cross_coin_dispersion' in df.columns:
        df['nx_news_in_dispersion'] = df['news_count_24h'] * df['cross_coin_dispersion']

    n_after = len([c for c in df.columns if c not in EXCLUDE_COLS])
    n_new = n_after - n_before
    nx_cols = [c for c in df.columns if c.startswith('nx_')]
    print(f"   ✅ Added {len(nx_cols)} news interaction features")
    return df


# ============================================================
# HPO (XGBoost-specific parameter space)
# ============================================================

def run_optuna_hpo(X_train, y_train, X_val, y_val, val_dates, n_trials=50):
    """Optuna HPO with Rank ICIR objective for XGBoost."""
    try:
        import optuna
        import xgboost as xgb
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("   ⚠️  Optuna or XGBoost not installed")
        return None

    print(f"   🔍 Running XGBoost HPO ({n_trials} trials)...")
    unique_dates = np.unique(val_dates)

    def objective(trial):
        params = {
            'n_estimators': 5000,
            'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.05, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 10, 300),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'max_bin': trial.suggest_int('max_bin', 128, 512),
            'random_state': 42,
            'tree_method': _tree_method,
            'device': _device,
            'verbosity': 0,
            'objective': 'reg:squarederror',
            'early_stopping_rounds': 50,
        }

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
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

def train_xgboost(X_train, y_train, X_val, y_val,
                  feat_names=None, custom_params=None, seed=42):
    """Train a single XGBoost model."""
    import xgboost as xgb

    base_params = {
        'n_estimators': 5000,
        'learning_rate': 0.01,
        'max_depth': 6,
        'reg_lambda': 3.0,
        'reg_alpha': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 100,
        'gamma': 1.0,
        'max_bin': 256,
        'random_state': seed,
        'tree_method': _tree_method,
        'device': _device,
        'verbosity': 0,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'early_stopping_rounds': 100,
    }
    if custom_params:
        for k, v in custom_params.items():
            if k in base_params or k in ('learning_rate', 'max_depth', 'reg_lambda',
                                          'reg_alpha', 'subsample', 'colsample_bytree',
                                          'min_child_weight', 'gamma', 'max_bin'):
                base_params[k] = v
    base_params['random_state'] = seed

    model = xgb.XGBRegressor(**base_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=200,
    )
    return model


def train_multi_seed(X_train, y_train, X_val, y_val, X_test,
                     feat_names=None, params=None, seeds=None):
    """Train XGBoost ensemble with multiple seeds."""
    seeds = seeds or SEEDS
    print(f"\n   🌱 XGBoost multi-seed ensemble ({len(seeds)} seeds)...")

    all_preds = []
    all_models = []
    for i, seed in enumerate(seeds):
        print(f"      Seed {seed} ({i+1}/{len(seeds)})...", end=" ")
        model = train_xgboost(X_train, y_train, X_val, y_val,
                              feat_names=feat_names,
                              custom_params=params, seed=seed)
        preds = model.predict(X_test)
        all_preds.append(preds)
        all_models.append(model)
        print(f"iters={model.best_iteration}")

    ensemble_pred = np.mean(all_preds, axis=0)
    return ensemble_pred, all_models


def feature_selection_xgb(model, feat_cols, threshold_pct=20):
    """Feature selection based on XGBoost importance (gain)."""
    imp = pd.Series(
        model.feature_importances_,
        index=feat_cols
    )
    threshold = np.percentile(imp.values, threshold_pct)
    keep = imp[imp > threshold].index.tolist()

    # Always keep interaction features if they passed the threshold filter
    # (they are the key innovation of this model)
    nx_feats = [c for c in feat_cols if c.startswith('nx_')]
    for f in nx_feats:
        if f not in keep and imp.get(f, 0) > 0:
            keep.append(f)

    print(f"   🔪 Feature selection: {len(feat_cols)} → {len(keep)}")
    nx_kept = sum(1 for f in keep if f.startswith('nx_'))
    print(f"      (including {nx_kept} news interaction features)")
    return keep


# ============================================================
# MAIN
# ============================================================

def main():
    import xgboost as xgb

    parser = argparse.ArgumentParser(description="XGBoost + News Interactions model")
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
                        help='Use GPU for XGBoost training (requires CUDA)')
    parser.add_argument('--residual-target', action='store_true',
                        help='Use beta-residual returns (remove BTC factor) for target')
    parser.add_argument('--hybrid-norm', action='store_true',
                        help='Hybrid normalization: CS-rank + TS-zscore for spike features')
    args = parser.parse_args()

    global _tree_method, _device
    if args.gpu:
        _tree_method = 'hist'
        _device = 'cuda'

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data or os.path.join(project_root, 'data', 'features')
    if args.production:
        results_dir = args.results or os.path.join(project_root, 'results_xgboost_prod')
    else:
        results_dir = args.results or os.path.join(project_root, 'results_xgboost')
    os.makedirs(results_dir, exist_ok=True)

    feat_path = os.path.join(data_dir, 'crypto_features_1h.parquet')
    if not os.path.exists(feat_path):
        print(f"❌ Feature file not found: {feat_path}")
        sys.exit(1)

    print("=" * 70)
    print("  XGBOOST + NEWS INTERACTIONS — CRYPTO ALPHA MODEL")
    print("  12h Target + Level-wise Trees + News × Price Interactions")
    print("=" * 70)

    # ========================================
    # 1. LOAD & ENRICH DATA (same pipeline as v6 + news interactions)
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
    df = add_sentiment_features(df, project_root)
    df = add_derivatives_features(df, project_root)

    # ★ News interaction features — the key differentiator of this model
    df = add_news_interaction_features(df)

    # Clean infinities
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=['target_ret_12h'])

    # Feature columns (include nx_ interaction features)
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS
                 and not c.startswith('target_')]
    feat_cols = [c for c in feat_cols if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    nx_count = sum(1 for c in feat_cols if c.startswith('nx_'))
    print(f"   Features: {len(feat_cols)} ({nx_count} news interactions)")

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
        model_base = train_xgboost(X_train, y_train, X_val, y_val,
                                   feat_names=feat_cols,
                                   custom_params=best_params)
        selected_feats = feature_selection_xgb(model_base, feat_cols,
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
            test['pred_xgb'] = ensemble_pred
        last_model = all_models[-1]

        # --- Save trained models ---
        for i, mdl in enumerate(all_models):
            seed = SEEDS[:args.seeds][i]
            model_path = os.path.join(results_dir, f'xgb_model_seed_{seed}.json')
            mdl.save_model(model_path)

        # Save selected feature names (for loading in fast_sim)
        with open(os.path.join(results_dir, 'feature_names.json'), 'w') as f:
            json.dump(selected_feats, f)
        print(f"   💾 Saved {len(all_models)} XGBoost models + feature names")

        if not has_test:
            print(f"\n   ✅ Production models saved (no test evaluation)")
            if args.production:
                prod_meta = {
                    'mode': 'production',
                    'model_type': 'XGBoost',
                    'train_end': window['train_end'],
                    'val_end': window['val_end'],
                    'n_seeds': args.seeds,
                    'n_features': len(selected_feats),
                    'n_nx_features': sum(1 for f in selected_feats if f.startswith('nx_')),
                    'train_rows': len(train),
                    'val_rows': len(val),
                    'timestamp': datetime.now().isoformat(),
                }
                with open(os.path.join(results_dir, 'production_meta.json'), 'w') as f:
                    json.dump(prod_meta, f, indent=2)
            continue

        # --- Evaluate ---
        metrics, ls_net, ls_vt, ls_dd, timestamps = evaluate_model(
            test, 'pred_xgb', target_col, HORIZON, label=window['name']
        )
        metrics['window'] = window['name']
        all_window_metrics.append(metrics)

        save_cols = ['timestamp', 'symbol', target_col, 'pred_xgb']
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

    if len(all_window_metrics) > 1:
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
            'importance': last_model.feature_importances_,
        }).sort_values('importance', ascending=False)

        print(f"\n🏆 Top 30 Features (XGBoost, last window):")
        for _, row in importance.head(30).iterrows():
            if row['feature'].startswith('nx_'):
                marker = "🧬"
            elif any(s in row['feature'] for s in
                     ['news', 'fng', 'funding', 'reversal', 'surge', 'dispersion',
                      'long_short', 'beta', 'sentiment']):
                marker = "📰"
            else:
                marker = "  "
            print(f"   {marker} {row['feature']:40s} {row['importance']:.4f}")

        nx_in_top30 = sum(1 for _, r in importance.head(30).iterrows()
                          if r['feature'].startswith('nx_'))
        news_in_top30 = sum(1 for _, r in importance.head(30).iterrows()
                            if r['feature'].startswith('nx_') or 'news' in r['feature'])
        print(f"\n   🧬 News interaction features in top 30: {nx_in_top30}/30")
        print(f"   📰 All news-related features in top 30: {news_in_top30}/30")

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
            'model_type': 'XGBoost + News Interactions',
            'horizon': HORIZON,
            'n_features': len(feat_cols),
            'n_selected': len(selected_feats) if 'selected_feats' in dir() else 0,
            'n_nx_features': sum(1 for f in (selected_feats if 'selected_feats' in dir() else [])
                                 if f.startswith('nx_')),
            'n_seeds': args.seeds,
            'n_windows': len(windows),
            'hpo_trials': args.hpo_trials if not args.skip_hpo else 0,
            'xgboost_version': xgb.__version__,
        },
    }

    with open(os.path.join(results_dir, 'all_results_xgboost.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    if 'importance' in dir():
        importance.to_csv(os.path.join(results_dir, 'feature_importance_xgboost.csv'),
                          index=False)

    if all_test_predictions:
        combined_preds = pd.concat(all_test_predictions, ignore_index=True)
        combined_preds.to_parquet(
            os.path.join(results_dir, 'test_predictions_xgboost.parquet'), index=False)

    eq = pd.DataFrame({
        'ls_net': np.cumprod(1 + combined_ls) * 1000,
        'ls_vol_target': np.cumprod(1 + combined_vt) * 1000,
        'ls_dd_stop': np.cumprod(1 + combined_dd) * 1000,
    })
    eq.to_parquet(os.path.join(results_dir, 'equity_curves_xgboost.parquet'), index=False)

    # ========================================
    # FINAL VERDICT
    # ========================================
    best_sharpe = avg_metrics.get('LS_DDStop_Sharpe', avg_metrics.get('LS_Sharpe_net', 0))
    best_dd = avg_metrics.get('LS_DDStop_MaxDD_%', avg_metrics.get('LS_MaxDD_net_%', 0))

    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY — XGBoost + News Interactions (12h target)")
    print(f"{'='*70}")
    print(f"   Rank IC (avg):            {avg_metrics.get('Rank_IC', 0):+.4f}")
    print(f"   Rank ICIR (avg):          {avg_metrics.get('Rank_ICIR', 0):+.4f}")
    print(f"   LS Sharpe net (avg):      {avg_metrics.get('LS_Sharpe_net', 0):+.2f}")
    print(f"   LS MaxDD net (avg):       {avg_metrics.get('LS_MaxDD_net_%', 0):.1f}%")
    print(f"   LS DDStop Sharpe:         {avg_metrics.get('LS_DDStop_Sharpe', 0):+.2f}")
    print(f"   LS DDStop MaxDD:          {avg_metrics.get('LS_DDStop_MaxDD_%', 0):.1f}%")
    print(f"   Cost model:               {COST_MODEL['taker_fee']*100:.2f}% taker + "
          f"{COST_MODEL['slippage']*100:.2f}% slip + "
          f"{COST_MODEL['funding_per_8h']*100:.3f}%/8h funding")
    print(f"{'='*70}")

    if best_sharpe > 1.5:
        print("🟢 STRONG — XGBoost news model adds alpha. Ready for ensemble.")
    elif best_sharpe > 1.0:
        print("🟡 DECENT — XGBoost adds diversity, useful as ensemble member.")
    elif best_sharpe > 0.5:
        print("🟠 MARGINAL — small alpha, test in ensemble before committing.")
    else:
        print("🔴 WEAK — news interactions not helping, skip.")

    print(f"\n✅ Results saved to {results_dir}/")
    print(f"   Models: xgb_model_seed_*.json")
    print(f"   Run ensemble: python run_fast_sim.py --ensemble --edge-boost --days 365")


if __name__ == '__main__':
    main()
