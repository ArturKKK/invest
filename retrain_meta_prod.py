#!/usr/bin/env python3
"""
Retrain meta-model on production L0 model predictions.

Generates fresh OOS predictions from all 4 production L0 models
(v6, v7, CatBoost, XGBoost) and retrains:
  1. lgb_minimal — LightGBM on 25 meta-features (preds + spreads + ranks)
  2. ridge_3 — Ridge on 3 features (pred_v6, pred_v7, pred_cb) as fallback

Usage:
    python retrain_meta_prod.py
"""

import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = Path(__file__).resolve().parent


def main():
    print("=" * 70)
    print("  META-MODEL RETRAIN — Production L0 Models")
    print("=" * 70)

    # ── 1. Load the feature dataset ──
    feat_path = ROOT / 'data' / 'features' / 'crypto_features_1h.parquet'
    if not feat_path.exists():
        print(f"❌ {feat_path} not found. Need historical feature data.")
        sys.exit(1)

    print(f"\n📦 Loading features from {feat_path}...")
    df = pd.read_parquet(feat_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Shape: {df.shape}, symbols: {df['symbol'].nunique()}")
    print(f"   Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # ── 2. Enrich with full pipeline (same as run_fast_sim / run_trading) ──
    from run_pipeline_v6 import (
        add_multi_horizon_targets, add_cross_asset_features,
        add_advanced_regime_features,
        add_derivatives_features, add_sentiment_features,
    )
    from run_trading import (
        add_12h_features, cross_sectional_rank, EXCLUDE_COLS,
        build_features, load_lgb_models, load_catboost_models,
    )

    print("\n🔧 Enriching features (full pipeline)...")
    df = add_multi_horizon_targets(df)
    df = add_cross_asset_features(df)
    df = add_advanced_regime_features(df)
    df = add_12h_features(df)
    df = add_sentiment_features(df, str(ROOT), news_mode='all')
    df = add_derivatives_features(df, str(ROOT))

    # Cross-sectional rank
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS
                 and not c.startswith('target_')
                 and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]
    df = cross_sectional_rank(df, feat_cols)

    # Clean infinities + NaN
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[feat_cols] = df[feat_cols].fillna(0)
    print(f"   Enriched: {df.shape}, {len(feat_cols)} feature columns")

    # ── Filter to OOS period early (speeds up predictions dramatically) ──
    oos_start = pd.Timestamp('2025-09-01', tz='UTC')
    print(f"\n🔪 Filtering to OOS period (>= {oos_start.date()}) before L0 prediction...")
    df = df[df['timestamp'] >= oos_start].copy()
    print(f"   Filtered: {df.shape[0]:,} rows, {df['timestamp'].min()} → {df['timestamp'].max()}")

    # ── 3. Load production L0 models ──
    print("\n📡 Loading production L0 models...")

    # V6
    v6_dir = None
    for d in ["results/production/lgb_v6_no_news", "results_v6_prod", "results_v6"]:
        p = ROOT / d
        if p.is_dir() and list(p.glob('lgb_model_seed_*.txt')):
            v6_dir = p; break
    if not v6_dir:
        print("❌ V6 models not found"); sys.exit(1)
    v6_models = load_lgb_models(str(v6_dir))
    v6_feats = v6_models[0].feature_name()
    print(f"   v6: {len(v6_models)} models, {len(v6_feats)} feats from {v6_dir}")

    # V7
    v7_dir = None
    for d in ["results/production/lgb_v7_no_news", "results_v7_prod", "results_v7"]:
        p = ROOT / d
        if p.is_dir() and list(p.glob('lgb_model_seed_*.txt')):
            v7_dir = p; break
    if not v7_dir:
        print("❌ V7 models not found"); sys.exit(1)
    v7_models = load_lgb_models(str(v7_dir))
    v7_feats = v7_models[0].feature_name()
    print(f"   v7: {len(v7_models)} models, {len(v7_feats)} feats from {v7_dir}")

    # CatBoost
    cb_dir = None
    for d in ["results/production/catboost_with_news", "results_catboost_prod", "results_catboost"]:
        p = ROOT / d
        if p.is_dir() and list(p.glob('cb_model_seed_*.cbm')):
            cb_dir = p; break
    if not cb_dir:
        print("❌ CatBoost models not found"); sys.exit(1)
    cb_models = load_catboost_models(str(cb_dir))
    fn_path = cb_dir / 'feature_names.json'
    if fn_path.exists():
        with open(fn_path) as f:
            cb_feats = json.load(f)
    else:
        cb_feats = cb_models[0].feature_names_
    print(f"   cb: {len(cb_models)} models, {len(cb_feats)} feats from {cb_dir}")

    # XGBoost
    xgb_dir = None
    for d in ["results/production/xgboost", "results_xgboost_prod", "results_xgboost"]:
        p = ROOT / d
        if p.is_dir() and list(p.glob('xgb_model_seed_*.json')):
            xgb_dir = p; break
    xgb_models = []
    xgb_feats = []
    if xgb_dir:
        try:
            import xgboost as xgb_lib
            xgb_files = sorted(xgb_dir.glob('xgb_model_seed_*.json'))
            xgb_models = [xgb_lib.Booster(model_file=str(f)) for f in xgb_files]
            fn_path = xgb_dir / 'feature_names.json'
            if fn_path.exists():
                with open(fn_path) as f:
                    xgb_feats = json.load(f)
            else:
                xgb_feats = xgb_models[0].feature_names
            print(f"   xgb: {len(xgb_models)} models, {len(xgb_feats)} feats from {xgb_dir}")
        except ImportError:
            print("   ⚠️  xgboost not installed, training meta without pred_xgb")
    else:
        print("   ⚠️  XGBoost models not found, training meta without pred_xgb")

    has_xgb = len(xgb_models) > 0

    # ── 4. Generate L0 predictions ──
    print("\n🔮 Generating L0 predictions on full dataset...")

    # Pad missing features with 0
    for flist in [v6_feats, v7_feats, cb_feats] + ([xgb_feats] if has_xgb else []):
        for c in flist:
            if c not in df.columns:
                df[c] = 0.0

    # Vectorized predictions (all rows at once)
    print("   Predicting v6...")
    pred_v6 = np.mean([m.predict(df[v6_feats].values) for m in v6_models], axis=0)
    print("   Predicting v7...")
    pred_v7 = np.mean([m.predict(df[v7_feats].values) for m in v7_models], axis=0)
    print("   Predicting cb...")
    pred_cb = np.mean([m.predict(df[cb_feats].values) for m in cb_models], axis=0)

    if has_xgb:
        import xgboost as xgb_lib
        print("   Predicting xgb...")
        xgb_preds = []
        dm = xgb_lib.DMatrix(df[xgb_feats].values, feature_names=xgb_feats)
        for m in xgb_models:
            xgb_preds.append(m.predict(dm))
        pred_xgb = np.mean(xgb_preds, axis=0)
    else:
        pred_xgb = None

    df['pred_v6'] = pred_v6
    df['pred_v7'] = pred_v7
    df['pred_cb'] = pred_cb
    if pred_xgb is not None:
        df['pred_xgb'] = pred_xgb
    n_l0 = 4 if has_xgb else 3
    print(f"   Done: {len(df):,} rows × {n_l0} L0 predictions")

    # ── 5. Build meta-features & prepare training data ──
    # Production L0 models trained through 2025-09-01. Only data AFTER that
    # is truly out-of-sample for L0. Use that as meta-train period.
    # Split: meta-train 2025-09-09→2026-01-01, meta-test 2026-01-01→latest.

    if 'target_ret_12h' not in df.columns:
        print("❌ target_ret_12h not found"); sys.exit(1)

    # Build meta-features using the shared module (same as inference)
    from src.models.meta_model import build_meta_features_live

    print("\n🧱 Building meta-features per timestamp snapshot...")
    timestamps = sorted(df['timestamp'].unique())
    meta_dfs = []
    for ts in timestamps:
        snap = df[df['timestamp'] == ts].copy()
        if len(snap) < 5:
            continue
        mf = build_meta_features_live(
            snap,
            pred_v6=snap['pred_v6'].values,
            pred_v7=snap['pred_v7'].values,
            pred_cb=snap['pred_cb'].values,
            pred_xgb=snap['pred_xgb'].values if has_xgb else None,
        )
        mf['timestamp'] = ts
        mf['symbol'] = snap['symbol'].values
        mf['target_ret_12h'] = snap['target_ret_12h'].values
        meta_dfs.append(mf)

    meta_df = pd.concat(meta_dfs, ignore_index=True)
    print(f"   Built meta-features: {meta_df.shape}, {meta_df['timestamp'].nunique()} snapshots")

    # Filter to OOS period only
    oos_start = pd.Timestamp('2025-09-09', tz='UTC')
    meta_df = meta_df[meta_df['timestamp'] >= oos_start].copy()
    print(f"\n📊 OOS data (after L0 train cutoff): {meta_df.shape[0]:,} rows")
    print(f"   Period: {meta_df['timestamp'].min()} → {meta_df['timestamp'].max()}")

    # Winsorize target
    q_lo = meta_df['target_ret_12h'].quantile(0.005)
    q_hi = meta_df['target_ret_12h'].quantile(0.995)
    meta_df['target_ret_12h'] = meta_df['target_ret_12h'].clip(q_lo, q_hi)

    # Cross-sectional target rank (per timestamp)
    meta_df['target_rank'] = meta_df.groupby('timestamp')['target_ret_12h'].rank(pct=True)

    # Train/test split
    cutoff = pd.Timestamp('2026-01-01', tz='UTC')
    meta_train = meta_df[meta_df['timestamp'] < cutoff].copy()
    meta_test = meta_df[meta_df['timestamp'] >= cutoff].copy()

    # Drop NaN in target
    meta_train = meta_train.dropna(subset=['target_rank'])
    meta_test = meta_test.dropna(subset=['target_rank'])

    print(f"\n📊 Walk-forward split:")
    print(f"   Meta-train: {meta_train.shape[0]:,} rows "
          f"({meta_train['timestamp'].min()} → {meta_train['timestamp'].max()})")
    print(f"   Meta-test:  {meta_test.shape[0]:,} rows "
          f"({meta_test['timestamp'].min()} → {meta_test['timestamp'].max()})")

    # ── Helper ──
    from scipy import stats as scipy_stats
    def rank_ic(pred, target):
        mask = np.isfinite(pred) & np.isfinite(target)
        if mask.sum() < 10:
            return 0.0
        return scipy_stats.spearmanr(pred[mask], target[mask]).statistic

    # ── 6. Train LGB-MINIMAL (25 features with XGBoost) ──
    from src.models.meta_model import META_FEATURES_MINIMAL

    minimal_cols = [c for c in META_FEATURES_MINIMAL if c in meta_train.columns]
    print(f"\n🏋️ Training LGB-MINIMAL meta-model ({len(minimal_cols)} features)...")
    print(f"   Features: {minimal_cols}")

    # Fill any NaN in meta-features
    for c in minimal_cols:
        meta_train[c] = meta_train[c].fillna(0)
        meta_test[c] = meta_test[c].fillna(0)

    X_train_lgb = meta_train[minimal_cols].values
    y_train_lgb = meta_train['target_rank'].values
    X_test_lgb = meta_test[minimal_cols].values

    # LGB training with multi-seed + TimeSeriesSplit (same as run_meta_stack.py)
    lgb_params = {
        'objective': 'regression',
        'metric': 'l2',
        'learning_rate': 0.03,
        'num_leaves': 15,
        'max_depth': 5,
        'min_child_samples': 500,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'verbose': -1,
    }

    seeds = [42, 123, 456, 789, 2024]
    n_cv_folds = 3

    # TimeSeriesSplit CV to find best_round
    tscv_lgb = TimeSeriesSplit(n_splits=n_cv_folds)
    best_iters = []
    for fold_idx, (tr_idx, val_idx) in enumerate(tscv_lgb.split(X_train_lgb)):
        p0 = {**lgb_params, 'seed': seeds[0], 'bagging_seed': seeds[0],
               'feature_fraction_seed': seeds[0]}
        dtrain_cv = lgb.Dataset(X_train_lgb[tr_idx], y_train_lgb[tr_idx],
                                feature_name=minimal_cols)
        dval_cv = lgb.Dataset(X_train_lgb[val_idx], y_train_lgb[val_idx],
                              feature_name=minimal_cols, reference=dtrain_cv)
        m_cv = lgb.train(
            p0, dtrain_cv,
            num_boost_round=2000,
            valid_sets=[dval_cv],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        best_iters.append(m_cv.best_iteration)

    best_round = int(np.median(best_iters))
    print(f"   LGB CV best_iters: {best_iters} → using {best_round}")

    # Train final models on ALL training data
    lgb_models_minimal = []
    lgb_preds = []
    dtrain_full = lgb.Dataset(X_train_lgb, y_train_lgb, feature_name=minimal_cols)

    for seed in seeds:
        p = {**lgb_params, 'seed': seed, 'bagging_seed': seed,
             'feature_fraction_seed': seed}
        model = lgb.train(p, dtrain_full, num_boost_round=best_round)
        pred = model.predict(X_test_lgb)
        lgb_preds.append(pred)
        lgb_models_minimal.append(model)

    lgb_pred_mean = np.mean(lgb_preds, axis=0)

    # Feature importance
    fi = lgb_models_minimal[0].feature_importance(importance_type='gain')
    fi_sorted = sorted(zip(minimal_cols, fi), key=lambda x: -x[1])
    print(f"   LGB: {len(seeds)} seeds, best_round={best_round}")
    print(f"   Top features: {[(n, round(v, 1)) for n, v in fi_sorted[:8]]}")

    # ── 7. Train Ridge (3 features, fallback) ──
    ridge_cols = ['pred_v6', 'pred_v7', 'pred_cb']
    print(f"\n🏋️ Training Ridge meta-model ({len(ridge_cols)} features, fallback)...")

    X_train_r = meta_train[ridge_cols].values
    y_train_r = meta_train['target_rank'].values
    X_test_r = meta_test[ridge_cols].values

    tscv = TimeSeriesSplit(n_splits=5)
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=tscv)
    ridge.fit(X_train_r, y_train_r)
    pred_ridge = ridge.predict(X_test_r)

    print(f"   Ridge alpha: {ridge.alpha_}")
    print(f"   Ridge coefs: {dict(zip(ridge_cols, [round(c, 4) for c in ridge.coef_]))}")

    # ── 8. Evaluate ──
    meta_test = meta_test.copy()
    meta_test['pred_lgb_minimal'] = lgb_pred_mean
    meta_test['pred_ridge'] = pred_ridge
    meta_test['pred_mean'] = meta_test[['pred_v6', 'pred_v7', 'pred_cb']].mean(axis=1)

    print(f"\n📊 OOS Evaluation ({meta_test['timestamp'].min().date()} → {meta_test['timestamp'].max().date()}):")

    for col in ['pred_v6', 'pred_v7', 'pred_cb'] + (['pred_xgb'] if has_xgb else []):
        ic = rank_ic(meta_test[col].values, meta_test['target_ret_12h'].values)
        print(f"   {col:20s} RankIC: {ic:.4f}")
    print(f"   {'---':20s} -------")

    ic_mean = rank_ic(meta_test['pred_mean'].values, meta_test['target_ret_12h'].values)
    ic_ridge = rank_ic(meta_test['pred_ridge'].values, meta_test['target_ret_12h'].values)
    ic_lgb = rank_ic(meta_test['pred_lgb_minimal'].values, meta_test['target_ret_12h'].values)

    print(f"   {'Simple Mean':20s} RankIC: {ic_mean:.4f}")
    print(f"   {'Ridge (3 feat)':20s} RankIC: {ic_ridge:.4f}")
    print(f"   {'LGB-MINIMAL':20s} RankIC: {ic_lgb:.4f}  ← production")

    # ── 9. Save updated meta-model ──
    output_dir = ROOT / 'results' / 'meta_stack'
    output_dir.mkdir(parents=True, exist_ok=True)

    old_pkl_path = output_dir / 'meta_model.pkl'
    if old_pkl_path.exists():
        old_obj = joblib.load(old_pkl_path)
        # Backup old model (once)
        backup_path = output_dir / 'meta_model_pre_retrain.pkl'
        if not backup_path.exists():
            joblib.dump(old_obj, backup_path)
            print(f"\n   📦 Backed up old meta-model → {backup_path.name}")
    else:
        old_obj = {}

    # Update all variants
    old_obj['lgb_models_minimal'] = lgb_models_minimal
    old_obj['minimal_cols'] = minimal_cols
    old_obj['ridge_model_3'] = ridge
    old_obj['ridge_cols_3'] = ridge_cols

    joblib.dump(old_obj, old_pkl_path)
    print(f"   💾 Updated meta-model saved → {old_pkl_path}")
    print(f"   LGB-MINIMAL: {len(lgb_models_minimal)} models, {len(minimal_cols)} features")
    print(f"   Ridge: {len(ridge_cols)} features")

    # Save metadata
    info = {
        'retrained_on': f'production L0 models (v6+v7+CB{"+XGB" if has_xgb else ""})',
        'meta_train_period': f"{meta_train['timestamp'].min()} → {meta_train['timestamp'].max()}",
        'meta_test_period': f"{meta_test['timestamp'].min()} → {meta_test['timestamp'].max()}",
        'meta_train_rows': int(meta_train.shape[0]),
        'meta_test_rows': int(meta_test.shape[0]),
        'lgb_minimal_features': minimal_cols,
        'lgb_best_round': best_round,
        'lgb_cv_iters': best_iters,
        'lgb_top_features': [(n, float(v)) for n, v in fi_sorted[:10]],
        'ridge_alpha': float(ridge.alpha_),
        'ridge_coefs': dict(zip(ridge_cols, [float(c) for c in ridge.coef_])),
        'rank_ic': {
            'lgb_minimal': float(ic_lgb),
            'ridge': float(ic_ridge),
            'simple_mean': float(ic_mean),
        },
        'per_model_rank_ic': {},
        'models_used': {
            'v6': str(v6_dir), 'v7': str(v7_dir), 'cb': str(cb_dir),
            'xgb': str(xgb_dir) if has_xgb else None,
        },
        'has_xgb': has_xgb,
    }
    for col in ['pred_v6', 'pred_v7', 'pred_cb'] + (['pred_xgb'] if has_xgb else []):
        info['per_model_rank_ic'][col] = float(rank_ic(
            meta_test[col].values, meta_test['target_ret_12h'].values))

    with open(output_dir / 'meta_retrain_info.json', 'w') as f:
        json.dump(info, f, indent=2, default=str)

    print("\n✅ Meta-model retrained successfully!")
    print(f"   LGB-MINIMAL RankIC: {ic_lgb:.4f} (production variant)")
    print(f"   Ridge RankIC:       {ic_ridge:.4f} (fallback)")


if __name__ == '__main__':
    main()
