#!/usr/bin/env python3
"""
Retrain meta-model on production L0 model predictions.

Problem: The current Ridge meta-model was trained on exp15/exp12 L0 predictions
which include pred_xgb (weight=0.288, the HIGHEST). But XGBoost is removed from
the production ensemble → pred_xgb=0 at inference → ~29% of signal weight is dead.

Solution: Generate fresh predictions from production L0 models (v6_prod, v7_prod,
catboost_prod) on the full historical feature dataset with proper enrichment,
then retrain Ridge on just 3 features (pred_v6, pred_v7, pred_cb).

Usage:
    python retrain_meta_prod.py
"""

import json
import os
import sys
import warnings
from pathlib import Path

import joblib
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

    # ── 4. Generate L0 predictions ──
    print("\n🔮 Generating L0 predictions on full dataset...")

    # Pad missing features with 0
    for flist in [v6_feats, v7_feats, cb_feats]:
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

    df['pred_v6'] = pred_v6
    df['pred_v7'] = pred_v7
    df['pred_cb'] = pred_cb
    print(f"   Done: {len(df):,} rows × 3 L0 predictions")

    # ── 5. Prepare meta-training data ──
    # Production L0 models trained through 2025-09-01. Only data AFTER that
    # is truly out-of-sample for L0. Use that as meta-train period.
    # Split: meta-train 2025-09-09→2026-01-01, meta-test 2026-01-01→latest.

    if 'target_ret_12h' not in df.columns:
        print("❌ target_ret_12h not found"); sys.exit(1)

    # Filter to OOS period only (already filtered early, just apply validation cutoff)
    oos_start = pd.Timestamp('2025-09-09', tz='UTC')
    meta_df = df[df['timestamp'] >= oos_start].copy()
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

    ridge_cols = ['pred_v6', 'pred_v7', 'pred_cb']

    # Drop NaN
    meta_train = meta_train.dropna(subset=ridge_cols + ['target_rank'])
    meta_test = meta_test.dropna(subset=ridge_cols + ['target_rank'])

    print(f"\n📊 Walk-forward split:")
    print(f"   Meta-train: {meta_train.shape[0]:,} rows "
          f"({meta_train['timestamp'].min()} → {meta_train['timestamp'].max()})")
    print(f"   Meta-test:  {meta_test.shape[0]:,} rows "
          f"({meta_test['timestamp'].min()} → {meta_test['timestamp'].max()})")

    # ── 6. Train Ridge ──
    print("\n🏋️ Training Ridge meta-model (3 features, no XGBoost)...")

    X_train = meta_train[ridge_cols].values
    y_train = meta_train['target_rank'].values
    X_test = meta_test[ridge_cols].values

    tscv = TimeSeriesSplit(n_splits=5)
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=tscv)
    ridge.fit(X_train, y_train)

    print(f"   Ridge alpha: {ridge.alpha_}")
    print(f"   Ridge coefs: {dict(zip(ridge_cols, [round(c, 4) for c in ridge.coef_]))}")
    print(f"   Ridge intercept: {ridge.intercept_:.4f}")

    # ── 7. Evaluate ──
    pred_ridge = ridge.predict(X_test)
    meta_test = meta_test.copy()
    meta_test['pred_ridge'] = pred_ridge
    meta_test['pred_mean'] = meta_test[ridge_cols].mean(axis=1)

    from scipy import stats as scipy_stats
    def rank_ic(pred, target):
        mask = np.isfinite(pred) & np.isfinite(target)
        return scipy_stats.spearmanr(pred[mask], target[mask]).statistic

    ic_ridge = rank_ic(meta_test['pred_ridge'].values, meta_test['target_ret_12h'].values)
    ic_mean = rank_ic(meta_test['pred_mean'].values, meta_test['target_ret_12h'].values)

    # Per-model RankIC
    for col in ridge_cols:
        ic = rank_ic(meta_test[col].values, meta_test['target_ret_12h'].values)
        print(f"   {col} RankIC: {ic:.4f}")

    print(f"\n📊 OOS Meta-test ({meta_test['timestamp'].min().date()} → {meta_test['timestamp'].max().date()}):")
    print(f"   Simple Mean RankIC: {ic_mean:.4f}")
    print(f"   Ridge RankIC:       {ic_ridge:.4f}")
    if ic_mean != 0:
        print(f"   Improvement:        {(ic_ridge - ic_mean) / abs(ic_mean) * 100:+.1f}%")

    # ── 8. Save updated meta-model ──
    output_dir = ROOT / 'results' / 'meta_stack'
    output_dir.mkdir(parents=True, exist_ok=True)

    old_pkl_path = output_dir / 'meta_model.pkl'
    if old_pkl_path.exists():
        old_obj = joblib.load(old_pkl_path)
        # Backup old model (once)
        backup_path = output_dir / 'meta_model_exp15_backup.pkl'
        if not backup_path.exists():
            joblib.dump(old_obj, backup_path)
            print(f"\n   📦 Backed up old meta-model → {backup_path.name}")
        old_ridge_cols = old_obj.get('ridge_cols_3', [])
        old_coefs = dict(zip(old_ridge_cols, old_obj['ridge_model_3'].coef_)) if 'ridge_model_3' in old_obj else {}
    else:
        old_obj = {}
        old_ridge_cols = []
        old_coefs = {}

    # Update ridge_3 with production-retrained version
    old_obj['ridge_model_3'] = ridge
    old_obj['ridge_cols_3'] = ridge_cols  # Now 3 cols, no pred_xgb

    joblib.dump(old_obj, old_pkl_path)
    print(f"   💾 Updated meta-model saved → {old_pkl_path}")
    print(f"   Ridge cols: {ridge_cols}")
    if old_ridge_cols:
        print(f"   Was:        {old_ridge_cols}")
        print(f"   Old coefs:  {old_coefs}")

    # Save metadata
    info = {
        'retrained_on': 'production L0 models (OOS predictions)',
        'meta_train_period': f"{meta_train['timestamp'].min()} → {meta_train['timestamp'].max()}",
        'meta_test_period': f"{meta_test['timestamp'].min()} → {meta_test['timestamp'].max()}",
        'meta_train_rows': int(meta_train.shape[0]),
        'meta_test_rows': int(meta_test.shape[0]),
        'ridge_alpha': float(ridge.alpha_),
        'ridge_coefs': dict(zip(ridge_cols, [float(c) for c in ridge.coef_])),
        'ridge_intercept': float(ridge.intercept_),
        'rank_ic_ridge': float(ic_ridge),
        'rank_ic_mean': float(ic_mean),
        'per_model_rank_ic': {},
        'models_used': {'v6': str(v6_dir), 'v7': str(v7_dir), 'cb': str(cb_dir)},
    }
    for col in ridge_cols:
        info['per_model_rank_ic'][col] = float(rank_ic(meta_test[col].values, meta_test['target_ret_12h'].values))

    with open(output_dir / 'meta_retrain_info.json', 'w') as f:
        json.dump(info, f, indent=2, default=str)

    print("\n✅ Meta-model retrained successfully!")


if __name__ == '__main__':
    main()
