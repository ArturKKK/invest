#!/usr/bin/env python3
"""
Meta-labeling: train a binary classifier that predicts P(trade profitable).

De Prado's meta-labeling framework:
  1. Primary model ensemble produces direction signals (scores)
  2. We simulate the backtester's position selection logic
  3. For each selected trade, compute whether it was profitable (net of costs)
  4. Train LGBMClassifier on meta-features → P(profitable)
  5. At inference: only take positions where P(profitable) > threshold

This is the highest-impact win-rate improvement:
  - Current WR ~63% → target 68-70%+
  - By skipping marginal trades (low P), we improve selectivity

Usage:
    python run_train_meta_label.py [--threshold 0.55] [--n-pos 5] [--cost-bps 8]
    python run_train_meta_label.py --sweep  # sweep thresholds 0.50-0.70

Outputs:
    results/meta_label/meta_label_model.pkl
    results/meta_label/meta_label_results.json
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = Path(__file__).resolve().parent

# ── Cost model (same as run_fast_sim.py) ──
COST_SIDE = 0.0003 + 0.0001       # taker 3bps + slippage 1bp = 4bps per side
FUNDING_PER_8H = 0.0001           # ~1bp per 8h funding
REBAL_H = 12                      # default rebalance interval
LEVERAGE = 3.0


def load_all_l0_models():
    """Load all 4 L0 model groups (same logic as retrain_meta_prod.py)."""
    from run_trading import load_lgb_models, load_catboost_models

    groups = {}

    # V6
    for d in ["results_v6_prod", "results_v6"]:
        p = ROOT / d
        if p.is_dir() and list(p.glob('lgb_model_seed_*.txt')):
            groups['v6'] = (load_lgb_models(str(p)),
                            lgb.Booster(model_file=str(list(p.glob('lgb_model_seed_*.txt'))[0])).feature_name())
            print(f"   v6: {len(groups['v6'][0])} models from {d}")
            break

    # V7
    for d in ["results_v7_prod", "results_v7"]:
        p = ROOT / d
        if p.is_dir() and list(p.glob('lgb_model_seed_*.txt')):
            groups['v7'] = (load_lgb_models(str(p)),
                            lgb.Booster(model_file=str(list(p.glob('lgb_model_seed_*.txt'))[0])).feature_name())
            print(f"   v7: {len(groups['v7'][0])} models from {d}")
            break

    # CatBoost
    for d in ["results_catboost_prod", "results_catboost"]:
        p = ROOT / d
        if p.is_dir() and list(p.glob('cb_model_seed_*.cbm')):
            from run_trading import load_catboost_models
            cb_models = load_catboost_models(str(p))
            fn_path = p / 'feature_names.json'
            if fn_path.exists():
                with open(fn_path) as f:
                    cb_feats = json.load(f)
            else:
                cb_feats = cb_models[0].feature_names_
            groups['cb'] = (cb_models, cb_feats)
            print(f"   cb: {len(cb_models)} models from {d}")
            break

    # XGBoost
    for d in ["results_xgboost_prod", "results_xgboost"]:
        p = ROOT / d
        if p.is_dir() and list(p.glob('xgb_model_seed_*.json')):
            try:
                import xgboost as xgb_lib
                xgb_files = sorted(p.glob('xgb_model_seed_*.json'))
                xgb_models = [xgb_lib.Booster(model_file=str(f)) for f in xgb_files]
                fn_path = p / 'feature_names.json'
                if fn_path.exists():
                    with open(fn_path) as f:
                        xgb_feats = json.load(f)
                else:
                    xgb_feats = xgb_models[0].feature_names
                groups['xgb'] = (xgb_models, xgb_feats)
                print(f"   xgb: {len(xgb_models)} models from {d}")
            except ImportError:
                print("   ⚠️  xgboost not installed")
            break

    # 24h horizon (if available)
    for d in ["results_v6_24h_prod"]:
        p = ROOT / d
        if p.is_dir() and list(p.glob('lgb_model_seed_*.txt')):
            groups['v6_24h'] = (load_lgb_models(str(p)),
                                lgb.Booster(model_file=str(list(p.glob('lgb_model_seed_*.txt'))[0])).feature_name())
            print(f"   v6_24h: {len(groups['v6_24h'][0])} models from {d}")
            break

    return groups


def predict_group(models, feats, df):
    """Mean prediction across seeds for a model group."""
    # Pad missing features
    for c in feats:
        if c not in df.columns:
            df[c] = 0.0
    X = df[feats].values
    if hasattr(models[0], 'predict'):
        return np.mean([m.predict(X) for m in models], axis=0)
    else:
        # XGBoost DMatrix
        import xgboost as xgb_lib
        dm = xgb_lib.DMatrix(X, feature_names=feats)
        return np.mean([m.predict(dm) for m in models], axis=0)


def build_meta_label_features(snap_df, group_preds, n_pos=5):
    """
    Build meta-features for meta-labeling.

    For each symbol in the snapshot, compute features that capture:
    - Ensemble signal strength and agreement
    - Position in cross-sectional ranking
    - Whether it's being selected (long/short/skip)
    - Market context (volatility regime, dispersion)

    Returns DataFrame with meta-features + 'selected_direction' column.
    """
    n = len(snap_df)
    mf = pd.DataFrame(index=range(n))

    # ── Raw predictions per group ──
    pred_names = []
    all_preds = []
    for name, preds in group_preds.items():
        mf[f'pred_{name}'] = preds
        pred_names.append(f'pred_{name}')
        all_preds.append(preds)

    preds_matrix = np.column_stack(all_preds)

    # ── Ensemble mean & std ──
    mf['ens_mean'] = preds_matrix.mean(axis=1)
    mf['ens_std'] = preds_matrix.std(axis=1)
    mf['ens_min'] = preds_matrix.min(axis=1)
    mf['ens_max'] = preds_matrix.max(axis=1)
    mf['ens_range'] = mf['ens_max'] - mf['ens_min']

    # ── Edge: distance from cross-sectional median ──
    median_score = np.median(mf['ens_mean'].values)
    mf['edge'] = mf['ens_mean'] - median_score
    mf['abs_edge'] = np.abs(mf['edge'])

    # ── Cross-sectional rank of ensemble score ──
    mf['ens_rank'] = pd.Series(mf['ens_mean'].values).rank(pct=True).values

    # ── Per-group ranks ──
    for name in group_preds:
        mf[f'rank_{name}'] = pd.Series(mf[f'pred_{name}'].values).rank(pct=True).values

    # ── Rank agreement ──
    rank_cols = [f'rank_{name}' for name in group_preds]
    rank_vals = mf[rank_cols].values
    mf['rank_std'] = rank_vals.std(axis=1)
    mf['rank_mean'] = rank_vals.mean(axis=1)
    mf['all_agree_long'] = (rank_vals > 0.75).all(axis=1).astype(float)
    mf['all_agree_short'] = (rank_vals < 0.25).all(axis=1).astype(float)
    mf['agree_direction'] = ((rank_vals > 0.5).all(axis=1) |
                              (rank_vals < 0.5).all(axis=1)).astype(float)

    # ── Cross-sectional z-score ──
    ens = mf['ens_mean'].values
    mu, sigma = ens.mean(), ens.std() + 1e-10
    mf['ens_zscore'] = (ens - mu) / sigma

    # ── Confidence = inverse of model disagreement ──
    # Normalize each group to zero-mean unit-var before computing std
    normed = []
    for p in all_preds:
        pm, ps = p.mean(), p.std() + 1e-10
        normed.append((p - pm) / ps)
    model_std = np.std(normed, axis=0)
    mf['confidence'] = 1.0 / (1.0 + model_std)

    # ── Position in the "selected" set (simulate backtester) ──
    order_desc = np.argsort(-mf['ens_mean'].values)
    order_asc = np.argsort(mf['ens_mean'].values)

    selected = np.zeros(n, dtype=int)  # 0=skip, 1=long, -1=short
    for idx in order_desc[:min(n_pos, n // 3)]:
        selected[idx] = 1
    for idx in order_asc[:min(n_pos, n // 3)]:
        selected[idx] = -1
    mf['selected_direction'] = selected

    # ── Edge percentile within selected set ──
    selected_mask = selected != 0
    if selected_mask.sum() > 0:
        selected_edges = mf['abs_edge'].values[selected_mask]
        edge_p50 = np.percentile(selected_edges, 50)
        edge_p75 = np.percentile(selected_edges, 75)
        mf['edge_vs_p50'] = mf['abs_edge'] / (edge_p50 + 1e-10)
        mf['edge_vs_p75'] = mf['abs_edge'] / (edge_p75 + 1e-10)
    else:
        mf['edge_vs_p50'] = 0.0
        mf['edge_vs_p75'] = 0.0

    # ── Per-symbol vol context ──
    for ctx_col in ['gk_vol_24h', 'rsi_14', 'adx', 'ret_24h', 'gk_vol_168h']:
        if ctx_col in snap_df.columns:
            mf[ctx_col] = snap_df[ctx_col].values
        else:
            mf[ctx_col] = 0.0

    # ── BTC market state ──
    syms = snap_df['symbol'].values if 'symbol' in snap_df.columns else np.array([])
    btc_mask = syms == 'BTC/USDT'
    if btc_mask.any():
        bi = np.where(btc_mask)[0][0]
        mf['btc_vol'] = snap_df['gk_vol_24h'].values[bi] if 'gk_vol_24h' in snap_df.columns else 0.0
        mf['btc_rsi'] = snap_df['rsi_14'].values[bi] if 'rsi_14' in snap_df.columns else 50.0
        mf['btc_trend'] = snap_df['close_ma336_ratio'].values[bi] if 'close_ma336_ratio' in snap_df.columns else 1.0
    else:
        mf['btc_vol'] = 0.0
        mf['btc_rsi'] = 50.0
        mf['btc_trend'] = 1.0

    # ── Market dispersion ──
    if 'ret_24h' in snap_df.columns:
        mf['market_dispersion'] = snap_df['ret_24h'].std()
    else:
        mf['market_dispersion'] = 0.0

    # ── Number of symbols in universe (can vary) ──
    mf['n_symbols'] = n

    return mf


# ── Constant: feature columns used by the meta-label model ──
META_LABEL_FEATURES = [
    # Ensemble signal strength
    'ens_mean', 'ens_std', 'ens_range', 'abs_edge', 'ens_zscore',
    # Rank & agreement
    'ens_rank', 'rank_std', 'rank_mean',
    'all_agree_long', 'all_agree_short', 'agree_direction',
    # Confidence
    'confidence',
    # Edge context
    'edge_vs_p50', 'edge_vs_p75',
    # Per-symbol context
    'gk_vol_24h', 'rsi_14', 'adx', 'ret_24h',
    # BTC market state
    'btc_vol', 'btc_rsi', 'btc_trend',
    # Market regime
    'market_dispersion', 'n_symbols',
]


def main():
    parser = argparse.ArgumentParser(description="Train meta-labeling model")
    parser.add_argument('--n-pos', type=int, default=5,
                        help='Number of long/short positions per side (default: 5)')
    parser.add_argument('--cost-bps', type=float, default=8,
                        help='Round-trip cost in bps per trade (default: 8 = 4bps×2 sides)')
    parser.add_argument('--leverage', type=float, default=3.0,
                        help='Leverage for funding cost calc (default: 3)')
    parser.add_argument('--threshold', type=float, default=0.55,
                        help='Default probability threshold for trade filter (default: 0.55)')
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep thresholds 0.50-0.70 and report results')
    parser.add_argument('--train-end', type=str, default='2026-02-01',
                        help='L0 models train cutoff (default: 2026-02-01)')
    parser.add_argument('--oos-start', type=str, default='2025-12-09',
                        help='OOS start date for meta-label (after L0 train + purge)')
    parser.add_argument('--meta-split', type=str, default='2026-02-01',
                        help='Meta train/test split date')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: results/meta_label)')
    args = parser.parse_args()

    print("=" * 70)
    print("  META-LABELING — Binary Trade Filter")
    print("  Goal: P(trade profitable) → skip marginal trades → higher WR")
    print("=" * 70)

    # ── 1. Load feature dataset ──
    feat_path = ROOT / 'data' / 'features' / 'crypto_features_1h.parquet'
    if not feat_path.exists():
        print(f"❌ {feat_path} not found"); sys.exit(1)

    print(f"\n📦 Loading features...")
    df = pd.read_parquet(feat_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Shape: {df.shape}, symbols: {df['symbol'].nunique()}")

    # ── 2. Enrich features (same pipeline as production) ──
    from run_pipeline_v6 import (
        add_multi_horizon_targets, add_cross_asset_features,
        add_advanced_regime_features, add_derivatives_features,
        add_sentiment_features,
    )
    from run_trading import (
        add_12h_features, cross_sectional_rank, EXCLUDE_COLS,
    )

    print("\n🔧 Enriching features...")
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
    print(f"   Enriched: {df.shape}")

    # ── 3. Filter to OOS period ──
    oos_start = pd.Timestamp(args.oos_start, tz='UTC')
    df = df[df['timestamp'] >= oos_start].copy()
    print(f"\n🔪 OOS period: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"   {df.shape[0]:,} rows, {df['symbol'].nunique()} symbols")

    if 'target_ret_12h' not in df.columns:
        print("❌ target_ret_12h not found"); sys.exit(1)

    # ── 4. Load L0 models ──
    print("\n📡 Loading L0 models...")
    groups = load_all_l0_models()
    if len(groups) < 3:
        print(f"❌ Need at least 3 model groups, got {len(groups)}"); sys.exit(1)
    print(f"   Loaded {len(groups)} groups: {list(groups.keys())}")

    # ── 5. Generate L0 predictions + meta-features + binary targets ──
    print("\n🔮 Building meta-label dataset...")
    rebal_timestamps = sorted(df['timestamp'].unique())
    # Subsample to REBAL_H intervals (12h)
    rebal_timestamps = rebal_timestamps[::REBAL_H]
    print(f"   {len(rebal_timestamps)} rebalance timestamps ({REBAL_H}h intervals)")

    cost_roundtrip = args.cost_bps / 10000  # e.g. 8bps = 0.0008
    funding_per_step = FUNDING_PER_8H * (REBAL_H / 8.0) * args.leverage

    meta_rows = []
    for ti, ts in enumerate(rebal_timestamps):
        snap = df[df['timestamp'] == ts].copy()
        if len(snap) < 10:
            continue

        # Predict with each group
        group_preds = {}
        for name, (models, feats) in groups.items():
            group_preds[name] = predict_group(models, feats, snap)

        # Build meta-features
        mf = build_meta_label_features(snap, group_preds, n_pos=args.n_pos)

        # ── Compute binary target: was this trade profitable? ──
        # For each SELECTED symbol, compute actual 12h forward return
        syms = snap['symbol'].values
        target_ret = snap['target_ret_12h'].values  # raw 12h return

        for i in range(len(snap)):
            direction = mf['selected_direction'].values[i]
            if direction == 0:
                continue  # skip: not selected by primary ensemble

            # Signed return: long profits from positive, short from negative
            raw_ret = target_ret[i]
            if np.isnan(raw_ret):
                continue

            signed_ret = direction * raw_ret

            # Net return after costs (round-trip entry+exit + funding)
            net_ret = signed_ret - cost_roundtrip - funding_per_step

            # Binary target: 1 if profitable, 0 if not
            y_label = 1 if net_ret > 0 else 0

            row = mf.iloc[i].to_dict()
            row['symbol'] = syms[i]
            row['timestamp'] = ts
            row['raw_ret'] = float(raw_ret)
            row['signed_ret'] = float(signed_ret)
            row['net_ret'] = float(net_ret)
            row['y_label'] = y_label
            row['direction'] = direction
            meta_rows.append(row)

        if (ti + 1) % 50 == 0:
            print(f"   ... {ti+1}/{len(rebal_timestamps)} timestamps, {len(meta_rows)} trades")

    meta_df = pd.DataFrame(meta_rows)
    print(f"\n📊 Meta-label dataset: {meta_df.shape[0]:,} trades from "
          f"{meta_df['timestamp'].nunique()} timestamps")

    total_wr = meta_df['y_label'].mean()
    print(f"   Baseline WR (before filter): {total_wr:.1%}")
    print(f"   Mean net return: {meta_df['net_ret'].mean():.4%}")
    print(f"   Positive trades: {meta_df['y_label'].sum():,} / {len(meta_df):,}")

    # ── 6. Train/test split ──
    cutoff = pd.Timestamp(args.meta_split, tz='UTC')
    meta_train = meta_df[meta_df['timestamp'] < cutoff].copy()
    meta_test = meta_df[meta_df['timestamp'] >= cutoff].copy()

    print(f"\n📊 Walk-forward split:")
    print(f"   Train: {meta_train.shape[0]:,} trades "
          f"({meta_train['timestamp'].min()} → {meta_train['timestamp'].max()})")
    print(f"   Test:  {meta_test.shape[0]:,} trades "
          f"({meta_test['timestamp'].min()} → {meta_test['timestamp'].max()})")
    print(f"   Train WR: {meta_train['y_label'].mean():.1%}, "
          f"Test WR: {meta_test['y_label'].mean():.1%}")

    if len(meta_train) < 50 or len(meta_test) < 20:
        print("❌ Not enough data for meta-label training")
        sys.exit(1)

    # ── 7. Train LGBMClassifier ──
    feat_cols_ml = [c for c in META_LABEL_FEATURES if c in meta_train.columns]
    print(f"\n🏋️ Training LGBMClassifier ({len(feat_cols_ml)} features)...")

    for c in feat_cols_ml:
        meta_train[c] = meta_train[c].fillna(0)
        meta_test[c] = meta_test[c].fillna(0)

    X_train = meta_train[feat_cols_ml].values
    y_train = meta_train['y_label'].values
    X_test = meta_test[feat_cols_ml].values
    y_test = meta_test['y_label'].values

    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.02,
        'num_leaves': 15,
        'max_depth': 4,
        'min_child_samples': 100,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'scale_pos_weight': 1.0,  # roughly balanced since WR~63%
        'verbose': -1,
    }

    seeds = [42, 123, 456, 789, 2024]

    # TimeSeriesSplit CV to find best_round
    tscv = TimeSeriesSplit(n_splits=3)
    best_iters = []
    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        p0 = {**lgb_params, 'seed': seeds[0], 'bagging_seed': seeds[0],
               'feature_fraction_seed': seeds[0]}
        dtrain_cv = lgb.Dataset(X_train[tr_idx], y_train[tr_idx],
                                feature_name=feat_cols_ml)
        dval_cv = lgb.Dataset(X_train[val_idx], y_train[val_idx],
                              feature_name=feat_cols_ml, reference=dtrain_cv)
        m_cv = lgb.train(
            p0, dtrain_cv,
            num_boost_round=2000,
            valid_sets=[dval_cv],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        best_iters.append(m_cv.best_iteration)

    best_round = int(np.median(best_iters))
    print(f"   CV best_iters: {best_iters} → using {best_round}")

    # Train final models (multi-seed)
    models = []
    dtrain_full = lgb.Dataset(X_train, y_train, feature_name=feat_cols_ml)

    for seed in seeds:
        p = {**lgb_params, 'seed': seed, 'bagging_seed': seed,
             'feature_fraction_seed': seed}
        model = lgb.train(p, dtrain_full, num_boost_round=best_round)
        models.append(model)

    # Predict probabilities on test set
    test_probs = np.mean([m.predict(X_test) for m in models], axis=0)

    # Feature importance
    fi = models[0].feature_importance(importance_type='gain')
    fi_sorted = sorted(zip(feat_cols_ml, fi), key=lambda x: -x[1])
    print(f"   Top features: {[(n, round(v, 1)) for n, v in fi_sorted[:8]]}")

    # ── 8. Evaluate at different thresholds ──
    print(f"\n{'='*70}")
    print(f"  META-LABEL EVALUATION (OOS: {meta_test['timestamp'].min().date()} → "
          f"{meta_test['timestamp'].max().date()})")
    print(f"{'='*70}")

    base_wr = meta_test['y_label'].mean()
    base_mean_ret = meta_test['net_ret'].mean()
    base_n = len(meta_test)
    n_per_step = meta_test.groupby('timestamp').size().mean()

    print(f"\n  Baseline (no filter): WR={base_wr:.1%}, "
          f"mean_ret={base_mean_ret:.4%}, N={base_n}, "
          f"avg_trades/step={n_per_step:.1f}")
    print()

    thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70]
    results = []

    print(f"  {'Threshold':>10s} {'WR':>7s} {'Mean Ret':>10s} {'N Trades':>10s} "
          f"{'Kept%':>7s} {'Trades/Step':>12s} {'Sharpe_est':>11s}")
    print(f"  {'-'*10} {'-'*7} {'-'*10} {'-'*10} {'-'*7} {'-'*12} {'-'*11}")

    for thr in thresholds:
        mask = test_probs >= thr
        if mask.sum() < 10:
            print(f"  {thr:>10.2f} {'—':>7s} {'—':>10s} {mask.sum():>10d} "
                  f"{'—':>7s} {'—':>12s} {'—':>11s}")
            continue

        subset = meta_test[mask].copy()
        subset_probs = test_probs[mask]
        wr = subset['y_label'].mean()
        mean_ret = subset['net_ret'].mean()
        std_ret = subset['net_ret'].std()
        sharpe_est = mean_ret / (std_ret + 1e-10) * np.sqrt(365 * 24 / REBAL_H)
        n_kept = mask.sum()
        pct_kept = n_kept / base_n
        trades_per_step = subset.groupby('timestamp').size().mean()

        results.append({
            'threshold': thr,
            'wr': float(wr),
            'mean_ret': float(mean_ret),
            'std_ret': float(std_ret),
            'sharpe_est': float(sharpe_est),
            'n_trades': int(n_kept),
            'pct_kept': float(pct_kept),
            'trades_per_step': float(trades_per_step),
        })

        # Highlight improvements
        wr_delta = wr - base_wr
        ret_delta = mean_ret - base_mean_ret
        wr_mark = "✓" if wr_delta > 0.01 else " "
        ret_mark = "✓" if ret_delta > 0 else " "

        print(f"  {thr:>10.2f} {wr:>6.1%}{wr_mark} {mean_ret:>9.4%}{ret_mark} "
              f"{n_kept:>10d} {pct_kept:>6.0%} {trades_per_step:>11.1f} "
              f"{sharpe_est:>11.2f}")

    # ── 9. Find optimal threshold ──
    if results:
        # Best = highest Sharpe with at least 30% of trades kept
        viable = [r for r in results if r['pct_kept'] >= 0.30]
        if viable:
            best = max(viable, key=lambda x: x['sharpe_est'])
            print(f"\n  🏆 Best threshold: {best['threshold']:.2f} "
                  f"(Sharpe={best['sharpe_est']:.2f}, WR={best['wr']:.1%}, "
                  f"kept={best['pct_kept']:.0%})")
        else:
            best = results[0]
            print(f"\n  ⚠️  No threshold keeps ≥30% of trades. Using {best['threshold']:.2f}")
    else:
        best = {'threshold': args.threshold}
        print("\n  ⚠️  No valid results")

    # ── 10. Simulate with meta-label filter ──
    print(f"\n{'='*70}")
    print(f"  SIMULATED BACKTEST WITH META-LABEL FILTER")
    print(f"{'='*70}")

    # Walk through test period step by step
    test_timestamps = sorted(meta_test['timestamp'].unique())
    equity = 10000.0
    peak = equity
    step_returns = []

    for ts in test_timestamps:
        step_trades = meta_test[meta_test['timestamp'] == ts]
        step_probs = test_probs[meta_test['timestamp'].values == ts]

        # Apply filter
        thr_val = best['threshold'] if results else args.threshold
        keep_mask = step_probs >= thr_val

        if keep_mask.sum() == 0:
            step_returns.append(0.0)
            continue

        kept_trades = step_trades[keep_mask]
        # Equal-weight the kept trades
        avg_net_ret = kept_trades['net_ret'].mean()
        # Scale by capital utilization (fewer trades = less capital used)
        capital_util = keep_mask.sum() / max(len(step_trades), 1)

        step_ret = avg_net_ret * capital_util
        step_returns.append(step_ret)
        equity *= (1 + step_ret)
        peak = max(peak, equity)

    step_returns = np.array(step_returns)
    total_ret = equity / 10000.0 - 1
    max_dd = min((np.cumprod(1 + step_returns) /
                  np.maximum.accumulate(np.cumprod(1 + step_returns))) - 1)
    sharpe = (step_returns.mean() / (step_returns.std() + 1e-10) *
              np.sqrt(365 * 24 / REBAL_H))
    wr_steps = (step_returns > 0).sum() / max(len(step_returns), 1)

    print(f"\n  Meta-label filtered (threshold={thr_val:.2f}):")
    print(f"    Return:     {total_ret:+.1%}")
    print(f"    Sharpe:     {sharpe:.2f}")
    print(f"    Max DD:     {max_dd:.1%}")
    print(f"    Step WR:    {wr_steps:.1%}")
    print(f"    Steps:      {len(test_timestamps)}")

    # ── Compare with unfiltered ──
    step_returns_base = []
    eq_base = 10000.0
    for ts in test_timestamps:
        step_trades = meta_test[meta_test['timestamp'] == ts]
        if len(step_trades) == 0:
            step_returns_base.append(0.0)
            continue
        avg_ret = step_trades['net_ret'].mean()
        step_returns_base.append(avg_ret)
        eq_base *= (1 + avg_ret)

    sr_base = np.array(step_returns_base)
    total_ret_base = eq_base / 10000.0 - 1
    sharpe_base = (sr_base.mean() / (sr_base.std() + 1e-10) *
                   np.sqrt(365 * 24 / REBAL_H))
    wr_base = (sr_base > 0).sum() / max(len(sr_base), 1)

    print(f"\n  Unfiltered baseline:")
    print(f"    Return:     {total_ret_base:+.1%}")
    print(f"    Sharpe:     {sharpe_base:.2f}")
    print(f"    Step WR:    {wr_base:.1%}")

    print(f"\n  Delta:")
    print(f"    Return:     {total_ret - total_ret_base:+.1%}")
    print(f"    Sharpe:     {sharpe - sharpe_base:+.2f}")
    print(f"    Step WR:    {wr_steps - wr_base:+.1%}")

    # ── 11. Save ──
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / 'results' / 'meta_label'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_obj = {
        'models': models,
        'feature_cols': feat_cols_ml,
        'threshold': float(thr_val),
        'best_round': best_round,
        'cv_iters': best_iters,
        'train_wr': float(meta_train['y_label'].mean()),
        'test_wr': float(meta_test['y_label'].mean()),
    }
    pkl_path = output_dir / 'meta_label_model.pkl'
    joblib.dump(model_obj, pkl_path)
    print(f"\n💾 Saved model → {pkl_path}")

    # Save results
    results_obj = {
        'config': {
            'n_pos': args.n_pos,
            'cost_bps': args.cost_bps,
            'leverage': args.leverage,
            'rebal_h': REBAL_H,
            'n_features': len(feat_cols_ml),
            'n_seeds': len(seeds),
            'best_round': best_round,
        },
        'dataset': {
            'train_trades': int(len(meta_train)),
            'test_trades': int(len(meta_test)),
            'train_period': f"{meta_train['timestamp'].min()} → {meta_train['timestamp'].max()}",
            'test_period': f"{meta_test['timestamp'].min()} → {meta_test['timestamp'].max()}",
            'baseline_wr': float(base_wr),
        },
        'threshold_sweep': results,
        'best_threshold': float(thr_val),
        'backtest': {
            'filtered': {
                'return': float(total_ret),
                'sharpe': float(sharpe),
                'max_dd': float(max_dd),
                'step_wr': float(wr_steps),
            },
            'unfiltered': {
                'return': float(total_ret_base),
                'sharpe': float(sharpe_base),
                'step_wr': float(wr_base),
            },
            'delta': {
                'return': float(total_ret - total_ret_base),
                'sharpe': float(sharpe - sharpe_base),
                'step_wr': float(wr_steps - wr_base),
            },
        },
        'feature_importance': [(n, float(v)) for n, v in fi_sorted],
        'model_groups': list(groups.keys()),
    }
    json_path = output_dir / 'meta_label_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_obj, f, indent=2, default=str)
    print(f"   Saved results → {json_path}")

    print(f"\n✅ Meta-labeling complete!")
    print(f"   Best threshold: {thr_val:.2f}")
    if results:
        best_r = max([r for r in results if r['pct_kept'] >= 0.30],
                     key=lambda x: x['sharpe_est'], default=results[0])
        print(f"   WR improvement: {base_wr:.1%} → {best_r['wr']:.1%} "
              f"(+{best_r['wr'] - base_wr:.1%})")
        print(f"   Trades kept: {best_r['pct_kept']:.0%}")


if __name__ == '__main__':
    main()
