#!/usr/bin/env python3
"""
Meta-model stacking: Level-1 learner over v6/v7/CatBoost OOS predictions.

Reads test_predictions_*.parquet from exp12_full (walk-forward OOS),
merges them, builds meta-features, and trains Ridge + LightGBM stackers.

Walk-forward for meta:
  - Meta-train = W1 test period (2024-07-01 → 2024-12-31) — L0 OOS
  - Meta-test  = W3 test period (2025-01-01 → 2026-03-07) — L0 OOS from later window
  - W2/W3 overlap (2025-01 → 2025-12) is resolved by keeping W3 preds
    (trained on more data, more realistic for production)

Usage:
  python run_meta_stack.py                          # default: exp12_full baselines
  python run_meta_stack.py --variant res_hyb        # use res_hyb variants
  python run_meta_stack.py --save-model             # persist meta-model for fast_sim
  python run_meta_stack.py --no-context             # skip market context features
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

# Walk-forward window boundaries (must match run_pipeline_v6.py)
WF_WINDOWS = {
    'W1': {'test_start': '2024-07-01', 'test_end': '2024-12-31'},
    'W2': {'test_start': '2025-01-01', 'test_end': '2025-12-31'},
    'W3': {'test_start': '2025-01-01', 'test_end': '2026-12-31'},
}

# Cost model (mirrors pipeline)
COST_MODEL = {
    'taker_fee': 0.0003,
    'slippage': 0.0001,
    'funding_per_8h': 0.00005,
    'turnover_pct': 0.35,
}
HORIZON = 12
PERIODS_PER_YEAR = 365 * 24 / HORIZON

# L0 model directories in exp12_full (keyed by variant)
# NOTE: v7 pipeline reuses the v6 save path (test_predictions_v6.parquet)
# but writes a 'pred_v7' column. This is intentional — v7 is a v6 fork.
L0_CONFIGS = {
    'baseline': {
        'v6': ('v6_baseline', 'test_predictions_v6.parquet', 'pred_v6'),
        'v7': ('v7_baseline', 'test_predictions_v6.parquet', 'pred_v7'),
        'cb': ('catboost_baseline', 'test_predictions_catboost.parquet', 'pred_cb'),
    },
    'res_hyb': {
        'v6': ('v6_res_hyb_no_news', 'test_predictions_v6.parquet', 'pred_v6'),
        'v7': ('v7_res_hyb', 'test_predictions_v6.parquet', 'pred_v7'),
        'cb': ('catboost_res_hyb_no_news', 'test_predictions_catboost.parquet', 'pred_cb'),
    },
    'exp15': {
        'v6':  ('v6',       'test_predictions_v6.parquet',       'pred_v6'),
        'v7':  ('v7',       'test_predictions_v6.parquet',       'pred_v7'),
        'cb':  ('catboost',  'test_predictions_catboost.parquet', 'pred_cb'),
        'xgb': ('xgboost',  'test_predictions_xgboost.parquet',  'pred_xgb'),
    },
}

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def compute_ic(pred, target):
    mask = np.isfinite(pred) & np.isfinite(target)
    if mask.sum() < 10:
        return np.nan
    return np.corrcoef(pred[mask], target[mask])[0, 1]


def compute_rank_ic(pred, target):
    mask = np.isfinite(pred) & np.isfinite(target)
    if mask.sum() < 10:
        return np.nan
    return scipy_stats.spearmanr(pred[mask], target[mask]).statistic


def sharpe(r, ppyr):
    if len(r) == 0:
        return 0.0
    return float((r.mean() / (r.std() + 1e-10)) * np.sqrt(ppyr))


def max_dd(r):
    if len(r) == 0:
        return 0.0
    cum = np.cumprod(1 + r)
    return float(np.min(cum / np.maximum.accumulate(cum) - 1))


def total_ret(r):
    if len(r) == 0:
        return 0.0
    return float(np.prod(1 + r) - 1)


def vol_target_returns(raw_rets, lookback=48, target_vol=0.02, cost_per_period=0.0):
    n = len(raw_rets)
    vt = np.zeros(n)
    for i in range(n):
        if i < lookback:
            scale = 1.0
        else:
            rv = np.std(raw_rets[max(0, i - lookback):i])
            scale = (target_vol / rv) if rv > 1e-6 else 1.0
        scale = np.clip(scale, 0.1, 2.0)
        vt[i] = raw_rets[i] * scale - cost_per_period * 2 * scale
    return vt


def drawdown_stop_returns(net_rets, max_dd_thresh=-0.25, recovery_thresh=-0.10):
    n = len(net_rets)
    out = np.zeros(n)
    eq = pk = 1.0
    stopped = False
    for i in range(n):
        if stopped:
            eq *= (1 + net_rets[i])
            if eq / pk - 1 > recovery_thresh:
                stopped = False
                out[i] = net_rets[i]
        else:
            eq *= (1 + net_rets[i])
            if eq > pk:
                pk = eq
            if eq / pk - 1 < max_dd_thresh:
                stopped = True
            else:
                out[i] = net_rets[i]
    return out


def ls_evaluation(df_eval, pred_col, target_col, cost_per_period):
    """Full L/S evaluation matching pipeline."""
    ls_raw, ls_net = [], []
    for ts, grp in df_eval.groupby('timestamp'):
        if len(grp) < 10:
            continue
        grp = grp.sort_values(pred_col, ascending=False)
        n = max(len(grp) // 5, 1)
        lr = grp.head(n)[target_col].mean()
        sr = grp.tail(n)[target_col].mean()
        ls_raw.append(lr - sr)
        ls_net.append(lr - sr - cost_per_period * 2)

    ls_raw = np.array(ls_raw)
    ls_net = np.array(ls_net)
    ls_vt = vol_target_returns(ls_raw, cost_per_period=cost_per_period)
    ls_dd = drawdown_stop_returns(ls_net)

    return {
        'LS_Sharpe_net': sharpe(ls_net, PERIODS_PER_YEAR),
        'LS_Ann_Return_net_%': ls_net.mean() * PERIODS_PER_YEAR * 100 if len(ls_net) > 0 else 0,
        'LS_MaxDD_net_%': max_dd(ls_net) * 100,
        'LS_Total_net_%': total_ret(ls_net) * 100,
        'LS_VolTarget_Sharpe': sharpe(ls_vt, PERIODS_PER_YEAR),
        'LS_DDStop_Sharpe': sharpe(ls_dd, PERIODS_PER_YEAR),
        'LS_DDStop_MaxDD_%': max_dd(ls_dd) * 100,
        'N_periods': len(ls_raw),
    }


# ──────────────────────────────────────────────────────────────────────
# 1. LOAD & MERGE L0 predictions
# ──────────────────────────────────────────────────────────────────────

def load_l0_predictions(variant='baseline', exp_dir=None):
    """Load and merge L0 OOS predictions from 3 model types."""
    if exp_dir is None:
        exp_dir = ROOT / 'results' / 'exp12_full'

    cfg = L0_CONFIGS[variant]
    dfs = {}

    for model_key, (subdir, fname, pred_col) in cfg.items():
        path = exp_dir / subdir / fname
        if not path.exists():
            print(f"  ❌ {path} not found")
            sys.exit(1)
        df = pd.read_parquet(path)
        # Dedup W2/W3 overlap BEFORE merge to avoid cross-product explosion
        n_before = len(df)
        df = df.drop_duplicates(subset=['timestamp', 'symbol'], keep='last')
        n_dupes = n_before - len(df)
        dup_str = f" (deduped {n_dupes:,})" if n_dupes > 0 else ""
        print(f"  {model_key}: {df.shape[0]:,} rows from {subdir}/{fname}{dup_str}")
        dfs[model_key] = df

    # Dynamic merge: start with first model, iteratively join the rest
    keys = list(cfg.keys())
    first_key = keys[0]
    first_pred = cfg[first_key][2]
    merged = dfs[first_key][['timestamp', 'symbol', 'target_ret_12h', first_pred]].copy()
    for key in keys[1:]:
        pred_col = cfg[key][2]
        merged = merged.merge(
            dfs[key][['timestamp', 'symbol', pred_col]],
            on=['timestamp', 'symbol'], how='inner'
        )

    # Sanity check: no duplicates should remain after per-model dedup
    n_dupes_after = merged.duplicated(subset=['timestamp', 'symbol']).sum()
    if n_dupes_after > 0:
        print(f"  ⚠️  {n_dupes_after:,} unexpected duplicates after merge, dropping")
        merged = merged.drop_duplicates(subset=['timestamp', 'symbol'], keep='last')

    merged = merged.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
    print(f"  Merged: {merged.shape[0]:,} rows, {merged['symbol'].nunique()} symbols")
    print(f"  Dates: {merged['timestamp'].min()} → {merged['timestamp'].max()}")

    return merged


# ──────────────────────────────────────────────────────────────────────
# 2. BUILD META-FEATURES
# ──────────────────────────────────────────────────────────────────────

def build_meta_features(merged, add_context=True):
    """
    Build meta-features from L0 predictions + optional market context.

    Meta-features:
      - 3 raw predictions (pred_v6, pred_v7, pred_cb)
      - 3 cross-sectional ranks of predictions
      - 3 pairwise absolute spreads
      - 1 cross-model disagreement (std)
      - 1 simple mean (baseline reference)
      - 2 min/max of predictions
      - Optional: market context (regime, vol, dispersion, time)
    """
    df = merged.copy()
    pred_cols = sorted([c for c in df.columns if c.startswith('pred_')])

    # ── Pairwise spreads (dynamic for 3 or 4 L0 models) ──
    from itertools import combinations
    for c1, c2 in combinations(pred_cols, 2):
        name = f'spread_{c1.replace("pred_", "")}_{c2.replace("pred_", "")}'
        df[name] = (df[c1] - df[c2]).abs()

    # ── Cross-model stats ──
    preds = df[pred_cols].values
    df['pred_mean'] = preds.mean(axis=1)
    df['pred_std'] = preds.std(axis=1)
    df['pred_min'] = preds.min(axis=1)
    df['pred_max'] = preds.max(axis=1)
    df['pred_range'] = df['pred_max'] - df['pred_min']

    # ── Cross-sectional ranks (per timestamp) ──
    for col in pred_cols:
        df[f'{col}_rank'] = df.groupby('timestamp')[col].rank(pct=True)

    # ── Cross-sectional z-scores (per timestamp) ──
    for col in pred_cols:
        g = df.groupby('timestamp')[col]
        df[f'{col}_zscore'] = (df[col] - g.transform('mean')) / (g.transform('std') + 1e-10)

    # ── Rank agreement: do all models agree on direction (top/bottom)? ──
    rank_cols = [f'{c}_rank' for c in pred_cols]
    rank_vals = df[rank_cols].values
    df['rank_std'] = rank_vals.std(axis=1)
    df['rank_min'] = rank_vals.min(axis=1)
    # All models agree this coin is top quartile
    df['all_top_q'] = (rank_vals > 0.75).all(axis=1).astype(float)
    # All models agree this coin is bottom quartile
    df['all_bot_q'] = (rank_vals < 0.25).all(axis=1).astype(float)

    if add_context:
        df = _add_market_context(df)

    return df


def _add_market_context(df):
    """Add market-level context features (regime, vol, dispersion, time)."""
    features_path = ROOT / 'data' / 'features' / 'crypto_features_1h.parquet'
    if not features_path.exists():
        print("  ⚠️  Features parquet not found, skipping context")
        return df

    # Load only the columns we need
    ctx_cols = ['timestamp', 'symbol', 'gk_vol_24h', 'gk_vol_168h',
                'ret_24h', 'ret_168h', 'close_ma336_ratio', 'rsi_14',
                'bb_width_20', 'adx']
    ctx = pd.read_parquet(features_path, columns=ctx_cols)
    ctx = ctx[(ctx['timestamp'] >= df['timestamp'].min()) &
              (ctx['timestamp'] <= df['timestamp'].max())]

    # Per-symbol context: join directly
    df = df.merge(ctx[['timestamp', 'symbol', 'gk_vol_24h', 'rsi_14', 'adx']],
                  on=['timestamp', 'symbol'], how='left')

    # Market-level context: BTC regime + market aggregates
    btc = ctx[ctx['symbol'] == 'BTC/USDT'].copy()
    btc_ctx = btc[['timestamp']].copy()
    btc_ctx['btc_vol_24h'] = btc['gk_vol_24h'].values
    btc_ctx['btc_vol_ratio'] = (btc['gk_vol_24h'].values /
                                 (btc['gk_vol_168h'].values + 1e-10))
    btc_ctx['btc_trend'] = btc['close_ma336_ratio'].values  # >1 = above MA336 = bullish
    btc_ctx['btc_rsi'] = btc['rsi_14'].values

    # Market dispersion: std of cross-sectional returns
    mkt_disp = ctx.groupby('timestamp')['ret_24h'].std().reset_index()
    mkt_disp.columns = ['timestamp', 'market_dispersion']

    # Market breadth: % of coins with positive 7d return
    breadth = ctx.groupby('timestamp')['ret_168h'].apply(
        lambda x: (x > 0).mean()
    ).reset_index()
    breadth.columns = ['timestamp', 'market_breadth']

    df = df.merge(btc_ctx, on='timestamp', how='left')
    df = df.merge(mkt_disp, on='timestamp', how='left')
    df = df.merge(breadth, on='timestamp', how='left')

    # Forward-fill market context NaN to avoid dropping entire periods
    ctx_fill_cols = ['gk_vol_24h', 'rsi_14', 'adx',
                     'btc_vol_24h', 'btc_vol_ratio', 'btc_trend', 'btc_rsi',
                     'market_dispersion', 'market_breadth']
    df = df.sort_values('timestamp')
    for col in ctx_fill_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # Time features
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)

    return df


# ──────────────────────────────────────────────────────────────────────
# 3. TRAIN & EVALUATE
# ──────────────────────────────────────────────────────────────────────

# Explicit feature lists for reproducibility
META_FEATURES_FULL = [
    # L0 predictions (3–4, xgb present only in exp15+)
    'pred_v6', 'pred_v7', 'pred_cb', 'pred_xgb',
    # Pairwise spreads (3–6)
    'spread_v6_v7', 'spread_v6_cb', 'spread_v6_xgb',
    'spread_v7_cb', 'spread_v7_xgb', 'spread_cb_xgb',
    # Cross-model aggregate stats (5)
    'pred_mean', 'pred_std', 'pred_min', 'pred_max', 'pred_range',
    # Cross-sectional ranks (3–4)
    'pred_v6_rank', 'pred_v7_rank', 'pred_cb_rank', 'pred_xgb_rank',
    # Cross-sectional z-scores (3–4)
    'pred_v6_zscore', 'pred_v7_zscore', 'pred_cb_zscore', 'pred_xgb_zscore',
    # Rank agreement (4)
    'rank_std', 'rank_min', 'all_top_q', 'all_bot_q',
    # Per-symbol context (3)
    'gk_vol_24h', 'rsi_14', 'adx',
    # BTC market context (4)
    'btc_vol_24h', 'btc_vol_ratio', 'btc_trend', 'btc_rsi',
    # Market aggregates (2)
    'market_dispersion', 'market_breadth',
    # Time features (3)
    'hour_sin', 'hour_cos', 'dow_sin',
]

META_FEATURES_MINIMAL = [
    'pred_v6', 'pred_v7', 'pred_cb', 'pred_xgb',
    'spread_v6_v7', 'spread_v6_cb', 'spread_v6_xgb',
    'spread_v7_cb', 'spread_v7_xgb', 'spread_cb_xgb',
    'pred_mean', 'pred_std', 'pred_min', 'pred_max', 'pred_range',
    'pred_v6_rank', 'pred_v7_rank', 'pred_cb_rank', 'pred_xgb_rank',
    'pred_v6_zscore', 'pred_v7_zscore', 'pred_cb_zscore', 'pred_xgb_zscore',
    'rank_std', 'rank_min', 'all_top_q', 'all_bot_q',
]


def get_feature_cols(df, explicit=True):
    """Return list of meta-feature columns.
    If explicit=True, use hardcoded list (reproducible).
    If explicit=False, discover from DataFrame (legacy)."""
    if explicit:
        return [c for c in META_FEATURES_FULL if c in df.columns]
    exclude = {'timestamp', 'symbol', 'target_ret_12h', 'target_rank'}
    return [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'float32', 'int64']]


def train_ridge(X_train, y_train, X_test, alphas=None):
    """Ridge regression baseline with time-series-aware CV."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import TimeSeriesSplit
    if alphas is None:
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    tscv = TimeSeriesSplit(n_splits=5)
    model = RidgeCV(alphas=alphas, cv=tscv)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"    Ridge: alpha={model.alpha_:.2f}, "
          f"coefs={dict(zip(range(X_train.shape[1]), np.round(model.coef_, 4)))}")
    return pred, model


def train_lgb(X_train, y_train, X_test, feat_names, n_seeds=5, n_cv_folds=3):
    """LightGBM meta-model with multi-seed + TimeSeriesSplit early stopping."""
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit

    params = {
        'objective': 'regression',
        'metric': 'l2',
        'learning_rate': 0.03,
        'num_leaves': 15,          # reduced from 31 — less overfitting on short meta-train
        'max_depth': 5,            # explicit depth cap
        'min_child_samples': 500,  # increased from 200 — more conservative
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'verbose': -1,
    }

    seeds = [42, 123, 456, 789, 2024][:n_seeds]

    # Determine best num_boost_round via TimeSeriesSplit CV (using first seed)
    tscv = TimeSeriesSplit(n_splits=n_cv_folds)
    best_iters = []
    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        p0 = {**params, 'seed': seeds[0], 'bagging_seed': seeds[0], 'feature_fraction_seed': seeds[0]}
        dtrain_cv = lgb.Dataset(X_train[tr_idx], y_train[tr_idx], feature_name=feat_names)
        dval_cv = lgb.Dataset(X_train[val_idx], y_train[val_idx], feature_name=feat_names, reference=dtrain_cv)
        m_cv = lgb.train(
            p0, dtrain_cv,
            num_boost_round=2000,
            valid_sets=[dval_cv],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        best_iters.append(m_cv.best_iteration)

    # Use median best iteration across folds (robust to single-fold variance)
    best_round = int(np.median(best_iters))
    print(f"    LGB CV best_iters per fold: {best_iters} → using {best_round}")

    # Train final models on ALL training data with fixed num_boost_round
    preds = []
    models = []
    dtrain_full = lgb.Dataset(X_train, y_train, feature_name=feat_names)

    for seed in seeds:
        p = {**params, 'seed': seed, 'bagging_seed': seed, 'feature_fraction_seed': seed}
        model = lgb.train(
            p, dtrain_full,
            num_boost_round=best_round,
        )
        pred = model.predict(X_test)
        preds.append(pred)
        models.append(model)

    mean_pred = np.mean(preds, axis=0)
    fi = models[0].feature_importance(importance_type='gain')
    fi_sorted = sorted(zip(feat_names, fi), key=lambda x: -x[1])
    print(f"    LGB: {n_seeds} seeds, best_iter≈{models[0].best_iteration}")
    print(f"    Top features: {[(n, round(v, 1)) for n, v in fi_sorted[:10]]}")

    return mean_pred, models


def compute_cost_per_period():
    """Realistic cost per rebalance period (mirrors run_pipeline_v6.py)."""
    trade_cost = (COST_MODEL['taker_fee'] + COST_MODEL['slippage']) * 2  # round-trip
    cost_pp = trade_cost * COST_MODEL['turnover_pct']
    cost_pp += COST_MODEL['funding_per_8h'] / (8 / HORIZON)
    return cost_pp


def evaluate_meta(df, pred_col, label, target_col='target_ret_12h'):
    """Evaluate predictions with IC, Rank IC, and L/S metrics."""
    cost_pp = compute_cost_per_period()

    ic = compute_ic(df[pred_col].values, df[target_col].values)
    ric = compute_rank_ic(df[pred_col].values, df[target_col].values)

    # Per-day IC for ICIR
    daily_ics = []
    for ts, grp in df.groupby(df['timestamp'].dt.date):
        if len(grp) > 10:
            daily_ics.append(compute_rank_ic(grp[pred_col].values, grp[target_col].values))
    daily_ics = np.array([x for x in daily_ics if np.isfinite(x)])
    icir = daily_ics.mean() / (daily_ics.std() + 1e-10) if len(daily_ics) > 0 else 0

    ls_metrics = ls_evaluation(df, pred_col, target_col, cost_pp)

    print(f"\n  {'='*60}")
    print(f"  {label}")
    print(f"  {'='*60}")
    print(f"    IC:          {ic:.4f}")
    print(f"    Rank IC:     {ric:.4f}")
    print(f"    ICIR:        {icir:.4f}")
    print(f"    LS Sharpe:   {ls_metrics['LS_Sharpe_net']:.2f}")
    print(f"    LS Ann Ret:  {ls_metrics['LS_Ann_Return_net_%']:.1f}%")
    print(f"    LS MaxDD:    {ls_metrics['LS_MaxDD_net_%']:.1f}%")
    print(f"    LS Total:    {ls_metrics['LS_Total_net_%']:.1f}%")
    print(f"    VT Sharpe:   {ls_metrics['LS_VolTarget_Sharpe']:.2f}")
    print(f"    DDStop Sharpe: {ls_metrics['LS_DDStop_Sharpe']:.2f}")
    print(f"    DDStop MaxDD:  {ls_metrics['LS_DDStop_MaxDD_%']:.1f}%")
    print(f"    N periods:   {ls_metrics['N_periods']:,}")

    return {'IC': round(ic, 4), 'Rank_IC': round(ric, 4), 'ICIR': round(icir, 4),
            **{k: round(v, 2) for k, v in ls_metrics.items()}}


# ──────────────────────────────────────────────────────────────────────
# 4. MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Meta-model stacking")
    parser.add_argument("--variant", default="baseline",
                        choices=list(L0_CONFIGS.keys()),
                        help="Which L0 model variant to use")
    parser.add_argument("--exp-dir", default=None,
                        help="Override L0 experiment directory (e.g., results/exp15_new_features)")
    parser.add_argument("--no-context", action="store_true",
                        help="Skip market context features (pure stacking)")
    parser.add_argument("--save-model", action="store_true", default=True,
                        help="Save trained meta-model for use in fast_sim (default: True)")
    parser.add_argument("--no-save-model", dest="save_model", action="store_false",
                        help="Disable auto-saving meta-model")
    parser.add_argument("--output-dir", default="results/meta_stack",
                        help="Output directory")
    parser.add_argument("--winsorize", type=float, default=0.005,
                        help="Winsorize target_ret before ranking: clip at [q, 1-q] (0=off, default 0.005)")
    parser.add_argument("--expanding", action="store_true",
                        help="Expanding window: use ALL data up to 2025-01-01 as meta-train "
                             "(default uses only W1 test period)")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  META-MODEL STACKING — variant={args.variant}")
    print("=" * 70)

    # ── 1. Load L0 predictions ──
    print("\n📦 Loading L0 predictions...")
    exp_dir = Path(args.exp_dir) if args.exp_dir else None
    merged = load_l0_predictions(variant=args.variant, exp_dir=exp_dir)
    l0_pred_cols = sorted([c for c in merged.columns if c.startswith('pred_')])

    # ── 2. Build meta-features ──
    print("\n🔧 Building meta-features...")
    df = build_meta_features(merged, add_context=not args.no_context)
    feat_cols = get_feature_cols(df)
    print(f"  Meta-features: {len(feat_cols)}")
    print(f"  Columns: {feat_cols}")

    # ── 3. Target preprocessing ──
    # Winsorize extreme returns to reduce noise before ranking
    if args.winsorize > 0:
        q_lo = df['target_ret_12h'].quantile(args.winsorize)
        q_hi = df['target_ret_12h'].quantile(1 - args.winsorize)
        n_clipped = ((df['target_ret_12h'] < q_lo) | (df['target_ret_12h'] > q_hi)).sum()
        df['target_ret_12h'] = df['target_ret_12h'].clip(q_lo, q_hi)
        print(f"  Winsorized target at [{q_lo:.4f}, {q_hi:.4f}], clipped {n_clipped:,} rows")

    # Cross-sectional target rank
    df['target_rank'] = df.groupby('timestamp')['target_ret_12h'].rank(pct=True)

    # ── 4. Walk-forward split ──
    # Meta-train: W1 test period (2024-07-01 → 2024-12-31)
    #   These are L0-OOS predictions from W1 models
    # Meta-test: W3 period (2025-01-01 → latest)
    #   These are L0-OOS predictions from W3 models

    cutoff = '2025-01-01'
    if args.expanding:
        # Expanding window: ALL available data before cutoff
        meta_train = df[df['timestamp'] < cutoff].copy()
        print(f"  ✅ Expanding window: using all data before {cutoff}")
    else:
        # Default: only W1 test period (2024-07 → 2024-12)
        meta_train = df[(df['timestamp'] >= '2024-07-01') & (df['timestamp'] < cutoff)].copy()
    meta_test = df[df['timestamp'] >= cutoff].copy()

    print(f"\n📊 Walk-forward split:")
    print(f"  Meta-train: {meta_train.shape[0]:,} rows "
          f"({meta_train['timestamp'].min()} → {meta_train['timestamp'].max()})")
    print(f"  Meta-test:  {meta_test.shape[0]:,} rows "
          f"({meta_test['timestamp'].min()} → {meta_test['timestamp'].max()})")

    # Drop NaN rows
    meta_train = meta_train.dropna(subset=feat_cols + ['target_rank'])
    meta_test = meta_test.dropna(subset=feat_cols + ['target_rank'])
    print(f"  After dropna: train={meta_train.shape[0]:,}, test={meta_test.shape[0]:,}")

    X_train = meta_train[feat_cols].values
    y_train = meta_train['target_rank'].values
    X_test = meta_test[feat_cols].values

    # ── 5. Baselines ──
    print(f"\n{'━'*70}")
    print(f"  BASELINES (on meta-test)")
    print(f"{'━'*70}")

    # Cache evaluate_meta results to avoid recomputing L/S in summary
    eval_cache = {}

    # Baseline 0: simple mean of L0 predictions
    eval_cache['pred_mean'] = evaluate_meta(meta_test, 'pred_mean', 'BASELINE-0: Simple Mean (current production)')

    # Baseline per model
    for col in l0_pred_cols:
        eval_cache[col] = evaluate_meta(meta_test, col, f'Individual: {col}')

    # ── 6. Ridge ──
    print(f"\n{'━'*70}")
    print(f"  RIDGE META-MODEL")
    print(f"{'━'*70}")

    # Ridge on raw L0 preds only (sanity check)
    ridge_cols_minimal = l0_pred_cols
    X_tr_r3 = meta_train[ridge_cols_minimal].values
    X_te_r3 = meta_test[ridge_cols_minimal].values
    ridge_pred_3, ridge_model_3 = train_ridge(X_tr_r3, y_train, X_te_r3)
    meta_test = meta_test.copy()
    meta_test['pred_ridge_3'] = ridge_pred_3
    eval_cache['pred_ridge_3'] = evaluate_meta(meta_test, 'pred_ridge_3',
        f'RIDGE-{len(l0_pred_cols)}: Ridge on {l0_pred_cols}')

    # Ridge on all meta-features
    ridge_pred_all, ridge_model_all = train_ridge(X_train, y_train, X_test)
    meta_test['pred_ridge_all'] = ridge_pred_all
    eval_cache['pred_ridge_all'] = evaluate_meta(meta_test, 'pred_ridge_all', f'RIDGE-ALL: Ridge on {len(feat_cols)} features')

    # ── 7. LightGBM ──
    print(f"\n{'━'*70}")
    print(f"  LIGHTGBM META-MODEL")
    print(f"{'━'*70}")

    lgb_pred, lgb_models = train_lgb(X_train, y_train, X_test, feat_cols)
    meta_test['pred_lgb_meta'] = lgb_pred
    eval_cache['pred_lgb_meta'] = evaluate_meta(meta_test, 'pred_lgb_meta',
                                                 f'LGB-META: LightGBM on {len(feat_cols)} features')

    # ── 8. LightGBM on minimal features (no context) ──
    print(f"\n{'━'*70}")
    print(f"  LIGHTGBM META (minimal — preds + spreads + ranks only)")
    print(f"{'━'*70}")

    minimal_cols = [c for c in META_FEATURES_MINIMAL if c in meta_train.columns]
    X_tr_min = meta_train[minimal_cols].values
    X_te_min = meta_test[minimal_cols].values
    y_train_min = meta_train['target_rank'].values
    lgb_pred_min, lgb_models_min = train_lgb(X_tr_min, y_train_min, X_te_min, minimal_cols)
    meta_test['pred_lgb_minimal'] = lgb_pred_min
    eval_cache['pred_lgb_minimal'] = evaluate_meta(meta_test, 'pred_lgb_minimal',
                                                    f'LGB-MINIMAL: LightGBM on {len(minimal_cols)} features (no context)')

    # ── 9. Summary comparison ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    # Build summary from cached results (no redundant L/S recomputation)
    summary = {}
    summary_items = [('pred_mean', 'Simple Mean')]
    for col in l0_pred_cols:
        summary_items.append((col, f'{col.replace("pred_", "")} only'))
    summary_items.extend([
        ('pred_ridge_3', f'Ridge-{len(l0_pred_cols)}'),
        ('pred_ridge_all', 'Ridge-ALL'),
        ('pred_lgb_meta', 'LGB-META'),
        ('pred_lgb_minimal', 'LGB-MINIMAL'),
    ])
    for col, label in summary_items:
        if col not in eval_cache:
            continue
        m = eval_cache[col]
        summary[label] = {
            'IC': m['IC'],
            'Rank_IC': m['Rank_IC'],
            'LS_Sharpe': m['LS_Sharpe_net'],
            'VT_Sharpe': m['LS_VolTarget_Sharpe'],
            'DDStop_Sharpe': m['LS_DDStop_Sharpe'],
            'MaxDD_%': m['LS_MaxDD_net_%'],
            'Total_%': m['LS_Total_net_%'],
        }
        print(f"  {label:20s}: IC={m['IC']:.4f}  RankIC={m['Rank_IC']:.4f}  "
              f"LS_Sharpe={m['LS_Sharpe_net']:+.2f}  "
              f"VT_Sharpe={m['LS_VolTarget_Sharpe']:+.2f}  "
              f"DDStop={m['LS_DDStop_Sharpe']:+.2f}  "
              f"MaxDD={m['LS_MaxDD_net_%']:.1f}%")

    # ── 10. Save outputs ──
    # Save predictions
    save_cols = ['timestamp', 'symbol', 'target_ret_12h'] + l0_pred_cols + [
                 'pred_mean', 'pred_ridge_3', 'pred_ridge_all', 'pred_lgb_meta', 'pred_lgb_minimal']
    save_cols = [c for c in save_cols if c in meta_test.columns]
    meta_test[save_cols].to_parquet(output_dir / 'meta_test_predictions.parquet', index=False)

    # Save summary
    result = {
        'variant': args.variant,
        'meta_features': len(feat_cols),
        'feature_names': feat_cols,
        'meta_train_rows': len(meta_train),
        'meta_test_rows': len(meta_test),
        'meta_train_period': f"{meta_train['timestamp'].min()} → {meta_train['timestamp'].max()}",
        'meta_test_period': f"{meta_test['timestamp'].min()} → {meta_test['timestamp'].max()}",
        'summary': summary,
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / 'meta_stack_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    # Save model if requested
    if args.save_model:
        import joblib
        joblib.dump({
            'lgb_models': lgb_models,
            'feature_cols': feat_cols,
            'minimal_cols': minimal_cols,
            'lgb_models_minimal': lgb_models_min,
            'ridge_model_3': ridge_model_3,
            'ridge_cols_3': ridge_cols_minimal,
            'ridge_model_all': ridge_model_all,
            'ridge_cols_all': feat_cols,
        }, output_dir / 'meta_model.pkl')
        print(f"\n  💾 Meta-model saved to {output_dir / 'meta_model.pkl'}")

    print(f"\n  📁 Results saved to {output_dir}/")
    print(f"     meta_test_predictions.parquet")
    print(f"     meta_stack_results.json")


if __name__ == '__main__':
    main()
