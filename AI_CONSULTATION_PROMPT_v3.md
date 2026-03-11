# Code Review: Meta-Model Stacking System (Round 2)

## Context

Crypto L/S trading system on OKX perpetual futures. 50 symbols, 12h rebalance. Three L0 model groups (LGB v6 × 5 seeds, LGB v7 × 5 seeds, CatBoost × 5 seeds = 15 models). A meta-model (Level-1 stacker) is trained on L0 OOS predictions and used at inference time.

**Round 1 review found 5 issues — all fixed:**
1. Cross-product on merge → fixed with per-model dedup before merge (`drop_duplicates(subset=['timestamp','symbol'], keep='last')`)
2. NaN bias from dropna → fixed with ffill + bfill on context cols before dropna
3. RidgeCV K-fold → replaced with TimeSeriesSplit(n_splits=5)
4. Implicit feature lists → replaced with explicit `META_FEATURES_FULL` (33) and `META_FEATURES_MINIMAL` (21)
5. LGB overfitting → reduced num_leaves=15, max_depth=5, min_child_samples=500

**Changes since Round 1:**
- LGB training: TimeSeriesSplit 3-fold CV for `best_num_boost_round`, then train final models on ALL training data with fixed iterations (not on last fold)
- Added `--winsorize 0.005` (clip extreme target returns before ranking)
- Added `--expanding` window option (use all data before cutoff vs only W1)
- Added `eval_cache` to avoid redundant L/S computation in summary
- Saved `ridge_model_all` alongside other models in pkl
- **NEW FILE: `src/models/meta_model.py`** — shared inference module used by both `run_fast_sim.py` and `run_trading.py`
- **`run_fast_sim.py`**: imports `MetaModelInference`, loads via `.load()`, `predict_ensemble()` calls `_meta_model_inf.predict()` when meta-model is loaded, falls back to simple mean
- **`run_trading.py`**: integrated meta-model — `--meta-model` / `--meta-variant` CLI args, loads `MetaModelInference`, passes to `generate_signal()`, overrides score when meta-model succeeds

## Latest Results (meta-test: 2025-01 → 2026-03)

| Model           | IC    | Rank IC | LS Sharpe | VT Sharpe | DDStop Sharpe | MaxDD % |
|-----------------|-------|---------|-----------|-----------|---------------|---------|
| Simple Mean     | 0.049 | 0.063   | 1.92      | 2.03      | 2.15          | -10.2   |
| Ridge-ALL (33f) | 0.041 | 0.057   | 1.40      | 2.07      | 1.28          | -20.1   |
| LGB-META (33f)  | 0.058 | 0.075   | 1.59      | 1.70      | 1.88          | -13.3   |
| LGB-MINIMAL (21f) | 0.052 | 0.069 | 1.59      | 2.24      | 2.35          | -13.9   |

60-day fast_sim backtest: meta-model +13.3% (Sharpe 2.85) vs baseline +13.4% (Sharpe 2.93) — effectively equal.

## Files for Review

### 1. `run_meta_stack.py` (738 lines) — Meta-model training pipeline

```python
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

WF_WINDOWS = {
    'W1': {'test_start': '2024-07-01', 'test_end': '2024-12-31'},
    'W2': {'test_start': '2025-01-01', 'test_end': '2025-12-31'},
    'W3': {'test_start': '2025-01-01', 'test_end': '2026-12-31'},
}

COST_MODEL = {
    'taker_fee': 0.0003,
    'slippage': 0.0001,
    'funding_per_8h': 0.00005,
    'turnover_pct': 0.35,
}
HORIZON = 12
PERIODS_PER_YEAR = 365 * 24 / HORIZON

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
    if exp_dir is None:
        exp_dir = ROOT / 'results' / 'exp12_full'

    cfg = L0_CONFIGS[variant]
    dfs = {}

    for model_key, (subdir, fname, pred_col) in cfg.items():
        path = exp_dir / subdir / fname
        if not path.exists():
            print(f"  ❌ {path} not found"); sys.exit(1)
        df = pd.read_parquet(path)
        n_before = len(df)
        df = df.drop_duplicates(subset=['timestamp', 'symbol'], keep='last')
        n_dupes = n_before - len(df)
        dup_str = f" (deduped {n_dupes:,})" if n_dupes > 0 else ""
        print(f"  {model_key}: {df.shape[0]:,} rows from {subdir}/{fname}{dup_str}")
        dfs[model_key] = df

    merged = dfs['v6'][['timestamp', 'symbol', 'target_ret_12h', 'pred_v6']].copy()
    merged = merged.merge(
        dfs['v7'][['timestamp', 'symbol', 'pred_v7']],
        on=['timestamp', 'symbol'], how='inner'
    )
    merged = merged.merge(
        dfs['cb'][['timestamp', 'symbol', 'pred_cb']],
        on=['timestamp', 'symbol'], how='inner'
    )

    n_dupes_after = merged.duplicated(subset=['timestamp', 'symbol']).sum()
    if n_dupes_after > 0:
        print(f"  ⚠️  {n_dupes_after:,} unexpected duplicates after merge, dropping")
        merged = merged.drop_duplicates(subset=['timestamp', 'symbol'], keep='last')

    merged = merged.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
    print(f"  Merged: {merged.shape[0]:,} rows, {merged['symbol'].nunique()} symbols")
    return merged


# ──────────────────────────────────────────────────────────────────────
# 2. BUILD META-FEATURES
# ──────────────────────────────────────────────────────────────────────

def build_meta_features(merged, add_context=True):
    df = merged.copy()
    pred_cols = ['pred_v6', 'pred_v7', 'pred_cb']

    df['spread_v6_v7'] = (df['pred_v6'] - df['pred_v7']).abs()
    df['spread_v6_cb'] = (df['pred_v6'] - df['pred_cb']).abs()
    df['spread_v7_cb'] = (df['pred_v7'] - df['pred_cb']).abs()

    preds = df[pred_cols].values
    df['pred_mean'] = preds.mean(axis=1)
    df['pred_std'] = preds.std(axis=1)
    df['pred_min'] = preds.min(axis=1)
    df['pred_max'] = preds.max(axis=1)
    df['pred_range'] = df['pred_max'] - df['pred_min']

    for col in pred_cols:
        df[f'{col}_rank'] = df.groupby('timestamp')[col].rank(pct=True)

    for col in pred_cols:
        g = df.groupby('timestamp')[col]
        df[f'{col}_zscore'] = (df[col] - g.transform('mean')) / (g.transform('std') + 1e-10)

    rank_cols = [f'{c}_rank' for c in pred_cols]
    rank_vals = df[rank_cols].values
    df['rank_std'] = rank_vals.std(axis=1)
    df['rank_min'] = rank_vals.min(axis=1)
    df['all_top_q'] = (rank_vals > 0.75).all(axis=1).astype(float)
    df['all_bot_q'] = (rank_vals < 0.25).all(axis=1).astype(float)

    if add_context:
        df = _add_market_context(df)
    return df


def _add_market_context(df):
    features_path = ROOT / 'data' / 'features' / 'crypto_features_1h.parquet'
    if not features_path.exists():
        print("  ⚠️  Features parquet not found, skipping context")
        return df

    ctx_cols = ['timestamp', 'symbol', 'gk_vol_24h', 'gk_vol_168h',
                'ret_24h', 'ret_168h', 'close_ma336_ratio', 'rsi_14',
                'bb_width_20', 'adx']
    ctx = pd.read_parquet(features_path, columns=ctx_cols)
    ctx = ctx[(ctx['timestamp'] >= df['timestamp'].min()) &
              (ctx['timestamp'] <= df['timestamp'].max())]

    df = df.merge(ctx[['timestamp', 'symbol', 'gk_vol_24h', 'rsi_14', 'adx']],
                  on=['timestamp', 'symbol'], how='left')

    btc = ctx[ctx['symbol'] == 'BTC/USDT'].copy()
    btc_ctx = btc[['timestamp']].copy()
    btc_ctx['btc_vol_24h'] = btc['gk_vol_24h'].values
    btc_ctx['btc_vol_ratio'] = btc['gk_vol_24h'].values / (btc['gk_vol_168h'].values + 1e-10)
    btc_ctx['btc_trend'] = btc['close_ma336_ratio'].values
    btc_ctx['btc_rsi'] = btc['rsi_14'].values

    mkt_disp = ctx.groupby('timestamp')['ret_24h'].std().reset_index()
    mkt_disp.columns = ['timestamp', 'market_dispersion']
    breadth = ctx.groupby('timestamp')['ret_168h'].apply(lambda x: (x > 0).mean()).reset_index()
    breadth.columns = ['timestamp', 'market_breadth']

    df = df.merge(btc_ctx, on='timestamp', how='left')
    df = df.merge(mkt_disp, on='timestamp', how='left')
    df = df.merge(breadth, on='timestamp', how='left')

    ctx_fill_cols = ['gk_vol_24h', 'rsi_14', 'adx',
                     'btc_vol_24h', 'btc_vol_ratio', 'btc_trend', 'btc_rsi',
                     'market_dispersion', 'market_breadth']
    df = df.sort_values('timestamp')
    for col in ctx_fill_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
    return df


# ──────────────────────────────────────────────────────────────────────
# 3. TRAIN & EVALUATE
# ──────────────────────────────────────────────────────────────────────

META_FEATURES_FULL = [
    'pred_v6', 'pred_v7', 'pred_cb',
    'spread_v6_v7', 'spread_v6_cb', 'spread_v7_cb',
    'pred_mean', 'pred_std', 'pred_min', 'pred_max', 'pred_range',
    'pred_v6_rank', 'pred_v7_rank', 'pred_cb_rank',
    'pred_v6_zscore', 'pred_v7_zscore', 'pred_cb_zscore',
    'rank_std', 'rank_min', 'all_top_q', 'all_bot_q',
    'gk_vol_24h', 'rsi_14', 'adx',
    'btc_vol_24h', 'btc_vol_ratio', 'btc_trend', 'btc_rsi',
    'market_dispersion', 'market_breadth',
    'hour_sin', 'hour_cos', 'dow_sin',
]

META_FEATURES_MINIMAL = [
    'pred_v6', 'pred_v7', 'pred_cb',
    'spread_v6_v7', 'spread_v6_cb', 'spread_v7_cb',
    'pred_mean', 'pred_std', 'pred_min', 'pred_max', 'pred_range',
    'pred_v6_rank', 'pred_v7_rank', 'pred_cb_rank',
    'pred_v6_zscore', 'pred_v7_zscore', 'pred_cb_zscore',
    'rank_std', 'rank_min', 'all_top_q', 'all_bot_q',
]


def get_feature_cols(df, explicit=True):
    if explicit:
        return [c for c in META_FEATURES_FULL if c in df.columns]
    exclude = {'timestamp', 'symbol', 'target_ret_12h', 'target_rank'}
    return [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'float32', 'int64']]


def train_ridge(X_train, y_train, X_test, alphas=None):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import TimeSeriesSplit
    if alphas is None:
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    tscv = TimeSeriesSplit(n_splits=5)
    model = RidgeCV(alphas=alphas, cv=tscv)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"    Ridge: alpha={model.alpha_:.2f}")
    return pred, model


def train_lgb(X_train, y_train, X_test, feat_names, n_seeds=5, n_cv_folds=3):
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit

    params = {
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

    seeds = [42, 123, 456, 789, 2024][:n_seeds]

    # TimeSeriesSplit CV to find best_num_boost_round
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

    best_round = int(np.median(best_iters))
    print(f"    LGB CV best_iters per fold: {best_iters} → using {best_round}")

    # Train final models on ALL training data
    preds = []
    models = []
    dtrain_full = lgb.Dataset(X_train, y_train, feature_name=feat_names)
    for seed in seeds:
        p = {**params, 'seed': seed, 'bagging_seed': seed, 'feature_fraction_seed': seed}
        model = lgb.train(p, dtrain_full, num_boost_round=best_round)
        pred = model.predict(X_test)
        preds.append(pred)
        models.append(model)

    mean_pred = np.mean(preds, axis=0)
    fi = models[0].feature_importance(importance_type='gain')
    fi_sorted = sorted(zip(feat_names, fi), key=lambda x: -x[1])
    print(f"    LGB: {n_seeds} seeds, iters={best_round}")
    print(f"    Top features: {[(n, round(v, 1)) for n, v in fi_sorted[:10]]}")
    return mean_pred, models


def compute_cost_per_period():
    trade_cost = (COST_MODEL['taker_fee'] + COST_MODEL['slippage']) * 2
    cost_pp = trade_cost * COST_MODEL['turnover_pct']
    cost_pp += COST_MODEL['funding_per_8h'] / (8 / HORIZON)
    return cost_pp


def evaluate_meta(df, pred_col, label, target_col='target_ret_12h'):
    cost_pp = compute_cost_per_period()
    ic = compute_ic(df[pred_col].values, df[target_col].values)
    ric = compute_rank_ic(df[pred_col].values, df[target_col].values)
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
    print(f"    VT Sharpe:   {ls_metrics['LS_VolTarget_Sharpe']:.2f}")
    print(f"    DDStop Sharpe: {ls_metrics['LS_DDStop_Sharpe']:.2f}")
    print(f"    N periods:   {ls_metrics['N_periods']:,}")

    return {'IC': round(ic, 4), 'Rank_IC': round(ric, 4), 'ICIR': round(icir, 4),
            **{k: round(v, 2) for k, v in ls_metrics.items()}}


# ──────────────────────────────────────────────────────────────────────
# 4. MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Meta-model stacking")
    parser.add_argument("--variant", default="baseline", choices=list(L0_CONFIGS.keys()))
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--save-model", action="store_true", default=True)
    parser.add_argument("--no-save-model", dest="save_model", action="store_false")
    parser.add_argument("--output-dir", default="results/meta_stack")
    parser.add_argument("--winsorize", type=float, default=0.005)
    parser.add_argument("--expanding", action="store_true")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load L0 predictions
    merged = load_l0_predictions(variant=args.variant)

    # 2. Build meta-features
    df = build_meta_features(merged, add_context=not args.no_context)
    feat_cols = get_feature_cols(df)

    # 3. Target preprocessing (winsorize + rank)
    if args.winsorize > 0:
        q_lo = df['target_ret_12h'].quantile(args.winsorize)
        q_hi = df['target_ret_12h'].quantile(1 - args.winsorize)
        df['target_ret_12h'] = df['target_ret_12h'].clip(q_lo, q_hi)
    df['target_rank'] = df.groupby('timestamp')['target_ret_12h'].rank(pct=True)

    # 4. Walk-forward split
    cutoff = '2025-01-01'
    if args.expanding:
        meta_train = df[df['timestamp'] < cutoff].copy()
    else:
        meta_train = df[(df['timestamp'] >= '2024-07-01') & (df['timestamp'] < cutoff)].copy()
    meta_test = df[df['timestamp'] >= cutoff].copy()

    meta_train = meta_train.dropna(subset=feat_cols + ['target_rank'])
    meta_test = meta_test.dropna(subset=feat_cols + ['target_rank'])

    X_train = meta_train[feat_cols].values
    y_train = meta_train['target_rank'].values
    X_test = meta_test[feat_cols].values

    # 5. Baselines
    eval_cache = {}
    eval_cache['pred_mean'] = evaluate_meta(meta_test, 'pred_mean', 'Simple Mean')

    # 6. Ridge
    ridge_cols_minimal = ['pred_v6', 'pred_v7', 'pred_cb']
    X_tr_r3 = meta_train[ridge_cols_minimal].values
    X_te_r3 = meta_test[ridge_cols_minimal].values
    ridge_pred_3, ridge_model_3 = train_ridge(X_tr_r3, y_train, X_te_r3)
    meta_test = meta_test.copy()
    meta_test['pred_ridge_3'] = ridge_pred_3
    eval_cache['pred_ridge_3'] = evaluate_meta(meta_test, 'pred_ridge_3', 'Ridge-3')

    ridge_pred_all, ridge_model_all = train_ridge(X_train, y_train, X_test)
    meta_test['pred_ridge_all'] = ridge_pred_all
    eval_cache['pred_ridge_all'] = evaluate_meta(meta_test, 'pred_ridge_all', 'Ridge-ALL')

    # 7. LightGBM
    lgb_pred, lgb_models = train_lgb(X_train, y_train, X_test, feat_cols)
    meta_test['pred_lgb_meta'] = lgb_pred
    eval_cache['pred_lgb_meta'] = evaluate_meta(meta_test, 'pred_lgb_meta', 'LGB-META')

    # 8. LGB minimal
    minimal_cols = [c for c in META_FEATURES_MINIMAL if c in meta_train.columns]
    X_tr_min = meta_train[minimal_cols].values
    X_te_min = meta_test[minimal_cols].values
    lgb_pred_min, lgb_models_min = train_lgb(X_tr_min, meta_train['target_rank'].values, X_te_min, minimal_cols)
    meta_test['pred_lgb_minimal'] = lgb_pred_min
    eval_cache['pred_lgb_minimal'] = evaluate_meta(meta_test, 'pred_lgb_minimal', 'LGB-MINIMAL')

    # 9. Summary & save
    # ... summary printing, save parquet + json + model pkl
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


if __name__ == '__main__':
    main()
```

### 2. `src/models/meta_model.py` (263 lines) — Shared inference module

```python
"""
Meta-model stacking inference module.

Shared between run_trading.py (production) and run_fast_sim.py (backtesting).
Trained by run_meta_stack.py, loaded from results/meta_stack/meta_model.pkl.
"""

import os
import numpy as np
import pandas as pd

META_FEATURES_FULL = [
    'pred_v6', 'pred_v7', 'pred_cb',
    'spread_v6_v7', 'spread_v6_cb', 'spread_v7_cb',
    'pred_mean', 'pred_std', 'pred_min', 'pred_max', 'pred_range',
    'pred_v6_rank', 'pred_v7_rank', 'pred_cb_rank',
    'pred_v6_zscore', 'pred_v7_zscore', 'pred_cb_zscore',
    'rank_std', 'rank_min', 'all_top_q', 'all_bot_q',
    'gk_vol_24h', 'rsi_14', 'adx',
    'btc_vol_24h', 'btc_vol_ratio', 'btc_trend', 'btc_rsi',
    'market_dispersion', 'market_breadth',
    'hour_sin', 'hour_cos', 'dow_sin',
]

META_FEATURES_MINIMAL = [
    'pred_v6', 'pred_v7', 'pred_cb',
    'spread_v6_v7', 'spread_v6_cb', 'spread_v7_cb',
    'pred_mean', 'pred_std', 'pred_min', 'pred_max', 'pred_range',
    'pred_v6_rank', 'pred_v7_rank', 'pred_cb_rank',
    'pred_v6_zscore', 'pred_v7_zscore', 'pred_cb_zscore',
    'rank_std', 'rank_min', 'all_top_q', 'all_bot_q',
]

VALID_VARIANTS = ('lgb', 'lgb_minimal', 'ridge', 'ridge_all')


def build_meta_features_live(snap_df, pred_v6, pred_v7, pred_cb):
    """
    Build meta-features from L0 predictions at inference time.
    Must match run_meta_stack.py build_meta_features() exactly.
    """
    n = len(snap_df)
    mf = pd.DataFrame(index=range(n))

    # Raw predictions
    mf['pred_v6'] = pred_v6
    mf['pred_v7'] = pred_v7
    mf['pred_cb'] = pred_cb

    # Pairwise spreads
    mf['spread_v6_v7'] = np.abs(pred_v6 - pred_v7)
    mf['spread_v6_cb'] = np.abs(pred_v6 - pred_cb)
    mf['spread_v7_cb'] = np.abs(pred_v7 - pred_cb)

    # Cross-model stats
    preds = np.column_stack([pred_v6, pred_v7, pred_cb])
    mf['pred_mean'] = preds.mean(axis=1)
    mf['pred_std'] = preds.std(axis=1)
    mf['pred_min'] = preds.min(axis=1)
    mf['pred_max'] = preds.max(axis=1)
    mf['pred_range'] = mf['pred_max'] - mf['pred_min']

    # Cross-sectional ranks (per snapshot = single timestamp → all symbols)
    for col in ['pred_v6', 'pred_v7', 'pred_cb']:
        mf[f'{col}_rank'] = pd.Series(mf[col].values).rank(pct=True).values

    # Cross-sectional z-scores
    for col in ['pred_v6', 'pred_v7', 'pred_cb']:
        vals = mf[col].values
        mu, sigma = vals.mean(), vals.std() + 1e-10
        mf[f'{col}_zscore'] = (vals - mu) / sigma

    # Rank agreement
    rank_cols = ['pred_v6_rank', 'pred_v7_rank', 'pred_cb_rank']
    rank_vals = mf[rank_cols].values
    mf['rank_std'] = rank_vals.std(axis=1)
    mf['rank_min'] = rank_vals.min(axis=1)
    mf['all_top_q'] = (rank_vals > 0.75).all(axis=1).astype(float)
    mf['all_bot_q'] = (rank_vals < 0.25).all(axis=1).astype(float)

    # Per-symbol context (from snap_df)
    for ctx_col in ['gk_vol_24h', 'rsi_14', 'adx']:
        if ctx_col in snap_df.columns:
            mf[ctx_col] = snap_df[ctx_col].values
        else:
            mf[ctx_col] = 0.0

    # BTC market context
    syms = snap_df['symbol'].values if 'symbol' in snap_df.columns else np.array([])
    btc_mask = syms == 'BTC/USDT'
    if btc_mask.any():
        btc_idx = np.where(btc_mask)[0][0]
        btc_vol = snap_df['gk_vol_24h'].values[btc_idx] if 'gk_vol_24h' in snap_df.columns else 0.0
        btc_vol_168 = snap_df['gk_vol_168h'].values[btc_idx] if 'gk_vol_168h' in snap_df.columns else btc_vol
        mf['btc_vol_24h'] = btc_vol
        mf['btc_vol_ratio'] = btc_vol / (btc_vol_168 + 1e-10)
        mf['btc_trend'] = snap_df['close_ma336_ratio'].values[btc_idx] if 'close_ma336_ratio' in snap_df.columns else 1.0
        mf['btc_rsi'] = snap_df['rsi_14'].values[btc_idx] if 'rsi_14' in snap_df.columns else 50.0
    else:
        mf['btc_vol_24h'] = 0.0
        mf['btc_vol_ratio'] = 1.0
        mf['btc_trend'] = 1.0
        mf['btc_rsi'] = 50.0

    # Market dispersion & breadth
    if 'ret_24h' in snap_df.columns:
        mf['market_dispersion'] = snap_df['ret_24h'].std()
    else:
        mf['market_dispersion'] = 0.0
    if 'ret_168h' in snap_df.columns:
        mf['market_breadth'] = (snap_df['ret_168h'].values > 0).mean()
    else:
        mf['market_breadth'] = 0.5

    # Time features
    if 'timestamp' in snap_df.columns:
        ts = snap_df['timestamp'].iloc[0]
        hour = ts.hour if hasattr(ts, 'hour') else 0
        dow = ts.dayofweek if hasattr(ts, 'dayofweek') else 0
    else:
        hour, dow = 0, 0
    mf['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    mf['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    mf['dow_sin'] = np.sin(2 * np.pi * dow / 7)

    return mf


class MetaModelInference:
    def __init__(self, models, feature_cols, variant, is_ridge=False):
        self.models = models
        self.feature_cols = feature_cols
        self.variant = variant
        self.is_ridge = is_ridge

    @classmethod
    def load(cls, pkl_path, variant='lgb_minimal', root=None):
        import joblib
        if pkl_path == 'auto':
            if root is None:
                root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            auto_path = os.path.join(root, 'results', 'meta_stack', 'meta_model.pkl')
            if os.path.exists(auto_path):
                pkl_path = auto_path
            else:
                return None

        if not os.path.exists(pkl_path):
            return None

        obj = joblib.load(pkl_path)
        if variant == 'lgb':
            models = obj['lgb_models']
            feat_cols = obj['feature_cols']
            is_ridge = False
        elif variant == 'lgb_minimal':
            models = obj['lgb_models_minimal']
            feat_cols = obj['minimal_cols']
            is_ridge = False
        elif variant == 'ridge':
            models = [obj['ridge_model_3']]
            feat_cols = obj['ridge_cols_3']
            is_ridge = True
        elif variant == 'ridge_all':
            models = [obj['ridge_model_all']]
            feat_cols = obj['ridge_cols_all']
            is_ridge = True
        else:
            raise ValueError(f"Unknown variant: {variant}")

        return cls(models=models, feature_cols=feat_cols, variant=variant, is_ridge=is_ridge)

    def predict(self, snap_df, pred_v6, pred_v7, pred_cb):
        mf = build_meta_features_live(snap_df, pred_v6, pred_v7, pred_cb)
        for col in self.feature_cols:
            if col not in mf.columns:
                mf[col] = 0.0
        X = mf[self.feature_cols].values
        if self.is_ridge:
            return self.models[0].predict(X)
        else:
            return np.mean([m.predict(X) for m in self.models], axis=0)

    def __repr__(self):
        return (f"MetaModelInference(variant={self.variant!r}, "
                f"features={len(self.feature_cols)}, models={len(self.models)})")
```

### 3. `run_fast_sim.py` — Meta-model integration (relevant sections only)

```python
# Line 33-36: Import
try:
    from src.models.meta_model import MetaModelInference, build_meta_features_live
except ImportError:
    MetaModelInference = None

# Lines 414-421: Loading
_meta_model_inf = None
if getattr(args, 'meta_model', None) and MetaModelInference is not None:
    _meta_model_inf = MetaModelInference.load(
        args.meta_model, variant=args.meta_variant, root=root
    )
    if _meta_model_inf is not None:
        arch_parts.append(f"meta-{args.meta_variant}")

# Lines 430-467: predict_ensemble() with meta-model branch
def predict_ensemble(snap_df):
    """Predict using L0 models, optionally refined by meta-model stacking."""
    # Step 1: Get per-group L0 predictions (always needed)
    all_scores = []
    all_individual = []
    per_group_scores = []  # [v6_mean, v7_mean, cb_mean]
    for ms, mf_g in model_groups:
        X = snap_df[mf_g].values
        preds = [m.predict(X) for m in ms]
        all_individual.extend(preds)
        scores = np.mean(preds, axis=0)
        all_scores.append(scores)
        per_group_scores.append(scores)

    # Confidence (always computed)
    if len(all_individual) > 1:
        normed = [(p - p.mean()) / (p.std() + 1e-10) for p in all_individual]
        model_std = np.std(normed, axis=0)
        confidence = 1.0 / (1.0 + model_std)
    else:
        confidence = np.ones(len(snap_df)) * 0.5

    # Step 2: Meta-model stacking (if enabled and 3 groups loaded)
    if _meta_model_inf is not None and len(per_group_scores) == 3:
        meta_scores = _meta_model_inf.predict(
            snap_df,
            pred_v6=per_group_scores[0],
            pred_v7=per_group_scores[1],
            pred_cb=per_group_scores[2],
        )
        return meta_scores, confidence

    # Fallback: simple mean (original behavior)
    mean_scores = np.mean(all_scores, axis=0)
    return mean_scores, confidence
```

### 4. `run_trading.py` — Meta-model integration (relevant sections only)

```python
# Lines 59-62: Import
try:
    from src.models.meta_model import MetaModelInference
except ImportError:
    MetaModelInference = None

# Line 828: generate_signal signature
def generate_signal(df, feat_cols, root, meta_model=None):
    """
    Generate ensemble signal.
    Args:
        meta_model: Optional MetaModelInference — if provided, replaces simple mean
                    with meta-model stacking (requires v6 + v7 + CB all loaded).
    """

# Lines 976-1003: After simple mean score, meta-model override
    result['score'] = sum(result[c] for c in pred_cols) / len(pred_cols)

    # ── Meta-model stacking: replace simple mean with learned combination ──
    if (meta_model is not None
            and 'pred_lgb_v6' in result.columns
            and 'pred_lgb_v7' in result.columns
            and 'pred_cb' in result.columns):
        try:
            latest_snap = df.groupby('symbol').last().reset_index()
            snap_aligned = latest_snap[latest_snap['symbol'].isin(result['symbol'])]
            snap_aligned = snap_aligned.set_index('symbol').loc[result['symbol'].values].reset_index()

            meta_scores = meta_model.predict(
                snap_aligned,
                pred_v6=result['pred_lgb_v6'].values,
                pred_v7=result['pred_lgb_v7'].values,
                pred_cb=result['pred_cb'].values,
            )
            result['score'] = meta_scores
            result['score'] = (result['score'] - result['score'].mean()) / (result['score'].std() + 1e-10)
            print(f"   🧠 Meta-model applied: {meta_model}")
        except Exception as e:
            print(f"   ⚠️  Meta-model failed, using simple mean: {e}")

# Lines 2056-2063: CLI args
    parser.add_argument('--meta-model', type=str, default=None, nargs='?', const='auto',
                        help="Use meta-model stacking. Pass path to pkl or 'auto'.")
    parser.add_argument('--meta-variant', type=str, default='lgb_minimal',
                        choices=['lgb', 'lgb_minimal', 'ridge', 'ridge_all'],
                        help='Meta-model variant (default: lgb_minimal)')

# Lines 2237-2239: Loading & passing
    _meta = None
    if getattr(args, 'meta_model', None) and MetaModelInference is not None:
        _meta = MetaModelInference.load(args.meta_model, variant=args.meta_variant, root=root)
    signals = generate_signal(df, feat_cols, root, meta_model=_meta)
```

## Key Concerns for Review

1. **Train/inference feature parity**: `build_meta_features()` in run_meta_stack.py uses `groupby('timestamp')` for cross-sectional ranks/z-scores (multiple timestamps in training data). `build_meta_features_live()` in meta_model.py uses single-timestamp data (all symbols at once). Are these equivalent? Both should produce per-snapshot cross-sectional stats, but the groupby path handles multiple snapshots while live handles just one.

2. **Predictions fed to meta-model in run_trading.py**: The L0 predictions stored as `result['pred_lgb_v6']` are already z-score normalized (`(x - mean) / std`). In run_fast_sim.py, `per_group_scores` are RAW un-normalized group means. Are these consistent with what the meta-model was trained on? (Training in run_meta_stack.py uses raw L0 OOS predictions from parquet.)

3. **snap_df alignment in run_trading.py**: `latest_snap = df.groupby('symbol').last()` — is this reliably the latest timestamp? If df is sorted by timestamp, yes. But `build_features()` does not guarantee sort order. Could this silently give wrong context features?

4. **Meta-model adds marginal value**: LGB-MINIMAL VT Sharpe=2.24 vs Simple Mean=2.03 on meta-test, but 60d fast_sim showed no improvement. Is the meta-model overfitting to the meta-test period structure? Only 6 months of meta-train data (216K rows, ~50 symbols × ~4K timestamps). Should we be concerned?

5. **Feature duplication**: `META_FEATURES_FULL` and `META_FEATURES_MINIMAL` are defined in both `run_meta_stack.py` and `src/models/meta_model.py`. If one changes, the other must change too. Should these be imported from one place?

6. **LGB trains with median CV iterations on full data**: The CV folds determine `best_num_boost_round`, then we train on ALL meta-train data. But if the last CV fold has more data than the full dataset (impossible here, but conceptually), the iteration count could be wrong. Is the median-of-CV-iters approach sound?

7. **Missing error handling**: `MetaModelInference.load()` catches file-not-found but not pkl format errors (e.g., missing keys). If the pkl was saved with an older format (no `ridge_model_all`), it will crash.

## What I Need

Please review all 4 files for:
- **Correctness bugs** (especially train/inference feature mismatch)
- **Data leakage** (meta-model shouldn't see future data)
- **Robustness** (edge cases, error handling, alignment issues)
- **Architecture** (is the 3-file split clean? should anything be restructured?)
- **Statistical issues** (is 6-month meta-train sufficient? is the evaluation methodology sound?)

Be specific — point to exact code locations and suggest concrete fixes.
