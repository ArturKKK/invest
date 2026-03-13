#!/usr/bin/env python3
"""Analyze multi-horizon LGB models: correlation with baseline GBDT, standalone IC,
and ensemble IC comparison.

Loads 4h / 24h horizon LGB models alongside the standard 4-group GBDT and
computes cross-model correlations and information coefficients on OOS data.
"""

import os, sys, json, warnings, glob
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── 1. Load data & build ALL features ──────────────────────────────
print("=" * 70)
print("  MULTI-HORIZON ENSEMBLE ANALYSIS")
print("=" * 70)

data_path = os.path.join(ROOT, "data/features/crypto_features_1h.parquet")
df = pd.read_parquet(data_path)
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
print(f"\n📊 Data: {len(df):,} rows, {df.shape[1]} cols")
print(f"   Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

from run_pipeline_v6 import (
    add_multi_horizon_targets, add_cross_asset_features,
    add_advanced_regime_features,
    add_derivatives_features, add_sentiment_features,
    add_calendar_features,
)
from run_trading import add_12h_features, cross_sectional_rank
from run_pipeline_xgboost import add_news_interaction_features

EXCLUDE_COLS = {'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume',
                'date'}

print("\n🔧 Building ALL features...")
df = add_multi_horizon_targets(df)
df = add_cross_asset_features(df)
df = add_advanced_regime_features(df)
df = add_12h_features(df)
df = add_calendar_features(df)
df = add_sentiment_features(df, ROOT, news_mode='all')
df = add_derivatives_features(df, ROOT)
df = add_news_interaction_features(df)

fc = [c for c in df.columns if c not in EXCLUDE_COLS
      and not c.startswith("target_")
      and df[c].dtype in ("float64", "float32", "int64", "int32")]
df = cross_sectional_rank(df, fc)

for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
df[fc] = df[fc].fillna(0)
print(f"   Final: {df.shape}, {len(fc)} features")

# OOS period
oos_start = pd.Timestamp("2026-02-09", tz='UTC')
oos_end   = pd.Timestamp("2026-03-07", tz='UTC')
mask = (df['timestamp'] >= oos_start) & (df['timestamp'] <= oos_end)
df_oos = df[mask].copy()
print(f"\n📅 OOS: {oos_start.date()} → {oos_end.date()} ({len(df_oos):,} rows)")

target_col = "target_ret_12h"

# ── 2. Load models & predict ───────────────────────────────────────
import lightgbm as lgb

def _ensure_cols(df_slice, feat_names):
    missing = [c for c in feat_names if c not in df_slice.columns]
    if missing:
        for c in missing:
            df_slice[c] = 0.0
        print(f"      ⚠️  {len(missing)} missing feats filled with 0")
    return df_slice[feat_names].fillna(0)

def load_lgb(model_dir, label):
    fn_path = os.path.join(model_dir, "feature_names.json")
    if not os.path.exists(fn_path):
        print(f"   {label}: ❌ not found ({model_dir})")
        return None
    feat_names = json.load(open(fn_path))
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.txt')])
    if not model_files:
        return None
    X = _ensure_cols(df_oos, feat_names).values
    preds = []
    for mf in model_files:
        m = lgb.Booster(model_file=os.path.join(model_dir, mf))
        preds.append(m.predict(X))
    avg = np.mean(preds, axis=0)
    print(f"   {label}: {len(model_files)} models, {len(feat_names)} feats")
    return avg

def load_cb(model_dir, label):
    fn_path = os.path.join(model_dir, "feature_names.json")
    if not os.path.exists(fn_path):
        print(f"   {label}: ❌ not found")
        return None
    feat_names = json.load(open(fn_path))
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.cbm')])
    if not model_files:
        return None
    from catboost import CatBoostRegressor
    X = _ensure_cols(df_oos, feat_names).values
    preds = []
    for mf in model_files:
        m = CatBoostRegressor()
        m.load_model(os.path.join(model_dir, mf))
        preds.append(m.predict(X))
    avg = np.mean(preds, axis=0)
    print(f"   {label}: {len(model_files)} models, {len(feat_names)} feats")
    return avg

def load_xgb(model_dir, label):
    fn_path = os.path.join(model_dir, "feature_names.json")
    if not os.path.exists(fn_path):
        print(f"   {label}: ❌ not found")
        return None
    feat_names = json.load(open(fn_path))
    model_files = sorted([f for f in os.listdir(model_dir)
                          if f.endswith('.json') and f.startswith('xgb_model')])
    if not model_files:
        return None
    import xgboost as xgb
    X = _ensure_cols(df_oos, feat_names)
    dm = xgb.DMatrix(X, feature_names=feat_names)
    preds = []
    for mf in model_files:
        m = xgb.Booster()
        m.load_model(os.path.join(model_dir, mf))
        preds.append(m.predict(dm))
    avg = np.mean(preds, axis=0)
    print(f"   {label}: {len(model_files)} models, {len(feat_names)} feats")
    return avg

print("\n🔧 Loading models & predicting on OOS...")
predictions = {}

# Baseline 4-group
p = load_lgb(os.path.join(ROOT, "results_v6_prod"), "lgb_v6 (12h)")
if p is not None: predictions['lgb_v6'] = p

p = load_lgb(os.path.join(ROOT, "results_v7_prod"), "lgb_v7 (12h)")
if p is not None: predictions['lgb_v7'] = p

p = load_cb(os.path.join(ROOT, "results_catboost_prod"), "catboost (12h)")
if p is not None: predictions['catboost'] = p

p = load_xgb(os.path.join(ROOT, "results_xgboost_prod"), "xgboost (12h)")
if p is not None: predictions['xgboost'] = p

# Multi-horizon candidates
for h_dir in sorted(glob.glob(os.path.join(ROOT, "results_v6_*h_prod"))):
    dirname = os.path.basename(h_dir)
    # e.g. results_v6_4h_prod → lgb_v6_4h
    horizon_tag = dirname.replace("results_v6_", "").replace("_prod", "")
    label = f"lgb_v6_{horizon_tag}"
    p = load_lgb(h_dir, label)
    if p is not None:
        predictions[label] = p

baseline_cols = ['lgb_v6', 'lgb_v7', 'catboost', 'xgboost']
baseline_cols = [c for c in baseline_cols if c in predictions]
new_cols = [c for c in predictions if c not in baseline_cols]

print(f"\n✅ Loaded {len(predictions)} model groups")
print(f"   Baseline: {baseline_cols}")
print(f"   New candidates: {new_cols}")

# ── 3. Correlation matrix ──────────────────────────────────────────
print("\n" + "=" * 70)
print("  PREDICTION CORRELATION MATRIX")
print("=" * 70)

pred_df = pd.DataFrame(predictions, index=df_oos['timestamp'].values)

corr = pred_df.corr()
print("\nPearson:")
print(corr.round(4).to_string())

scorr = pred_df.corr(method='spearman')
print("\nSpearman:")
print(scorr.round(4).to_string())

# Per-candidate vs baseline ensemble
baseline_ens = pred_df[baseline_cols].mean(axis=1)
print("\n🔑 New model vs Baseline ensemble:")
for nc in new_cols:
    p_corr = pred_df[nc].corr(baseline_ens)
    s_corr = pred_df[nc].corr(baseline_ens, method='spearman')
    print(f"   {nc:15s}  Pearson={p_corr:.4f}  Spearman={s_corr:.4f}")

# ── 4. Per-model IC ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PER-MODEL IC (OOS, target=target_ret_12h)")
print("=" * 70)

y = df_oos[target_col].values

def calc_ic(pred, actual, timestamps):
    temp = pd.DataFrame({'pred': pred, 'actual': actual, 'time': timestamps})
    ics = []
    for _, grp in temp.groupby('time'):
        if len(grp) > 5:
            ic = grp['pred'].corr(grp['actual'], method='spearman')
            if not np.isnan(ic):
                ics.append(ic)
    return np.mean(ics) if ics else 0, np.std(ics) if ics else 0, len(ics)

ts = df_oos['timestamp'].values

print(f"\n{'Model':18s} {'Mean IC':>10s} {'Std IC':>10s} {'ICIR':>10s} {'N':>6s}")
print("-" * 58)
model_ics = {}
for name, pred in predictions.items():
    ic_m, ic_s, n = calc_ic(pred, y, ts)
    icir = ic_m / ic_s if ic_s > 0 else 0
    model_ics[name] = ic_m
    marker = " ★" if name in new_cols else ""
    print(f"{name:18s} {ic_m:10.4f} {ic_s:10.4f} {icir:10.4f} {n:6d}{marker}")

# ── 5. Ensemble IC comparisons ─────────────────────────────────────
print("\n" + "=" * 70)
print("  ENSEMBLE IC COMPARISON")
print("=" * 70)

combos = {}
# Baseline 4-group
combos['4-grp baseline'] = pred_df[baseline_cols].mean(axis=1).values

# Add each new candidate individually
for nc in new_cols:
    label = f"5-grp +{nc}"
    combos[label] = pred_df[baseline_cols + [nc]].mean(axis=1).values

# Add all new candidates together
if len(new_cols) > 1:
    label = f"{4+len(new_cols)}-grp +all_new"
    combos[label] = pred_df[baseline_cols + new_cols].mean(axis=1).values

# Weighted blends (give new models less weight)
for nc in new_cols:
    label = f"80/20 +{nc}"
    baseline_part = pred_df[baseline_cols].mean(axis=1)
    combos[label] = (baseline_part * 0.8 + pred_df[nc] * 0.2).values

print(f"\n{'Ensemble':22s} {'Mean IC':>10s} {'Std IC':>10s} {'ICIR':>10s} {'N':>6s}")
print("-" * 58)
for label, ens_pred in combos.items():
    ic_m, ic_s, n = calc_ic(ens_pred, y, ts)
    icir = ic_m / ic_s if ic_s > 0 else 0
    print(f"{label:22s} {ic_m:10.4f} {ic_s:10.4f} {icir:10.4f} {n:6d}")

# ── 6. Additional: IC on native target for multi-horizon ───────────
# The 4h model was trained on target_ret_4h — does it have better IC
# on its own native target?
print("\n" + "=" * 70)
print("  NATIVE-TARGET IC (model evaluated on its own target)")
print("=" * 70)

for nc in new_cols:
    # Extract horizon from name like "lgb_v6_4h"
    parts = nc.split('_')
    horizon_str = [p for p in parts if p.endswith('h')]
    if not horizon_str:
        continue
    native_target = f"target_ret_{horizon_str[0]}"
    if native_target in df_oos.columns:
        y_native = df_oos[native_target].values
        ic_m, ic_s, n = calc_ic(predictions[nc], y_native, ts)
        icir = ic_m / ic_s if ic_s > 0 else 0
        print(f"   {nc:18s} on {native_target:18s}: IC={ic_m:.4f} ± {ic_s:.4f}  ICIR={icir:.4f}")
    else:
        print(f"   {nc:18s}: target {native_target} not in data")

# Baseline models on their 12h target for comparison
ic_m, ic_s, n = calc_ic(pred_df[baseline_cols].mean(axis=1).values, y, ts)
print(f"   {'4-grp baseline':18s} on {'target_ret_12h':18s}: IC={ic_m:.4f} ± {ic_s:.4f}")

# ── 7. Verdict ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  VERDICT")
print("=" * 70)

baseline_ic = model_ics.get('lgb_v6', 0)  # reference single-model IC
baseline_ens_ic, _, _ = calc_ic(pred_df[baseline_cols].mean(axis=1).values, y, ts)

for nc in new_cols:
    nc_ic = model_ics.get(nc, 0)
    nc_corr = pred_df[nc].corr(baseline_ens)
    ens_5_ic, _, _ = calc_ic(pred_df[baseline_cols + [nc]].mean(axis=1).values, y, ts)
    
    print(f"\n   {nc}:")
    print(f"   ├── Standalone IC:     {nc_ic:.4f}")
    print(f"   ├── Corr with baseline: {nc_corr:.4f}")
    print(f"   ├── 4-grp ensemble IC:  {baseline_ens_ic:.4f}")
    print(f"   └── 5-grp ensemble IC:  {ens_5_ic:.4f}  (Δ = {ens_5_ic - baseline_ens_ic:+.4f})")
    
    # Decision
    ic_ok = nc_ic >= 0.03
    improves_ens = ens_5_ic > baseline_ens_ic
    low_corr = nc_corr < 0.85
    
    if ic_ok and improves_ens:
        print(f"   ✅ RECOMMENDATION: ADD to ensemble")
        if low_corr:
            print(f"      • Good diversity (corr={nc_corr:.2f})")
    elif ic_ok and not improves_ens:
        print(f"   ⚠️  MAYBE: IC decent but ensemble IC doesn't improve")
        print(f"      Try weighted blend or keep for robustness")
    else:
        print(f"   ❌ DO NOT add: IC too low ({nc_ic:.4f} < 0.03)")

print("\n" + "=" * 70)
print("Done.")
