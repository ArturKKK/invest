#!/usr/bin/env python3
"""Analyze MLP model: correlation with GBDT, standalone IC, ensemble comparison.

Uses the SAME feature pipeline as run_fast_sim.py to ensure all 177+ features
are built correctly (cross-asset, regime, 12h, calendar, sentiment, derivatives).
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── 1. Load data & build ALL features (mirror run_fast_sim offline) ─
print("=" * 70)
print("  MLP ENSEMBLE ANALYSIS")
print("=" * 70)

data_path = os.path.join(ROOT, "data/features/crypto_features_1h.parquet")
df = pd.read_parquet(data_path)
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
print(f"\n📊 Data: {len(df):,} rows, {df.shape[1]} cols")
print(f"   Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

# Full feature engineering — same as run_fast_sim.py offline path
from run_pipeline_v6 import (
    add_multi_horizon_targets, add_cross_asset_features,
    add_advanced_regime_features,
    add_derivatives_features, add_sentiment_features,
    add_calendar_features,
)
from run_trading import add_12h_features, cross_sectional_rank
from run_pipeline_xgboost import add_news_interaction_features

# Columns to exclude from features
EXCLUDE_COLS = {'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume',
                'date'}

print("\n🔧 Building ALL features (same pipeline as run_fast_sim)...")
df = add_multi_horizon_targets(df)
df = add_cross_asset_features(df)
df = add_advanced_regime_features(df)
df = add_12h_features(df)
df = add_calendar_features(df)
df = add_sentiment_features(df, ROOT, news_mode='all')
df = add_derivatives_features(df, ROOT)
df = add_news_interaction_features(df)

# Cross-sectional rank (after all features built)
fc = [c for c in df.columns if c not in EXCLUDE_COLS
      and not c.startswith("target_")
      and df[c].dtype in ("float64", "float32", "int64", "int32")]
df = cross_sectional_rank(df, fc)

# Clean infinities
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
df[fc] = df[fc].fillna(0)
print(f"   Final: {df.shape}, {len(fc)} features")

# OOS period: Feb 9 → Mar 7 (true out-of-sample)
oos_start = pd.Timestamp("2026-02-09", tz='UTC')
oos_end = pd.Timestamp("2026-03-07", tz='UTC')
mask = (df['timestamp'] >= oos_start) & (df['timestamp'] <= oos_end)
df_oos = df[mask].copy()
print(f"\n📅 OOS window: {oos_start.date()} → {oos_end.date()}")
print(f"   OOS rows: {len(df_oos):,}")

# Target
target_col = "target_ret_12h"
print(f"   Target: {target_col}")

# ── 2. Load all models and predict ─────────────────────────────────
import lightgbm as lgb
import torch

def _ensure_cols(df_slice, feat_names):
    """Add missing feature columns as 0 and return X matrix."""
    missing = [c for c in feat_names if c not in df_slice.columns]
    if missing:
        for c in missing:
            df_slice[c] = 0.0
        print(f"      ⚠️  {len(missing)} missing features filled with 0: {missing[:5]}...")
    return df_slice[feat_names].fillna(0)

def load_lgb_models(model_dir, label):
    """Load LightGBM models and predict."""
    fn_path = os.path.join(model_dir, "feature_names.json")
    if not os.path.exists(fn_path):
        return None, None, None
    feat_names = json.load(open(fn_path))
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.txt')])
    if not model_files:
        return None, None, None
    
    X = _ensure_cols(df_oos, feat_names).values
    preds = []
    for mf in model_files:
        m = lgb.Booster(model_file=os.path.join(model_dir, mf))
        p = m.predict(X)
        preds.append(p)
    
    avg_pred = np.mean(preds, axis=0)
    print(f"   {label}: {len(model_files)} models, {len(feat_names)} feats")
    return avg_pred, feat_names, model_files

def load_cb_models(model_dir, label):
    """Load CatBoost models and predict."""
    fn_path = os.path.join(model_dir, "feature_names.json")
    if not os.path.exists(fn_path):
        return None, None, None
    feat_names = json.load(open(fn_path))
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.cbm')])
    if not model_files:
        return None, None, None
    
    from catboost import CatBoostRegressor
    X = _ensure_cols(df_oos, feat_names).values
    preds = []
    for mf in model_files:
        m = CatBoostRegressor()
        m.load_model(os.path.join(model_dir, mf))
        p = m.predict(X)
        preds.append(p)
    
    avg_pred = np.mean(preds, axis=0)
    print(f"   {label}: {len(model_files)} models, {len(feat_names)} feats")
    return avg_pred, feat_names, model_files

def load_xgb_models(model_dir, label):
    """Load XGBoost models and predict."""
    fn_path = os.path.join(model_dir, "feature_names.json")
    if not os.path.exists(fn_path):
        return None, None, None
    feat_names = json.load(open(fn_path))
    model_files = sorted([f for f in os.listdir(model_dir) 
                          if f.endswith('.json') and f.startswith('xgb_model')])
    if not model_files:
        return None, None, None
    
    import xgboost as xgb
    X = _ensure_cols(df_oos, feat_names)
    dm = xgb.DMatrix(X, feature_names=feat_names)
    preds = []
    for mf in model_files:
        m = xgb.Booster()
        m.load_model(os.path.join(model_dir, mf))
        p = m.predict(dm)
        preds.append(p)
    
    avg_pred = np.mean(preds, axis=0)
    print(f"   {label}: {len(model_files)} models, {len(feat_names)} feats")
    return avg_pred, feat_names, model_files

def load_mlp_models(model_dir, label):
    """Load MLP models and predict."""
    fn_path = os.path.join(model_dir, "feature_names.json")
    if not os.path.exists(fn_path):
        return None, None, None
    feat_names = json.load(open(fn_path))
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pt')])
    if not model_files:
        return None, None, None
    
    from run_pipeline_mlp import AlphaMLP
    
    X = _ensure_cols(df_oos, feat_names).values
    X_tensor = torch.FloatTensor(X)
    
    preds = []
    for mf in model_files:
        ckpt = torch.load(os.path.join(model_dir, mf), map_location='cpu', weights_only=False)
        cfg = ckpt['config']
        hdims = cfg.get('hidden_dims', (256, 128, 64))
        if isinstance(hdims, list):
            hdims = tuple(hdims)
        m = AlphaMLP(input_dim=ckpt['input_dim'], hidden_dims=hdims,
                     dropout=cfg.get('dropout', 0.3))
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        with torch.no_grad():
            p = m(X_tensor).numpy()
        preds.append(p)
    
    avg_pred = np.mean(preds, axis=0)
    print(f"   {label}: {len(model_files)} models, {len(feat_names)} feats")
    return avg_pred, feat_names, model_files

print("\n🔧 Loading models & predicting on OOS...")
predictions = {}

# V6  
p, _, _ = load_lgb_models(os.path.join(ROOT, "results_v6_prod"), "lgb_v6")
if p is not None: predictions['lgb_v6'] = p

# V7
p, _, _ = load_lgb_models(os.path.join(ROOT, "results_v7_prod"), "lgb_v7")
if p is not None: predictions['lgb_v7'] = p

# CatBoost
p, _, _ = load_cb_models(os.path.join(ROOT, "results_catboost_prod"), "catboost")
if p is not None: predictions['catboost'] = p

# XGBoost  
p, _, _ = load_xgb_models(os.path.join(ROOT, "results_xgboost_prod"), "xgboost")
if p is not None: predictions['xgboost'] = p

# MLP
p, _, _ = load_mlp_models(os.path.join(ROOT, "results_mlp_prod"), "mlp")
if p is not None: predictions['mlp'] = p

print(f"\n✅ Loaded {len(predictions)} model groups")

# ── 3. Correlation matrix ──────────────────────────────────────────
print("\n" + "=" * 70)
print("  PREDICTION CORRELATION MATRIX")
print("=" * 70)

pred_df = pd.DataFrame(predictions, index=df_oos['timestamp'].values)

# Pairwise Pearson correlation
corr = pred_df.corr()
print("\nPearson correlation:")
print(corr.round(4).to_string())

# Pairwise Spearman (rank) correlation
scorr = pred_df.corr(method='spearman')
print("\nSpearman (rank) correlation:")
print(scorr.round(4).to_string())

# MLP vs GBDT average
gbdt_cols = [c for c in pred_df.columns if c != 'mlp']
if 'mlp' in pred_df.columns and gbdt_cols:
    gbdt_avg = pred_df[gbdt_cols].mean(axis=1)
    mlp_vs_gbdt_pearson = pred_df['mlp'].corr(gbdt_avg)
    mlp_vs_gbdt_spearman = pred_df['mlp'].corr(gbdt_avg, method='spearman')
    print(f"\n🔑 MLP vs GBDT-ensemble (avg of {len(gbdt_cols)}):")
    print(f"   Pearson:  {mlp_vs_gbdt_pearson:.4f}")
    print(f"   Spearman: {mlp_vs_gbdt_spearman:.4f}")

# ── 4. Per-model IC on OOS ──────────────────────────────────────────
print("\n" + "=" * 70)
print("  PER-MODEL IC (OOS)")
print("=" * 70)

y = df_oos[target_col].values

def calc_ic(pred, actual):
    """Cross-sectional IC: mean of per-timestamp rank correlations."""
    temp = pd.DataFrame({'pred': pred, 'actual': actual, 'time': df_oos['timestamp'].values})
    ics = []
    for t, grp in temp.groupby('time'):
        if len(grp) > 5:
            ic = grp['pred'].corr(grp['actual'], method='spearman')
            if not np.isnan(ic):
                ics.append(ic)
    return np.mean(ics) if ics else 0, np.std(ics) if ics else 0, len(ics)

print(f"\n{'Model':12s} {'Mean IC':>10s} {'Std IC':>10s} {'ICIR':>10s} {'N_periods':>10s}")
print("-" * 55)
model_ics = {}
for name, pred in predictions.items():
    ic_mean, ic_std, n = calc_ic(pred, y)
    icir = ic_mean / ic_std if ic_std > 0 else 0
    model_ics[name] = ic_mean
    print(f"{name:12s} {ic_mean:10.4f} {ic_std:10.4f} {icir:10.4f} {n:10d}")

# Ensemble ICs
print("\n--- Ensemble ICs ---")
# 4-group (current)
gbdt_ens = pred_df[gbdt_cols].mean(axis=1).values
ic_mean, ic_std, n = calc_ic(gbdt_ens, y)
icir = ic_mean / ic_std if ic_std > 0 else 0
print(f"{'4-grp GBDT':12s} {ic_mean:10.4f} {ic_std:10.4f} {icir:10.4f} {n:10d}")

# 5-group (with MLP)
if 'mlp' in predictions:
    all_ens = pred_df.mean(axis=1).values
    ic_mean_5, ic_std_5, n_5 = calc_ic(all_ens, y)
    icir_5 = ic_mean_5 / ic_std_5 if ic_std_5 > 0 else 0
    print(f"{'5-grp +MLP':12s} {ic_mean_5:10.4f} {ic_std_5:10.4f} {icir_5:10.4f} {n_5:10d}")
    
    # Weighted: give MLP less weight (0.5x) since it's a new model
    wt_ens = pred_df[gbdt_cols].mean(axis=1) * 0.8 + pred_df['mlp'] * 0.2
    ic_wt, ic_wt_std, n_wt = calc_ic(wt_ens.values, y)
    icir_wt = ic_wt / ic_wt_std if ic_wt_std > 0 else 0
    print(f"{'80/20 blend':12s} {ic_wt:10.4f} {ic_wt_std:10.4f} {icir_wt:10.4f} {n_wt:10d}")

# ── 5. Position overlap analysis ───────────────────────────────────
print("\n" + "=" * 70)
print("  POSITION OVERLAP (top 10 / bottom 10)")
print("=" * 70)

# For each timestamp, check which coins MLP and GBDT agree on
if 'mlp' in predictions:
    overlaps_long = []
    overlaps_short = []
    timestamps = df_oos['timestamp'].unique()
    
    for t in timestamps:
        mask_t = pred_df.index == t
        if mask_t.sum() < 20:
            continue
        
        gbdt_scores_t = gbdt_avg[mask_t].values
        mlp_scores_t = pred_df.loc[mask_t, 'mlp'].values
        symbols_t = df_oos.loc[mask_t, 'symbol'].values if 'symbol' in df_oos.columns else np.arange(mask_t.sum())
        
        n_pos = min(10, mask_t.sum() // 2)
        
        gbdt_top = set(np.argsort(gbdt_scores_t)[-n_pos:])
        mlp_top = set(np.argsort(mlp_scores_t)[-n_pos:])
        gbdt_bot = set(np.argsort(gbdt_scores_t)[:n_pos])
        mlp_bot = set(np.argsort(mlp_scores_t)[:n_pos])
        
        overlaps_long.append(len(gbdt_top & mlp_top) / n_pos)
        overlaps_short.append(len(gbdt_bot & mlp_bot) / n_pos)
    
    if overlaps_long:
        print(f"\n   Long overlap  (MLP ∩ GBDT top-10):  {np.mean(overlaps_long):.1%} avg")
        print(f"   Short overlap (MLP ∩ GBDT bot-10):  {np.mean(overlaps_short):.1%} avg")
        print(f"   → 100% = MLP identical to GBDT (useless)")
        print(f"   → 50-70% = good diversity while agreeing on key picks")
        print(f"   → <30% = too different, may hurt ensemble")

# ── 6. Summary & Recommendation ────────────────────────────────────
print("\n" + "=" * 70)
print("  VERDICT")
print("=" * 70)

if 'mlp' in model_ics:
    mlp_ic = model_ics['mlp']
    gbdt_avg_ic = np.mean([model_ics[k] for k in gbdt_cols])
    
    print(f"\n   MLP standalone IC:     {mlp_ic:.4f}")
    print(f"   GBDT average IC:      {gbdt_avg_ic:.4f}")
    if 'mlp' in pred_df.columns:
        print(f"   MLP↔GBDT correlation: {mlp_vs_gbdt_pearson:.4f}")
    
    # Decision logic
    add_to_ensemble = True
    reasons = []
    
    if mlp_ic < 0.03:
        add_to_ensemble = False
        reasons.append(f"IC too low ({mlp_ic:.4f} < 0.03)")
    
    if mlp_vs_gbdt_pearson > 0.90:
        add_to_ensemble = False
        reasons.append(f"Too correlated with GBDT ({mlp_vs_gbdt_pearson:.2f} > 0.90)")
    
    if mlp_ic > 0.05 and mlp_vs_gbdt_pearson < 0.85:
        reasons.append(f"Good IC ({mlp_ic:.4f}) + low correlation ({mlp_vs_gbdt_pearson:.2f})")
    
    if add_to_ensemble:
        print(f"\n   ✅ RECOMMENDATION: ADD MLP to ensemble")
        for r in reasons:
            print(f"      • {r}")
    else:
        print(f"\n   ❌ RECOMMENDATION: DO NOT add MLP")
        for r in reasons:
            print(f"      • {r}")

print("\n" + "=" * 70)
print("Done.")
