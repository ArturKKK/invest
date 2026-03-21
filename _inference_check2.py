#!/usr/bin/env python3
"""Lightweight inference check — build features, run all 4 models, compare with bot output."""
import json, os, sys, warnings, time
warnings.filterwarnings("ignore")
os.chdir("/home/trader/invest")
sys.path.insert(0, "/home/trader/invest")

import numpy as np
import pandas as pd

t0 = time.time()

# ─── Build features exactly as bot does ───
print("=" * 60)
print("STEP 1: Build features (same as run_trading.py)")
print("=" * 60)

from run_trading import (fetch_ohlcv, build_features, add_12h_features,
                         cross_sectional_rank, SYMBOLS)
from run_pipeline_v6 import add_sentiment_features, add_derivatives_features
from run_pipeline_v7 import add_12h_features as add_12h_v7

root = "/home/trader/invest"

# 1. Fetch OHLCV data
print("Fetching data...")
try:
    df = fetch_ohlcv(SYMBOLS, hours=800)
    print(f"  OHLCV: {df.shape}, symbols: {df['symbol'].nunique()}, latest: {df['timestamp'].max()}")
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

# 2. Build core features
print("Building core features...")
df = build_features(df)
print(f"  After build_features: {df.shape}")

# 3. Add 12h features
print("Adding 12h features...")
df = add_12h_features(df)
print(f"  After 12h: {df.shape}")

# 4. Enrichment (sentiment, derivatives)
# Check how run_trading.py does it
print("Adding sentiment features...")
try:
    df = add_sentiment_features(df, root, news_mode='all')
except Exception as e:
    print(f"  Sentiment error: {e}")

print("Adding derivatives features...")
try:
    df = add_derivatives_features(df, root)
except Exception as e:
    print(f"  Derivatives error: {e}")

# 5. Cross-sectional ranking
feat_cols = [c for c in df.columns if c not in ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'date']]
print(f"  Total features: {len(feat_cols)}")

# Replace inf/nan (same as build_features)
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)

# Latest row per symbol
latest = df.groupby('symbol').last().reset_index()
print(f"  Symbols in latest: {len(latest)}")

# ─── Feature health check ───
print("\n" + "=" * 60)
print("STEP 2: Feature health")
print("=" * 60)

for model_dir, name in [
    ("results_v6_huber_prod", "v6_lgb"),
    ("results_v7_huber_prod", "v7_lgb"),
    ("results_catboost_prod", "catboost"),
    ("results_xgboost_prod", "xgboost"),
]:
    feat_path = os.path.join(model_dir, "feature_names.json")
    if not os.path.exists(feat_path):
        continue
    feats = json.load(open(feat_path))
    available = [f for f in feats if f in latest.columns]
    missing = [f for f in feats if f not in latest.columns]
    
    X = latest.reindex(columns=feats, fill_value=0)
    zero_cols = [c for c in feats if (X[c] == 0).all()]
    nan_cols = [c for c in feats if X[c].isna().all()]
    
    print(f"\n  {name}: expected={len(feats)}, available={len(available)}, MISSING={len(missing)}, zero={len(zero_cols)}")
    if missing:
        print(f"    ⚠️  MISSING: {missing[:20]}")
    if zero_cols:
        print(f"    ZERO cols: {zero_cols[:20]}")

# ─── Run all 4 models ───
print("\n" + "=" * 60)
print("STEP 3: Model inference")
print("=" * 60)

import lightgbm as lgb

all_preds = {}

for model_dir, name, ext in [
    ("results_v6_huber_prod", "v6", ".txt"),
    ("results_v7_huber_prod", "v7", ".txt"),
]:
    feats = json.load(open(os.path.join(model_dir, "feature_names.json")))
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith(ext) and "model" in f])
    models = [lgb.Booster(model_file=os.path.join(model_dir, f)) for f in model_files]
    X = latest.reindex(columns=feats, fill_value=0).values
    preds = np.mean([m.predict(X) for m in models], axis=0)
    all_preds[name] = preds
    print(f"  {name}: {len(models)} models, preds [{preds.min():.6f}, {preds.max():.6f}], mean={preds.mean():.6f}, std={preds.std():.6f}")

# CatBoost
try:
    from catboost import CatBoostRegressor
    feats = json.load(open("results_catboost_prod/feature_names.json"))
    model_files = sorted([f for f in os.listdir("results_catboost_prod") if f.endswith(".cbm")])
    models = []
    for mf in model_files:
        m = CatBoostRegressor()
        m.load_model(os.path.join("results_catboost_prod", mf))
        models.append(m)
    X = latest.reindex(columns=feats, fill_value=0).values
    preds = np.mean([m.predict(X) for m in models], axis=0)
    all_preds["cb"] = preds
    print(f"  cb: {len(models)} models, preds [{preds.min():.6f}, {preds.max():.6f}], mean={preds.mean():.6f}, std={preds.std():.6f}")
except Exception as e:
    print(f"  cb ERROR: {e}")

# XGBoost
try:
    import xgboost as xgb_lib
    feats = json.load(open("results_xgboost_prod/feature_names.json"))
    model_files = sorted([f for f in os.listdir("results_xgboost_prod") if f.endswith(".json") and "model" in f])
    models = [xgb_lib.Booster(model_file=os.path.join("results_xgboost_prod", f)) for f in model_files]
    X = latest.reindex(columns=feats, fill_value=0)
    preds = np.mean([m.predict(xgb_lib.DMatrix(X.values, feature_names=feats)) for m in models], axis=0)
    all_preds["xgb"] = preds
    print(f"  xgb: {len(models)} models, preds [{preds.min():.6f}, {preds.max():.6f}], mean={preds.mean():.6f}, std={preds.std():.6f}")
except Exception as e:
    print(f"  xgb ERROR: {e}")

# ─── Ensemble (same as bot: simple mean) ───
print("\n" + "=" * 60)
print("STEP 4: Ensemble + Z-normalize (same as bot)")
print("=" * 60)

if all_preds:
    ensemble = np.mean(list(all_preds.values()), axis=0)
    
    # Z-normalize (same as generate_signal)
    score_std = np.std(ensemble)
    if score_std > 1e-10:
        z_scores = (ensemble - np.mean(ensemble)) / score_std
    else:
        z_scores = ensemble
    
    latest["score"] = z_scores
    sorted_df = latest.sort_values("score", ascending=False)
    
    print("\n  TOP 10 (LONG):")
    for _, row in sorted_df.head(10).iterrows():
        print(f"    {row['symbol']:12s} score={row['score']:+.4f}")
    
    print("\n  BOTTOM 10 (SHORT):")
    for _, row in sorted_df.tail(10).iterrows():
        print(f"    {row['symbol']:12s} score={row['score']:+.4f}")
    
    # Per-model agreement
    print("\n  Per-model correlation:")
    for n1 in all_preds:
        for n2 in all_preds:
            if n1 < n2:
                corr = np.corrcoef(all_preds[n1], all_preds[n2])[0, 1]
                print(f"    {n1} vs {n2}: {corr:.4f}")

# ─── Compare with bot signals ───
print("\n" + "=" * 60)
print("STEP 5: Compare with bot's last signals")
print("=" * 60)

import glob
latest_log = sorted(glob.glob("/home/trader/invest/trading_logs/trade_*.json"))[-1]
bot_data = json.load(open(latest_log))
print(f"  Bot log: {os.path.basename(latest_log)}")

top5 = bot_data.get("signals_top5", [])
bot5 = bot_data.get("signals_bot5", [])
print(f"  Bot TOP signals:")
for s in top5:
    sym = s.get("symbol", "?")
    bscore = s.get("score", "?")
    # Find our score for same symbol
    our = latest[latest["symbol"] == sym]
    our_score = our["score"].values[0] if len(our) > 0 else "N/A"
    print(f"    {sym:12s} bot={bscore:+.4f}  us={our_score:+.4f}" if isinstance(our_score, float) else f"    {sym:12s} bot={bscore:+.4f}  us={our_score}")

print(f"  Bot BOTTOM signals:")
for s in bot5:
    sym = s.get("symbol", "?")
    bscore = s.get("score", "?")
    our = latest[latest["symbol"] == sym]
    our_score = our["score"].values[0] if len(our) > 0 else "N/A"
    print(f"    {sym:12s} bot={bscore:+.4f}  us={our_score:+.4f}" if isinstance(our_score, float) else f"    {sym:12s} bot={bscore:+.4f}  us={our_score}")

elapsed = time.time() - t0
print(f"\n  Elapsed: {elapsed:.1f}s")
