#!/usr/bin/env python3
"""Deep model inference diagnostic — compare training feature stats with live inference."""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/trader/invest")

import numpy as np
import pandas as pd

# ========================
# 1. Build features exactly as the bot does
# ========================
print("=" * 60)
print("STEP 1: Building features (same as bot)")
print("=" * 60)

from run_pipeline_v6 import load_data as load_data_v6, build_features as build_features_v6
from run_pipeline_v7 import load_data as load_data_v7, build_features as build_features_v7

# Load raw data
print("Loading data...")
try:
    prices_v6, funding_v6, sentiment_v6 = load_data_v6()
    print(f"  v6: prices={prices_v6.shape}, funding={funding_v6.shape if funding_v6 is not None else None}")
except Exception as e:
    print(f"  v6 load error: {e}")
    prices_v6 = None

try:
    prices_v7, funding_v7, sentiment_v7 = load_data_v7()
    print(f"  v7: prices={prices_v7.shape}, funding={funding_v7.shape if funding_v7 is not None else None}")
except Exception as e:
    print(f"  v7 load error: {e}")
    prices_v7 = None

# Build features
if prices_v6 is not None:
    print("\nBuilding v6 features...")
    try:
        df_v6 = build_features_v6(prices_v6, funding_v6, sentiment_v6)
        print(f"  v6 features shape: {df_v6.shape}")
        # Get the latest row per symbol
        if "symbol" in df_v6.columns:
            latest_v6 = df_v6.groupby("symbol").last()
        else:
            latest_v6 = df_v6.iloc[-20:]  # last 20 rows
    except Exception as e:
        print(f"  v6 build error: {e}")
        import traceback; traceback.print_exc()
        latest_v6 = None

# ========================
# 2. Check feature distributions
# ========================
print("\n" + "=" * 60)
print("STEP 2: Feature distributions (latest data)")
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
    
    if latest_v6 is not None:
        available = [f for f in feats if f in latest_v6.columns]
        missing = [f for f in feats if f not in latest_v6.columns]
        
        if available:
            subset = latest_v6[available]
            zero_cols = [c for c in available if (subset[c] == 0).all()]
            nan_cols = [c for c in available if subset[c].isna().all()]
            
            # Feature stats
            desc = subset.describe()
            
            print(f"\n  {name} ({model_dir}):")
            print(f"    Expected: {len(feats)}, Available: {len(available)}, Missing: {len(missing)}")
            print(f"    All-zero: {len(zero_cols)}, All-NaN: {len(nan_cols)}")
            if missing:
                print(f"    MISSING features: {missing[:15]}")
            if zero_cols:
                print(f"    ZERO features: {zero_cols[:15]}")

            # Check for features with suspicious distributions
            if len(available) > 0:
                means = subset.mean()
                stds = subset.std()
                # Extremely large values
                extreme = means[means.abs() > 100]
                if len(extreme) > 0:
                    print(f"    ⚠️ EXTREME means (>100): {dict(extreme.head(10))}")
                # Very low variance
                low_var = stds[stds < 1e-10]
                if len(low_var) > 0:
                    print(f"    ⚠️ LOW variance (<1e-10): {list(low_var.index[:10])}")
        else:
            print(f"\n  {name}: NO matching features in built data")
    else:
        print(f"\n  {name}: No v6 feature data available")

# ========================
# 3. Run actual model inference
# ========================
print("\n" + "=" * 60)
print("STEP 3: Running model inference")
print("=" * 60)

import lightgbm as lgb

for model_dir, name in [
    ("results_v6_huber_prod", "v6_lgb"),
    ("results_v7_huber_prod", "v7_lgb"),
]:
    feat_path = os.path.join(model_dir, "feature_names.json")
    if not os.path.exists(feat_path):
        continue
    feats = json.load(open(feat_path))
    
    # Load one model
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".txt")])
    if not model_files:
        continue
    
    model_path = os.path.join(model_dir, model_files[0])
    model = lgb.Booster(model_file=model_path)
    
    if latest_v6 is not None:
        # Prepare features exactly as bot does
        X = latest_v6.reindex(columns=feats, fill_value=0)
        preds = model.predict(X.values)
        
        print(f"\n  {name} ({model_files[0]}):")
        print(f"    Predictions shape: {preds.shape}")
        print(f"    Range: [{preds.min():.6f}, {preds.max():.6f}]")
        print(f"    Mean: {preds.mean():.6f}, Std: {np.std(preds):.6f}")
        
        # Show top/bottom predictions with symbols
        if hasattr(X, "index"):
            pred_series = pd.Series(preds, index=X.index)
            top5 = pred_series.nlargest(5)
            bot5 = pred_series.nsmallest(5)
            print(f"    TOP 5 (LONG): {dict(top5)}")
            print(f"    BOT 5 (SHORT): {dict(bot5)}")

# ========================
# 4. Check CatBoost & XGBoost
# ========================
print("\n  --- CatBoost ---")
try:
    from catboost import CatBoostRegressor
    feat_path = "results_catboost_prod/feature_names.json"
    feats = json.load(open(feat_path))
    model_files = sorted([f for f in os.listdir("results_catboost_prod") if f.endswith(".cbm")])
    if model_files and latest_v6 is not None:
        model = CatBoostRegressor()
        model.load_model(os.path.join("results_catboost_prod", model_files[0]))
        X = latest_v6.reindex(columns=feats, fill_value=0)
        preds = model.predict(X.values)
        print(f"    Predictions: [{preds.min():.6f}, {preds.max():.6f}], mean={preds.mean():.6f}, std={np.std(preds):.6f}")
        pred_series = pd.Series(preds, index=X.index)
        print(f"    TOP 5: {dict(pred_series.nlargest(5))}")
        print(f"    BOT 5: {dict(pred_series.nsmallest(5))}")
except Exception as e:
    print(f"    Error: {e}")

print("\n  --- XGBoost ---")
try:
    import xgboost as xgb
    feat_path = "results_xgboost_prod/feature_names.json"
    feats = json.load(open(feat_path))
    model_files = sorted([f for f in os.listdir("results_xgboost_prod") if f.endswith(".json") and "model" in f])
    if model_files and latest_v6 is not None:
        model = xgb.Booster()
        model.load_model(os.path.join("results_xgboost_prod", model_files[0]))
        X = latest_v6.reindex(columns=feats, fill_value=0)
        dmat = xgb.DMatrix(X.values, feature_names=feats)
        preds = model.predict(dmat)
        print(f"    Predictions: [{preds.min():.6f}, {preds.max():.6f}], mean={preds.mean():.6f}, std={np.std(preds):.6f}")
        pred_series = pd.Series(preds, index=X.index)
        print(f"    TOP 5: {dict(pred_series.nlargest(5))}")
        print(f"    BOT 5: {dict(pred_series.nsmallest(5))}")
except Exception as e:
    print(f"    Error: {e}")

# ========================
# 5. Compare with what bot produced
# ========================
print("\n" + "=" * 60)
print("STEP 4: Compare with bot signals")
print("=" * 60)
import glob
latest_log = sorted(glob.glob("/home/trader/invest/trading_logs/trade_*.json"))[-1]
bot_data = json.load(open(latest_log))
top5 = bot_data.get("signals_top5", [])
bot5 = bot_data.get("signals_bot5", [])
print(f"Bot signals from {os.path.basename(latest_log)}:")
for s in top5:
    sym = s.get("symbol", "?")
    score = s.get("score", "?")
    print(f"  LONG:  {sym:12s} score={score:+.4f}")
for s in bot5:
    sym = s.get("symbol", "?")
    score = s.get("score", "?")
    print(f"  SHORT: {sym:12s} score={score:+.4f}")
