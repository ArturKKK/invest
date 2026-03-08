#!/usr/bin/env python3
"""Check feature alignment between trained models and production."""
import lightgbm as lgb
import os, sys
sys.path.insert(0, '.')

# Load model features
v6_models = sorted([f for f in os.listdir('results_v6') if f.endswith('.txt')])
m6 = lgb.Booster(model_file=f'results_v6/{v6_models[0]}')
v6_feats = set(m6.feature_name())

v7_models = sorted([f for f in os.listdir('results_v7') if f.endswith('.txt')])
m7 = lgb.Booster(model_file=f'results_v7/{v7_models[0]}')
v7_feats = set(m7.feature_name())

from catboost import CatBoostRegressor
cb_models = sorted([f for f in os.listdir('results_catboost') if f.endswith('.cbm')])
cb = CatBoostRegressor()
cb.load_model(f'results_catboost/{cb_models[0]}')
cb_feats = set(cb.feature_names_)

all_model_feats = v6_feats | v7_feats | cb_feats

# Production features
from run_trading import fetch_ohlcv, build_features, SYMBOLS, EXCLUDE_COLS
raw = fetch_ohlcv(SYMBOLS, 800)
df = build_features(raw)
prod_cols = set(c for c in df.columns if c not in EXCLUDE_COLS
                and not c.startswith('target_')
                and df[c].dtype in ('float64','float32','int64','int32'))

print(f"Production features: {len(prod_cols)}")
print(f"V6: {len(v6_feats)}, V7: {len(v7_feats)}, CB: {len(cb_feats)}")
print(f"Model needs (union): {len(all_model_feats)}")

# MISSING
missing_in_prod = all_model_feats - prod_cols
print(f"\n❌ MISSING in production ({len(missing_in_prod)}):")
for f in sorted(missing_in_prod):
    tags = []
    if f in v6_feats: tags.append("v6")
    if f in v7_feats: tags.append("v7")
    if f in cb_feats: tags.append("cb")
    print(f"   [{','.join(tags):8s}]  {f}")

# EXTRA (production has but models don't use)
extra = prod_cols - all_model_feats
print(f"\n📦 EXTRA in production, unused by models ({len(extra)}):")
for f in sorted(extra):
    print(f"   {f}")

# Also check: what does the TRAINING pipeline (run_pipeline_v8) produce?
print("\n" + "="*60)
print("Checking training pipeline features (crypto_features_1h.parquet)...")
import pandas as pd
try:
    train_df = pd.read_parquet('data/features/crypto_features_1h.parquet', columns=None)
    train_cols = set(c for c in train_df.columns if c not in EXCLUDE_COLS
                     and not c.startswith('target_')
                     and train_df[c].dtype in ('float64','float32','int64','int32'))
    print(f"Training data features: {len(train_cols)}")
    
    missing_from_train = all_model_feats - train_cols
    print(f"\nModel features NOT in training parquet ({len(missing_from_train)}):")
    for f in sorted(missing_from_train):
        print(f"   {f}")
    
    extra_in_train = train_cols - all_model_feats
    print(f"\nTraining parquet features NOT used by any model ({len(extra_in_train)}):")
    for f in sorted(extra_in_train):
        print(f"   {f}")
except Exception as e:
    print(f"Could not load training parquet: {e}")
