#!/usr/bin/env python3
"""Diagnostic: check zero features and model health on VPS."""
import sys, os, json, glob
sys.path.insert(0, '/home/trader/invest')
os.chdir('/home/trader/invest')

print("=" * 60)
print("FEATURE & MODEL DIAGNOSTIC")  
print("=" * 60)

# 1. Check which features models expect
print("\n--- 1. MODEL FEATURE LISTS ---")
for d in ['results_v6_huber_prod', 'results_v7_huber_prod', 'results_catboost_prod', 'results_xgboost_prod']:
    feat_file = f'/home/trader/invest/{d}/feature_names.json'
    if os.path.exists(feat_file):
        feats = json.load(open(feat_file))
        print(f"  {d}: {len(feats)} features")
        # Show first/last 5
        print(f"    first 5: {feats[:5]}")
        print(f"    last 5: {feats[-5:]}")
    else:
        print(f"  {d}: NO feature_names.json!")

# 2. Quick data pipeline check
print("\n--- 2. BUILDING FEATURES (like bot does) ---")
try:
    from src.features.build_features import engineer_features
    from src.data.download_data import download_multi_symbol
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    df = download_multi_symbol(symbols, hours=100, exchange='binance', interval='1h')
    print(f"  Raw: {df.shape}")
    
    df = engineer_features(df)
    print(f"  After engineer_features: {df.shape}")
    
    from run_pipeline_v6 import add_all_sentiment
    df = add_all_sentiment(df)
    print(f"  After sentiment: {df.shape}")
    
    from run_pipeline_v6 import add_derivatives_features
    df = add_derivatives_features(df)
    print(f"  After derivatives: {df.shape}")
    
    # Check zero columns
    numeric = df.select_dtypes('number')
    all_zero = [c for c in numeric.columns if (numeric[c] == 0).all()]
    all_nan = [c for c in numeric.columns if numeric[c].isna().all()]
    
    print(f"\n  ALL-ZERO columns: {len(all_zero)}")
    for c in sorted(all_zero):
        print(f"    {c}")
    
    print(f"\n  ALL-NaN columns: {len(all_nan)}")
    for c in sorted(all_nan):
        print(f"    {c}")
    
    # Check last row for each model's features
    print("\n--- 3. FEATURES MISSING/ZERO PER MODEL ---")
    last = df.iloc[-1]
    for d in ['results_v6_huber_prod', 'results_v7_huber_prod', 'results_catboost_prod', 'results_xgboost_prod']:
        feat_file = f'/home/trader/invest/{d}/feature_names.json'
        if not os.path.exists(feat_file):
            continue
        model_feats = json.load(open(feat_file))
        missing = [f for f in model_feats if f not in df.columns]
        present = [f for f in model_feats if f in df.columns]
        zero_feats = [f for f in present if last[f] == 0]
        nan_feats = [f for f in present if str(last[f]) == 'nan']
        
        print(f"\n  {d} ({len(model_feats)} features):")
        print(f"    Missing from data: {len(missing)}")
        if missing:
            print(f"      {missing[:10]}")
        print(f"    Zero in last row: {len(zero_feats)}")
        if zero_feats:
            print(f"      {zero_feats[:15]}")
        print(f"    NaN in last row: {len(nan_feats)}")
        if nan_feats:
            print(f"      {nan_feats[:15]}")

except Exception as e:
    import traceback
    print(f"  ERROR: {e}")
    traceback.print_exc()

# 4. Check model file integrity
print("\n--- 4. MODEL FILE SIZES ---")
for d in ['results_v6_huber_prod', 'results_v7_huber_prod', 'results_catboost_prod', 'results_xgboost_prod']:
    path = f'/home/trader/invest/{d}'
    if not os.path.isdir(path):
        print(f"  {d}: MISSING!")
        continue
    files = sorted(os.listdir(path))
    print(f"  {d}/:")
    for f in files:
        fp = os.path.join(path, f)
        sz = os.path.getsize(fp)
        print(f"    {f}: {sz:,} bytes")

# 5. Check last few trade logs for score distribution
print("\n--- 5. RECENT TRADE SCORES ---")
logs = sorted(glob.glob('/home/trader/invest/trading_logs/trade_*.json'))
for log_path in logs[-3:]:
    d = json.load(open(log_path))
    ts = d.get('timestamp', '?')[:19]
    positions = d.get('positions', [])
    scores = [p.get('score', 0) for p in positions]
    sides = [p.get('side', '?') for p in positions]
    
    n_long = sum(1 for s in sides if s == 'long')
    n_short = sum(1 for s in sides if s == 'short')
    
    print(f"\n  {os.path.basename(log_path)} ({ts}):")
    print(f"    Positions: {len(positions)} (L={n_long}, S={n_short})")
    if scores:
        print(f"    Score range: [{min(scores):.4f}, {max(scores):.4f}]")
        print(f"    Score mean: {sum(scores)/len(scores):.4f}")
    
    for p in positions[:3]:
        print(f"      {p.get('symbol')} {p.get('side')} score={p.get('score',0):.4f} usd={p.get('usd',0):.0f}")

print("\n" + "=" * 60)
print("DONE")
