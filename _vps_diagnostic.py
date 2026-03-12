#!/usr/bin/env python3
"""VPS diagnostic: run on server to check feature health."""
import json, os, sys, pathlib

root = '/home/trader/invest'
os.chdir(root)

print("=== MODEL FEATURES ===")
for name, path in [("v6", "results_v6_prod/feature_names.json"),
                    ("v7", "results_v7_prod/feature_names.json"),
                    ("cb", "results_catboost_prod/feature_names.json")]:
    if os.path.exists(path):
        feats = json.load(open(path))
        print(f"  {name}: {len(feats)} features")
    else:
        print(f"  {name}: MISSING feature_names.json")

# Deriv model
deriv_path = "results/production/deriv_only"
if os.path.isdir(deriv_path):
    files = os.listdir(deriv_path)
    print(f"  deriv: {files}")
else:
    deriv_dirs = []
    for d in ["results_deriv", "results/exp15_new_features/deriv_only"]:
        if os.path.isdir(d):
            deriv_dirs.append(d)
    print(f"  deriv: dir missing, alternatives: {deriv_dirs}")

# Symlinks
print("\n=== SYMLINKS ===")
for name in ["lgb_v6_no_news", "lgb_v7_no_news", "catboost_with_news", "catboost_no_news", "deriv_only", "xgboost"]:
    p = pathlib.Path(f"results/production/{name}")
    if p.is_symlink():
        target = os.readlink(p)
        resolved = p.resolve()
        print(f"  {name} -> {target} (resolved={resolved}, exists={resolved.exists()})")
    elif p.exists():
        print(f"  {name}: regular dir")
    else:
        print(f"  {name}: MISSING!")

# Check data files
print("\n=== DATA FILES ===")
import pandas as pd
for name, path in [
    ("funding_rates (OKX)", "data/sentiment/funding_rates.parquet"),
    ("long_short_ratio", "data/sentiment/long_short_ratio.parquet"),
    ("binance_futures_metrics", "data/sentiment/binance_futures_metrics.parquet"),
    ("binance_funding_rates", "data/sentiment/binance_funding_rates.parquet"),
    ("binance_premium_index", "data/sentiment/binance_premium_index.parquet"),
    ("fear_greed", "data/sentiment/fear_greed.parquet"),
    ("crypto_news", "data/sentiment/crypto_news.parquet"),
]:
    if os.path.exists(path):
        df = pd.read_parquet(path)
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
            print(f"  {name}: {len(df):,} rows, {ts.min()} → {ts.max()}")
        else:
            print(f"  {name}: {len(df):,} rows, cols={list(df.columns)[:5]}")
    else:
        print(f"  {name}: MISSING!")

# Now compare VPS model features with local production model features
# Focus: features model expects vs what run_trading can produce
print("\n=== FEATURE AVAILABILITY CHECK ===")
sys.path.insert(0, root)

import numpy as np 

v6_feats = json.load(open("results_v6_prod/feature_names.json")) if os.path.exists("results_v6_prod/feature_names.json") else []
v7_feats = json.load(open("results_v7_prod/feature_names.json")) if os.path.exists("results_v7_prod/feature_names.json") else []
cb_feats = json.load(open("results_catboost_prod/feature_names.json")) if os.path.exists("results_catboost_prod/feature_names.json") else []

# Check if feature_names saved in model files match the JSON
import lightgbm as lgb
for name, fdir in [("v6", "results_v6_prod"), ("v7", "results_v7_prod")]:
    model_files = sorted(pathlib.Path(fdir).glob("lgb_model_seed_*.txt"))
    if model_files:
        m = lgb.Booster(model_file=str(model_files[0]))
        model_feats = m.feature_name()
        json_feats = json.load(open(f"{fdir}/feature_names.json"))
        if model_feats == json_feats:
            print(f"  {name}: model feats == JSON feats ({len(model_feats)})")
        else:
            diff_m = set(model_feats) - set(json_feats)
            diff_j = set(json_feats) - set(model_feats)
            print(f"  {name}: MISMATCH! model has extra: {diff_m}, json has extra: {diff_j}")

# Check latest trade log for feature health
print("\n=== LATEST TRADE LOG ===")
log_dir = os.path.join(root, "trading_logs")
logs = sorted([f for f in os.listdir(log_dir) if f.startswith("trade_") and f.endswith(".json")])
if logs:
    latest_log = os.path.join(log_dir, logs[-1])
    with open(latest_log) as f:
        log_data = json.load(f)
    print(f"  Latest: {logs[-1]}")
    print(f"  Timestamp: {log_data.get('timestamp')}")
    print(f"  Mode: {log_data.get('mode')}")
    positions = log_data.get('positions', [])
    print(f"  Positions: {len(positions)}")
    if positions:
        for p in positions[:3]:
            print(f"    {p['symbol']} {p['side']} ${p['usd']:.0f} score={p['score']}")
    signals = log_data.get('signals_top5', [])
    if signals:
        print(f"  Top signals: {signals[:3]}")
    signals_bot = log_data.get('signals_bot5', [])
    if signals_bot:
        print(f"  Bottom signals: {signals_bot[:3]}")
else:
    print("  No trade logs found")

# Check systemd service status
print("\n=== SERVICE STATUS ===")
import subprocess
result = subprocess.run(["systemctl", "is-active", "crypto-trader"], capture_output=True, text=True)
print(f"  crypto-trader: {result.stdout.strip()}")
result = subprocess.run(["journalctl", "-u", "crypto-trader", "-n", "50", "--no-pager"], capture_output=True, text=True)
lines = result.stdout.strip().split('\n')
# Show last 50 lines of relevant output
for line in lines[-50:]:
    print(f"  {line}")
