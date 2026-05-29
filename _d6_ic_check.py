#!/usr/bin/env python3
"""Quick IC check for d6 orderbook depth features - using mid_price for fwd return."""
import os, warnings
os.chdir("/home/trader/invest")
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# Load orderbook depth features (has mid_price)
ob = pd.read_parquet("data/features/binance_orderbook_depth_features.parquet")
print(f"OB features: {ob.shape}")
print(f"Time range: {ob.timestamp.min()} -> {ob.timestamp.max()}")
print(f"Symbols: {ob.symbol.nunique()}")
print(f"Unique hours: {ob.timestamp.nunique()}")
print(f"Columns: {list(ob.columns)}")
print()

# Forward return from mid_price
ob = ob.sort_values(["symbol", "timestamp"])
ob["fwd_ret_12h"] = ob.groupby("symbol")["mid_price"].pct_change(12).shift(-12)
ob["fwd_ret_1h"] = ob.groupby("symbol")["mid_price"].pct_change(1).shift(-1)
ob["fwd_ret_6h"] = ob.groupby("symbol")["mid_price"].pct_change(6).shift(-6)
ob["fwd_ret_24h"] = ob.groupby("symbol")["mid_price"].pct_change(24).shift(-24)

for h in ["1h", "6h", "12h", "24h"]:
    valid = ob.dropna(subset=[f"fwd_ret_{h}", "mid_price"])
    valid = valid[valid["mid_price"] > 0]
    print(f"\n=== IC with fwd_ret_{h} (valid rows: {len(valid)}, hours: {valid.timestamp.nunique()}) ===")
    ob_feats = [c for c in ob.columns if c not in ["timestamp", "symbol", "mid_price",
                "fwd_ret_1h", "fwd_ret_6h", "fwd_ret_12h", "fwd_ret_24h"]]
    
    if len(valid) < 50:
        print("  Not enough data")
        continue
    
    print(f"  {'Feature':40s} {'IC':>8s} {'ICIR':>8s} {'RankIC':>8s} {'RICIR':>8s} {'n':>6s}")
    print("  " + "-" * 80)
    
    results = []
    for feat in ob_feats:
        if feat not in valid.columns or valid[feat].notna().sum() < 50:
            continue
        target = f"fwd_ret_{h}"
        ic_s = valid.groupby("timestamp").apply(lambda g: g[feat].corr(g[target])).dropna()
        ric_s = valid.groupby("timestamp").apply(
            lambda g: g[feat].rank().corr(g[target].rank())
        ).dropna()
        if len(ic_s) > 3:
            ic, icir = ic_s.mean(), ic_s.mean()/(ic_s.std()+1e-10)
            ric, ricir = ric_s.mean(), ric_s.mean()/(ric_s.std()+1e-10)
            results.append((feat, ic, icir, ric, ricir, len(ic_s)))
    
    results.sort(key=lambda x: abs(x[4]), reverse=True)
    for feat, ic, icir, ric, ricir, n in results:
        marker = " ***" if abs(ricir) > 0.15 else ""
        print(f"  {feat:40s} {ic:+.4f}   {icir:+.3f}   {ric:+.4f}   {ricir:+.3f}   {n:>5d}{marker}")

# Cross-correlation between OB features
print("\n=== Cross-correlation between OB features ===")
ob_num = [c for c in ob.columns if c not in ["timestamp", "symbol", "mid_price",
          "fwd_ret_1h", "fwd_ret_6h", "fwd_ret_12h", "fwd_ret_24h"] and ob[c].dtype in ["float64","float32","int64"]]
corr = ob[ob_num].corr()
print(corr.round(2).to_string())

# Data continuity check
print("\n=== Data continuity ===")
for sym in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
    subset = ob[ob.symbol == sym].sort_values("timestamp")
    if len(subset) > 0:
        gaps = subset["timestamp"].diff().dt.total_seconds() / 3600
        print(f"  {sym}: {len(subset)} rows, max_gap={gaps.max():.1f}h, mean_gap={gaps.mean():.1f}h")

print("\nDone.")
