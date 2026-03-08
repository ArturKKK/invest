"""Run inference + analyze signals. Save results to experiments log."""
import run_trading as rt
import pandas as pd, numpy as np, os, json
from datetime import datetime, timezone

root = os.path.dirname(os.path.abspath(rt.__file__))

# 1. Fetch data
print("Fetching data...")
df = rt.fetch_ohlcv(rt.SYMBOLS, hours=800)
print(f"  Shape: {df.shape}")

# 2. Build features
print("Building features...")
df_feat = rt.build_features(df)
feat_cols = [c for c in df_feat.columns if c not in rt.EXCLUDE_COLS]
print(f"  Features: {len(feat_cols)}")

# 3. Generate signals
print("Generating signals...")
signals = rt.generate_signal(df_feat, feat_cols, root)
print(f"  Signals: {len(signals)} coins")

# 4. Full signal table
print("\n=== ALL SIGNALS ===")
for _, row in signals.iterrows():
    sym = row["symbol"]
    sc = row["score"]
    tradeable = sym in rt.SYMBOLS_TO_OKX
    mark = "T" if tradeable else "X"
    above = "*" if abs(sc) >= 1.0 else " "
    print(f"  [{mark}]{above} {sym:15s} score={sc:+.4f}")

# 5. Tradeable shorts analysis
print("\n=== TRADEABLE SHORTS ===")
shorts = signals[
    (signals["score"] < 0) & (signals["symbol"].isin(rt.SYMBOLS_TO_OKX.keys()))
]
for _, r in shorts.iterrows():
    print(f"  {r['symbol']:15s} score={r['score']:+.4f}")
strong_shorts = shorts[shorts["score"].abs() >= 1.0]
print(f"\nTradeable shorts total: {len(shorts)}")
print(f"Tradeable shorts |score| >= 1.0: {len(strong_shorts)}")

# 6. Summary
all_pos = signals[signals["score"] > 0]
all_neg = signals[signals["score"] < 0]
print(f"\n=== SUMMARY ===")
print(f"Positive signals: {len(all_pos)}, Negative: {len(all_neg)}")
print(f"Mean positive score: {all_pos['score'].mean():.4f}")
print(f"Mean negative score: {all_neg['score'].mean():.4f}")
print(f"Max score: {signals['score'].max():.4f} ({signals.iloc[0]['symbol']})")
print(f"Min score: {signals['score'].min():.4f} ({signals.iloc[-1]['symbol']})")

tradeable_mask = signals["symbol"].isin(rt.SYMBOLS_TO_OKX.keys())
tradeable_sigs = signals[tradeable_mask]
print(f"\nTradeable coins: {len(tradeable_sigs)} / {len(signals)}")
t_longs = tradeable_sigs[tradeable_sigs["score"] > 1.0]
t_shorts = tradeable_sigs[tradeable_sigs["score"] < -1.0]
print(f"Tradeable longs |score|>1.0: {len(t_longs)}")
print(f"Tradeable shorts |score|>1.0: {len(t_shorts)}")

# 7. Save experiment log
now = datetime.now(timezone.utc)
log_entry = {
    "timestamp": now.isoformat(),
    "features_count": len(feat_cols),
    "models": "LGB_v6(5) + LGB_v7(5) + CatBoost(5) = 15",
    "config": {
        "min_score": rt.DEFAULT_RISK["min_score"],
        "edge_boost": rt.DEFAULT_RISK["edge_boost"],
        "leverage": rt.DEFAULT_RISK["leverage"],
        "rebal": "12h",
    },
    "signals_count": len(signals),
    "tradeable_count": len(tradeable_sigs),
    "mean_pos_score": round(float(all_pos["score"].mean()), 4),
    "mean_neg_score": round(float(all_neg["score"].mean()), 4),
    "tradeable_longs_gt1": len(t_longs),
    "tradeable_shorts_gt1": len(t_shorts),
    "top_5_longs": [
        {"symbol": r["symbol"], "score": round(r["score"], 4)}
        for _, r in signals.head(5).iterrows()
    ],
    "top_5_shorts": [
        {"symbol": r["symbol"], "score": round(r["score"], 4)}
        for _, r in signals.tail(5).iterrows()
    ],
    "notes": "Post feature-fix: funding_rate, long_short_ratio, cross_coin_dispersion now included",
}

log_path = os.path.join(root, "experiments_log.jsonl")
with open(log_path, "a") as f:
    f.write(json.dumps(log_entry, default=str) + "\n")
print(f"\n📝 Experiment logged to {log_path}")
