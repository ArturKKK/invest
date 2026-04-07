#!/usr/bin/env python3
"""Quick check: what timeframe is research data? What 4h data exists?"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from _research_r35_new_features import load_research_frame
df, _ = load_research_frame()

d = df.groupby("symbol")["timestamp"].diff().dropna()
mode_sec = d.dt.total_seconds().mode().values

fwd_cols = [c for c in df.columns if "fwd" in c]

lines = [
    f"ROWS: {len(df)}",
    f"SYMBOLS: {df.symbol.nunique()}",
    f"MODE_DIFF_SEC: {mode_sec}",
    f"FWD_COLS: {fwd_cols}",
    f"DATE_RANGE: {df.timestamp.min()} → {df.timestamp.max()}",
    f"ALL_COLS: {len(df.columns)}",
    "DONE",
]
with open("/tmp/tf_check.txt", "w") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
