#!/usr/bin/env python3
"""Quick check of downloaded news data."""
import pandas as pd, os
from datetime import datetime, timezone

for name, path in [
    ("checkpoint", "data/sentiment/raw_news.parquet.checkpoint"),
    ("raw_news", "data/sentiment/raw_news.parquet"),
]:
    if not os.path.exists(path):
        print(f"{name}: NOT FOUND\n")
        continue
    df = pd.read_parquet(path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    min_ts = df["published_on"].min()
    max_ts = df["published_on"].max()
    min_date = datetime.fromtimestamp(min_ts, tz=timezone.utc).strftime("%Y-%m-%d")
    max_date = datetime.fromtimestamp(max_ts, tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"{name}: {len(df):,} rows, {size_mb:.1f} MB")
    print(f"  Range: {min_date} -> {max_date}")
    print(f"  Unique titles: {df['title'].nunique():,}")
    df["date"] = pd.to_datetime(df["published_on"], unit="s", utc=True)
    monthly = df.set_index("date").resample("ME").size()
    print("  Monthly distribution:")
    for m, cnt in monthly.items():
        marker = " << LOW!" if cnt < 500 else ""
        print(f"    {m.strftime('%Y-%m')}: {cnt:>6,}{marker}")
    print()
