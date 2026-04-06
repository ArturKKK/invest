#!/usr/bin/env python3
"""Check all parquet files in data/raw for corruption."""
import pandas as pd
import glob
import sys

raw_dir = "/data/datasets/data/raw"
files = sorted(glob.glob(f"{raw_dir}/*.parquet"))
print(f"Checking {len(files)} parquet files in {raw_dir}...")

bad = []
for f in files:
    try:
        df = pd.read_parquet(f)
        if len(df) < 100:
            print(f"  SMALL: {f.split('/')[-1]:40s} {len(df)} rows")
    except Exception as e:
        bad.append((f.split('/')[-1], str(e)[:120]))
        print(f"  BAD:  {f.split('/')[-1]:40s} {str(e)[:80]}")

if bad:
    print(f"\n{len(bad)} corrupted files: {[b[0] for b in bad]}")
    sys.exit(1)
else:
    print(f"\nAll {len(files)} files OK")
