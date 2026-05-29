#!/usr/bin/env python3
"""Check if inf→nan fix causes survivorship bias in old results."""
import numpy as np, pandas as pd
from _research_r68_continuous_wf import load_data, CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, ORIGINAL_WINDOWS

df, _ = load_data()

# After the fix, inf is already nan. Check current state:
for col in ['oi_chg_12h', 'oi_chg_24h']:
    if col in df.columns:
        print(f"{col}: inf={np.isinf(df[col]).sum()}, nan={df[col].isna().sum()}, zero={(df[col]==0).sum()}")

# Now simulate OLD behavior: put inf back where oi_chg was computed from zero OI
# We can detect these as: oi_chg is NaN AND the symbol had data at that time
# Actually, just count how many rows have NaN in oi_chg (these were inf before fix)
tz = df["timestamp"].dt.tz

print("\n=== Simulating OLD behavior (inf drops rows) ===")
for label, windows in [("CONTINUOUS", CONTINUOUS_WINDOWS), ("ORIGINAL", ORIGINAL_WINDOWS)]:
    all_ts_new = set()
    all_ts_old = set()
    for w in windows:
        te_s = pd.Timestamp(w["test_start"], tz=tz)
        te_e = pd.Timestamp(w["test_end"], tz=tz)
        test = df[(df["timestamp"] >= te_s) & (df["timestamp"] <= te_e)].copy()
        
        # NEW behavior: all rows kept (inf already nan, fillna(0) handles it)
        total_ts = test["timestamp"].nunique()
        all_ts_new.update(test["timestamp"].unique())
        
        # OLD behavior: rows with NaN in oi_chg would have had inf → dropped after replace+dropna
        # Simulate: which rows would have inf?
        oi_nan_mask = test["oi_chg_12h"].isna() | test["oi_chg_24h"].isna()
        test_old = test[~oi_nan_mask]
        old_ts = test_old["timestamp"].nunique()
        all_ts_old.update(test_old["timestamp"].unique())
        
        print(f"  {w['name']} {label}: new_ts={total_ts}, old_ts={old_ts}, "
              f"lost={total_ts-old_ts} ({100*(total_ts-old_ts)/total_ts:.1f}%), "
              f"rows_dropped={oi_nan_mask.sum()}/{len(test)}")
    
    new_periods = len(all_ts_new) // 12  # approximate
    old_periods = len(all_ts_old) // 12
    print(f"  TOTAL {label}: new_periods≈{new_periods}, old_periods≈{old_periods}")

print("\nDone")
