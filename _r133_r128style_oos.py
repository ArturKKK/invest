"""R133 — R128-style baseline on the fresh 39-day OOS window.

Difference from R132: train_end = 2025-07-01 (matches R128 W3 cutoff).
Tests whether the R128-deployed model architecture still generalises to
2026-03-18 → 2026-04-25, with no benefit from a newer training cutoff.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

import pandas as pd

import _research_r68_continuous_wf as r68
from _research_r22_models import SEEDS
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r68_continuous_wf import CHAMPION_FEAT_31

# R128 W3 cutoff (train_end=2025-07-01). Same model used in deployed R128 baseline.
# val window kept the same 2.5-month buffer style as W3 original.
W4_R128 = {
    "name": "W4_R128",
    "train_end": "2025-07-01",
    "val_start": "2025-07-01",
    "val_end": "2025-09-15",
    "test_start": "2026-03-18",
    "test_end": "2026-04-25",
}


def main():
    t0 = time.time()
    print("=" * 70)
    print("  R133 — R128-style baseline on fresh OOS")
    print("=" * 70)
    print(f"  W4_R128 = {W4_R128}")

    df, regime_df = r68.load_data()
    print(f"\n  Frame range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    r_idx = regime_df.index if regime_df.index.name == "timestamp" else regime_df["timestamp"]
    print(f"  Regime range: {r_idx.min()} → {r_idx.max()}")

    feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]
    print(f"  Features: {len(feats)}/31 (no_rank: {len(no_rank)})")

    print(f"\n  Training ensemble on W4_R128 (seeds={SEEDS})...")
    t1 = time.time()
    preds = r68.train_ensemble(
        df, feats, [W4_R128], seeds=SEEDS, cs_rank_exclude=no_rank
    )
    elapsed = time.time() - t1

    if preds is None or len(preds) == 0:
        print("  ❌ No predictions generated — aborting.")
        sys.exit(1)

    print(f"\n  Done in {elapsed:.0f}s")
    print(f"  OOS preds: {len(preds):,} rows")
    print(f"  Range: {preds['timestamp'].min()} → {preds['timestamp'].max()}")
    print(f"  Symbols: {preds['symbol'].nunique()}")

    out = "cache/r133_r128style_preds.parquet"
    preds.to_parquet(out, index=False)
    print(f"  ✅ Saved: {out}")

    print(f"\n  TOTAL: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    import _preflight_check  # noqa
    main()
