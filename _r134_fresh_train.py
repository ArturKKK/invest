"""R134 — baseline trained right up to OOS start.

train_end = 2026-03-15 (3 days before OOS), val = 2026-03-15..2026-03-17 (2 days),
test = 2026-03-18..2026-04-25. Most honest baseline = use ALL available data
right before the test period.
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

W4_FRESH = {
    "name": "W4_FRESH",
    "train_end": "2026-03-15",
    "val_start": "2026-03-15",
    "val_end": "2026-03-17",
    "test_start": "2026-03-18",
    "test_end": "2026-04-25",
}


def main():
    t0 = time.time()
    print("=" * 70)
    print("  R134 — baseline trained up to 2026-03-15 (3d before OOS)")
    print("=" * 70)
    print(f"  W4_FRESH = {W4_FRESH}")

    df, regime_df = r68.load_data()
    feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]
    print(f"  Features: {len(feats)}/31 (no_rank: {len(no_rank)})")

    print(f"\n  Training (seeds={SEEDS})...")
    t1 = time.time()
    preds = r68.train_ensemble(df, feats, [W4_FRESH], seeds=SEEDS, cs_rank_exclude=no_rank)
    elapsed = time.time() - t1

    if preds is None or len(preds) == 0:
        print("  ❌ No predictions"); sys.exit(1)

    print(f"\n  Done in {elapsed:.0f}s, preds: {len(preds):,} rows")
    out = "cache/r134_fresh_preds.parquet"
    preds.to_parquet(out, index=False)
    print(f"  ✅ Saved: {out}")
    print(f"  TOTAL: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    import _preflight_check  # noqa
    main()
