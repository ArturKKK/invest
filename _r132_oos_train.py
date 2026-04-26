"""R132 — OOS forward-test data generation.

Trains the r68 LGB+XGB ensemble on a fresh window W4_OOS that holds out
2026-03-18 → 2026-04-25 as the OOS test set. Reuses r68.train_ensemble
without modifying r68 itself (HARD invariant).

Inputs (refreshed 2026-04-26 morning UTC):
  data/raw/*_1h.parquet         (OHLCV → 2026-04-25 23:00)
  data/sentiment/binance_*.parquet (funding/futures/premium → 2026-04-26)
  data/sentiment/deribit_dvol.parquet (→ 2026-04-26)
  data/raw/coinglass/*.parquet  (taker → 2026-04-25, rest → 2026-04-05)

Outputs:
  cache/r132_oos_preds.parquet  (OOS test predictions for W4_OOS)
  cache/r132_regime_oos.parquet (regime frame extended to fresh data)
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

# Frozen W4 OOS window. test_start strictly > W3 test_end (2026-03-17)
# per critic R130: "OOS forward-test (40 дней после 2026-03-17) — единственный тест"
W4_OOS = {
    "name": "W4_OOS",
    "train_end": "2026-01-01",
    "val_start": "2026-01-01",
    "val_end": "2026-03-15",
    "test_start": "2026-03-18",
    "test_end": "2026-04-25",
}


def main():
    t0 = time.time()
    print("=" * 70)
    print("  R132 — OOS forward-test prediction generation")
    print("=" * 70)
    print(f"  W4_OOS = {W4_OOS}")

    df, regime_df = r68.load_data()
    print(f"\n  Frame range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    r_idx = regime_df.index if regime_df.index.name == "timestamp" else regime_df["timestamp"]
    print(f"  Regime range: {r_idx.min()} → {r_idx.max()}")

    feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]
    print(f"  Features: {len(feats)}/31 (no_rank: {len(no_rank)})")

    print(f"\n  Training ensemble on W4_OOS (seeds={SEEDS})...")
    t1 = time.time()
    preds = r68.train_ensemble(
        df, feats, [W4_OOS], seeds=SEEDS, cs_rank_exclude=no_rank
    )
    elapsed = time.time() - t1

    if preds is None or len(preds) == 0:
        print("  ❌ No predictions generated — aborting.")
        sys.exit(1)

    print(f"\n  Done in {elapsed:.0f}s")
    print(f"  OOS preds: {len(preds):,} rows")
    print(f"  Range: {preds['timestamp'].min()} → {preds['timestamp'].max()}")
    print(f"  Symbols: {preds['symbol'].nunique()}")

    out = "cache/r132_oos_preds.parquet"
    preds.to_parquet(out, index=False)
    print(f"  ✅ Saved: {out}")

    out_r = "cache/r132_regime_oos.parquet"
    regime_df.to_parquet(out_r)
    print(f"  ✅ Saved: {out_r}")

    print(f"\n  TOTAL: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
