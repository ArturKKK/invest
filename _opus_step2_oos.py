"""Opus STEP 2 — OOS forward-test on fresh data (2026-03-18 → 2026-04-25).

Trains R132 (cutoff 2026-01-01) and R134 (cutoff 2026-03-15) using the
SAME canonical pipeline as R128 (r68.train_ensemble — per-split CS-rank,
NOT global). Saves preds, then runs four configs through r131.simulate_prod
on the OOS slice:
  - baseline (no overlay)
  - A1 always (trend_thr=0.25, weak_scale=0.60)
  - A1 always (trend_thr=0.25, weak_scale=0.50)
  - Gated A1 (R129 frozen: L=720, q=0.20)

Output: a single FINAL_TABLE comparing all model×overlay combinations
on the same 39-day OOS window, plus PROD_deployed (current_prod_cls_oos_preds)
as reference.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from _preflight_check import check_versions
check_versions()

import _research_r68_continuous_wf as r68
from _research_r22_models import SEEDS
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r68_continuous_wf import CHAMPION_FEAT_31

import _r128_all_overlays_canonical as r128
import _r129_persistence_gate as r129
import _r130_validate_r129 as r130
import _r131_prod_sim_validate as r131

CACHE = Path("cache")

OOS_START = "2026-03-18"
OOS_END   = "2026-04-25"  # inclusive (test_end semantic in r68)

WINDOWS = {
    "R128_W4": {  # train_end=R128 (2025-07-01), eval on fresh OOS
        "name": "W4_R128",
        "train_end": "2025-07-01", "val_start": "2025-07-01", "val_end": "2025-09-15",
        "test_start": OOS_START, "test_end": OOS_END,
    },
    "R132_W4": {
        "name": "W4_R132",
        "train_end": "2026-01-01", "val_start": "2026-01-01", "val_end": "2026-03-15",
        "test_start": OOS_START, "test_end": OOS_END,
    },
    "R134_W4": {
        "name": "W4_R134",
        "train_end": "2026-03-15", "val_start": "2026-03-15", "val_end": "2026-03-17",
        "test_start": OOS_START, "test_end": OOS_END,
    },
}

OUTPUTS = {
    "R128_W4": CACHE / "opus_r128_w4_preds.parquet",
    "R132_W4": CACHE / "opus_r132_w4_preds.parquet",
    "R134_W4": CACHE / "opus_r134_w4_preds.parquet",
    "REGIME":  CACHE / "opus_oos_regime.parquet",
}


def line(c="="):
    print(c * 78)


def hdr(t):
    line()
    print(f"  {t}")
    line()


def train_one(df, feats, no_rank, key, win):
    out = OUTPUTS[key]
    if out.exists():
        print(f"  [{key}] reusing cached preds: {out}")
        return pd.read_parquet(out)
    print(f"\n  [{key}] train_ensemble: {win}")
    t0 = time.time()
    preds = r68.train_ensemble(df, feats, [win], seeds=SEEDS, cs_rank_exclude=no_rank)
    if preds is None or len(preds) == 0:
        print(f"  [{key}] EMPTY preds")
        return None
    preds.to_parquet(out, index=False)
    print(f"  [{key}] done in {time.time()-t0:.0f}s, rows={len(preds):,}")
    return preds


def metrics_row(label, port):
    arr = port["net_ret"].values
    return {
        "label": label,
        "n": len(port),
        "act": int((~port["risk_off"]).sum()) if "risk_off" in port.columns else len(port),
        "S":   round(r130.sharpe(arr), 3),
        "Sortino": round(r130.sortino(arr), 3),
        "DD%": round(r130.max_drawdown(arr) * 100, 2),
        "ret%": round(arr.sum() * 100, 2),
        "mean_bp": round(arr.mean() * 1e4, 2),
        "gate_on": int(port["gate_on"].sum()) if "gate_on" in port.columns else 0,
    }


def run_all_overlays(preds_full, regime_aug, persist_col, label_prefix, oos_only=True):
    rows = []
    # Slice OOS only? r131.simulate_prod loops over all preds timestamps anyway,
    # so we should pre-slice preds to OOS window so the cumulative DD makes sense.
    if oos_only:
        ws = pd.Timestamp(OOS_START, tz="UTC")
        we = pd.Timestamp(OOS_END,   tz="UTC") + pd.Timedelta(days=1)
        p = preds_full[(preds_full["timestamp"] >= ws) & (preds_full["timestamp"] < we)].copy()
    else:
        p = preds_full

    # baseline
    port = r131.simulate_prod(p, regime_aug, 4, 2)
    rows.append(metrics_row(f"{label_prefix} baseline", port))
    # A1 0.50
    port = r131.simulate_prod(p, regime_aug, 4, 2,
                              a1_cfg={"trend_thr": 0.25, "weak_scale": 0.50})
    rows.append(metrics_row(f"{label_prefix} A1(0.25,0.50)", port))
    # A1 0.60
    port = r131.simulate_prod(p, regime_aug, 4, 2,
                              a1_cfg={"trend_thr": 0.25, "weak_scale": 0.60})
    rows.append(metrics_row(f"{label_prefix} A1(0.25,0.60)", port))
    # gated
    thr = r129.expanding_quantile_threshold(regime_aug[persist_col], q=0.20, min_periods=720)
    port = r131.simulate_prod(p, regime_aug, 4, 2,
                              a1_cfg={"trend_thr": 0.25, "weak_scale": 0.60},
                              gate_persist_col=persist_col,
                              gate_threshold_series=thr)
    rows.append(metrics_row(f"{label_prefix} GatedA1(L720,q0.20)", port))
    return rows


def main():
    t0 = time.time()
    hdr("OPUS STEP 2 — OOS FORWARD-TEST (2026-03-18 → 2026-04-25)")

    print("\n  Loading research frame (this is the bottleneck)...")
    df, regime_df = r68.load_data()
    feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]
    print(f"  Frame: {len(df):,} rows, {df['symbol'].nunique()} syms")
    print(f"  Features: {len(feats)}/31  (no_rank: {len(no_rank)})")
    print(f"  ts range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    regime_df.reset_index().to_parquet(OUTPUTS["REGIME"])

    # Train each model
    preds_by_model = {}
    for key in ["R128_W4", "R132_W4", "R134_W4"]:
        preds = train_one(df, feats, no_rank, key, WINDOWS[key])
        if preds is not None:
            preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True)
            preds_by_model[key] = preds

    # Free df
    del df

    # Augment regime with persistence
    regime_aug = r129.add_persistence(regime_df, lookback=720)
    persist_col = "td_persist_720h"

    # PROD_deployed reference
    prod_path = CACHE / "current_prod_cls_oos_preds.parquet"
    if prod_path.exists():
        pp = pd.read_parquet(prod_path)
        tcol = "ts" if "ts" in pp.columns else "timestamp"
        pp = pp.rename(columns={tcol: "timestamp"})
        pp["timestamp"] = pd.to_datetime(pp["timestamp"], utc=True)
        preds_by_model["PROD_DEPLOYED"] = pp

    # Run overlays per model
    all_rows = []
    for key, preds in preds_by_model.items():
        hdr(f"OOS — {key}")
        rows = run_all_overlays(preds, regime_aug, persist_col, key, oos_only=True)
        for r in rows:
            print(f"  {r['label']:<38} S={r['S']:+7.3f}  Sortino={r['Sortino']:+7.3f}  "
                  f"DD={r['DD%']:+6.2f}%  ret={r['ret%']:+6.2f}%  mean={r['mean_bp']:+7.2f}bp  "
                  f"n={r['n']} act={r['act']} gate={r['gate_on']}")
        all_rows.extend(rows)

    # FINAL TABLE
    hdr("FINAL TABLE (all model × overlay combinations, OOS 2026-03-18→2026-04-25)")
    out = pd.DataFrame(all_rows).sort_values("S", ascending=False)
    print(out.to_string(index=False))

    out_csv = CACHE / "opus_oos_results.csv"
    out.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")
    print(f"  TOTAL ELAPSED: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
