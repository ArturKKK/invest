"""Opus full validation — single comprehensive runner for MLC.

Step 1: Reproduce R124/R128 canonical baseline (target: 4L/2S Sharpe ~3.78).
Step 2: A1 overlay variants (0.50, 0.60).
Step 3: R131 honest record-zero (target: 4L/2S Sharpe ~2.46).
Step 4: R129 frozen gated A1 (L=720, q=0.20).
Step 5: 6L/3S sanity check.

ALL on canonical preds (cache/r128_canonical_preds.parquet, n=688 active periods).
This is in-sample apples-to-apples reproduction. NO OOS yet.
"""
from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd

from _preflight_check import check_versions
check_versions()

import _research_r68_continuous_wf as r68
import _r128_all_overlays_canonical as r128
import _r131_prod_sim_validate as r131
import _r129_persistence_gate as r129
import _r130_validate_r129 as r130


def line(c="="):
    print(c * 78)


def hdr(t):
    line()
    print(f"  {t}")
    line()


def metrics(label, rets, n_extra=""):
    arr = np.asarray(rets, dtype=float)
    if len(arr) < 2:
        print(f"  {label:<42} EMPTY")
        return
    S = r130.sharpe(arr)
    So = r130.sortino(arr)
    DD = r130.max_drawdown(arr) * 100
    sum_pct = arr.sum() * 100
    mean_bp = arr.mean() * 1e4
    print(f"  {label:<42} S={S:+.3f}  Sortino={So:+.3f}  DD={DD:+.2f}%  "
          f"sum={sum_pct:+.2f}%  mean={mean_bp:+.2f}bp  n={len(arr)}{n_extra}")


def main():
    t0 = time.time()
    hdr("OPUS VALIDATION — STEP 1..5 (canonical, in-sample)")

    print("\n  Loading canonical cache...")
    preds = pd.read_parquet("cache/r128_canonical_preds.parquet")
    reg = pd.read_parquet("cache/r128_canonical_regime.parquet")
    if "timestamp" in reg.columns:
        reg = reg.set_index("timestamp")
    preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True)
    print(f"  preds: {len(preds):,} rows, {preds.timestamp.nunique()} ts, "
          f"{preds.symbol.nunique()} syms, range {preds.timestamp.min()} → {preds.timestamp.max()}")
    print(f"  regime: {len(reg):,} rows, range {reg.index.min()} → {reg.index.max()}")

    # ── STEP 1: CANONICAL skip-risk-off (target 3.78) ───────────────
    hdr("STEP 1 — CANONICAL R124/R128 (skip-risk-off, n=688, target 3.78)")
    port = r128.simulate_full(preds, reg, 4, 2)
    metrics("R128 canonical 4L/2S (skip)", port.net_ret.values)

    print()
    for w in r68.CONTINUOUS_WINDOWS:
        ws = pd.Timestamp(w["test_start"], tz="UTC")
        we = pd.Timestamp(w["test_end"], tz="UTC")
        p = port[(port.timestamp >= ws) & (port.timestamp <= we)]
        metrics(f"  {w['name']} ({w['test_start']}→{w['test_end']})", p.net_ret.values)

    # ── STEP 2: A1 OVERLAYS (target +0.3 → ~4.10) ────────────────────
    hdr("STEP 2 — A1 OVERLAY on canonical (target Δ +0.3)")
    for trend_thr, weak_scale in [(0.25, 0.50), (0.25, 0.60)]:
        p = r128.simulate_full(preds, reg, 4, 2,
                               overlay={"a1": {"trend_thr": trend_thr, "weak_scale": weak_scale}})
        metrics(f"4L/2S A1(thr={trend_thr},scale={weak_scale})", p.net_ret.values)

    # ── STEP 3: HONEST RECORD-ZERO (target ~2.46) ────────────────────
    hdr("STEP 3 — HONEST RECORD-ZERO PROD SIM (n=1013, target 2.46)")
    pH = r131.simulate_prod(preds, reg, 4, 2)
    metrics("R131 baseline 4L/2S (honest)", pH.net_ret.values,
            n_extra=f"  act={int((~pH.risk_off).sum())}")
    pHa = r131.simulate_prod(preds, reg, 4, 2,
                              a1_cfg={"trend_thr": 0.25, "weak_scale": 0.60})
    metrics("R131 + A1-always 4L/2S", pHa.net_ret.values)

    # ── STEP 4: GATED A1 (R129 frozen) ───────────────────────────────
    hdr("STEP 4 — R129 GATED A1 (L=720, q=0.20, frozen)")
    try:
        regime_aug = r129.add_persistence(reg, lookback=720)
        persist_col = "td_persist_720h"
        thr = r129.expanding_quantile_threshold(regime_aug[persist_col], q=0.20, min_periods=720)
        pG = r131.simulate_prod(preds, regime_aug, 4, 2,
                                a1_cfg={"trend_thr": 0.25, "weak_scale": 0.60},
                                gate_persist_col=persist_col,
                                gate_threshold_series=thr)
        metrics("R131 + Gated A1 (L=720,q=0.20)", pG.net_ret.values,
                n_extra=f"  gate_fires={int(pG.gate_on.sum())}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  Gated A1 skipped: {e!r}")

    # ── STEP 5: 6L/3S sanity ─────────────────────────────────────────
    hdr("STEP 5 — 6L/3S sanity (R114b lineage)")
    p63 = r128.simulate_full(preds, reg, 6, 3)
    metrics("6L/3S canonical (skip)", p63.net_ret.values)

    line()
    print(f"  TOTAL: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
