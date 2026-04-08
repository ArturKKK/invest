#!/usr/bin/env python3
"""
R116 — 8h Rebalance A/B Test.

Compare 12h vs 8h rebalance frequency, keeping everything else the same.
Model still trained on 12h prediction target; only rebalance cadence changes.

Key considerations:
  - 8h holding → use fwd_ret_8h (not 12h) for actual portfolio returns
  - Costs: holding_cost scales with rebal_hours (already handled)
  - Trend filter: same cutoff applied at 8h intervals
  - 8h = 3 rebalances/day vs 2 → 50% more trades

Grid:
  - rebal_hours ∈ {8, 12}
  - cutoff_on ∈ {0.9, 1.0, 1.2, None}

Acceptance: net Sharpe and Calmar better than R113 baseline,
            otherwise reject (higher turnover not worth it).
"""
import time, json, os, warnings
import numpy as np, pandas as pd
from typing import Set, Dict
warnings.filterwarnings("ignore")

from _research_r22_models import SEEDS, log
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r68_continuous_wf import (
    CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, PROD_CFG,
    load_data, train_ensemble, sharpe, _cost_for_sym,
)
from _research_r113_trend_cutoff_reopt import simulate_v2, analyze_config, print_result


def main():
    t0 = time.time()
    log("=" * 70)
    log("R116 — 8h Rebalance A/B Test")
    log("=" * 70)

    os.makedirs("results", exist_ok=True)

    # ── Load data ──
    log("\nLoading data...")
    df, regime_df = load_data()
    base_feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in base_feats if f in MARKET_LEVEL_FEATURES]

    # ── Compute fwd_ret_8h on raw df (hourly close prices) ──
    log("\nComputing fwd_ret_8h...")
    df["fwd_ret_8h"] = df.groupby("symbol")["close"].transform(
        lambda x: x.pct_change(8).shift(-8))
    fwd_8h_lookup = (df[["timestamp", "symbol", "fwd_ret_8h"]]
                     .drop_duplicates(subset=["timestamp", "symbol"]))
    log(f"  fwd_ret_8h computed: {fwd_8h_lookup.dropna().shape[0]:,} rows")

    # ── Train ensemble (same 12h target) ──
    log("\nTraining ensemble...")
    t1 = time.time()
    preds = train_ensemble(df, base_feats, CONTINUOUS_WINDOWS,
                           seeds=SEEDS, cs_rank_exclude=no_rank)
    log(f"  Trained in {time.time()-t1:.0f}s")

    # ── Merge fwd_ret_8h into preds ──
    preds = preds.merge(fwd_8h_lookup, on=["timestamp", "symbol"], how="left")
    log(f"  Preds with fwd_ret_8h: {preds['fwd_ret_8h'].notna().sum():,} / {len(preds):,}")

    # ── Prepare 8h preds (swap fwd_ret) ──
    preds_8h = preds.copy()
    preds_8h["fwd_ret_12h_orig"] = preds_8h["fwd_ret"]
    preds_8h["fwd_ret"] = preds_8h["fwd_ret_8h"]
    preds_8h = preds_8h.dropna(subset=["fwd_ret"])

    # ── Grid search ──
    log("\n" + "=" * 70)
    log("R116 Grid: rebal_hours × cutoff_on")
    log("=" * 70)

    CUTOFF_GRID = [0.9, 1.0, 1.2, None]
    REBAL_GRID  = [12, 8]

    results = []

    for rh in REBAL_GRID:
        for co in CUTOFF_GRID:
            co_str = f"{co}" if co is not None else "None"
            label = f"rebal{rh}h_co{co_str}"
            log(f"\n  {label}...")

            cfg = dict(PROD_CFG)
            cfg["rebal_hours"] = rh

            # Use appropriate preds (12h returns for 12h rebal, 8h for 8h)
            p = preds if rh == 12 else preds_8h

            port = simulate_v2(p, regime_df, 4, 2, cfg,
                               cutoff_on=co, cutoff_off=(co - 0.1 if co else None))
            m = analyze_config(port, label)
            m["rebal_hours"] = rh
            m["cutoff_on"] = co
            print_result(m)
            results.append(m)

    # ── Results table ──
    log("\n" + "=" * 70)
    log("R116 RESULTS")
    log("=" * 70)

    hdr = (f"  {'Config':<24} {'NetSh':>7} {'GrSh':>7} {'Ret%':>7} "
           f"{'DD%':>7} {'Calmar':>7} {'%flat':>6} {'#off':>5} "
           f"{'Cost%':>6} {'Turn':>5}")
    sep = (f"  {'-'*24} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} "
           f"{'-'*6} {'-'*5} {'-'*6} {'-'*5}")
    log(hdr)
    log(sep)

    for m in results:
        log(f"  {m['label']:<24} {m['net_sharpe']:>7.3f} "
            f"{m['gross_sharpe']:>7.3f} {m['total_ret_pct']:>7.1f} "
            f"{m['max_dd_pct']:>7.1f} {m['calmar']:>7.2f} "
            f"{m['pct_flat']:>5.1f}% {m['n_off_events']:>5} "
            f"{m['total_cost_pct']:>6.2f} {m['avg_turnover']:>5.1f}")

    # ── 12h vs 8h comparison (same cutoff) ──
    log("\n" + "=" * 70)
    log("12h vs 8h comparison (same cutoff_on)")
    log("=" * 70)

    for co in CUTOFF_GRID:
        co_str = f"{co}" if co is not None else "None"
        m12 = next((m for m in results
                     if m["rebal_hours"] == 12 and m["cutoff_on"] == co), None)
        m8  = next((m for m in results
                     if m["rebal_hours"] == 8 and m["cutoff_on"] == co), None)
        if m12 and m8:
            log(f"\n  cutoff_on={co_str}:")
            log(f"    {'Metric':<18} {'12h':>10} {'8h':>10} {'Delta':>10}")
            log(f"    {'-'*18} {'-'*10} {'-'*10} {'-'*10}")
            for k in ['net_sharpe', 'total_ret_pct', 'max_dd_pct',
                       'calmar', 'pct_flat', 'total_cost_pct', 'avg_turnover']:
                v12 = m12[k]
                v8 = m8[k]
                log(f"    {k:<18} {v12:>10.3f} {v8:>10.3f} {v8 - v12:>+10.3f}")

    # ── Best configs ──
    best_12 = max([m for m in results if m["rebal_hours"] == 12],
                  key=lambda x: x["calmar"])
    best_8  = max([m for m in results if m["rebal_hours"] == 8],
                  key=lambda x: x["calmar"])

    log(f"\n  Best 12h: {best_12['label']} → Sharpe={best_12['net_sharpe']:.3f}, "
        f"DD={best_12['max_dd_pct']:.1f}%, Calmar={best_12['calmar']:.2f}")
    log(f"  Best  8h: {best_8['label']} → Sharpe={best_8['net_sharpe']:.3f}, "
        f"DD={best_8['max_dd_pct']:.1f}%, Calmar={best_8['calmar']:.2f}")

    if best_8["calmar"] > best_12["calmar"]:
        log(f"\n  ✓ 8h rebalance BEATS 12h (Calmar {best_8['calmar']:.2f} "
            f"vs {best_12['calmar']:.2f})")
        best = best_8
    else:
        log(f"\n  ✗ 8h rebalance LOSES to 12h (Calmar {best_8['calmar']:.2f} "
            f"vs {best_12['calmar']:.2f})")
        best = best_12

    # ── Save ──
    df_res = pd.DataFrame(results)
    df_res.to_csv("results/r116_grid.csv", index=False)
    with open("results/r116_best.json", "w") as f:
        json.dump(best, f, indent=2)

    log(f"\nSaved: results/r116_grid.csv, r116_best.json")
    log(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
