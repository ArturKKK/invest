#!/usr/bin/env python3
"""
Quick backtest to measure Sharpe impact of trend_cutoff fix.
Compares: old behavior (skip=free) vs new (close-to-flat with costs).
"""
import time
import warnings
warnings.filterwarnings("ignore")

from _research_r22_models import SEEDS, log
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r68_continuous_wf import (
    CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, PROD_CFG,
    load_data, train_ensemble, simulate, sharpe, analyze,
)


def main():
    t0 = time.time()
    log("=" * 70)
    log("Trend Cutoff Fix — Backtest Comparison")
    log("=" * 70)

    log("\nLoading data...")
    df, regime_df = load_data()

    base_feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in base_feats if f in MARKET_LEVEL_FEATURES]

    log("\nTraining ensemble (R68 champion)...")
    t1 = time.time()
    preds = train_ensemble(df, base_feats, CONTINUOUS_WINDOWS,
                           seeds=SEEDS, cs_rank_exclude=no_rank)
    log(f"  Trained in {time.time()-t1:.0f}s")

    # ── Run simulation with FIXED code (close-to-flat) ──
    log("\n--- NEW behavior (close-to-flat on trend_cutoff) ---")
    port_new = simulate(preds, regime_df, 4, 2, PROD_CFG)
    m_new = analyze(port_new, "R68_new_trend_fix")

    # Count trend_cutoff events  
    n_flat = (port_new["n_long"] == 0).sum()
    n_total = len(port_new)
    log(f"\n  Periods: {n_total} total, {n_flat} flat (trend_cutoff)")
    log(f"  Flat %: {n_flat/n_total*100:.1f}%")

    # Per-period cost analysis
    flat_periods = port_new[port_new["n_long"] == 0]
    if len(flat_periods) > 0:
        flat_cost = flat_periods["cost"].sum()
        log(f"  Total closing cost during flat: {flat_cost*100:.4f}%")

    elapsed = time.time() - t0
    log(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    log("Done.")


if __name__ == "__main__":
    main()
