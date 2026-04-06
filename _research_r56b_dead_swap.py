#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R56b — Dead Feature Substitution: replace gain=0 features with R55 candidates.

Consultant recommendation: test new features INSTEAD OF dead features (dow_cos,
hour_cos, dow_sin) rather than replacing top-1 (cum_funding_24h).

Experiments:
  B0: Baseline (champion 31f, correct cs_rank_exclude)
  B1: dow_cos  → cg_fr_disagreement
  B2: hour_cos → cg_fr_disagreement
  B3: dow_cos  → cg_oi_chg_1d

Usage:
  python _research_r56b_dead_swap.py
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── project imports ───────────────────────────────────────────

from _research_round7 import WINDOWS, SYM_35
from _research_r22_models import SEEDS, LEVERAGE, CAPITAL, log
from _research_r30b_fixed import train_ensemble, eval_with_costs
from _research_r35_new_features import (
    add_r35_features,
    load_research_frame,
    MARKET_LEVEL_FEATURES,
)
from _research_r47_coinglass import (
    CANONICAL_EXEC_CFG,
    CHAMPION_FEAT_30,
    add_cg_features,
    compute_cg_features,
    load_cg_daily,
    make_feature_set,
)
from _research_r48_cost import simulate_with_hybrid_costs
from _research_r55_cg_features import (
    build_all_r55_features,
    merge_r55_into_model,
)

# True champion = 30 base + cg_taker_imb
CHAMPION_FEAT_31 = CHAMPION_FEAT_30 + ["cg_taker_imb"]

# Dead features (gain=0 in R56 Phase 0)
DEAD_FEATURES = ["dow_cos", "dow_sin", "hour_cos"]

# ── Experiments ──
EXPERIMENTS = [
    {"label": "B1_dow_cos→fr_disagree",   "old": "dow_cos",  "new": "cg_fr_disagreement"},
    {"label": "B2_hour_cos→fr_disagree",  "old": "hour_cos", "new": "cg_fr_disagreement"},
    {"label": "B3_dow_cos→oi_chg",        "old": "dow_cos",  "new": "cg_oi_chg_1d"},
]


# ═════════════════════════════════════════════════════════════
#  Hybrid-cost evaluation (from R56 fixed)
# ═════════════════════════════════════════════════════════════

def eval_per_window_hybrid(preds, regime_df, cfg, label=""):
    """Per-window evaluation using tiered hybrid costs."""
    results = {}
    for wname in ["W1", "W2", "W3"]:
        sub = preds[preds["window"] == wname]
        if len(sub) < 10:
            results[wname] = {"sharpe": 0, "sharpe_gross": 0}
            continue
        port = simulate_with_hybrid_costs(sub, regime_df, cfg)
        r = eval_with_costs(port, f"{label}_{wname}")
        results[wname] = r
        log(f"  {wname}: Sh={r['sharpe']:>5.2f} (gross={r['sharpe_gross']:>5.2f})  "
            f"Eq=${r['equity']:>6.0f}  DD={r['max_dd_pct']:>+5.1f}%  "
            f"Cost={r['total_cost_pct']:.1f}%")

    port_all = simulate_with_hybrid_costs(preds, regime_df, cfg)
    r_all = eval_with_costs(port_all, label)
    log(f"  ALL: Sh={r_all['sharpe']:>5.2f} (gross={r_all['sharpe_gross']:>5.2f})  "
        f"Eq=${r_all['equity']:>6.0f}  DD={r_all['max_dd_pct']:>+5.1f}%  "
        f"Cost={r_all['total_cost_pct']:.1f}%")
    results["ALL"] = r_all
    return results


def compute_no_rank(feats: List[str]) -> List[str]:
    """Compute which features should be excluded from CS ranking."""
    return [f for f in feats if f in MARKET_LEVEL_FEATURES]


def run_experiment(df, regime_df, feats, label, no_rank):
    """Train + evaluate one configuration."""
    present = [f for f in feats if f in df.columns]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        print(f"  [{label}] Missing: {missing}")
    if len(present) < 25:
        print(f"  [{label}] Too few features ({len(present)}), skipping")
        return None

    print(f"  [{label}] Training {len(present)}f, no_rank={no_rank}")
    preds = train_ensemble(df, present, WINDOWS, l2=1.0, rolling=False,
                           label=label, cs_rank_exclude=no_rank)
    if preds is None:
        print(f"  [{label}] Training failed")
        return None

    results = eval_per_window_hybrid(preds, regime_df, CANONICAL_EXEC_CFG, label)
    return results


def main():
    print("=" * 70)
    print("  R56b — Dead Feature Substitution (with correct cs_rank_exclude)")
    print("=" * 70)

    # ── Load data ──
    print("\n[LOAD] Building model frame...")
    base_df, regime_df = load_research_frame()
    base_df, _ = add_r35_features(base_df)

    cg = load_cg_daily()
    cg_daily = compute_cg_features(cg)
    if not cg_daily.empty:
        base_df, _, mkt_cols = add_cg_features(base_df, cg_daily)

    # Merge R55 features (needed for cg_fr_disagreement, cg_oi_chg_1d)
    r55_feats = build_all_r55_features()
    df, r55_cols = merge_r55_into_model(base_df, r55_feats)

    print(f"\n  Frame: {len(df):,} rows, {df['symbol'].nunique()} symbols")
    print(f"  R55 features: {r55_cols}")

    # Verify candidates are present
    for exp in EXPERIMENTS:
        if exp["new"] not in df.columns:
            print(f"  ⚠️  {exp['new']} NOT in dataframe!")

    all_results = {}

    # ══════════════════════════════════════════════════════════
    #  B0: Baseline (with correct cs_rank_exclude)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  B0 — Baseline: Champion 31f (correct cs_rank_exclude)")
    print("=" * 70)

    no_rank_baseline = compute_no_rank(CHAMPION_FEAT_31)
    print(f"  cs_rank_exclude: {no_rank_baseline}")

    results_b0 = run_experiment(df, regime_df, CHAMPION_FEAT_31,
                                "baseline_31f", no_rank_baseline)
    all_results["B0"] = results_b0

    if results_b0 is None:
        print("  ✗ Baseline failed, aborting")
        return

    baseline_sharpe = results_b0["ALL"]["sharpe"]
    baseline_w1 = results_b0.get("W1", {}).get("sharpe", 0)
    baseline_w2 = results_b0.get("W2", {}).get("sharpe", 0)
    baseline_w3 = results_b0.get("W3", {}).get("sharpe", 0)

    # ══════════════════════════════════════════════════════════
    #  B1-B3: Substitution experiments
    # ══════════════════════════════════════════════════════════
    for exp in EXPERIMENTS:
        label = exp["label"]
        old_f = exp["old"]
        new_f = exp["new"]

        print(f"\n" + "=" * 70)
        print(f"  {label}: {old_f} → {new_f}")
        print("=" * 70)

        if new_f not in df.columns:
            print(f"  ✗ {new_f} not available, skipping")
            all_results[label] = None
            continue

        # Build feature set with substitution
        feats = [(new_f if f == old_f else f) for f in CHAMPION_FEAT_31]
        no_rank = compute_no_rank(feats)

        results = run_experiment(df, regime_df, feats, label, no_rank)
        all_results[label] = results

    # ══════════════════════════════════════════════════════════
    #  RESULTS TABLE
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  R56b RESULTS TABLE")
    print("=" * 70)

    header = "| # | Experiment | ALL | W1 | W2 | W3 | Cost% | ΔSharpe | Decision |"
    sep = "|---|-----------|-----|----|----|----|----|---------|----------|"
    print(f"\n{header}")
    print(sep)

    labels = [("B0", "Baseline 31f (fixed)")] + \
             [(e["label"], f"{e['old']}→{e['new']}") for e in EXPERIMENTS]

    for key, display in labels:
        r = all_results.get(key)
        if r is None:
            print(f"| {key} | {display} | FAIL | | | | | | |")
            continue

        s_all = r["ALL"]["sharpe"]
        s_w1 = r.get("W1", {}).get("sharpe", 0)
        s_w2 = r.get("W2", {}).get("sharpe", 0)
        s_w3 = r.get("W3", {}).get("sharpe", 0)
        cost = r["ALL"].get("total_cost_pct", 0)

        if key == "B0":
            delta_str = "—"
            decision = "baseline"
        else:
            delta = s_all - baseline_sharpe
            delta_str = f"{delta:+.2f}"

            # Decision logic
            if delta > 0.05:
                decision = "✅ ACCEPT"
            elif delta < -0.05:
                decision = "❌ REJECT"
            else:
                decision = "— neutral"

            # W1 veto: new feature shouldn't destroy W1
            if baseline_w1 > 0 and s_w1 < baseline_w1 - 0.50:
                decision = "🚫 VETO(W1)"
            # W2 veto
            if baseline_w2 > 0 and s_w2 < baseline_w2 - 0.30:
                decision = "🚫 VETO(W2)"

        print(f"| {key} | {display} | {s_all:+.2f} | {s_w1:+.2f} | "
              f"{s_w2:+.2f} | {s_w3:+.2f} | {cost:.1f}% | {delta_str} | {decision} |")

    # Summary
    print(f"\n  Baseline ALL: {baseline_sharpe:+.2f}")
    winners = []
    for exp in EXPERIMENTS:
        r = all_results.get(exp["label"])
        if r:
            d = r["ALL"]["sharpe"] - baseline_sharpe
            if d > 0.05:
                winners.append((exp["label"], d))
    if winners:
        print(f"  Winners: {[(k, f'Δ={d:+.2f}') for k, d in winners]}")
    else:
        print("  No winners (all ΔSharpe ≤ 0.05)")

    print("\n" + "=" * 70)
    print("  R56b COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
