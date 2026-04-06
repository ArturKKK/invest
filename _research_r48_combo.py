#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R48 Phase 4 — Best Combo from Phase 1-3

Takes best features from Phase 1 (taker derivatives) and Phase 2 (residualized liq),
combines with hybrid cost model from Phase 3.

Reads winner info from:
  - results_r48_phase12_winners.json  (best features from Phase 1+2)
  - results_r48_phase3_best.json      (cost model comparison)

Usage:
  python _research_r48_combo.py          # full Phase 4
  python _research_r48_combo.py --quick  # BTC/ETH/SOL only
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── project imports ───────────────────────────────────────────

from _research_round7 import WINDOWS, SYM_35
from _research_r30b_fixed import (
    eval_with_costs,
    simulate_with_costs,
    train_ensemble,
)
from _research_r33_creative_features import FEAT_28
from _research_r35_new_features import (
    MARKET_LEVEL_FEATURES,
    add_r35_features,
    load_research_frame,
)
from _research_r47_coinglass import (
    CANONICAL_EXEC_CFG,
    CHAMPION_FEAT_30,
    add_cg_features,
    compute_cg_features,
    load_cg_daily,
    make_feature_set,
)
from _research_r48_features import (
    add_taker_derivatives,
    add_residualized_liq,
)
from _research_r48_cost import simulate_with_hybrid_costs

# ── config ─────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
CHAMPION_FEAT_31 = CHAMPION_FEAT_30 + ["cg_taker_imb"]


# ═══════════════════════════════════════════════════════════════
#  Load phase results
# ═══════════════════════════════════════════════════════════════

def load_phase_results() -> Tuple[List[str], bool]:
    """
    Load best features from Phase 1+2 and cost model decision from Phase 3.
    Returns (extra_features, use_hybrid_costs).
    """
    extra_feats = []

    # Phase 1+2 winners
    p12_path = BASE_DIR / "results_r48_phase12_winners.json"
    if p12_path.exists():
        with open(p12_path) as f:
            winners = json.load(f)
        for phase, feat_str in winners.items():
            for feat in feat_str.split("|"):
                if feat and feat not in extra_feats:
                    extra_feats.append(feat)
        print(f"  Phase 1+2 winners: {extra_feats}")
    else:
        print("  ⚠️  No Phase 1+2 winners file — using cg_taker_imb only")

    # Phase 3 cost model
    use_hybrid = False
    p3_path = BASE_DIR / "results_r48_phase3_best.json"
    if p3_path.exists():
        with open(p3_path) as f:
            cost_info = json.load(f)
        use_hybrid = bool(cost_info.get("improved_sharpe", False))
        print(f"  Phase 3 hybrid costs: {'YES' if use_hybrid else 'NO'} "
              f"(Sharpe improved: {cost_info.get('hybrid_all_sharpe', '?')})")
    else:
        print("  ⚠️  No Phase 3 results — using uniform 7bp costs")

    return extra_feats, use_hybrid


# ═══════════════════════════════════════════════════════════════
#  Combo WF
# ═══════════════════════════════════════════════════════════════

def run_combo_wf(df, regime_df, mkt_cols, extra_feats, use_hybrid):
    """Run WF for various combinations."""
    rows = []
    no_rank_market = list(MARKET_LEVEL_FEATURES)

    def _train_and_eval(feats_list, label, mkt_extra=None):
        f = list(feats_list)
        nr = [c for c in f if c in no_rank_market or c in mkt_cols]
        if mkt_extra:
            nr.extend([c for c in mkt_extra if c not in nr])
        preds = train_ensemble(df, f, WINDOWS, l2=1.0, rolling=False,
                               label=label, cs_rank_exclude=nr)
        if preds is None or preds.empty:
            return None

        result = {}
        for window in ["W1", "W2", "W3", "ALL"]:
            subset = preds if window == "ALL" else preds[preds["window"] == window]
            if use_hybrid:
                port = simulate_with_hybrid_costs(subset, regime_df, CANONICAL_EXEC_CFG)
            else:
                port = simulate_with_costs(subset, regime_df, CANONICAL_EXEC_CFG)
            result[window] = eval_with_costs(port, f"{label}_{window}")
        return result

    # Config A: baseline 31f (champion + cg_taker_imb)
    print("\n  [A] champion_31f ...")
    r = _train_and_eval(CHAMPION_FEAT_31, "A_31f")
    if r:
        row = _mk(r, "A_31f", [])
        rows.append(row)
        _pr(row)
        baseline_all = row["ALL_sh"]
    else:
        print("  ❌ Baseline failed")
        return pd.DataFrame()

    # Config B: 31f + best from Phase 1
    p1_feats = [f for f in extra_feats if "taker" in f]
    if p1_feats:
        for feat in p1_feats[:2]:
            label = f"B_31f+{feat[-20:]}"
            print(f"\n  [{label}] ...")
            feats = list(CHAMPION_FEAT_31)
            if feat not in feats:
                feats.append(feat)
            r = _train_and_eval(feats, label)
            if r:
                row = _mk(r, label, [feat])
                row["delta_all"] = row["ALL_sh"] - baseline_all
                rows.append(row)
                _pr(row)

    # Config C: 31f + best from Phase 2
    p2_feats = [f for f in extra_feats if "resid" in f or "liq" in f]
    if p2_feats:
        for feat in p2_feats[:2]:
            label = f"C_31f+{feat[-20:]}"
            print(f"\n  [{label}] ...")
            feats = list(CHAMPION_FEAT_31)
            if feat not in feats:
                feats.append(feat)
            mkt_extra = [feat] if "mkt_" in feat else []
            r = _train_and_eval(feats, label, mkt_extra)
            if r:
                row = _mk(r, label, [feat])
                row["delta_all"] = row["ALL_sh"] - baseline_all
                rows.append(row)
                _pr(row)

    # Config D: 31f + best P1 + best P2 (if both exist)
    if p1_feats and p2_feats:
        best_p1 = p1_feats[0]
        best_p2 = p2_feats[0]
        label = f"D_combo_{best_p1[-10:]}+{best_p2[-10:]}"
        print(f"\n  [{label}] ...")
        feats = list(CHAMPION_FEAT_31)
        for f in [best_p1, best_p2]:
            if f not in feats:
                feats.append(f)
        mkt_extra = [f for f in [best_p1, best_p2] if "mkt_" in f]
        r = _train_and_eval(feats, label, mkt_extra)
        if r:
            row = _mk(r, label, [best_p1, best_p2])
            row["delta_all"] = row["ALL_sh"] - baseline_all
            rows.append(row)
            _pr(row)

    # Config E: 30f only (no CG) for reference with potentially better costs
    print("\n  [E] champion_30f (no CG, reference) ...")
    r = _train_and_eval(CHAMPION_FEAT_30, "E_30f_ref")
    if r:
        row = _mk(r, "E_30f_ref", [])
        row["delta_all"] = row["ALL_sh"] - baseline_all
        rows.append(row)
        _pr(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("ALL_sh", ascending=False).reset_index(drop=True)
    return summary


def _mk(results, config, extra_feats):
    row = {"config": config, "extra_feats": "|".join(extra_feats)}
    for window in ["W1", "W2", "W3", "ALL"]:
        m = results[window]
        row[f"{window}_sh"] = m.get("sharpe", 0.0)
        row[f"{window}_sh_gr"] = m.get("sharpe_gross", 0.0)
        row[f"{window}_dd"] = m.get("max_dd_pct", 0.0)
        row[f"{window}_cost"] = m.get("total_cost_pct", 0.0)
        row[f"{window}_turn"] = m.get("avg_turnover", 0.0)
    return row


def _pr(row):
    w = (row.get("W1_sh", 0), row.get("W2_sh", 0),
         row.get("W3_sh", 0), row.get("ALL_sh", 0))
    cost = row.get("ALL_cost", 0)
    delta = row.get("delta_all", 0)
    delta_str = f"Δ{delta:+.2f}" if delta else ""
    print(f"    W1={w[0]:+.2f}  W2={w[1]:+.2f}  W3={w[2]:+.2f}  ALL={w[3]:+.2f}  "
          f"cost={cost:.1f}%  {delta_str}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main(quick: bool = False) -> None:
    print("=" * 80)
    print("R48 Phase 4 — BEST COMBO")
    print("=" * 80)

    # ── Load phase results ──────────────────────────────────────
    print("\n[0] Loading phase results ...")
    extra_feats, use_hybrid = load_phase_results()
    cost_label = "HYBRID" if use_hybrid else "UNIFORM 7bp"
    print(f"  Cost model: {cost_label}")

    # ── Load CG daily features ──────────────────────────────────
    print("\n[1] Loading CoinGlass daily data ...")
    cg = load_cg_daily()
    cg_feats_daily = compute_cg_features(cg)
    if cg_feats_daily.empty:
        print("❌ No CG features — aborting")
        return

    # ── Load research frame ──────────────────────────────────────
    print("\n[2] Loading research frame ...")
    df, regime_df = load_research_frame()
    df, _ = add_r35_features(df)
    regime_df = regime_df.sort_index()
    print(f"  Base frame: {len(df):,} rows × {len(df.columns)} cols")

    if quick:
        print("  ⚡ Quick mode: BTC/ETH/SOL only ...")
        df = df[df["symbol"].isin(["BTC/USDT", "ETH/USDT", "SOL/USDT"])].copy()

    # ── Merge CG features ───────────────────────────────────────
    print("\n[3] Merging CG features ...")
    df, per_sym_cols, mkt_cols = add_cg_features(df, cg_feats_daily)

    # ── Add Phase 1+2 features if needed ─────────────────────────
    needs_taker_deriv = any("taker_imb_ma" in f or "taker_imb_delta" in f or
                            "taker_imb_cs_demean" in f for f in extra_feats)
    needs_resid = any("resid" in f for f in extra_feats)

    if needs_taker_deriv:
        print("\n  Adding taker derivatives ...")
        df, _ = add_taker_derivatives(df)

    if needs_resid:
        print("\n  Adding residualized liquidations ...")
        df, _ = add_residualized_liq(df)

    # ── Run combo WF ────────────────────────────────────────────
    print("\n[4] Running combo walk-forward tests ...")
    summary = run_combo_wf(df, regime_df, mkt_cols, extra_feats, use_hybrid)

    if not summary.empty:
        print("\n" + "=" * 70)
        print("  R48 FINAL SUMMARY")
        print("=" * 70)
        print(f"\n  Cost model: {cost_label}")
        print(f"\n  {'Config':<30} {'W1':>6} {'W2':>6} {'W3':>6} {'ALL':>6} {'Cost%':>7} {'Δ_ALL':>7}")
        print(f"  {'─'*30} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*7} {'─'*7}")
        for _, r in summary.iterrows():
            delta = r.get("delta_all", 0)
            delta_str = f"{delta:+.2f}" if delta else ""
            print(f"  {r['config']:<30} {r['W1_sh']:>+5.2f} {r['W2_sh']:>+5.2f} "
                  f"{r['W3_sh']:>+5.2f} {r['ALL_sh']:>+5.2f} {r['ALL_cost']:>6.1f}% {delta_str:>7}")

        summary.to_csv("results_r48_phase4_summary.csv", index=False)
        print("\n  → Saved results_r48_phase4_summary.csv")

        # Determine champion
        best = summary.iloc[0]
        current_best = 1.31  # R47 champion
        if best["ALL_sh"] > current_best:
            print(f"\n  🏆 NEW CHAMPION: {best['config']} → ALL={best['ALL_sh']:.2f} "
                  f"(was {current_best:.2f})")
        elif best["ALL_sh"] >= current_best and best["ALL_cost"] < 19.0:
            print(f"\n  ✅ COST IMPROVEMENT: {best['config']} → ALL={best['ALL_sh']:.2f}, "
                  f"Cost={best['ALL_cost']:.1f}% (was 19.2%)")
        else:
            print(f"\n  ℹ️  No improvement over R47 champion (ALL={current_best:.2f})")

    print("\n" + "=" * 80)
    print("R48 Phase 4 — COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(quick=args.quick)
