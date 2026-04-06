#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R48 Continuation — Phase 2 (missing) + Phase 3 + Phase 4

Phase 1 already done (results in results_r48_phase1_summary.csv).
Phase 2 was cut off — runs only the missing mkt_cg_liq_imb_resid WF test.
Phase 3 — hybrid cost model (new script).
Phase 4 — best combo.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

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
from _research_r48_features import add_residualized_liq
from _research_r48_cost import (
    TIER1_SYMS, TIER2_SYMS, TIER3_SYMS,
    simulate_with_hybrid_costs,
    simulate_liq_weighted,
)

BASE_DIR = Path(__file__).resolve().parent
CHAMPION_FEAT_31 = CHAMPION_FEAT_30 + ["cg_taker_imb"]

# ── helpers ───────────────────────────────────────────────────

def _eval(preds, regime_df, cost_fn=None):
    out = {}
    for window in ["W1", "W2", "W3", "ALL"]:
        subset = preds if window == "ALL" else preds[preds["window"] == window]
        if cost_fn is not None:
            port = cost_fn(subset, regime_df, CANONICAL_EXEC_CFG)
        else:
            port = simulate_with_costs(subset, regime_df, CANONICAL_EXEC_CFG)
        out[window] = eval_with_costs(port, window)
    return out


def _mk(results, config, extra_feats):
    row = {"config": config, "extra_feats": "|".join(extra_feats)}
    for w in ["W1", "W2", "W3", "ALL"]:
        m = results[w]
        row[f"{w}_sh"] = m.get("sharpe", 0.0)
        row[f"{w}_sh_gr"] = m.get("sharpe_gross", 0.0)
        row[f"{w}_dd"] = m.get("max_dd_pct", 0.0)
        row[f"{w}_cost"] = m.get("total_cost_pct", 0.0)
        row[f"{w}_turn"] = m.get("avg_turnover", 0.0)
    return row


def _pr(row):
    w = (row.get("W1_sh", 0), row.get("W2_sh", 0),
         row.get("W3_sh", 0), row.get("ALL_sh", 0))
    cost = row.get("ALL_cost", 0)
    delta = row.get("delta_all", 0)
    ds = f"Δ{delta:+.2f}" if delta else ""
    print(f"    W1={w[0]:+.2f}  W2={w[1]:+.2f}  W3={w[2]:+.2f}  ALL={w[3]:+.2f}  "
          f"cost={cost:.1f}%  {ds}")


def _train_eval(df, feats, no_rank, mkt_cols, regime_df, label, cost_fn=None):
    nr = list(no_rank)
    for f in feats:
        if f in mkt_cols and f not in nr:
            nr.append(f)
    preds = train_ensemble(df, feats, WINDOWS, l2=1.0, rolling=False,
                           label=label, cs_rank_exclude=nr)
    if preds is None or preds.empty:
        return None, None
    results = _eval(preds, regime_df, cost_fn)
    return preds, results


# ═══════════════════════════════════════════════════════════════
#  LOAD DATA (shared for all phases)
# ═══════════════════════════════════════════════════════════════

def load_data():
    print("[DATA] Loading CoinGlass + research frame ...")
    cg = load_cg_daily()
    cg_feats = compute_cg_features(cg)

    df, regime_df = load_research_frame()
    df, _ = add_r35_features(df)
    regime_df = regime_df.sort_index()
    print(f"  Base frame: {len(df):,} rows × {len(df.columns)} cols")

    df, per_sym_cols, mkt_cols = add_cg_features(df, cg_feats)
    print(f"  CG features merged: {len(per_sym_cols)} per-sym + {len(mkt_cols)} mkt")
    return df, regime_df, mkt_cols


# ═══════════════════════════════════════════════════════════════
#  PHASE 2 COMPLETION — only mkt_cg_liq_imb_resid
# ═══════════════════════════════════════════════════════════════

def run_phase2_cont(df, regime_df, mkt_cols):
    print("\n" + "=" * 80)
    print("  PHASE 2 CONT — mkt_cg_liq_imb_resid WF test")
    print("=" * 80)

    df2, resid_cols = add_residualized_liq(df)

    # Known from earlier run:
    # cg_liq_imb_resid_bin = ALL 0.71 (WORSE — skip)
    # cg_liq_imb_resid_roll  IC=-0.002 (NOISE — skip)
    # mkt_cg_liq_imb_resid  IC=+0.059 — TEST THIS

    target = "mkt_cg_liq_imb_resid"
    if target not in df2.columns:
        print(f"  ⚠️  {target} not found — skipping")
        return {}

    # Baseline
    print("\n  [baseline] champion_31f ...")
    feats_bl = list(CHAMPION_FEAT_31)
    nr_bl = [f for f in feats_bl if f in MARKET_LEVEL_FEATURES or f in mkt_cols]
    preds_bl, res_bl = _train_eval(df2, feats_bl, nr_bl, mkt_cols, regime_df, "p2c_baseline")
    if res_bl is None:
        print("  ❌ Baseline failed")
        return {}
    row_bl = _mk(res_bl, "champion_31f", [])
    _pr(row_bl)
    base_all = row_bl["ALL_sh"]

    # mkt_cg_liq_imb_resid
    print(f"\n  [{target}] ...")
    feats_t = list(CHAMPION_FEAT_31) + [target]
    nr_t = [f for f in feats_t if f in MARKET_LEVEL_FEATURES or f in mkt_cols]
    nr_t.append(target)  # market-level → no cs-rank
    preds_t, res_t = _train_eval(df2, feats_t, nr_t, mkt_cols, regime_df, f"p2c_{target}")
    if res_t is None:
        print(f"  ⚠️  {target}: no predictions")
        return {}
    row_t = _mk(res_t, f"+{target}", [target])
    row_t["delta_all"] = row_t["ALL_sh"] - base_all
    _pr(row_t)

    rows = [row_bl, row_t]
    summary = pd.DataFrame(rows)
    summary.to_csv("results_r48_phase2_cont_summary.csv", index=False)
    print("  → Saved results_r48_phase2_cont_summary.csv")

    # Return winner info
    winners = {}
    if row_t["ALL_sh"] > base_all:
        winners["phase2"] = target
        print(f"  ✅ Phase 2 winner: {target} → ALL={row_t['ALL_sh']:.2f}")
    else:
        print(f"  Phase 2: no improvement (ALL={row_t['ALL_sh']:.2f} vs {base_all:.2f})")

    return winners


# ═══════════════════════════════════════════════════════════════
#  PHASE 3 — Hybrid costs + liq-weighted
# ═══════════════════════════════════════════════════════════════

def run_phase3(df, regime_df, mkt_cols):
    print("\n" + "=" * 80)
    print("  PHASE 3 — HYBRID COST MODEL")
    print("=" * 80)

    feats, no_rank = make_feature_set(["cg_taker_imb"], mkt_cols)
    print("\n  Training champion_31f predictions ...")
    preds = train_ensemble(df, feats, WINDOWS, l2=1.0, rolling=False,
                           label="p3_31f", cs_rank_exclude=no_rank)
    if preds is None or preds.empty:
        print("  ❌ Training failed")
        return {}

    volume_df = df[["timestamp", "symbol", "volume"]].copy()

    rows = []
    for window in ["W1", "W2", "W3", "ALL"]:
        subset = preds if window == "ALL" else preds[preds["window"] == window]

        port_u = simulate_with_costs(subset, regime_df, CANONICAL_EXEC_CFG)
        m_u = eval_with_costs(port_u, f"uniform_{window}")

        port_h = simulate_with_hybrid_costs(subset, regime_df, CANONICAL_EXEC_CFG)
        m_h = eval_with_costs(port_h, f"hybrid_{window}")

        port_lw = simulate_liq_weighted(subset, regime_df, CANONICAL_EXEC_CFG, volume_df)
        m_lw = eval_with_costs(port_lw, f"liqwt_{window}")

        rows.append({
            "window": window,
            "uniform_sh": m_u["sharpe"],
            "hybrid_sh": m_h["sharpe"],
            "liqwt_sh": m_lw["sharpe"],
            "uniform_cost": m_u.get("total_cost_pct", 0),
            "hybrid_cost": m_h.get("total_cost_pct", 0),
            "liqwt_cost": m_lw.get("total_cost_pct", 0),
        })

    df3 = pd.DataFrame(rows)
    print(f"\n  {'Win':<5} {'Unif_Sh':>8} {'Hyb_Sh':>8} {'LiqWt_Sh':>9} "
          f"{'Unif_C%':>8} {'Hyb_C%':>7} {'LiqWt_C%':>9}")
    print(f"  {'─'*5} {'─'*8} {'─'*8} {'─'*9} {'─'*8} {'─'*7} {'─'*9}")
    for _, r in df3.iterrows():
        print(f"  {r['window']:<5} {r['uniform_sh']:>+7.2f} {r['hybrid_sh']:>+7.2f} "
              f"{r['liqwt_sh']:>+8.2f} {r['uniform_cost']:>7.1f}% {r['hybrid_cost']:>6.1f}% "
              f"{r['liqwt_cost']:>8.1f}%")

    df3.to_csv("results_r48_phase3_summary.csv", index=False)
    print("  → Saved results_r48_phase3_summary.csv")

    all_row = df3[df3["window"] == "ALL"].iloc[0]
    best_cfg = max(
        [("uniform", all_row["uniform_sh"]),
         ("hybrid", all_row["hybrid_sh"]),
         ("liqwt", all_row["liqwt_sh"])],
        key=lambda x: x[1]
    )
    use_hybrid = best_cfg[0] == "hybrid"
    print(f"\n  Best cost model: {best_cfg[0]} (ALL={best_cfg[1]:.2f})")

    result = {
        "uniform_all_sh": float(all_row["uniform_sh"]),
        "hybrid_all_sh": float(all_row["hybrid_sh"]),
        "liqwt_all_sh": float(all_row["liqwt_sh"]),
        "uniform_all_cost": float(all_row["uniform_cost"]),
        "hybrid_all_cost": float(all_row["hybrid_cost"]),
        "best_cost_model": best_cfg[0],
        "improved_sharpe": bool(all_row["hybrid_sh"] > all_row["uniform_sh"]),
    }
    with open("results_r48_phase3_best.json", "w") as f:
        json.dump(result, f, indent=2)
    print("  → Saved results_r48_phase3_best.json")
    return result


# ═══════════════════════════════════════════════════════════════
#  PHASE 4 — Best combo
# ═══════════════════════════════════════════════════════════════

def run_phase4(df, regime_df, mkt_cols, p2_winners, p3_info):
    print("\n" + "=" * 80)
    print("  PHASE 4 — BEST COMBO")
    print("=" * 80)

    use_hybrid = p3_info.get("best_cost_model", "uniform") == "hybrid"
    cost_fn = simulate_with_hybrid_costs if use_hybrid else None
    cost_label = "HYBRID" if use_hybrid else "UNIFORM"
    print(f"\n  Cost model: {cost_label}")

    p2_feat = p2_winners.get("phase2")
    rows = []

    def _run(label, feats, extra_for_mkt=None):
        nr = [f for f in feats if f in MARKET_LEVEL_FEATURES or f in mkt_cols]
        if extra_for_mkt:
            nr.extend([f for f in extra_for_mkt if f not in nr])
        preds = train_ensemble(df, feats, WINDOWS, l2=1.0, rolling=False,
                               label=label, cs_rank_exclude=nr)
        if preds is None or preds.empty:
            print(f"  ⚠️  {label}: failed")
            return None
        results = _eval(preds, regime_df, cost_fn)
        return results

    # A — champion_30f (no CG) with best cost model
    print("\n  [A] champion_30f (no CG) ...")
    r = _run("A_30f", list(CHAMPION_FEAT_30))
    if r:
        row = _mk(r, "A_30f_noCG", [])
        rows.append(row)
        _pr(row)

    # B — champion_31f (+ cg_taker_imb) — MAIN CANDIDATE
    print("\n  [B] champion_31f (+cg_taker_imb) ...")
    r = _run("B_31f", list(CHAMPION_FEAT_31))
    if r:
        row = _mk(r, "B_31f_taker", ["cg_taker_imb"])
        rows.append(row)
        _pr(row)
        baseline_all = row["ALL_sh"]
    else:
        baseline_all = 1.31

    # C — champion_31f + best Phase 2 feature (if any)
    if p2_feat:
        print(f"\n  [C] champion_31f + {p2_feat} ...")
        feats_c = list(CHAMPION_FEAT_31) + [p2_feat]
        mkt_extra = [p2_feat] if "mkt_" in p2_feat else []
        r = _run(f"C_31f+p2", feats_c, mkt_extra)
        if r:
            row = _mk(r, f"C_31f+{p2_feat[-15:]}", [p2_feat])
            row["delta_all"] = row["ALL_sh"] - baseline_all
            rows.append(row)
            _pr(row)

    # D — 31f with uniform costs (for comparison if hybrid is bad)
    if use_hybrid:
        print("\n  [D] champion_31f with UNIFORM costs (comparison) ...")
        r2 = _eval(
            train_ensemble(df, list(CHAMPION_FEAT_31), WINDOWS, l2=1.0,
                           rolling=False, label="D_31f_uniform",
                           cs_rank_exclude=[f for f in CHAMPION_FEAT_31
                                            if f in MARKET_LEVEL_FEATURES or f in mkt_cols]),
            regime_df, None
        )
        if r2:
            row = _mk(r2, "D_31f_uniform_cost", ["cg_taker_imb"])
            row["delta_all"] = row["ALL_sh"] - baseline_all
            rows.append(row)
            _pr(row)

    if not rows:
        print("  ❌ No results")
        return

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("ALL_sh", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("  R48 FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  Cost model tested: {cost_label}")
    print(f"\n  {'Config':<30} {'W1':>6} {'W2':>6} {'W3':>6} {'ALL':>6} {'Cost%':>7} {'Δ_ALL':>7}")
    print(f"  {'─'*30} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*7} {'─'*7}")
    for _, r in summary.iterrows():
        delta = r.get("delta_all", 0)
        ds = f"{delta:+.2f}" if delta else "  ref"
        print(f"  {r['config']:<30} {r['W1_sh']:>+5.2f} {r['W2_sh']:>+5.2f} "
              f"{r['W3_sh']:>+5.2f} {r['ALL_sh']:>+5.2f} {r['ALL_cost']:>6.1f}% {ds:>7}")

    summary.to_csv("results_r48_phase4_summary.csv", index=False)
    print("\n  → Saved results_r48_phase4_summary.csv")

    current_best = 1.31
    best = summary.iloc[0]
    if best["ALL_sh"] > current_best:
        print(f"\n  🏆 NEW CHAMPION: {best['config']} → ALL={best['ALL_sh']:.2f}")
    elif best["ALL_cost"] < 17.0:
        print(f"\n  ✅ COST WIN: {best['config']} → cost={best['ALL_cost']:.1f}% (was 19.2%)")
    else:
        print(f"\n  ── R47 champion remains best: ALL=1.31, cost=19.2%")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("R48 CONTINUATION — Phase 2 (remain) + Phase 3 + Phase 4")
    print("=" * 80)

    df, regime_df, mkt_cols = load_data()

    # Phase 2 cont
    p2_winners = run_phase2_cont(df, regime_df, mkt_cols)

    # Save phase12 winners
    with open("results_r48_phase12_winners.json", "w") as f:
        json.dump(p2_winners, f, indent=2)
    print(f"\n  Phase 1+2 winners: {p2_winners}")

    # Phase 3
    p3_info = run_phase3(df, regime_df, mkt_cols)

    # Phase 4
    run_phase4(df, regime_df, mkt_cols, p2_winners, p3_info)

    print("\n" + "=" * 80)
    print("R48 CONTINUATION — COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
