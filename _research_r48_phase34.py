#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R48 Phase 3 + Phase 4 only  (Phase 1+2 выполнены отдельно)

Запускать ТОЛЬКО после завершения _research_r48_features.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from _research_round7 import WINDOWS, SYM_35
from _research_r30b_fixed import (
    eval_with_costs,
    simulate_with_costs,
    train_ensemble,
)
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
from _research_r48_features import add_taker_derivatives, add_residualized_liq
from _research_r48_cost import simulate_with_hybrid_costs, simulate_liq_weighted

BASE_DIR = Path(__file__).resolve().parent
CHAMPION_FEAT_31 = CHAMPION_FEAT_30 + ["cg_taker_imb"]


def _eval_all(preds, regime_df, cost_fn=None):
    out = {}
    for window in ["W1", "W2", "W3", "ALL"]:
        subset = preds if window == "ALL" else preds[preds["window"] == window]
        port = cost_fn(subset, regime_df, CANONICAL_EXEC_CFG) if cost_fn else \
               simulate_with_costs(subset, regime_df, CANONICAL_EXEC_CFG)
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


def main():
    print("=" * 80)
    print("R48 — Phase 3 + Phase 4")
    print("=" * 80)

    # ── Load data ───────────────────────────────────────────────
    print("\n[DATA] Loading ...")
    cg = load_cg_daily()
    cg_feats = compute_cg_features(cg)

    df, regime_df = load_research_frame()
    df, _ = add_r35_features(df)
    regime_df = regime_df.sort_index()
    print(f"  Base frame: {len(df):,} rows × {len(df.columns)} cols")

    df, per_sym_cols, mkt_cols = add_cg_features(df, cg_feats)

    # ── Load Phase 1+2 winners ──────────────────────────────────
    p12_path = BASE_DIR / "results_r48_phase12_winners.json"
    p12_winners: Dict = {}
    if p12_path.exists():
        with open(p12_path) as f:
            p12_winners = json.load(f)
        print(f"\n  Phase 1+2 winners loaded: {p12_winners}")
    else:
        print("\n  ⚠️  No Phase 1+2 winners file — no extra features for combo")

    # ── Prepare extra features if needed ───────────────────────
    all_extra_feats = []
    for v in p12_winners.values():
        for feat in v.split("|"):
            if feat and feat not in all_extra_feats:
                all_extra_feats.append(feat)

    if any("taker_imb_ma" in f or "taker_imb_delta" in f or
           "taker_imb_cs" in f for f in all_extra_feats):
        print("  Adding taker derivatives ...")
        df, _ = add_taker_derivatives(df)

    if any("resid" in f for f in all_extra_feats):
        print("  Adding residualized features ...")
        df, _ = add_residualized_liq(df)

    # ═══════════════════════════════════════════════════════════
    #  PHASE 3: Hybrid cost model
    # ═══════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("  PHASE 3 — HYBRID COST MODEL")
    print("=" * 80)

    print("\n  Training champion_31f predictions (once, for all cost tests) ...")
    feats_31, no_rank_31 = make_feature_set(["cg_taker_imb"], mkt_cols)
    preds_31 = train_ensemble(df, feats_31, WINDOWS, l2=1.0, rolling=False,
                              label="p3_31f", cs_rank_exclude=no_rank_31)

    p3_rows = []
    if preds_31 is not None and not preds_31.empty:
        volume_df = df[["timestamp", "symbol", "volume"]].copy()

        for window in ["W1", "W2", "W3", "ALL"]:
            sub = preds_31 if window == "ALL" else preds_31[preds_31["window"] == window]

            port_u = simulate_with_costs(sub, regime_df, CANONICAL_EXEC_CFG)
            m_u = eval_with_costs(port_u, f"uniform_{window}")

            port_h = simulate_with_hybrid_costs(sub, regime_df, CANONICAL_EXEC_CFG)
            m_h = eval_with_costs(port_h, f"hybrid_{window}")

            port_lw = simulate_liq_weighted(sub, regime_df, CANONICAL_EXEC_CFG, volume_df)
            m_lw = eval_with_costs(port_lw, f"liqwt_{window}")

            p3_rows.append({
                "window": window,
                "uniform_sh": m_u["sharpe"],
                "hybrid_sh": m_h["sharpe"],
                "liqwt_sh": m_lw["sharpe"],
                "uniform_cost": m_u.get("total_cost_pct", 0),
                "hybrid_cost": m_h.get("total_cost_pct", 0),
                "liqwt_cost": m_lw.get("total_cost_pct", 0),
            })

        df3 = pd.DataFrame(p3_rows)
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
        use_hybrid = best_cfg[0] in ("hybrid", "liqwt")
        cost_fn = simulate_with_hybrid_costs if best_cfg[0] == "hybrid" else \
                  (lambda s, r, c: simulate_liq_weighted(s, r, c, volume_df)) \
                  if best_cfg[0] == "liqwt" else None

        print(f"\n  Best cost model: {best_cfg[0]} (ALL={best_cfg[1]:.2f})")

        p3_best = {
            "uniform_all_sh": float(all_row["uniform_sh"]),
            "hybrid_all_sh": float(all_row["hybrid_sh"]),
            "liqwt_all_sh": float(all_row["liqwt_sh"]),
            "uniform_all_cost": float(all_row["uniform_cost"]),
            "hybrid_all_cost": float(all_row["hybrid_cost"]),
            "best_cost_model": best_cfg[0],
        }
        with open("results_r48_phase3_best.json", "w") as f:
            json.dump(p3_best, f, indent=2)
        print("  → Saved results_r48_phase3_best.json")
    else:
        print("  ❌ Training failed — skip Phase 3")
        cost_fn = None
        p3_best = {"best_cost_model": "uniform"}
        best_cfg = ("uniform", 1.31)

    # ═══════════════════════════════════════════════════════════
    #  PHASE 4: Best combo
    # ═══════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("  PHASE 4 — BEST COMBO")
    print("=" * 80)

    cost_label = p3_best.get("best_cost_model", "uniform")
    print(f"\n  Cost model: {cost_label}")

    p4_rows = []
    baseline_all = 1.31  # R47 champion

    def _run_config(label, feats_list, extra_mkt=None, cost_fn_=None):
        nr = [f for f in feats_list if f in MARKET_LEVEL_FEATURES or f in mkt_cols]
        if extra_mkt:
            nr.extend([f for f in extra_mkt if f not in nr])
        preds = train_ensemble(df, feats_list, WINDOWS, l2=1.0, rolling=False,
                               label=label, cs_rank_exclude=nr)
        if preds is None or preds.empty:
            print(f"  ⚠️  {label}: failed")
            return
        results = _eval_all(preds, regime_df, cost_fn_)
        row = _mk(results, label, [])
        row["delta_all"] = row["ALL_sh"] - baseline_all
        p4_rows.append(row)
        _pr(row)

    # A: 30f no CG с лучшим cost model
    print("\n  [A] champion_30f (no CG) ...")
    _run_config("A_30f_noCG", list(CHAMPION_FEAT_30), cost_fn_=cost_fn)

    # B: 31f с uniform cost (R47 baseline для сравнения)
    print("\n  [B] champion_31f uniform cost (R47 reference) ...")
    _run_config("B_31f_uniform", list(CHAMPION_FEAT_31), cost_fn_=None)

    # C: 31f с лучшим cost model из Phase 3
    if cost_fn is not None:
        print(f"\n  [C] champion_31f {cost_label} cost ...")
        _run_config(f"C_31f_{cost_label}", list(CHAMPION_FEAT_31), cost_fn_=cost_fn)

    # D+: 31f + winner из Phase 1 или 2 (если есть)
    if all_extra_feats:
        for feat in all_extra_feats[:2]:
            if feat not in df.columns:
                print(f"  ⚠️  {feat} not in df — skip")
                continue
            feats_d = list(CHAMPION_FEAT_31) + [feat]
            mkt_extra = [feat] if "mkt_" in feat else []
            label_d = f"D_31f+{feat[-15:]}"
            print(f"\n  [{label_d}] ...")
            nr_d = [f for f in feats_d if f in MARKET_LEVEL_FEATURES or f in mkt_cols]
            nr_d.extend(mkt_extra)
            preds_d = train_ensemble(df, feats_d, WINDOWS, l2=1.0, rolling=False,
                                     label=label_d, cs_rank_exclude=nr_d)
            if preds_d is not None and not preds_d.empty:
                results_d = _eval_all(preds_d, regime_df, cost_fn)
                row_d = _mk(results_d, label_d, [feat])
                row_d["delta_all"] = row_d["ALL_sh"] - baseline_all
                p4_rows.append(row_d)
                _pr(row_d)

    if not p4_rows:
        print("  ❌ No Phase 4 results")
        return

    summary = pd.DataFrame(p4_rows)
    summary = summary.sort_values("ALL_sh", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("  R48 FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  {'Config':<30} {'W1':>6} {'W2':>6} {'W3':>6} {'ALL':>6} {'Cost%':>7} {'Δ_ALL':>7}")
    print(f"  {'─'*30} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*7} {'─'*7}")
    for _, r in summary.iterrows():
        delta = r.get("delta_all", 0)
        ds = f"{delta:+.2f}"
        print(f"  {r['config']:<30} {r['W1_sh']:>+5.2f} {r['W2_sh']:>+5.2f} "
              f"{r['W3_sh']:>+5.2f} {r['ALL_sh']:>+5.2f} {r['ALL_cost']:>6.1f}% {ds:>7}")

    summary.to_csv("results_r48_phase4_summary.csv", index=False)
    print("\n  → Saved results_r48_phase4_summary.csv")

    best = summary.iloc[0]
    print(f"\n  {'─'*70}")
    if best["ALL_sh"] > 1.31:
        print(f"  🏆 NEW CHAMPION: {best['config']} → ALL={best['ALL_sh']:.2f} "
              f"(prev 1.31, Δ={best['ALL_sh']-1.31:+.2f})")
    elif best["ALL_cost"] < 15.0:
        print(f"  ✅ COST WIN: {best['config']} → ALL={best['ALL_sh']:.2f}, "
              f"cost={best['ALL_cost']:.1f}% (was 19.2%)")
    else:
        print(f"  ── R47 champion remains: ALL=1.31, cost=19.2%")
        print(f"     Best this round: {best['config']} → ALL={best['ALL_sh']:.2f}")


if __name__ == "__main__":
    main()
