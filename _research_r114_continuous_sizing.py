#!/usr/bin/env python3
"""
R114 — Continuous Trend Sizing

Instead of binary cutoff (trade at 100% / flat at 0%), use continuous
position sizing based on trend_strength:

  size = max(0, 1 - trend_str / flat_threshold)

Grid search over:
  - flat_threshold ∈ {1.5, 2.0, 2.5, 3.0, 4.0, 5.0}
  - dyn_start ∈ {0.3, 0.5, 0.7}  (where scaling begins)
  
Compare vs R113 baseline (binary cutoff_on=0.9):
  Net Sharpe=3.057, DD=-11.2%, Calmar=16.47, %flat=33.9%

Hypothesis: continuous sizing recovers diluted Sharpe by reducing flat time
while preserving DD protection for extreme trends.
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


# ─── Continuous sizing simulate ──────────────────────────────────────────

def simulate_continuous(merged, regime_df, n_long, n_short, cfg,
                        flat_threshold=2.0, dyn_start=0.5):
    """
    Continuous trend sizing: position size scales linearly with trend_strength.

    size_mult = clip(1 - (trend_str - dyn_start) / (flat_threshold - dyn_start), 0, 1)

    When trend_str <= dyn_start:  size_mult = 1.0 (full size)
    When trend_str >= flat_threshold: size_mult = 0.0 (flat)
    Between: linear interpolation

    Key differences from R113 binary:
    - No binary risk_off state machine
    - Positions stay open but at reduced size → less closing/reopening cost
    - EMA survives always (same as R113)
    - prev_longs/prev_shorts cleared only when size_mult = 0
    """
    rebal_hours  = cfg["rebal_hours"]
    ema_alpha    = cfg.get("ema_alpha", None)
    hysteresis   = cfg.get("hysteresis", 0)
    funding_per_12h = 0.00008

    all_rets = []
    prev_longs: Set[str] = set()
    prev_shorts: Set[str] = set()
    prev_preds: Dict[str, float] = {}
    was_flat = False  # track if previous period was flat (for cost accounting)

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}
    rebal_timestamps = timestamps_sorted[::rebal_hours]

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        grp = grouped[ts].copy()

        # ── Update EMA always ──
        if ema_alpha is not None and ema_alpha < 1.0:
            for idx, r in grp.iterrows():
                sym = r["symbol"]
                raw_pred = r["pred"]
                smoothed = ema_alpha * raw_pred + (1 - ema_alpha) * prev_preds.get(sym, raw_pred)
                prev_preds[sym] = smoothed
                grp.at[idx, "pred"] = smoothed

        # ── Compute size multiplier ──
        if trend_str <= dyn_start:
            size_mult = 1.0
        elif trend_str >= flat_threshold:
            size_mult = 0.0
        else:
            size_mult = 1.0 - (trend_str - dyn_start) / (flat_threshold - dyn_start)

        # ── Flat: go to cash ──
        if size_mult <= 0.0:
            if prev_longs or prev_shorts:
                n_prev = len(prev_longs) + len(prev_shorts)
                avg_w = 1.0 / n_prev
                close_cost = sum(_cost_for_sym(s) * avg_w for s in prev_longs | prev_shorts)
                all_rets.append({
                    "timestamp": ts, "gross_ret": 0.0, "net_ret": -close_cost,
                    "cost": close_cost, "n_long": 0, "n_short": 0,
                    "turnover": n_prev, "risk_off": True, "size_mult": 0.0,
                })
            else:
                all_rets.append({
                    "timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                    "cost": 0.0, "n_long": 0, "n_short": 0,
                    "turnover": 0, "risk_off": True, "size_mult": 0.0,
                })
            prev_longs, prev_shorts = set(), set()
            was_flat = True
            continue

        # ── Normal portfolio construction (with scaling) ──
        n = len(grp)
        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl == 0 and ns == 0:
            all_rets.append({
                "timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                "cost": 0.0, "n_long": 0, "n_short": 0,
                "turnover": 0, "risk_off": False, "size_mult": size_mult,
            })
            continue

        grp["pred_rank"] = grp["pred"].rank(ascending=False)

        # Hysteresis
        if hysteresis > 0 and (prev_longs or prev_shorts):
            new_longs: Set[str] = set()
            new_shorts: Set[str] = set()
            for idx, r in grp.iterrows():
                sym, rank = r["symbol"], r["pred_rank"]
                if sym in prev_longs and rank <= nl + hysteresis:
                    new_longs.add(sym)
                elif sym in prev_shorts and rank > (n - ns - hysteresis):
                    new_shorts.add(sym)
            for _, r in grp[~grp["symbol"].isin(new_longs | new_shorts)].sort_values("pred_rank").head(nl - len(new_longs)).iterrows():
                new_longs.add(r["symbol"])
            for _, r in grp[~grp["symbol"].isin(new_longs | new_shorts)].sort_values("pred_rank", ascending=False).head(ns - len(new_shorts)).iterrows():
                new_shorts.add(r["symbol"])
        else:
            new_longs = set(grp[grp["pred_rank"] <= nl]["symbol"].tolist()) if nl > 0 else set()
            new_shorts = set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist()) if ns > 0 else set()

        # Cost calculation
        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        total_positions = len(new_longs) + len(new_shorts)

        longs = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]
        long_ret = longs["fwd_ret"].mean() if len(longs) > 0 else 0
        short_ret = shorts["fwd_ret"].mean() if len(shorts) > 0 else 0

        nl_act, ns_act = len(new_longs), len(new_shorts)
        if nl_act > 0 and ns_act > 0:
            gross_ret = 0.5 * long_ret - 0.5 * short_ret
        elif ns_act > 0:
            gross_ret = -short_ret
        else:
            gross_ret = long_ret

        # Apply continuous sizing
        gross_ret *= size_mult

        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            turnover_cost = sum(_cost_for_sym(sym) * avg_weight for sym in new_opened)
            turnover_cost += sum(_cost_for_sym(sym) * avg_weight for sym in closed)
            holding_cost = funding_per_12h * (rebal_hours / 12)
            # Costs also scale with size (less capital deployed = less cost)
            total_cost = (turnover_cost + holding_cost) * size_mult
        else:
            total_cost = 0.0

        net_ret = gross_ret - total_cost
        prev_longs, prev_shorts = new_longs, new_shorts
        was_flat = False

        all_rets.append({
            "timestamp": ts, "gross_ret": gross_ret, "net_ret": net_ret,
            "cost": total_cost, "n_long": nl_act, "n_short": ns_act,
            "turnover": len(new_opened) + len(closed),
            "risk_off": False, "size_mult": round(size_mult, 3),
        })

    return pd.DataFrame(all_rets) if all_rets else pd.DataFrame()


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("=" * 70)
    log("R114 — Continuous Trend Sizing")
    log("=" * 70)

    os.makedirs("results", exist_ok=True)

    # ── Load & train (once) ──
    log("\nLoading data...")
    df, regime_df = load_data()
    base_feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in base_feats if f in MARKET_LEVEL_FEATURES]

    log("\nTraining ensemble...")
    t1 = time.time()
    preds = train_ensemble(df, base_feats, CONTINUOUS_WINDOWS,
                           seeds=SEEDS, cs_rank_exclude=no_rank)
    log(f"  Trained in {time.time()-t1:.0f}s")

    # ── R113 baseline for comparison ──
    log("\n" + "=" * 70)
    log("R113 baseline (binary cutoff_on=0.9)")
    log("=" * 70)
    cfg = dict(PROD_CFG)
    port_base = simulate_v2(preds, regime_df, 4, 2, cfg, cutoff_on=0.9)
    m_base = analyze_config(port_base, "R113_baseline")
    print_result(m_base)

    # ── Grid search ──
    log("\n" + "=" * 70)
    log("R114 Grid: continuous sizing")
    log("=" * 70)

    FLAT_THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    DYN_STARTS = [0.3, 0.5, 0.7]

    results = [m_base]  # include baseline in results

    for ft in FLAT_THRESHOLDS:
        for ds in DYN_STARTS:
            label = f"cont_ft{ft}_ds{ds}"
            log(f"\n  {label}...")
            port = simulate_continuous(preds, regime_df, 4, 2, cfg,
                                       flat_threshold=ft, dyn_start=ds)
            m = analyze_config(port, label)
            m["flat_threshold"] = ft
            m["dyn_start"] = ds

            # Additional stats
            if "size_mult" in port.columns:
                m["avg_size_mult"] = round(port["size_mult"].mean(), 3)
                m["pct_full_size"] = round((port["size_mult"] == 1.0).mean() * 100, 1)
                m["pct_reduced"] = round(((port["size_mult"] > 0) & (port["size_mult"] < 1.0)).mean() * 100, 1)
            print_result(m)
            results.append(m)

    # ── Results table ──
    log("\n" + "=" * 70)
    log("R114 RESULTS")
    log("=" * 70)
    log(f"  {'Config':<22} {'NetSh':>7} {'GrSh':>7} {'Ret%':>7} {'DD%':>7} {'Calmar':>7} {'%flat':>6} {'%reduced':>9} {'AvgSize':>8}")
    log(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*9} {'-'*8}")

    # Print baseline first
    log(f"  {'R113_baseline':<22} {m_base['net_sharpe']:>7.3f} {m_base['gross_sharpe']:>7.3f}"
        f" {m_base['total_ret_pct']:>7.1f} {m_base['max_dd_pct']:>7.1f} {m_base['calmar']:>7.2f}"
        f" {m_base['pct_flat']:>5.1f}% {'n/a':>9} {'n/a':>8}")

    best = m_base
    for m in results[1:]:
        pct_red = m.get('pct_reduced', 0)
        avg_sz = m.get('avg_size_mult', 1.0)
        log(f"  {m['label']:<22} {m['net_sharpe']:>7.3f} {m['gross_sharpe']:>7.3f}"
            f" {m['total_ret_pct']:>7.1f} {m['max_dd_pct']:>7.1f} {m['calmar']:>7.2f}"
            f" {m['pct_flat']:>5.1f}% {pct_red:>8.1f}% {avg_sz:>8.3f}")
        if m["net_sharpe"] > best["net_sharpe"]:
            best = m

    # Also best by Calmar
    best_calmar = max(results, key=lambda x: x["calmar"])

    log(f"\n  BEST by Sharpe: {best['label']} → Net={best['net_sharpe']:.3f}, DD={best['max_dd_pct']:.1f}%, Calmar={best['calmar']:.2f}")
    log(f"  BEST by Calmar: {best_calmar['label']} → Net={best_calmar['net_sharpe']:.3f}, DD={best_calmar['max_dd_pct']:.1f}%, Calmar={best_calmar['calmar']:.2f}")

    # ── Comparison vs baseline ──
    log(f"\n  {'Metric':<15} {'R113 baseline':>15} {'R114 best':>15} {'Delta':>10}")
    log(f"  {'-'*15} {'-'*15} {'-'*15} {'-'*10}")
    for metric in ['net_sharpe', 'gross_sharpe', 'total_ret_pct', 'max_dd_pct', 'calmar', 'pct_flat']:
        v_base = m_base[metric]
        v_best = best[metric]
        delta = v_best - v_base
        log(f"  {metric:<15} {v_base:>15.3f} {v_best:>15.3f} {delta:>+10.3f}")

    # ── Save ──
    df_results = pd.DataFrame(results)
    df_results.to_csv("results/r114_grid.csv", index=False)
    with open("results/r114_best.json", "w") as f:
        json.dump(best, f, indent=2)

    log(f"\nSaved: results/r114_grid.csv, r114_best.json")
    log(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
