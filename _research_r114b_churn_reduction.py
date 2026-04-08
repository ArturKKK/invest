#!/usr/bin/env python3
"""
R114b — Risk-off churn reduction (regime state machine stabilisation).

Keep R113 edge (Sharpe~3.06, DD~-11.2) while reducing unnecessary
risk_off entry/exit events and round-trips via timing hysteresis.

Grid:
  - cutoff_on = 0.9 (fixed)
  - cutoff_off ∈ {0.8, 0.75, 0.7, 0.65}
  - min_risk_off_periods ∈ {1, 2, 3}   (12/24/36h min stay)
  - min_risk_on_periods  ∈ {0, 1, 2}

Acceptance:
  - Sharpe >= 3.00 AND MaxDD >= -12.5%
  - #off_events reduced ≥25% vs 71 (R113), without hurting Calmar
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
from _research_r113_trend_cutoff_reopt import analyze_config, print_result


# ─── simulate with timing hysteresis ────────────────────────────────

def simulate_v2b(merged, regime_df, n_long, n_short, cfg,
                 cutoff_on=0.9, cutoff_off=None,
                 min_risk_off_periods=1, min_risk_on_periods=0):
    """
    R113 simulate_v2 + timing constraints:
      - min_risk_off_periods: once in risk_off, stay at least N periods
      - min_risk_on_periods:  once in risk_on,  stay at least N periods
                              before allowed to re-enter risk_off

    With (1, 0) → identical to R113 simulate_v2.
    """
    if cutoff_off is None and cutoff_on is not None:
        cutoff_off = cutoff_on - 0.1

    rebal_hours = cfg["rebal_hours"]
    ema_alpha    = cfg.get("ema_alpha", None)
    hysteresis   = cfg.get("hysteresis", 0)
    dyn_threshold = cfg.get("dyn_threshold", 0.5)
    funding_per_12h = 0.00008

    all_rets = []
    prev_longs: Set[str] = set()
    prev_shorts: Set[str] = set()
    prev_preds: Dict[str, float] = {}
    risk_off = False
    periods_in_off = 0
    periods_in_on = 999          # allow entering risk_off from the start

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
                smoothed = (ema_alpha * raw_pred
                            + (1 - ema_alpha) * prev_preds.get(sym, raw_pred))
                prev_preds[sym] = smoothed
                grp.at[idx, "pred"] = smoothed

        # ── State machine with timing constraints ──
        if cutoff_on is not None:
            if risk_off:
                periods_in_off += 1
                can_exit = (trend_str < cutoff_off
                            and periods_in_off >= min_risk_off_periods)
                if can_exit:
                    risk_off = False
                    periods_in_on = 0
                    # fall through → portfolio construction
                else:
                    all_rets.append({
                        "timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                        "cost": 0.0, "n_long": 0, "n_short": 0,
                        "turnover": 0, "risk_off": True,
                    })
                    continue
            else:
                periods_in_on += 1
                can_enter = (trend_str > cutoff_on
                             and periods_in_on >= min_risk_on_periods)
                if can_enter:
                    risk_off = True
                    periods_in_off = 0
                    periods_in_on = 0
                    if prev_longs or prev_shorts:
                        n_prev = len(prev_longs) + len(prev_shorts)
                        avg_w = 1.0 / n_prev
                        close_cost = sum(_cost_for_sym(s) * avg_w
                                         for s in prev_longs | prev_shorts)
                        all_rets.append({
                            "timestamp": ts, "gross_ret": 0.0,
                            "net_ret": -close_cost, "cost": close_cost,
                            "n_long": 0, "n_short": 0,
                            "turnover": n_prev, "risk_off": True,
                        })
                    else:
                        all_rets.append({
                            "timestamp": ts, "gross_ret": 0.0,
                            "net_ret": 0.0, "cost": 0.0,
                            "n_long": 0, "n_short": 0,
                            "turnover": 0, "risk_off": True,
                        })
                    prev_longs, prev_shorts = set(), set()
                    continue

        # ── Normal portfolio construction (risk_on) ──
        n = len(grp)
        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl == 0 and ns == 0:
            all_rets.append({
                "timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                "cost": 0.0, "n_long": 0, "n_short": 0,
                "turnover": 0, "risk_off": False,
            })
            continue

        exposure = 1.0
        if (cutoff_on is not None and dyn_threshold is not None
                and trend_str > dyn_threshold):
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) /
                           (cutoff_on - dyn_threshold + 1e-10) * 0.5)

        grp["pred_rank"] = grp["pred"].rank(ascending=False)

        if hysteresis > 0 and (prev_longs or prev_shorts):
            new_longs: Set[str] = set()
            new_shorts: Set[str] = set()
            for idx, r in grp.iterrows():
                sym, rank = r["symbol"], r["pred_rank"]
                if sym in prev_longs and rank <= nl + hysteresis:
                    new_longs.add(sym)
                elif sym in prev_shorts and rank > (n - ns - hysteresis):
                    new_shorts.add(sym)
            remaining = grp[~grp["symbol"].isin(new_longs | new_shorts)]
            for _, r in remaining.sort_values("pred_rank").head(
                    nl - len(new_longs)).iterrows():
                new_longs.add(r["symbol"])
            for _, r in remaining.sort_values("pred_rank", ascending=False).head(
                    ns - len(new_shorts)).iterrows():
                new_shorts.add(r["symbol"])
        else:
            new_longs = (set(grp[grp["pred_rank"] <= nl]["symbol"].tolist())
                         if nl > 0 else set())
            new_shorts = (set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist())
                          if ns > 0 else set())

        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed     = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        total_positions = len(new_longs) + len(new_shorts)

        longs  = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]
        long_ret  = longs["fwd_ret"].mean() if len(longs) > 0 else 0
        short_ret = shorts["fwd_ret"].mean() if len(shorts) > 0 else 0

        nl_act, ns_act = len(new_longs), len(new_shorts)
        if nl_act > 0 and ns_act > 0:
            gross_ret = 0.5 * long_ret - 0.5 * short_ret
        elif ns_act > 0:
            gross_ret = -short_ret
        else:
            gross_ret = long_ret
        gross_ret *= exposure

        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            turnover_cost = sum(_cost_for_sym(sym) * avg_weight
                                for sym in new_opened)
            turnover_cost += sum(_cost_for_sym(sym) * avg_weight
                                 for sym in closed)
            holding_cost = funding_per_12h * (rebal_hours / 12)
            total_cost = turnover_cost + holding_cost
        else:
            total_cost = 0.0

        net_ret = gross_ret - total_cost
        prev_longs, prev_shorts = new_longs, new_shorts

        all_rets.append({
            "timestamp": ts, "gross_ret": gross_ret, "net_ret": net_ret,
            "cost": total_cost, "n_long": nl_act, "n_short": ns_act,
            "turnover": len(new_opened) + len(closed), "risk_off": False,
        })

    return pd.DataFrame(all_rets) if all_rets else pd.DataFrame()


# ─── Main ────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("=" * 70)
    log("R114b — Risk-off Churn Reduction")
    log("=" * 70)

    os.makedirs("results", exist_ok=True)

    log("\nLoading data...")
    df, regime_df = load_data()
    base_feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in base_feats if f in MARKET_LEVEL_FEATURES]

    log("\nTraining ensemble...")
    t1 = time.time()
    preds = train_ensemble(df, base_feats, CONTINUOUS_WINDOWS,
                           seeds=SEEDS, cs_rank_exclude=no_rank)
    log(f"  Trained in {time.time()-t1:.0f}s")

    # ── R113 baseline ──
    log("\n" + "=" * 70)
    log("R113 baseline (cutoff_on=0.9, cutoff_off=0.8, min_off=1, min_on=0)")
    log("=" * 70)
    cfg = dict(PROD_CFG)
    port_base = simulate_v2b(preds, regime_df, 4, 2, cfg,
                             cutoff_on=0.9, cutoff_off=0.8,
                             min_risk_off_periods=1, min_risk_on_periods=0)
    m_base = analyze_config(port_base, "R113_baseline")
    print_result(m_base)
    base_off_events = m_base["n_off_events"]

    # ── Grid search ──
    log("\n" + "=" * 70)
    log("R114b Grid: churn reduction")
    log("=" * 70)

    CUTOFF_OFFS    = [0.8, 0.75, 0.7, 0.65]
    MIN_OFF_PERIODS = [1, 2, 3]
    MIN_ON_PERIODS  = [0, 1, 2]

    results = [m_base]

    for co_off in CUTOFF_OFFS:
        for min_off in MIN_OFF_PERIODS:
            for min_on in MIN_ON_PERIODS:
                label = f"off{co_off}_moff{min_off}_mon{min_on}"
                log(f"\n  {label}...")
                port = simulate_v2b(preds, regime_df, 4, 2, cfg,
                                    cutoff_on=0.9, cutoff_off=co_off,
                                    min_risk_off_periods=min_off,
                                    min_risk_on_periods=min_on)
                m = analyze_config(port, label)
                m["cutoff_off"] = co_off
                m["min_risk_off_periods"] = min_off
                m["min_risk_on_periods"] = min_on
                print_result(m)
                results.append(m)

    # ── Results table ──
    log("\n" + "=" * 70)
    log("R114b RESULTS")
    log("=" * 70)

    hdr = (f"  {'Config':<28} {'NetSh':>7} {'GrSh':>7} {'Ret%':>7} "
           f"{'DD%':>7} {'Calmar':>7} {'%flat':>6} {'#off':>5} "
           f"{'AvgDur':>7} {'Cost%':>6}")
    sep = (f"  {'-'*28} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} "
           f"{'-'*6} {'-'*5} {'-'*7} {'-'*6}")
    log(hdr)
    log(sep)

    def _row(m):
        log(f"  {m['label']:<28} {m['net_sharpe']:>7.3f} "
            f"{m['gross_sharpe']:>7.3f} {m['total_ret_pct']:>7.1f} "
            f"{m['max_dd_pct']:>7.1f} {m['calmar']:>7.2f} "
            f"{m['pct_flat']:>5.1f}% {m['n_off_events']:>5} "
            f"{m['avg_off_duration']:>7.1f} {m['total_cost_pct']:>6.2f}")

    _row(m_base)
    for m in results[1:]:
        _row(m)

    # ── Acceptance filter ──
    acceptable = [m for m in results[1:]
                  if m["net_sharpe"] >= 3.0 and m["max_dd_pct"] >= -12.5]
    churn_reduced = [m for m in acceptable
                     if m["n_off_events"] <= base_off_events * 0.75]

    log(f"\n  Acceptable (Sharpe>=3.0, DD>=-12.5%): "
        f"{len(acceptable)}/{len(results)-1}")
    log(f"  Churn reduced >=25%: {len(churn_reduced)}/{len(acceptable)}")

    if churn_reduced:
        best = max(churn_reduced, key=lambda x: x["calmar"])
        log(f"\n  BEST (churn-reduced + best Calmar): {best['label']}")
    elif acceptable:
        best = max(acceptable, key=lambda x: x["calmar"])
        log(f"\n  BEST (acceptable, best Calmar): {best['label']}")
    else:
        best = m_base
        log(f"\n  No configs beat acceptance. R113 baseline stays.")

    # ── Comparison ──
    log(f"\n  {'Metric':<22} {'R113':>12} {'Best R114b':>12} {'Delta':>10}")
    log(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*10}")
    for metric in ['net_sharpe', 'gross_sharpe', 'total_ret_pct',
                    'max_dd_pct', 'calmar', 'pct_flat',
                    'n_off_events', 'avg_off_duration', 'total_cost_pct']:
        v0 = m_base[metric]
        v1 = best[metric]
        log(f"  {metric:<22} {v0:>12.3f} {v1:>12.3f} {v1 - v0:>+10.3f}")

    # ── Save ──
    df_res = pd.DataFrame(results)
    df_res.to_csv("results/r114b_grid.csv", index=False)
    with open("results/r114b_best.json", "w") as f:
        json.dump(best, f, indent=2)

    log(f"\nSaved: results/r114b_grid.csv, r114b_best.json")
    log(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
