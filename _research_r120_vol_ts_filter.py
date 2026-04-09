#!/usr/bin/env python3
"""
R120 — Vol Scaling + Per-Coin TS Filter
========================================

Two experiments vs R114b baseline (Sharpe 3.266, DD -10.9%, Calmar 18.25):

  A) Inverse-vol position sizing: weight positions as 1/vol (normalized),
     instead of equal weight. Uses gk_vol_24h from features.
     Goal: reduce vol concentration, improve risk-adjusted returns.

  B) Per-coin time-series filter: exclude longs with negative own-momentum
     (and shorts with positive momentum). Uses ret_24h / ret_168h.
     Goal: avoid entering positions against own trend.

  C) Combined A + B.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from typing import Set, Dict, Optional

from _research_r22_models import SEEDS, log
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r68_continuous_wf import (
    CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, PROD_CFG,
    load_data, train_ensemble, sharpe, _cost_for_sym,
)
from _research_r113_trend_cutoff_reopt import analyze_config, print_result


# ─── R114b champion config ──────────────────────────────────

R114B_CFG = {
    "n_long": 4, "n_short": 2, "rebal_hours": 12,
    "trend_cutoff": 0.9, "dyn_threshold": 0.7,
    "ema_alpha": 0.5, "hysteresis": 3,
}
R114B_CUTOFF_ON = 0.9
R114B_CUTOFF_OFF = 0.8
R114B_MIN_OFF = 2
R114B_MIN_ON = 0


# ─── Modified simulation with vol-scaling + TS filter ────────

def simulate_r120(merged, regime_df, n_long, n_short, cfg,
                  cutoff_on=0.9, cutoff_off=None,
                  min_risk_off_periods=2, min_risk_on_periods=0,
                  # ── R120 extensions ──
                  vol_scaling=False,
                  vol_lookup=None,
                  vol_max_weight_mult=3.0,
                  ts_filter=False,
                  mom_lookup=None,
                  ts_filter_strict=False):
    """
    R114b simulate_v2b + optional vol-scaling and per-coin TS filter.

    vol_scaling: if True, weight positions as 1/vol (inverse realized vol).
    vol_lookup: dict of (timestamp, symbol) → vol value (e.g. gk_vol_24h).
    vol_max_weight_mult: max weight = mult * (1/n_positions). Caps outliers.

    ts_filter: if True, filter longs with negative momentum, shorts with positive.
    mom_lookup: dict of (timestamp, symbol) → momentum value (e.g. ret_24h).
    ts_filter_strict: if True, use ret_168h (weekly); else ret_24h.
    """
    if cutoff_off is None and cutoff_on is not None:
        cutoff_off = cutoff_on - 0.1

    rebal_hours   = cfg["rebal_hours"]
    ema_alpha     = cfg.get("ema_alpha", None)
    hysteresis    = cfg.get("hysteresis", 0)
    dyn_threshold = cfg.get("dyn_threshold", 0.5)
    funding_per_12h = 0.00008

    all_rets = []
    prev_longs: Set[str]       = set()
    prev_shorts: Set[str]      = set()
    prev_preds: Dict[str, float] = {}
    risk_off = False
    periods_in_off = 0
    periods_in_on  = 999

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}
    rebal_timestamps = timestamps_sorted[::rebal_hours]

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        grp = grouped[ts].copy()

        # ── EMA smoothing ──
        if ema_alpha is not None and ema_alpha < 1.0:
            for idx, r in grp.iterrows():
                sym = r["symbol"]
                raw_pred = r["pred"]
                smoothed = (ema_alpha * raw_pred
                            + (1 - ema_alpha) * prev_preds.get(sym, raw_pred))
                prev_preds[sym] = smoothed
                grp.at[idx, "pred"] = smoothed

        # ── State machine with timing constraints (R114b) ──
        if cutoff_on is not None:
            if risk_off:
                periods_in_off += 1
                can_exit = (trend_str < cutoff_off
                            and periods_in_off >= min_risk_off_periods)
                if can_exit:
                    risk_off = False
                    periods_in_on = 0
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

        # ── Portfolio construction ──
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

        # ── Ranking with hysteresis (unchanged from R114b) ──
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
            for _, r in remaining.sort_values(
                    "pred_rank", ascending=False).head(
                    ns - len(new_shorts)).iterrows():
                new_shorts.add(r["symbol"])
        else:
            new_longs = (set(grp[grp["pred_rank"] <= nl]["symbol"].tolist())
                         if nl > 0 else set())
            new_shorts = (set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist())
                          if ns > 0 else set())

        # ── R120-B: Per-coin TS filter ──
        if ts_filter and mom_lookup is not None:
            filtered_longs = set()
            for sym in new_longs:
                mom = mom_lookup.get((ts, sym), 0.0)
                if mom > 0:
                    filtered_longs.add(sym)
            filtered_shorts = set()
            for sym in new_shorts:
                mom = mom_lookup.get((ts, sym), 0.0)
                if mom < 0:
                    filtered_shorts.add(sym)
            new_longs = filtered_longs if filtered_longs else new_longs
            new_shorts = filtered_shorts if filtered_shorts else new_shorts

        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed     = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        total_positions = len(new_longs) + len(new_shorts)

        longs  = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]
        nl_act, ns_act = len(new_longs), len(new_shorts)

        # ── R120-A: Vol-weighted returns ──
        if vol_scaling and vol_lookup is not None and total_positions > 0:
            # Compute inverse-vol weights
            long_weights = {}
            for sym in new_longs:
                v = vol_lookup.get((ts, sym), None)
                if v is not None and v > 1e-8:
                    long_weights[sym] = 1.0 / v
                else:
                    long_weights[sym] = 1.0  # fallback: equal

            short_weights = {}
            for sym in new_shorts:
                v = vol_lookup.get((ts, sym), None)
                if v is not None and v > 1e-8:
                    short_weights[sym] = 1.0 / v
                else:
                    short_weights[sym] = 1.0

            # Normalize each side to sum to 0.5
            def _normalize_weights(w_dict, max_mult):
                if not w_dict:
                    return {}
                total = sum(w_dict.values())
                if total < 1e-12:
                    eq = 1.0 / len(w_dict)
                    return {s: eq for s in w_dict}
                # Normalize to 1.0
                normed = {s: v / total for s, v in w_dict.items()}
                # Cap at max_mult * (1/n)
                n_pos = len(normed)
                max_w = max_mult / n_pos
                capped = False
                for s in normed:
                    if normed[s] > max_w:
                        normed[s] = max_w
                        capped = True
                if capped:
                    # Re-normalize uncapped
                    total2 = sum(normed.values())
                    normed = {s: v / total2 for s, v in normed.items()}
                return normed

            long_w = _normalize_weights(long_weights, vol_max_weight_mult)
            short_w = _normalize_weights(short_weights, vol_max_weight_mult)

            # Weighted returns
            grp_idx = grp.set_index("symbol")
            long_ret = sum(long_w.get(sym, 0) * grp_idx.at[sym, "fwd_ret"]
                           for sym in new_longs) if nl_act > 0 else 0
            short_ret = sum(short_w.get(sym, 0) * grp_idx.at[sym, "fwd_ret"]
                            for sym in new_shorts) if ns_act > 0 else 0

            if nl_act > 0 and ns_act > 0:
                gross_ret = 0.5 * long_ret - 0.5 * short_ret
            elif ns_act > 0:
                gross_ret = -short_ret
            else:
                gross_ret = long_ret
            gross_ret *= exposure

            # Weighted costs
            all_weights = {}
            for sym in new_longs:
                all_weights[sym] = 0.5 * long_w.get(sym, 0) if nl_act > 0 and ns_act > 0 else long_w.get(sym, 0)
            for sym in new_shorts:
                all_weights[sym] = 0.5 * short_w.get(sym, 0) if nl_act > 0 and ns_act > 0 else short_w.get(sym, 0)

            turnover_cost = sum(_cost_for_sym(sym) * all_weights.get(sym, 0)
                                for sym in new_opened)
            turnover_cost += sum(_cost_for_sym(sym) * all_weights.get(sym, 0)
                                 for sym in closed
                                 if sym in all_weights)
            # For closed positions not in current weights, use prev equal weight
            for sym in closed:
                if sym not in all_weights:
                    n_prev = len(prev_longs) + len(prev_shorts)
                    if n_prev > 0:
                        turnover_cost += _cost_for_sym(sym) * (1.0 / n_prev)
            holding_cost = funding_per_12h * (rebal_hours / 12)
            total_cost = turnover_cost + holding_cost

        else:
            # ── Original equal-weight returns ──
            long_ret  = longs["fwd_ret"].mean() if nl_act > 0 else 0
            short_ret = shorts["fwd_ret"].mean() if ns_act > 0 else 0

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


# ─── Build lookup dicts from feature dataframe ──────────────

def build_vol_lookup(df, col="gk_vol_24h"):
    """Dict of (timestamp, symbol) → vol value."""
    sub = df[["timestamp", "symbol", col]].dropna(subset=[col])
    return dict(zip(zip(sub["timestamp"], sub["symbol"]), sub[col]))


def build_mom_lookup(df, col="ret_24h"):
    """Dict of (timestamp, symbol) → momentum value."""
    sub = df[["timestamp", "symbol", col]].dropna(subset=[col])
    return dict(zip(zip(sub["timestamp"], sub["symbol"]), sub[col]))


# ─── Per-window analysis ─────────────────────────────────────

def per_window_sharpe(port, merged):
    """Compute Sharpe per walk-forward window."""
    if port.empty or "window" not in merged.columns:
        return {}
    # Map timestamps to windows
    ts_window = merged.drop_duplicates("timestamp")[["timestamp", "window"]]
    ts_window = ts_window.set_index("timestamp")["window"]
    port = port.copy()
    port["window"] = port["timestamp"].map(ts_window)
    results = {}
    for w, wport in port.groupby("window"):
        if len(wport) > 10:
            results[w] = round(sharpe(wport["net_ret"]), 3)
    return results


# ─── Main ────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("=" * 70)
    log("R120 — Vol Scaling + Per-Coin TS Filter")
    log("=" * 70)

    os.makedirs("results", exist_ok=True)

    # ── Load data ──
    log("\nLoading data...")
    df, regime_df = load_data()
    base_feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in base_feats if f in MARKET_LEVEL_FEATURES]

    # ── Build feature lookups (before training to avoid holding full df later) ──
    log("\nBuilding feature lookups...")
    vol_lookup_gk24 = build_vol_lookup(df, "gk_vol_24h")
    vol_lookup_rv24 = build_vol_lookup(df, "rvol_24h")
    mom_lookup_24h  = build_mom_lookup(df, "ret_24h")
    mom_lookup_168h = build_mom_lookup(df, "ret_168h")
    log(f"  Vol lookup (gk_vol_24h): {len(vol_lookup_gk24):,} entries")
    log(f"  Vol lookup (rvol_24h):   {len(vol_lookup_rv24):,} entries")
    log(f"  Mom lookup (ret_24h):    {len(mom_lookup_24h):,} entries")
    log(f"  Mom lookup (ret_168h):   {len(mom_lookup_168h):,} entries")

    # ── Train ensemble ──
    log("\nTraining ensemble...")
    t1 = time.time()
    preds = train_ensemble(df, base_feats, CONTINUOUS_WINDOWS,
                           seeds=SEEDS, cs_rank_exclude=no_rank)
    log(f"  Trained in {time.time()-t1:.0f}s")

    if preds is None or preds.empty:
        log("ERROR: No predictions generated")
        return

    cfg = dict(R114B_CFG)

    # ── A0: R114b baseline ──
    log("\n" + "=" * 70)
    log("A0: R114b BASELINE (equal weight, no TS filter)")
    log("=" * 70)
    port_base = simulate_r120(preds, regime_df, 4, 2, cfg,
                              cutoff_on=R114B_CUTOFF_ON,
                              cutoff_off=R114B_CUTOFF_OFF,
                              min_risk_off_periods=R114B_MIN_OFF,
                              min_risk_on_periods=R114B_MIN_ON)
    m_base = analyze_config(port_base, "R114b_baseline")
    print_result(m_base)
    pw_base = per_window_sharpe(port_base, preds)
    log(f"    Per-window: {pw_base}")

    results = [m_base]

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT A: Inverse-Vol Position Sizing
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("EXPERIMENT A: Inverse-Vol Position Sizing")
    log("=" * 70)

    VOL_COLS = [("gk_vol_24h", vol_lookup_gk24), ("rvol_24h", vol_lookup_rv24)]
    MAX_WEIGHT_MULTS = [2.0, 3.0, 5.0]

    for vol_name, vol_lk in VOL_COLS:
        for mwm in MAX_WEIGHT_MULTS:
            label = f"A_vol_{vol_name}_cap{mwm}"
            log(f"\n  {label}...")
            port = simulate_r120(preds, regime_df, 4, 2, cfg,
                                 cutoff_on=R114B_CUTOFF_ON,
                                 cutoff_off=R114B_CUTOFF_OFF,
                                 min_risk_off_periods=R114B_MIN_OFF,
                                 min_risk_on_periods=R114B_MIN_ON,
                                 vol_scaling=True,
                                 vol_lookup=vol_lk,
                                 vol_max_weight_mult=mwm)
            m = analyze_config(port, label)
            print_result(m)
            pw = per_window_sharpe(port, preds)
            log(f"    Per-window: {pw}")
            m["vol_col"] = vol_name
            m["max_weight_mult"] = mwm
            results.append(m)

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT B: Per-Coin TS Filter
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("EXPERIMENT B: Per-Coin TS Filter")
    log("=" * 70)

    MOM_COLS = [("ret_24h", mom_lookup_24h), ("ret_168h", mom_lookup_168h)]

    for mom_name, mom_lk in MOM_COLS:
        label = f"B_tsfilter_{mom_name}"
        log(f"\n  {label}...")
        port = simulate_r120(preds, regime_df, 4, 2, cfg,
                             cutoff_on=R114B_CUTOFF_ON,
                             cutoff_off=R114B_CUTOFF_OFF,
                             min_risk_off_periods=R114B_MIN_OFF,
                             min_risk_on_periods=R114B_MIN_ON,
                             ts_filter=True,
                             mom_lookup=mom_lk)
        m = analyze_config(port, label)
        print_result(m)
        pw = per_window_sharpe(port, preds)
        log(f"    Per-window: {pw}")
        m["mom_col"] = mom_name
        results.append(m)

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT C: Combined Vol-Scaling + TS Filter
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("EXPERIMENT C: Combined (Vol-Scaling + TS Filter)")
    log("=" * 70)

    # Pick best vol col and best mom col for combination
    COMBO_CONFIGS = [
        ("gk_vol_24h", vol_lookup_gk24, 3.0, "ret_24h",  mom_lookup_24h),
        ("gk_vol_24h", vol_lookup_gk24, 3.0, "ret_168h", mom_lookup_168h),
        ("rvol_24h",   vol_lookup_rv24, 3.0, "ret_24h",  mom_lookup_24h),
        ("rvol_24h",   vol_lookup_rv24, 3.0, "ret_168h", mom_lookup_168h),
    ]

    for vol_name, vol_lk, mwm, mom_name, mom_lk in COMBO_CONFIGS:
        label = f"C_{vol_name}_cap{mwm}_{mom_name}"
        log(f"\n  {label}...")
        port = simulate_r120(preds, regime_df, 4, 2, cfg,
                             cutoff_on=R114B_CUTOFF_ON,
                             cutoff_off=R114B_CUTOFF_OFF,
                             min_risk_off_periods=R114B_MIN_OFF,
                             min_risk_on_periods=R114B_MIN_ON,
                             vol_scaling=True,
                             vol_lookup=vol_lk,
                             vol_max_weight_mult=mwm,
                             ts_filter=True,
                             mom_lookup=mom_lk)
        m = analyze_config(port, label)
        print_result(m)
        pw = per_window_sharpe(port, preds)
        log(f"    Per-window: {pw}")
        m["vol_col"] = vol_name
        m["max_weight_mult"] = mwm
        m["mom_col"] = mom_name
        results.append(m)

    # ══════════════════════════════════════════════════════════
    # RESULTS SUMMARY
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("R120 RESULTS SUMMARY")
    log("=" * 70)

    hdr = (f"  {'Config':<40} {'NetSh':>7} {'GrSh':>7} {'Ret%':>7} "
           f"{'DD%':>7} {'Calmar':>7} {'%flat':>6} {'Cost%':>6}")
    sep = (f"  {'-'*40} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} "
           f"{'-'*6} {'-'*6}")
    log(hdr)
    log(sep)

    for m in results:
        log(f"  {m['label']:<40} {m['net_sharpe']:>7.3f} "
            f"{m['gross_sharpe']:>7.3f} {m['total_ret_pct']:>7.1f} "
            f"{m['max_dd_pct']:>7.1f} {m['calmar']:>7.2f} "
            f"{m['pct_flat']:>5.1f}% {m['total_cost_pct']:>6.2f}")

    # ── Find best ──
    base_sharpe = m_base["net_sharpe"]
    improved = [m for m in results[1:]
                if m["net_sharpe"] > base_sharpe
                and m["max_dd_pct"] >= -15.0]

    log(f"\n  Baseline Sharpe: {base_sharpe:.3f}")
    log(f"  Improved configs (Sharpe > baseline, DD >= -15%): "
        f"{len(improved)}/{len(results)-1}")

    if improved:
        best = max(improved, key=lambda x: x["net_sharpe"])
        log(f"\n  BEST: {best['label']}")
        log(f"    Sharpe: {best['net_sharpe']:.3f} "
            f"(+{best['net_sharpe'] - base_sharpe:.3f})")
        log(f"    DD: {best['max_dd_pct']:.1f}%  "
            f"Calmar: {best['calmar']:.2f}")
    else:
        log("\n  No improvement over baseline. R114b stays champion.")

    # ── Comparison table ──
    if improved:
        best_m = best
    else:
        best_m = max(results[1:], key=lambda x: x["net_sharpe"])

    log(f"\n  {'Metric':<22} {'R114b':>12} {'Best R120':>12} {'Delta':>10}")
    log(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*10}")
    for metric in ['net_sharpe', 'gross_sharpe', 'total_ret_pct',
                    'max_dd_pct', 'calmar', 'pct_flat', 'total_cost_pct']:
        v0 = m_base[metric]
        v1 = best_m[metric]
        log(f"  {metric:<22} {v0:>12.3f} {v1:>12.3f} {v1 - v0:>+10.3f}")

    # ── Save ──
    df_res = pd.DataFrame(results)
    df_res.to_csv("results/r120_grid.csv", index=False)
    with open("results/r120_best.json", "w") as f:
        json.dump(best_m if improved else m_base, f, indent=2, default=str)

    log(f"\nSaved: results/r120_grid.csv, r120_best.json")
    log(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
