#!/usr/bin/env python3
"""
R113 — Trend Cutoff Reoptimization with correct time accounting.

P0 fix: simulate_v2() with risk-off state machine:
  - Every rebal timestamp records a return (no skipped periods)
  - Risk-off with hysteresis: enter_off when trend_str > cutoff_on,
    exit_off when trend_str < cutoff_off (= cutoff_on - 0.1)
  - selection_state (EMA/prev_preds) lives through risk_off
  - positions_state (prev_longs/prev_shorts) cleared on enter_off

Grid search: cutoff_on ∈ {0.9, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, None}
"""
import time, json, os, warnings
import numpy as np, pandas as pd
from typing import Set, Dict, Optional
warnings.filterwarnings("ignore")

from _research_r22_models import SEEDS, log
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r68_continuous_wf import (
    CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, PROD_CFG,
    load_data, train_ensemble, sharpe, _cost_for_sym,
)


# ─── P0 fixed simulate: risk-off state machine ──────────────────────────

def simulate_v2(merged, regime_df, n_long, n_short, cfg,
                cutoff_on=0.9, cutoff_off=None):
    """
    Fixed simulate with correct time accounting.

    - Every rebal timestamp → a record (no `continue` skip)
    - Risk-off state machine with hysteresis:
        enter risk_off when trend_str > cutoff_on
        exit  risk_off when trend_str < cutoff_off
    - EMA (prev_preds) stays alive during risk_off
    - prev_longs/prev_shorts cleared on enter_off
    - If cutoff_on is None → no risk-off filter at all
    """
    if cutoff_off is None and cutoff_on is not None:
        cutoff_off = cutoff_on - 0.1

    rebal_hours  = cfg["rebal_hours"]
    ema_alpha    = cfg.get("ema_alpha", None)
    hysteresis   = cfg.get("hysteresis", 0)
    dyn_threshold = cfg.get("dyn_threshold", 0.5)
    funding_per_12h = 0.00008

    all_rets = []
    prev_longs: Set[str] = set()
    prev_shorts: Set[str] = set()
    prev_preds: Dict[str, float] = {}       # ← survives risk_off
    risk_off = False

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}
    rebal_timestamps = timestamps_sorted[::rebal_hours]

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        grp = grouped[ts].copy()

        # ── Update EMA even in risk_off (selection_state lives) ──
        if ema_alpha is not None and ema_alpha < 1.0:
            for idx, r in grp.iterrows():
                sym = r["symbol"]
                raw_pred = r["pred"]
                smoothed = ema_alpha * raw_pred + (1 - ema_alpha) * prev_preds.get(sym, raw_pred)
                prev_preds[sym] = smoothed
                grp.at[idx, "pred"] = smoothed

        # ── Risk-off state machine ──
        if cutoff_on is not None:
            if not risk_off and trend_str > cutoff_on:
                # ENTER risk_off: close all positions
                risk_off = True
                if prev_longs or prev_shorts:
                    n_prev = len(prev_longs) + len(prev_shorts)
                    avg_w = 1.0 / n_prev
                    close_cost = sum(_cost_for_sym(s) * avg_w for s in prev_longs | prev_shorts)
                    all_rets.append({
                        "timestamp": ts, "gross_ret": 0.0, "net_ret": -close_cost,
                        "cost": close_cost, "n_long": 0, "n_short": 0,
                        "turnover": n_prev, "risk_off": True,
                    })
                else:
                    all_rets.append({
                        "timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                        "cost": 0.0, "n_long": 0, "n_short": 0,
                        "turnover": 0, "risk_off": True,
                    })
                prev_longs, prev_shorts = set(), set()
                continue

            if risk_off:
                if trend_str < cutoff_off:
                    risk_off = False
                    # fall through to normal portfolio construction
                else:
                    # STAY risk_off: record flat period
                    all_rets.append({
                        "timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                        "cost": 0.0, "n_long": 0, "n_short": 0,
                        "turnover": 0, "risk_off": True,
                    })
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

        # Dynamic exposure
        exposure = 1.0
        if cutoff_on is not None and dyn_threshold is not None and trend_str > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) /
                           (cutoff_on - dyn_threshold + 1e-10) * 0.5)

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
        gross_ret *= exposure

        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            turnover_cost = sum(_cost_for_sym(sym) * avg_weight for sym in new_opened)
            turnover_cost += sum(_cost_for_sym(sym) * avg_weight for sym in closed)
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


# ─── Analysis helpers ────────────────────────────────────────────────────

def calmar(port):
    if port.empty or len(port) < 2:
        return 0.0
    eq = (1 + port["net_ret"]).cumprod() * 100
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    maxdd = (eq / eq.cummax() - 1).min()
    if maxdd == 0:
        return 0.0
    return total_ret / abs(maxdd)


def analyze_config(port, label=None):
    """Return dict of metrics for a single config."""
    if port.empty:
        return {"label": label, "net_sharpe": 0, "gross_sharpe": 0}

    gs = sharpe(port["gross_ret"])
    ns = sharpe(port["net_ret"])
    eq = (1 + port["net_ret"]).cumprod() * 100
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    maxdd = (eq / eq.cummax() - 1).min()
    cal = total_ret / abs(maxdd) if maxdd != 0 else 0
    wr = (port["net_ret"] > 0).mean() * 100
    n_periods = len(port)
    n_flat = int((port.get("risk_off", pd.Series(dtype=bool))).sum()) if "risk_off" in port.columns else 0
    pct_flat = n_flat / n_periods * 100 if n_periods > 0 else 0
    total_cost = port["cost"].sum() * 100
    avg_turnover = port["turnover"].mean()

    # Risk-off event counting
    if "risk_off" in port.columns:
        ro = port["risk_off"].astype(int)
        n_off_events = (ro.diff() == 1).sum()
        # avg duration of each risk-off spell
        off_durations = []
        in_spell = False
        spell_len = 0
        for v in ro:
            if v == 1:
                in_spell = True
                spell_len += 1
            else:
                if in_spell:
                    off_durations.append(spell_len)
                    spell_len = 0
                    in_spell = False
        if in_spell:
            off_durations.append(spell_len)
        avg_off_duration = np.mean(off_durations) if off_durations else 0
    else:
        n_off_events = 0
        avg_off_duration = 0

    return {
        "label": label,
        "gross_sharpe": round(gs, 3),
        "net_sharpe": round(ns, 3),
        "total_ret_pct": round(total_ret * 100, 1),
        "max_dd_pct": round(maxdd * 100, 1),
        "calmar": round(cal, 2),
        "win_rate": round(wr, 1),
        "n_periods": n_periods,
        "n_flat": n_flat,
        "pct_flat": round(pct_flat, 1),
        "n_off_events": int(n_off_events),
        "avg_off_duration": round(avg_off_duration, 1),
        "total_cost_pct": round(total_cost, 2),
        "avg_turnover": round(avg_turnover, 2),
    }


def print_result(m):
    log(f"    Sharpe: G={m['gross_sharpe']:.3f}  N={m['net_sharpe']:.3f}"
        f"  Ret={m['total_ret_pct']:.1f}%  DD={m['max_dd_pct']:.1f}%"
        f"  Calmar={m['calmar']:.2f}")
    log(f"    Periods={m['n_periods']}  Flat={m['n_flat']} ({m['pct_flat']:.1f}%)"
        f"  OffEvents={m['n_off_events']}  AvgOffDur={m['avg_off_duration']:.1f}"
        f"  Cost={m['total_cost_pct']:.2f}%  Turnover={m['avg_turnover']:.1f}")


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("=" * 70)
    log("R113 — Trend Cutoff Reoptimization (P0 fix + grid)")
    log("=" * 70)

    os.makedirs("results", exist_ok=True)

    # ── Load data & train models once ──
    log("\nLoading data...")
    df, regime_df = load_data()
    base_feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in base_feats if f in MARKET_LEVEL_FEATURES]

    log("\nTraining ensemble...")
    t1 = time.time()
    preds = train_ensemble(df, base_feats, CONTINUOUS_WINDOWS,
                           seeds=SEEDS, cs_rank_exclude=no_rank)
    log(f"  Trained in {time.time()-t1:.0f}s")

    # ── P0 sanity: time accounting check ──
    log("\n" + "=" * 70)
    log("P0 sanity: time accounting with cutoff_on=0.9")
    log("=" * 70)

    cfg = dict(PROD_CFG)
    port_p0 = simulate_v2(preds, regime_df, 4, 2, cfg, cutoff_on=0.9)
    m_p0 = analyze_config(port_p0, "P0_cutoff_0.9")
    print_result(m_p0)

    # Calendar check
    rebal_ts = sorted(preds["timestamp"].unique())[::cfg["rebal_hours"]]
    expected_periods = sum(1 for ts in rebal_ts if ts in regime_df.index)
    log(f"  Expected periods (calendar): {expected_periods}")
    log(f"  Actual periods recorded:     {m_p0['n_periods']}")
    log(f"  Match: {'YES' if m_p0['n_periods'] == expected_periods else 'NO (delta=' + str(expected_periods - m_p0['n_periods']) + ')'}")

    # Save sanity check
    with open("results/r113a_time_accounting_check.json", "w") as f:
        json.dump({
            **m_p0,
            "expected_periods": expected_periods,
            "periods_match": m_p0["n_periods"] == expected_periods,
        }, f, indent=2)

    # ── R113 Grid Search ──
    log("\n" + "=" * 70)
    log("R113 Grid Search: cutoff_on sweep")
    log("=" * 70)

    CUTOFF_GRID = [0.9, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, None]

    results = []
    for co in CUTOFF_GRID:
        label = f"cutoff_{co}" if co is not None else "cutoff_None"
        co_off = co - 0.1 if co is not None else None
        log(f"\n  {label} (on={co}, off={co_off})...")

        port = simulate_v2(preds, regime_df, 4, 2, cfg,
                           cutoff_on=co, cutoff_off=co_off)
        m = analyze_config(port, label)
        m["cutoff_on"] = co
        m["cutoff_off"] = co_off
        print_result(m)
        results.append(m)

    # ── Results table ──
    log("\n" + "=" * 70)
    log("R113 RESULTS GRID")
    log("=" * 70)
    log(f"  {'cutoff_on':>10} {'cutoff_off':>10} {'NetSh':>7} {'GrSh':>7} {'Ret%':>7} {'DD%':>7} {'Calmar':>7} {'%flat':>6} {'#off':>5} {'AvgDur':>7}")
    log(f"  {'-'*10} {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*5} {'-'*7}")

    best = None
    for m in results:
        co_str = f"{m['cutoff_on']}" if m['cutoff_on'] is not None else "None"
        coff_str = f"{m['cutoff_off']}" if m['cutoff_off'] is not None else "None"
        log(f"  {co_str:>10} {coff_str:>10} {m['net_sharpe']:>7.3f} {m['gross_sharpe']:>7.3f}"
            f" {m['total_ret_pct']:>7.1f} {m['max_dd_pct']:>7.1f} {m['calmar']:>7.2f}"
            f" {m['pct_flat']:>5.1f}% {m['n_off_events']:>5} {m['avg_off_duration']:>7.1f}")
        # Best by net Sharpe (but prefer pct_flat 2-10%)
        if best is None or m["net_sharpe"] > best["net_sharpe"]:
            best = m

    # Also find best in sweet spot (pct_flat 2-15%)
    sweet = [m for m in results if 2 <= m["pct_flat"] <= 15]
    best_sweet = max(sweet, key=lambda x: x["net_sharpe"]) if sweet else None

    log(f"\n  BEST overall:      {best['label']} → Net Sharpe={best['net_sharpe']:.3f}, %flat={best['pct_flat']:.1f}%")
    if best_sweet:
        log(f"  BEST sweet (2-15%): {best_sweet['label']} → Net Sharpe={best_sweet['net_sharpe']:.3f}, %flat={best_sweet['pct_flat']:.1f}%")

    # ── Save results ──
    df_results = pd.DataFrame(results)
    df_results.to_csv("results/r113_grid.csv", index=False)

    best_config = best_sweet if best_sweet else best
    with open("results/r113_best.json", "w") as f:
        json.dump(best_config, f, indent=2)

    # Save equity curve of best config
    port_best = simulate_v2(preds, regime_df, 4, 2, cfg,
                            cutoff_on=best_config["cutoff_on"],
                            cutoff_off=best_config["cutoff_off"])
    eq = (1 + port_best["net_ret"]).cumprod() * 100
    eq_df = pd.DataFrame({"timestamp": port_best["timestamp"], "equity": eq.values})
    eq_df.to_csv("results/r113_equity_best.csv", index=False)

    log(f"\nSaved: results/r113_grid.csv, r113_best.json, r113_equity_best.csv")
    log(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
