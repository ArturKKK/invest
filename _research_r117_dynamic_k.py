#!/usr/bin/env python3
"""
R117 — Dynamic K (confidence-based position sizing).

Instead of fixed 4L/2S, scale positions based on model confidence:
  - confidence = std(predictions) across coins at each timestamp
  - High confidence (wide pred spread): more positions (6L/3S)
  - Low confidence (narrow pred spread): fewer positions (2L/1S) or flat

Grid:
  - method: std_pred, range_pred (max-min)
  - high_threshold quantile ∈ {0.6, 0.7}
  - low_threshold quantile ∈ {0.3, 0.4}
  - high_K / low_K combos

Uses R114b champion params: moff=2, mon=0, cutoff=0.9/0.8

Acceptance:
  - Sharpe >= R114b (3.266) or Calmar >= R114b (18.25)
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
from _research_r114b_churn_reduction import simulate_v2b


# ─── Simulate with dynamic K ────────────────────────────────────

def simulate_v2b_dynK(merged, regime_df, cfg,
                      cutoff_on=0.9, cutoff_off=0.8,
                      min_risk_off_periods=2, min_risk_on_periods=0,
                      confidence_method="std_pred",
                      high_q=0.7, low_q=0.3,
                      high_nl=6, high_ns=3,
                      mid_nl=4, mid_ns=2,
                      low_nl=2, low_ns=1):
    """
    R114b simulate_v2b with dynamic position count based on model confidence.

    At each rebalance:
      1. Compute confidence = std(pred) or range(pred) across coins
      2. Compare to historical quantiles (expanding window)
      3. If confidence > high_q quantile → use high_nl/high_ns
         If confidence < low_q quantile → use low_nl/low_ns
         Otherwise → use mid_nl/mid_ns
    """
    rebal_hours = cfg["rebal_hours"]
    ema_alpha = cfg.get("ema_alpha", None)
    hysteresis = cfg.get("hysteresis", 0)
    dyn_threshold = cfg.get("dyn_threshold", 0.5)
    funding_per_12h = 0.00008

    all_rets = []
    prev_longs: Set[str] = set()
    prev_shorts: Set[str] = set()
    prev_preds: Dict[str, float] = {}
    risk_off = False
    periods_in_off = 0
    periods_in_on = 999

    # Track confidence history for expanding-window quantiles
    confidence_history = []

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}
    rebal_timestamps = timestamps_sorted[::rebal_hours]

    k_choices = []  # diagnostic: track K choices

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        grp = grouped[ts].copy()

        # ── Update EMA ──
        if ema_alpha is not None and ema_alpha < 1.0:
            for idx, r in grp.iterrows():
                sym = r["symbol"]
                raw_pred = r["pred"]
                smoothed = (ema_alpha * raw_pred
                            + (1 - ema_alpha) * prev_preds.get(sym, raw_pred))
                prev_preds[sym] = smoothed
                grp.at[idx, "pred"] = smoothed

        # ── Compute confidence ──
        if confidence_method == "std_pred":
            conf = grp["pred"].std()
        else:  # range_pred
            conf = grp["pred"].max() - grp["pred"].min()
        confidence_history.append(conf)

        # Expanding-window quantiles (need ≥50 history points)
        if len(confidence_history) >= 50:
            hist = np.array(confidence_history)
            q_high = np.quantile(hist, high_q)
            q_low = np.quantile(hist, low_q)
            if conf >= q_high:
                n_long, n_short = high_nl, high_ns
                k_label = "high"
            elif conf <= q_low:
                n_long, n_short = low_nl, low_ns
                k_label = "low"
            else:
                n_long, n_short = mid_nl, mid_ns
                k_label = "mid"
        else:
            n_long, n_short = mid_nl, mid_ns
            k_label = "warmup"

        k_choices.append(k_label)

        # ── State machine (R114b) ──
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
                        "k_choice": k_label,
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
                            "k_choice": k_label,
                        })
                    else:
                        all_rets.append({
                            "timestamp": ts, "gross_ret": 0.0,
                            "net_ret": 0.0, "cost": 0.0,
                            "n_long": 0, "n_short": 0,
                            "turnover": 0, "risk_off": True,
                            "k_choice": k_label,
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
                "turnover": 0, "risk_off": False, "k_choice": k_label,
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
            "k_choice": k_label,
        })

    port = pd.DataFrame(all_rets) if all_rets else pd.DataFrame()

    # K-choice distribution diagnostic
    if k_choices:
        from collections import Counter
        counts = Counter(k_choices)
        port.attrs["k_distribution"] = dict(counts)

    return port


# ─── Main ────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("=" * 70)
    log("R117 — Dynamic K (confidence-based position sizing)")
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

    cfg = dict(PROD_CFG)

    # ── R114b baseline ──
    log("\n" + "=" * 70)
    log("R114b baseline (4L/2S fixed, moff=2)")
    log("=" * 70)
    port_base = simulate_v2b(preds, regime_df, 4, 2, cfg,
                             cutoff_on=0.9, cutoff_off=0.8,
                             min_risk_off_periods=2, min_risk_on_periods=0)
    m_base = analyze_config(port_base, "R114b_fixed4L2S")
    print_result(m_base)

    # ── Grid search ──
    log("\n" + "=" * 70)
    log("R117 Grid: Dynamic K")
    log("=" * 70)

    CONFIGS = [
        # (label, method, high_q, low_q, high_nl, high_ns, mid_nl, mid_ns, low_nl, low_ns)
        # Variant A: expand on high confidence, shrink on low
        ("std_q70_30_6L3S_2L1S", "std_pred", 0.7, 0.3, 6, 3, 4, 2, 2, 1),
        ("std_q60_40_6L3S_2L1S", "std_pred", 0.6, 0.4, 6, 3, 4, 2, 2, 1),
        # Variant B: go flat on low confidence
        ("std_q70_30_6L3S_flat", "std_pred", 0.7, 0.3, 6, 3, 4, 2, 0, 0),
        ("std_q60_40_6L3S_flat", "std_pred", 0.6, 0.4, 6, 3, 4, 2, 0, 0),
        # Variant C: only expand on high, keep 4L2S otherwise
        ("std_q70_6L3S_else4L2S", "std_pred", 0.7, 0.0, 6, 3, 4, 2, 4, 2),
        ("std_q60_6L3S_else4L2S", "std_pred", 0.6, 0.0, 6, 3, 4, 2, 4, 2),
        # Variant D: only shrink on low, keep 4L2S otherwise
        ("std_q30_4L2S_else2L1S", "std_pred", 1.0, 0.3, 4, 2, 4, 2, 2, 1),
        ("std_q40_4L2S_else2L1S", "std_pred", 1.0, 0.4, 4, 2, 4, 2, 2, 1),
        # Variant E: range-based confidence
        ("rng_q70_30_6L3S_2L1S", "range_pred", 0.7, 0.3, 6, 3, 4, 2, 2, 1),
        ("rng_q60_40_6L3S_2L1S", "range_pred", 0.6, 0.4, 6, 3, 4, 2, 2, 1),
    ]

    results = [m_base]

    for (label, method, hq, lq, h_nl, h_ns, m_nl, m_ns, l_nl, l_ns) in CONFIGS:
        log(f"\n  {label}...")
        port = simulate_v2b_dynK(
            preds, regime_df, cfg,
            cutoff_on=0.9, cutoff_off=0.8,
            min_risk_off_periods=2, min_risk_on_periods=0,
            confidence_method=method,
            high_q=hq, low_q=lq,
            high_nl=h_nl, high_ns=h_ns,
            mid_nl=m_nl, mid_ns=m_ns,
            low_nl=l_nl, low_ns=l_ns,
        )
        m = analyze_config(port, label)
        m["confidence_method"] = method
        m["high_q"] = hq
        m["low_q"] = lq
        m["high_K"] = f"{h_nl}L{h_ns}S"
        m["mid_K"] = f"{m_nl}L{m_ns}S"
        m["low_K"] = f"{l_nl}L{l_ns}S"

        # K distribution
        if hasattr(port, 'attrs') and 'k_distribution' in port.attrs:
            m["k_distribution"] = port.attrs["k_distribution"]
            log(f"    K dist: {port.attrs['k_distribution']}")

        print_result(m)
        results.append(m)

    # ── Results table ──
    log("\n" + "=" * 70)
    log("R117 RESULTS")
    log("=" * 70)

    hdr = (f"  {'Config':<30} {'NetSh':>7} {'GrSh':>7} {'Ret%':>7} "
           f"{'DD%':>7} {'Calmar':>7} {'%flat':>6} {'Cost%':>6}")
    sep = (f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} "
           f"{'-'*6} {'-'*6}")
    log(hdr)
    log(sep)

    for m in results:
        log(f"  {m['label']:<30} {m['net_sharpe']:>7.3f} "
            f"{m['gross_sharpe']:>7.3f} {m['total_ret_pct']:>7.1f} "
            f"{m['max_dd_pct']:>7.1f} {m['calmar']:>7.2f} "
            f"{m['pct_flat']:>5.1f}% {m['total_cost_pct']:>6.2f}")

    # ── Best config ──
    all_configs = [m for m in results[1:] if m["net_sharpe"] > 0]
    if all_configs:
        best = max(all_configs, key=lambda x: x["calmar"])
        log(f"\n  Best dynamic K: {best['label']}")
        for metric in ['net_sharpe', 'calmar', 'max_dd_pct', 'total_ret_pct', 'total_cost_pct']:
            v0 = m_base[metric]
            v1 = best[metric]
            log(f"    {metric}: {v0:.3f} → {v1:.3f}  Δ={v1-v0:+.3f}")

        if best["net_sharpe"] >= m_base["net_sharpe"]:
            log(f"\n  >>> RESULT: WIN — dynamic K improves over fixed 4L/2S")
        elif best["net_sharpe"] >= m_base["net_sharpe"] - 0.1:
            log(f"\n  >>> RESULT: MARGINAL — dynamic K similar to fixed")
        else:
            log(f"\n  >>> RESULT: FAIL — fixed 4L/2S is optimal")
    else:
        log(f"\n  >>> RESULT: FAIL — no valid dynamic K configs")
        best = m_base

    # ── Save ──
    df_res = pd.DataFrame(results)
    df_res.to_csv("results/r117_grid.csv", index=False)
    with open("results/r117_best.json", "w") as f:
        json.dump(best, f, indent=2, default=str)

    log(f"\nSaved: results/r117_grid.csv, r117_best.json")
    log(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
