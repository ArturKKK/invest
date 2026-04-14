#!/usr/bin/env python3
"""
R121 — Realistic Cost Model Audit
===================================

Compare R114b champion backtest under multiple cost scenarios:

  1. ORIGINAL  — current cost model (92% maker Tier1, lenient)
  2. OKX_TAKER — all market orders, OKX Futures Lv1 taker fees
  3. OKX_MAKER — limit-order execution (maker fees, achievable with code change)
  4. PESSIMISTIC — taker + wider spreads + execution delay penalty

Also models:
  - Correct OKX funding (every 8h, ~1.5 settlements per 12h hold)
  - Execution delay penalty (5-min price noise = ~2-5bp random drag)
"""

import time, json, os, warnings
import numpy as np, pandas as pd
from typing import Set, Dict
warnings.filterwarnings("ignore")

from _research_r22_models import SEEDS, log
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r68_continuous_wf import (
    CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, PROD_CFG,
    load_data, train_ensemble, sharpe,
)
from _research_r113_trend_cutoff_reopt import analyze_config, print_result


# ─── Cost models ─────────────────────────────────────────────

TIER1_SYMS = {"BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"}
TIER3_SYMS = {
    "SAND/USDT", "LDO/USDT", "INJ/USDT", "APT/USDT", "ARB/USDT",
    "GALA/USDT", "FTM/USDT", "MATIC/USDT",
}


def cost_original(sym):
    """Current backtest model (92% maker Tier1 assumption)."""
    if sym in TIER1_SYMS:
        return 0.92 * (-0.0001) + 0.08 * 0.0007   # -0.36 bps
    elif sym in TIER3_SYMS:
        return 0.0005 + 0.0002                      # 7 bps
    else:
        return 0.75 * 0.0001 + 0.25 * 0.0007        # 2.5 bps


def cost_okx_taker(sym):
    """OKX Futures Lv1, 100% market orders (current prod behavior)."""
    # taker fee = 0.05% = 5 bps for all tiers
    # spread varies by liquidity tier
    if sym in TIER1_SYMS:
        return 0.0005 + 0.0001   # 5 bps fee + 1 bps spread = 6 bps
    elif sym in TIER3_SYMS:
        return 0.0005 + 0.0005   # 5 bps fee + 5 bps spread = 10 bps
    else:
        return 0.0005 + 0.0002   # 5 bps fee + 2 bps spread = 7 bps


def cost_okx_maker(sym):
    """OKX Futures Lv1, limit orders (post-only, achievable but not yet implemented)."""
    # maker fee = 0.02% = 2 bps for all tiers
    # spread cost ≈ 0 (you ARE the spread with post-only)
    # but ~10% of orders may fallback to taker
    if sym in TIER1_SYMS:
        return 0.90 * 0.0002 + 0.10 * 0.0006   # 2.4 bps blended
    elif sym in TIER3_SYMS:
        return 0.80 * 0.0002 + 0.20 * 0.0010   # 3.6 bps blended
    else:
        return 0.85 * 0.0002 + 0.15 * 0.0007   # 2.75 bps blended


def cost_pessimistic(sym):
    """Worst case: taker + wide spreads + stress slippage."""
    if sym in TIER1_SYMS:
        return 0.0005 + 0.0002 + 0.0001   # 8 bps
    elif sym in TIER3_SYMS:
        return 0.0005 + 0.0010 + 0.0003   # 18 bps
    else:
        return 0.0005 + 0.0004 + 0.0002   # 11 bps


def cost_prod_blended(sym):
    """S6: Models actual prod execution mix (maker-first Tier1, aggressive limit Tier2, market Tier3).

    Tier1 (BTC,ETH,SOL,BNB,XRP): _maker_first_limit → post_only, 90s TTL, 3 retries
      → ~90% maker (2bp) + 10% fallback taker (5bp+1bp spread) = 2.4 bps
    Tier2 (mid-cap): _limit_with_fallback → aggressive limit 0.03% cross
      → ~50% maker-like (2bp fee + 2bp spread) + 50% taker (5bp fee + 2bp spread) = 5.5 bps
    Tier3 (small-cap): plain market
      → 100% taker (5bp fee + 5bp spread) = 10 bps
    """
    if sym in TIER1_SYMS:
        return 0.90 * 0.0002 + 0.10 * 0.0006   # 2.4 bps blended
    elif sym in TIER3_SYMS:
        return 0.0005 + 0.0005                   # 10 bps (pure taker)
    else:
        return 0.50 * 0.0004 + 0.50 * 0.0007     # 5.5 bps blended


COST_MODELS = {
    "original":      (cost_original,      0.00008),   # funding: 0.8bp/12h
    "okx_taker":     (cost_okx_taker,     0.00012),   # funding: 1.2bp/12h (1.5 settlements × 0.8bp)
    "okx_maker":     (cost_okx_maker,     0.00012),   # same funding
    "pessimistic":   (cost_pessimistic,   0.00015),   # funding: 1.5bp/12h (stress)
    "prod_blended":  (cost_prod_blended,  0.00012),   # funding: 1.2bp/12h
}


# ─── R114b champion config ──────────────────────────────────

R114B_CFG = {
    "n_long": 4, "n_short": 2, "rebal_hours": 12,
    "trend_cutoff": 0.9, "dyn_threshold": 0.7,
    "ema_alpha": 0.5, "hysteresis": 3,
}


# ─── Simulation with pluggable cost function ─────────────────

def simulate_r121(merged, regime_df, n_long, n_short, cfg,
                  cutoff_on=0.9, cutoff_off=0.8,
                  min_risk_off_periods=2, min_risk_on_periods=0,
                  cost_fn=cost_original,
                  funding_per_12h=0.00008,
                  exec_delay_penalty=0.0):
    """
    R114b simulate_v2b with pluggable cost function.

    exec_delay_penalty: per-period random penalty to simulate
    5-min execution delay noise (e.g. 0.0003 = ~3bp std both ways).
    Applied as symmetric noise to gross returns.
    """
    rebal_hours   = cfg["rebal_hours"]
    ema_alpha     = cfg.get("ema_alpha", None)
    hysteresis    = cfg.get("hysteresis", 0)
    dyn_threshold = cfg.get("dyn_threshold", 0.5)

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

    rng = np.random.RandomState(42)  # deterministic noise

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
                        close_cost = sum(cost_fn(s) * avg_w
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

        # ── Execution delay noise ──
        if exec_delay_penalty > 0 and total_positions > 0:
            noise = rng.normal(0, exec_delay_penalty)
            gross_ret += noise

        # ── Costs ──
        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            turnover_cost = sum(cost_fn(sym) * avg_weight
                                for sym in new_opened)
            turnover_cost += sum(cost_fn(sym) * avg_weight
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


# ─── Per-window breakdown ─────────────────────────────────────

def per_window_metrics(port, merged):
    """Sharpe and return per walk-forward window."""
    if port.empty or "window" not in merged.columns:
        return {}
    ts_window = merged.drop_duplicates("timestamp")[["timestamp", "window"]]
    ts_window = ts_window.set_index("timestamp")["window"]
    port = port.copy()
    port["window"] = port["timestamp"].map(ts_window)
    results = {}
    for w, wport in port.groupby("window"):
        if len(wport) > 10:
            eq = (1 + wport["net_ret"]).cumprod()
            ret_pct = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
            sh = sharpe(wport["net_ret"])
            results[w] = {"sharpe": round(sh, 3), "ret_pct": round(ret_pct, 1)}
    return results


# ─── Main ────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("=" * 70)
    log("R121 — Realistic Cost Model Audit")
    log("=" * 70)

    os.makedirs("results", exist_ok=True)

    # ── Load data ──
    log("\nLoading data...")
    df, regime_df = load_data()
    base_feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in base_feats if f in MARKET_LEVEL_FEATURES]

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
    results = []

    # ═══════════════════════════════════════════════════════════
    # SCENARIO RUNS
    # ═══════════════════════════════════════════════════════════

    SCENARIOS = [
        # (label, cost_model_key, exec_delay_penalty)
        ("S0_original",         "original",    0.0),
        ("S1_okx_taker",        "okx_taker",   0.0),
        ("S2_okx_taker+delay",  "okx_taker",   0.0003),  # 3bp noise std
        ("S3_okx_maker",        "okx_maker",   0.0),
        ("S4_okx_maker+delay",  "okx_maker",   0.0003),
        ("S5_pessimistic",      "pessimistic", 0.0005),   # 5bp noise std
        ("S6_prod_blended",     "prod_blended", 0.0003),  # actual prod execution mix
    ]

    for label, cm_key, delay in SCENARIOS:
        cost_fn, funding = COST_MODELS[cm_key]
        log(f"\n{'='*70}")
        log(f"SCENARIO: {label}")
        log(f"  cost_model={cm_key}, funding={funding*10000:.1f}bp/12h, "
            f"exec_delay_std={delay*10000:.1f}bp")
        log(f"{'='*70}")

        port = simulate_r121(preds, regime_df, 4, 2, cfg,
                             cutoff_on=0.9, cutoff_off=0.8,
                             min_risk_off_periods=2,
                             min_risk_on_periods=0,
                             cost_fn=cost_fn,
                             funding_per_12h=funding,
                             exec_delay_penalty=delay)
        m = analyze_config(port, label)
        print_result(m)
        pw = per_window_metrics(port, preds)
        for w, wm in pw.items():
            log(f"    {w}: Sharpe={wm['sharpe']:.3f}  Ret={wm['ret_pct']:.1f}%")

        m["cost_model"] = cm_key
        m["exec_delay"] = delay
        m["per_window"] = pw
        results.append(m)

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("R121 — COST MODEL COMPARISON")
    log("=" * 70)

    hdr = (f"  {'Scenario':<25} {'NetSh':>7} {'GrSh':>7} {'Ret%':>7} "
           f"{'DD%':>7} {'Calmar':>7} {'Cost%':>6} {'W1sh':>6} {'W2sh':>6} {'W3sh':>6}")
    log(hdr)
    log(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for m in results:
        pw = m.get("per_window", {})
        w1 = pw.get("W1", {}).get("sharpe", 0)
        w2 = pw.get("W2", {}).get("sharpe", 0)
        w3 = pw.get("W3", {}).get("sharpe", 0)
        log(f"  {m['label']:<25} {m['net_sharpe']:>7.3f} "
            f"{m['gross_sharpe']:>7.3f} {m['total_ret_pct']:>7.1f} "
            f"{m['max_dd_pct']:>7.1f} {m['calmar']:>7.2f} "
            f"{m['total_cost_pct']:>6.2f} {w1:>6.3f} {w2:>6.3f} {w3:>6.3f}")

    # ── Delta from baseline ──
    log(f"\n  {'Scenario':<25} {'ΔSharpe':>8} {'ΔRet%':>8} {'ΔDD%':>8} {'ΔCost%':>8}")
    log(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    base = results[0]
    for m in results[1:]:
        log(f"  {m['label']:<25} "
            f"{m['net_sharpe']-base['net_sharpe']:>+8.3f} "
            f"{m['total_ret_pct']-base['total_ret_pct']:>+8.1f} "
            f"{m['max_dd_pct']-base['max_dd_pct']:>+8.1f} "
            f"{m['total_cost_pct']-base['total_cost_pct']:>+8.2f}")

    # ── "What matters" analysis ──
    log("\n" + "=" * 70)
    log("ANALYSIS: What matters most?")
    log("=" * 70)

    # Cost delta: taker vs original
    if len(results) >= 2:
        d_cost = results[1]["net_sharpe"] - results[0]["net_sharpe"]
        log(f"  Cost model fix (taker vs original):       {d_cost:+.3f} Sharpe")
    if len(results) >= 3:
        d_delay = results[2]["net_sharpe"] - results[1]["net_sharpe"]
        log(f"  Execution delay (3bp noise):              {d_delay:+.3f} Sharpe")
    if len(results) >= 5:
        d_maker = results[3]["net_sharpe"] - results[1]["net_sharpe"]
        log(f"  Maker orders upgrade:                     {d_maker:+.3f} Sharpe")
    if len(results) >= 6:
        d_worst = results[5]["net_sharpe"] - results[0]["net_sharpe"]
        log(f"  Worst case vs original:                   {d_worst:+.3f} Sharpe")

    # Most realistic scenario
    s6 = next((m for m in results if m['label'] == 'S6_prod_blended'), None)
    log(f"\n  MOST REALISTIC scenario (actual prod mix):  S6 (prod_blended + delay)")
    if s6:
        log(f"    Sharpe: {s6['net_sharpe']:.3f}  (was {base['net_sharpe']:.3f})")
        log(f"    Return: {s6['total_ret_pct']:.1f}%  (was {base['total_ret_pct']:.1f}%)")
        log(f"    MaxDD:  {s6['max_dd_pct']:.1f}%  (was {base['max_dd_pct']:.1f}%)")
        log(f"    Calmar: {s6['calmar']:.2f}  (was {base['calmar']:.2f})")

    log(f"\n  S2 lower bound (100% taker):                S2 (okx_taker + delay)")
    if len(results) >= 3:
        real = results[2]
        log(f"    Sharpe: {real['net_sharpe']:.3f}")

    log(f"\n  IF all orders go maker:                     S4 (okx_maker + delay)")
    if len(results) >= 5:
        maker = results[4]
        log(f"    Sharpe: {maker['net_sharpe']:.3f}  (was {base['net_sharpe']:.3f})")
        log(f"    Return: {maker['total_ret_pct']:.1f}%  (was {base['total_ret_pct']:.1f}%)")

    # ── Save ──
    save_results = []
    for m in results:
        m_copy = {k: v for k, v in m.items() if k != "per_window"}
        pw = m.get("per_window", {})
        for w, wm in pw.items():
            m_copy[f"{w}_sharpe"] = wm["sharpe"]
            m_copy[f"{w}_ret"] = wm["ret_pct"]
        save_results.append(m_copy)

    df_res = pd.DataFrame(save_results)
    df_res.to_csv("results/r121_cost_audit.csv", index=False)

    best_realistic = next((m for m in results if m['label'] == 'S6_prod_blended'),
                          results[2] if len(results) >= 3 else results[-1])
    with open("results/r121_realistic.json", "w") as f:
        json.dump({
            "realistic_sharpe": best_realistic["net_sharpe"],
            "realistic_dd": best_realistic["max_dd_pct"],
            "realistic_calmar": best_realistic["calmar"],
            "realistic_ret": best_realistic["total_ret_pct"],
            "original_sharpe": base["net_sharpe"],
            "sharpe_drag": round(best_realistic["net_sharpe"] - base["net_sharpe"], 3),
        }, f, indent=2)

    log(f"\nSaved: results/r121_cost_audit.csv, r121_realistic.json")
    log(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
