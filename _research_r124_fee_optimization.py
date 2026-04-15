#!/usr/bin/env python3
"""
R124 — OKX Fee Optimization Parametric Sweep
==============================================

Goal:   Estimate Sharpe improvement from achievable fee reductions.
Method: Re-simulate R114b champion with different cost scenarios.
Baseline: S6 prod_blended (Sharpe 2.831, known).

Scenarios:
  S6           — Current prod_blended (Regular Lv1: maker 2bp, taker 5bp)
  REF10        — 10% referral cashback on base fees
  REF20        — 20% referral cashback on base fees
  REF30        — 30% referral cashback (max promo)
  MAKER_OPT    — Optimized maker ratio: 95% Tier1, 70% Tier2 (improved execution)
  REF20_MAKER  — 20% referral + optimized maker execution
  VIP1         — VIP1 tier (maker 1.8bp, taker 4.5bp) — requires $100K+

OKX Futures USDT-M Perpetual Fee Structure:
  Regular Lv1: maker 0.020%, taker 0.050%
  VIP 1:       maker 0.018%, taker 0.045%  (assets >= $100K or vol >= $5M)
  VIP 2:       maker 0.016%, taker 0.040%  (assets >= $500K or vol >= $25M)
"""

import os
import sys
import time
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── Project imports ──
sys.path.insert(0, os.path.dirname(__file__))
from _research_r121_realistic_costs import (
    simulate_r121, R114B_CFG, TIER1_SYMS, TIER3_SYMS,
    cost_prod_blended,
)
from _research_r123_news_sentiment import load_data, CHAMPION_FEAT_31, \
    MARKET_LEVEL_FEATURES, CONTINUOUS_WINDOWS, SEEDS, train_ensemble

TIMESTAMP = time.strftime("%Y%m%d_%H%M")


def log(msg=""):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─── Fee Scenario Definitions ───────────────────────────────

# OKX Futures base rates (Regular Lv1)
BASE_MAKER = 0.0002   # 2 bp
BASE_TAKER = 0.0005   # 5 bp

# VIP1 rates
VIP1_MAKER = 0.00018  # 1.8 bp
VIP1_TAKER = 0.00045  # 4.5 bp


def _make_cost_fn(maker, taker, spread_tier1=0.0001, spread_tier2=0.0002,
                  spread_tier3=0.0005, maker_pct_t1=0.90, maker_pct_t2=0.50):
    """
    Build a cost function with given maker/taker base rates and execution mix.

    Returns one-way cost per trade (fractional).
    """
    def cost_fn(sym):
        if sym in TIER1_SYMS:
            # Tier1: maker-first (post_only), fallback to taker
            maker_cost = maker  # maker fee, ~0 spread (you are the spread)
            taker_cost = taker + spread_tier1  # taker fee + spread
            return maker_pct_t1 * maker_cost + (1 - maker_pct_t1) * taker_cost
        elif sym in TIER3_SYMS:
            # Tier3: pure market orders (no change in execution, just fee rates)
            return taker + spread_tier3
        else:
            # Tier2: limit with fallback
            maker_like = maker + spread_tier2  # limit order filled: maker fee + spread
            taker_cost = taker + spread_tier2  # market fallback
            return maker_pct_t2 * maker_like + (1 - maker_pct_t2) * taker_cost
    return cost_fn


# ─── Scenario definitions ───────────────────────────────────

SCENARIOS = {
    "S6_current": {
        "desc": "Current prod_blended (Regular Lv1)",
        "cost_fn": cost_prod_blended,
        "funding": 0.00012,
        "exec_delay": 0.0003,
    },
    "REF10": {
        "desc": "10% referral cashback",
        "cost_fn": _make_cost_fn(
            maker=BASE_MAKER * 0.90,
            taker=BASE_TAKER * 0.90,
        ),
        "funding": 0.00012,
        "exec_delay": 0.0003,
    },
    "REF20": {
        "desc": "20% referral cashback",
        "cost_fn": _make_cost_fn(
            maker=BASE_MAKER * 0.80,
            taker=BASE_TAKER * 0.80,
        ),
        "funding": 0.00012,
        "exec_delay": 0.0003,
    },
    "REF30": {
        "desc": "30% referral cashback (max promo)",
        "cost_fn": _make_cost_fn(
            maker=BASE_MAKER * 0.70,
            taker=BASE_TAKER * 0.70,
        ),
        "funding": 0.00012,
        "exec_delay": 0.0003,
    },
    "MAKER_OPT": {
        "desc": "Optimized execution: 95% maker T1, 70% maker T2",
        "cost_fn": _make_cost_fn(
            maker=BASE_MAKER,
            taker=BASE_TAKER,
            maker_pct_t1=0.95,
            maker_pct_t2=0.70,
        ),
        "funding": 0.00012,
        "exec_delay": 0.0003,
    },
    "REF20_MAKER": {
        "desc": "20% referral + optimized execution",
        "cost_fn": _make_cost_fn(
            maker=BASE_MAKER * 0.80,
            taker=BASE_TAKER * 0.80,
            maker_pct_t1=0.95,
            maker_pct_t2=0.70,
        ),
        "funding": 0.00012,
        "exec_delay": 0.0003,
    },
    "VIP1": {
        "desc": "VIP1 tier (>=$100K assets or >=$5M vol)",
        "cost_fn": _make_cost_fn(
            maker=VIP1_MAKER,
            taker=VIP1_TAKER,
        ),
        "funding": 0.00012,
        "exec_delay": 0.0003,
    },
    "VIP1_MAKER": {
        "desc": "VIP1 + optimized execution",
        "cost_fn": _make_cost_fn(
            maker=VIP1_MAKER,
            taker=VIP1_TAKER,
            maker_pct_t1=0.95,
            maker_pct_t2=0.70,
        ),
        "funding": 0.00012,
        "exec_delay": 0.0003,
    },
}


def compute_metrics(port_df):
    """Compute Sharpe, return, DD, Calmar from portfolio returns."""
    if port_df.empty or "net_ret" not in port_df.columns:
        return {}

    rets = port_df["net_ret"].values
    gross = port_df["gross_ret"].values

    trading = port_df[~port_df["risk_off"]]
    if len(trading) == 0:
        return {}

    trading_rets = trading["net_ret"].values
    ann = np.sqrt(365 * 24 / 12)  # 12h periods

    mean_r = trading_rets.mean()
    std_r = trading_rets.std()
    sharpe = mean_r / std_r * ann if std_r > 0 else 0

    cumret = (1 + rets).cumprod()
    total_ret = cumret.iloc[-1] - 1 if hasattr(cumret, "iloc") else cumret[-1] - 1
    running_max = np.maximum.accumulate(cumret)
    dd = (cumret - running_max) / running_max
    max_dd = dd.min()
    calmar = (total_ret / abs(max_dd)) if max_dd != 0 else 0

    total_cost = port_df["cost"].sum()
    total_gross = gross.sum()
    cost_pct = total_cost / total_gross * 100 if total_gross > 0 else 0

    n_periods = len(port_df)
    n_flat = port_df["risk_off"].sum()

    return {
        "net_sharpe": round(sharpe, 3),
        "gross_ret_pct": round(total_gross * 100, 1),
        "net_ret_pct": round(total_ret * 100, 1),
        "max_dd_pct": round(max_dd * 100, 1),
        "calmar": round(calmar, 2),
        "cost_pct_of_gross": round(cost_pct, 1),
        "total_cost_bps": round(total_cost * 10000, 1),
        "periods": n_periods,
        "flat_pct": round(n_flat / n_periods * 100, 1) if n_periods > 0 else 0,
    }


def main():
    t0 = time.time()
    log("=" * 70)
    log("R124 — OKX Fee Optimization Parametric Sweep")
    log("=" * 70)

    os.makedirs("results", exist_ok=True)

    # ── 1. Load data & train baseline predictions (once) ──
    log("\n[1/3] Loading data & training baseline ensemble...")
    df, regime_df = load_data()
    log(f"  df: {df.shape}")

    base_feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in base_feats if f in MARKET_LEVEL_FEATURES]
    log(f"  Features: {len(base_feats)}/31")

    t1 = time.time()
    preds = train_ensemble(df, base_feats, CONTINUOUS_WINDOWS,
                           seeds=SEEDS, cs_rank_exclude=no_rank)
    log(f"  Trained in {time.time()-t1:.0f}s")
    log(f"  Predictions: {len(preds):,} rows")

    # ── 2. Run all scenarios ──
    log("\n[2/3] Running fee scenarios...")
    results = {}

    for name, scfg in SCENARIOS.items():
        log(f"\n  {'─' * 50}")
        log(f"  {name}: {scfg['desc']}")

        port = simulate_r121(
            preds, regime_df,
            n_long=R114B_CFG["n_long"],
            n_short=R114B_CFG["n_short"],
            cfg=R114B_CFG,
            cost_fn=scfg["cost_fn"],
            funding_per_12h=scfg["funding"],
            exec_delay_penalty=scfg["exec_delay"],
        )

        metrics = compute_metrics(port)
        results[name] = {**metrics, "desc": scfg["desc"]}
        log(f"    Sharpe={metrics.get('net_sharpe', 'N/A')}  "
            f"Ret={metrics.get('net_ret_pct', 'N/A')}%  "
            f"DD={metrics.get('max_dd_pct', 'N/A')}%  "
            f"Cost={metrics.get('cost_pct_of_gross', 'N/A')}%")

    # ── 3. Summary ──
    log("\n[3/3] Summary")
    log("=" * 70)

    base_sharpe = results.get("S6_current", {}).get("net_sharpe", 0)

    header = f"{'Scenario':<20} {'Sharpe':>7} {'ΔSharpe':>8} {'Return%':>8} {'DD%':>7} {'Cost%':>7}"
    log(header)
    log("─" * 70)

    for name, r in sorted(results.items(), key=lambda x: -x[1].get("net_sharpe", 0)):
        delta = r.get("net_sharpe", 0) - base_sharpe
        log(f"  {name:<18} {r.get('net_sharpe', 0):>7.3f} {delta:>+8.3f} "
            f"{r.get('net_ret_pct', 0):>7.1f}% {r.get('max_dd_pct', 0):>6.1f}% "
            f"{r.get('cost_pct_of_gross', 0):>6.1f}%")

    log("\n" + "=" * 70)
    log("Per-scenario cost breakdown (bps per trade, by tier):")
    log("─" * 70)
    for name, scfg in SCENARIOS.items():
        cfn = scfg["cost_fn"]
        t1_cost = cfn("BTC/USDT") * 10000
        t2_cost = cfn("LINK/USDT") * 10000  # example Tier2
        t3_cost = cfn("SAND/USDT") * 10000
        log(f"  {name:<20}  T1={t1_cost:.1f}bp  T2={t2_cost:.1f}bp  T3={t3_cost:.1f}bp")

    # ── Save ──
    log(f"\n  Total time: {time.time()-t0:.0f}s")

    out = {
        "experiment": "R124_fee_optimization",
        "timestamp": TIMESTAMP,
        "baseline": "S6_prod_blended",
        "results": results,
    }
    with open("results/r124_fee_optimization.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    log("  Saved: results/r124_fee_optimization.json")


if __name__ == "__main__":
    main()
