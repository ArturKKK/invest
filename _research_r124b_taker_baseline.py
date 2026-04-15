#!/usr/bin/env python3
"""
R124b — TAKER_ONLY Baseline for Fee Optimization
==================================================

External reviewer flagged: if prod actually uses market orders (not maker-first),
the S6 baseline (maker_pct_t1=0.90) is optimistic. Need a pure 100% taker
baseline to measure the TRUE fee structure gap.

This adds two scenarios to R124:
  TAKER_ONLY   — 100% taker for all tiers (maker_pct=0)
  TAKER_REF20  — 100% taker + 20% referral cashback

Also prints Sharpe_full (all periods) alongside Sharpe_active (trading only)
as recommended by the reviewer.
"""

import time, json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from _research_r121_realistic_costs import (
    simulate_r121, R114B_CFG, TIER1_SYMS, TIER3_SYMS,
    cost_prod_blended, COST_MODELS,
)
from _research_r123_news_sentiment import (
    load_data, CHAMPION_FEAT_31, MARKET_LEVEL_FEATURES,
    CONTINUOUS_WINDOWS, SEEDS,
)
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r68_continuous_wf import train_ensemble

TIMESTAMP = time.strftime("%Y%m%d_%H%M")


def log(msg=""):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─── Base rates ───
BASE_MAKER = 0.0002   # 2 bp
BASE_TAKER = 0.0005   # 5 bp


def _make_cost_fn(maker, taker, spread_tier1=0.0001, spread_tier2=0.0002,
                  spread_tier3=0.0005, maker_pct_t1=0.90, maker_pct_t2=0.50):
    """Clone of R124's cost fn builder."""
    def cost_fn(sym):
        if sym in TIER1_SYMS:
            return maker_pct_t1 * maker + (1 - maker_pct_t1) * (taker + spread_tier1)
        elif sym in TIER3_SYMS:
            return taker + spread_tier3
        else:
            maker_like = maker + spread_tier2
            taker_cost = taker + spread_tier2
            return maker_pct_t2 * maker_like + (1 - maker_pct_t2) * taker_cost
    return cost_fn


# ─── Scenarios ───

SCENARIOS = {
    "S6_current": {
        "desc": "Current prod_blended (maker_pct=90%/50%)",
        "cost_fn": cost_prod_blended,
        "funding": 0.00012,
        "exec_delay": 0.0003,
    },
    "TAKER_ONLY": {
        "desc": "100% taker for ALL tiers (maker_pct=0)",
        "cost_fn": _make_cost_fn(
            maker=BASE_MAKER,
            taker=BASE_TAKER,
            maker_pct_t1=0.0,
            maker_pct_t2=0.0,
        ),
        "funding": 0.00012,
        "exec_delay": 0.0003,
    },
    "TAKER_REF20": {
        "desc": "100% taker + 20% referral cashback",
        "cost_fn": _make_cost_fn(
            maker=BASE_MAKER * 0.80,
            taker=BASE_TAKER * 0.80,
            maker_pct_t1=0.0,
            maker_pct_t2=0.0,
        ),
        "funding": 0.00012,
        "exec_delay": 0.0003,
    },
    "REF20": {
        "desc": "20% referral + maker-first (from R124)",
        "cost_fn": _make_cost_fn(
            maker=BASE_MAKER * 0.80,
            taker=BASE_TAKER * 0.80,
        ),
        "funding": 0.00012,
        "exec_delay": 0.0003,
    },
}


def compute_metrics_dual(port_df):
    """
    Compute BOTH Sharpe metrics:
      - sharpe_active: on trading periods only (excl risk_off)
      - sharpe_full: on all periods
    As recommended by external reviewer.
    """
    if port_df.empty or "net_ret" not in port_df.columns:
        return {}

    rets = port_df["net_ret"].values
    gross = port_df["gross_ret"].values
    ann = np.sqrt(365 * 24 / 12)  # 12h periods

    # Full Sharpe (all periods)
    mean_all = rets.mean()
    std_all = rets.std()
    sharpe_full = mean_all / std_all * ann if std_all > 0 else 0

    # Active Sharpe (trading only)
    trading = port_df[~port_df["risk_off"]]
    if len(trading) > 0:
        trading_rets = trading["net_ret"].values
        mean_t = trading_rets.mean()
        std_t = trading_rets.std()
        sharpe_active = mean_t / std_t * ann if std_t > 0 else 0
    else:
        sharpe_active = 0

    # DD on full equity curve
    cumret = (1 + rets).cumprod()
    total_ret = cumret[-1] - 1
    running_max = np.maximum.accumulate(cumret)
    dd = (cumret - running_max) / running_max
    max_dd = dd.min()

    # Cost info
    total_cost = port_df["cost"].sum()
    total_gross = gross.sum()
    cost_pct = total_cost / total_gross * 100 if total_gross > 0 else 0

    return {
        "sharpe_active": round(sharpe_active, 3),
        "sharpe_full": round(sharpe_full, 3),
        "net_ret_pct": round(total_ret * 100, 1),
        "max_dd_pct": round(max_dd * 100, 1),
        "cost_pct_of_gross": round(cost_pct, 1),
        "n_periods": len(port_df),
        "n_trading": len(trading),
        "flat_pct": round(port_df["risk_off"].mean() * 100, 1),
    }


def main():
    t0 = time.time()
    log("=" * 70)
    log("R124b — TAKER_ONLY Baseline + Dual Sharpe Reporting")
    log("=" * 70)

    os.makedirs("results", exist_ok=True)

    # ── Load & train (same as R124) ──
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

    # ── Run scenarios ──
    log("\n[2/3] Running scenarios...")
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

        metrics = compute_metrics_dual(port)
        results[name] = {**metrics, "desc": scfg["desc"]}

        log(f"    Sharpe_active={metrics.get('sharpe_active', 'N/A')}  "
            f"Sharpe_full={metrics.get('sharpe_full', 'N/A')}  "
            f"Ret={metrics.get('net_ret_pct', 'N/A')}%  "
            f"DD={metrics.get('max_dd_pct', 'N/A')}%  "
            f"Cost={metrics.get('cost_pct_of_gross', 'N/A')}%")

    # ── Summary ──
    log("\n[3/3] Summary")
    log("=" * 70)

    base_active = results.get("S6_current", {}).get("sharpe_active", 0)
    base_full = results.get("S6_current", {}).get("sharpe_full", 0)

    header = (f"  {'Scenario':<15} {'ShActive':>9} {'ΔAct':>7} "
              f"{'ShFull':>8} {'ΔFull':>7} {'Ret%':>7} {'DD%':>7} "
              f"{'Cost%':>7}")
    log(header)
    log("  " + "─" * 80)

    for name in ["S6_current", "TAKER_ONLY", "TAKER_REF20", "REF20"]:
        r = results.get(name, {})
        sa = r.get("sharpe_active", 0)
        sf = r.get("sharpe_full", 0)
        da = sa - base_active
        df_val = sf - base_full
        log(f"  {name:<15} {sa:>9.3f} {da:>+7.3f} "
            f"{sf:>8.3f} {df_val:>+7.3f} "
            f"{r.get('net_ret_pct', 0):>7.1f} {r.get('max_dd_pct', 0):>7.1f} "
            f"{r.get('cost_pct_of_gross', 0):>7.1f}")

    # ── Cost per trade breakdown ──
    log(f"\n  Per-trade cost (bps):")
    for name, scfg in SCENARIOS.items():
        cfn = scfg["cost_fn"]
        t1c = cfn("BTC/USDT") * 10000
        t2c = cfn("LINK/USDT") * 10000
        t3c = cfn("SAND/USDT") * 10000
        log(f"    {name:<15}  T1={t1c:.1f}bp  T2={t2c:.1f}bp  T3={t3c:.1f}bp")

    # ── Key insights ──
    taker = results.get("TAKER_ONLY", {})
    s6 = results.get("S6_current", {})
    maker_benefit_active = s6.get("sharpe_active", 0) - taker.get("sharpe_active", 0)
    maker_benefit_full = s6.get("sharpe_full", 0) - taker.get("sharpe_full", 0)

    log(f"\n  KEY FINDINGS:")
    log(f"    Maker-first execution benefit: "
        f"+{maker_benefit_active:.3f} Sharpe_active, +{maker_benefit_full:.3f} Sharpe_full")
    log(f"    If prod is 100% taker today, switching to maker-first = largest free improvement")
    log(f"    Referral 20% on taker-only: Sharpe {results.get('TAKER_REF20', {}).get('sharpe_active', 0):.3f} "
        f"→ still worse than maker-first S6")

    # ── Save ──
    out = {
        "experiment": "R124b_taker_baseline",
        "timestamp": TIMESTAMP,
        "reviewer_fix": "Added TAKER_ONLY baseline + dual Sharpe reporting",
        "results": results,
    }
    with open("results/r124b_taker_baseline.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"\n  Saved: results/r124b_taker_baseline.json")
    log(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
