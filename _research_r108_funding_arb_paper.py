#!/usr/bin/env python3
"""
R108 — Funding Rate Arbitrage: Paper Trading Monitor (dry run)

Runs once and:
1. Fetches current FR for all symbols (from existing downloaded data)
2. Identifies current opportunities (FR > threshold)
3. Simulates paper positions opened at the last few FR windows
4. Compares current market conditions to R106 backtest assumptions

This is a "snapshot paper trade" — run it every 8h or daily to track.

Outputs:
  results/r108_paper_snapshot.json  — current opportunities + paper P&L
  results/r108_paper_log.jsonl      — append-only log of every run
"""

import json
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT))

SPOT_FEE = 0.0005
PERP_FEE = 0.0003
ROUND_TRIP = 2 * (SPOT_FEE + PERP_FEE)

# R106 best config
BEST_CONFIG = {
    "entry_threshold": 0.0008,
    "exit_threshold": 0.00005,
    "max_hold_periods": 24,
    "max_positions": 3,
}

# Also test the more active config
ALT_CONFIG = {
    "entry_threshold": 0.0005,
    "exit_threshold": 0.0001,
    "max_hold_periods": 24,
    "max_positions": 3,
}


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_latest_funding() -> pd.DataFrame:
    """Load funding data, take last 30 days."""
    path = DATA_DIR / "sentiment" / "binance_funding_rates.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.rename(columns={"funding_rate_binance": "fr"})
    df = df.sort_values(["symbol", "timestamp"])
    cutoff = df.timestamp.max() - pd.Timedelta(days=30)
    recent = df[df.timestamp >= cutoff].copy()
    log(f"  Funding data: last 30d, {len(recent):,} rows, up to {df.timestamp.max()}")
    return recent


def load_latest_premium() -> pd.DataFrame:
    """Load premium index, take last 30 days."""
    path = DATA_DIR / "sentiment" / "binance_premium_index.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol", "timestamp"])
    cutoff = df.timestamp.max() - pd.Timedelta(days=30)
    recent = df[df.timestamp >= cutoff].copy()
    log(f"  Premium data: last 30d, {len(recent):,} rows, up to {df.timestamp.max()}")
    return recent


def current_opportunities(fr_df: pd.DataFrame, config: dict) -> list:
    """Find current FR opportunities at latest timestamp."""
    latest_ts = fr_df.timestamp.max()
    latest = fr_df[fr_df.timestamp == latest_ts].copy()

    # Also get previous FR for trend
    prev_ts = fr_df[fr_df.timestamp < latest_ts].timestamp.max()
    prev = fr_df[fr_df.timestamp == prev_ts].set_index("symbol")["fr"]

    opps = []
    thr = config["entry_threshold"]
    for _, row in latest.iterrows():
        if row.fr > thr:
            prev_fr = prev.get(row.symbol, 0)
            opps.append({
                "symbol": row.symbol,
                "fr": round(row.fr * 100, 4),
                "fr_prev": round(prev_fr * 100, 4),
                "fr_trend": "↑" if row.fr > prev_fr else "↓",
                "ann_carry_pct": round(row.fr * 3 * 365 * 100, 1),
            })
    opps.sort(key=lambda x: x["fr"], reverse=True)
    return opps


def paper_simulation(fr_df: pd.DataFrame, premium_df: pd.DataFrame,
                     config: dict, lookback_days: int = 14) -> dict:
    """
    Simulate paper trading over last N days with the given config.
    This is a mini-backtest on recent data to compare with R106 assumptions.
    """
    cutoff = fr_df.timestamp.max() - pd.Timedelta(days=lookback_days)
    fr_recent = fr_df[fr_df.timestamp >= cutoff]
    prem_recent = premium_df[premium_df.timestamp >= cutoff]

    # Merge
    merged = fr_recent.merge(prem_recent, on=["timestamp", "symbol"], how="inner")
    merged = merged.sort_values(["symbol", "timestamp"])
    merged["basis_change"] = merged.groupby("symbol")["premium_index"].diff()

    # Simple simulation
    fr_lookup = dict(zip(zip(merged.timestamp, merged.symbol), merged.fr))
    basis_lookup = dict(zip(zip(merged.timestamp, merged.symbol),
                            merged.basis_change.fillna(0)))
    all_ts = sorted(merged.timestamp.unique())
    symbols = sorted(merged.symbol.unique())

    entry_thr = config["entry_threshold"]
    exit_thr = config["exit_threshold"]
    max_hold = config["max_hold_periods"]
    max_pos = config["max_positions"]

    positions = []
    equity = 100.0
    total_funding = 0.0
    total_basis = 0.0
    total_costs = 0.0
    n_entries = 0
    closed = []

    for ts in all_ts:
        # Collect for open positions
        for pos in positions:
            fr = fr_lookup.get((ts, pos["symbol"]), 0.0)
            bc = basis_lookup.get((ts, pos["symbol"]), 0.0)
            pos["funding"] += pos["size"] * fr
            pos["basis_pnl"] -= pos["size"] * bc
            pos["periods"] += 1
            equity += pos["size"] * fr - pos["size"] * bc
            total_funding += pos["size"] * fr
            total_basis -= pos["size"] * bc

        # Exits
        to_close = []
        for i, pos in enumerate(positions):
            fr = fr_lookup.get((ts, pos["symbol"]), 0.0)
            if fr < exit_thr or pos["periods"] >= max_hold:
                cost = pos["size"] * ROUND_TRIP
                equity -= cost
                total_costs += cost
                closed.append({
                    "symbol": pos["symbol"],
                    "periods": pos["periods"],
                    "funding": round(pos["funding"], 4),
                    "basis": round(pos["basis_pnl"], 4),
                    "net": round(pos["funding"] + pos["basis_pnl"] - pos["entry_cost"] - cost, 4),
                })
                to_close.append(i)
        for i in sorted(to_close, reverse=True):
            positions.pop(i)

        # Entries
        if len(positions) < max_pos:
            open_syms = {p["symbol"] for p in positions}
            cands = [(s, fr_lookup.get((ts, s), 0.0)) for s in symbols if s not in open_syms]
            cands = [(s, f) for s, f in cands if f > entry_thr]
            cands.sort(key=lambda x: x[1], reverse=True)
            for sym, fr in cands[:max_pos - len(positions)]:
                sz = 100.0 / max_pos / 2
                cost = sz * ROUND_TRIP
                equity -= cost
                total_costs += cost
                positions.append({
                    "symbol": sym, "size": sz, "periods": 0,
                    "funding": 0.0, "basis_pnl": 0.0, "entry_cost": cost,
                })
                n_entries += 1

    ret = equity / 100.0 - 1
    n_periods = len(all_ts)
    wins = sum(1 for c in closed if c["net"] > 0)

    return {
        "lookback_days": lookback_days,
        "n_periods": n_periods,
        "n_entries": n_entries,
        "n_closed": len(closed),
        "n_open": len(positions),
        "return_pct": round(ret * 100, 3),
        "total_funding_usd": round(total_funding, 4),
        "total_basis_usd": round(total_basis, 4),
        "total_costs_usd": round(total_costs, 4),
        "win_rate": round(wins / max(len(closed), 1), 3),
        "closed_trades": closed[-10:],  # last 10
        "open_positions": [{
            "symbol": p["symbol"], "periods": p["periods"],
            "funding": round(p["funding"], 4),
            "basis": round(p["basis_pnl"], 4),
        } for p in positions],
    }


def compare_to_backtest(paper_result: dict) -> dict:
    """Compare paper results to R106 backtest expectations."""
    # R106 best: 14.7% over ~3774 periods = ~0.0039% per period
    r106_per_period_ret = 14.73 / 3774  # % per period
    paper_per_period = paper_result["return_pct"] / max(paper_result["n_periods"], 1)

    return {
        "r106_per_period_pct": round(r106_per_period_ret, 4),
        "paper_per_period_pct": round(paper_per_period, 4),
        "ratio": round(paper_per_period / (r106_per_period_ret + 1e-10), 2),
        "deviation_pct": round((paper_per_period / (r106_per_period_ret + 1e-10) - 1) * 100, 1),
    }


def main():
    log("=" * 70)
    log("R108 — Paper Trading Monitor (Snapshot)")
    log("=" * 70)

    log("\n[1/4] Loading recent data...")
    fr_df = load_latest_funding()
    prem_df = load_latest_premium()

    # Current opportunities
    log("\n[2/4] Current opportunities...")
    for name, config in [("R106_best", BEST_CONFIG), ("R106_alt", ALT_CONFIG)]:
        opps = current_opportunities(fr_df, config)
        log(f"  {name} (entry>{config['entry_threshold']*100:.2f}%): {len(opps)} opportunities")
        for o in opps[:5]:
            log(f"    {o['symbol']:>12s}  FR={o['fr']:.4f}%  {o['fr_trend']}  "
                f"ann_carry={o['ann_carry_pct']:.1f}%")

    # Paper simulation
    log("\n[3/4] Paper simulation (last 14 days)...")
    for name, config in [("R106_best", BEST_CONFIG), ("R106_alt", ALT_CONFIG)]:
        paper = paper_simulation(fr_df, prem_df, config, lookback_days=14)
        log(f"\n  {name}:")
        log(f"    Periods={paper['n_periods']}  Entries={paper['n_entries']}  "
            f"Closed={paper['n_closed']}  Open={paper['n_open']}")
        log(f"    Return={paper['return_pct']:+.3f}%  "
            f"Funding=${paper['total_funding_usd']:.4f}  "
            f"Basis=${paper['total_basis_usd']:.4f}  "
            f"Costs=${paper['total_costs_usd']:.4f}")
        if paper["n_closed"] > 0:
            log(f"    Win rate={paper['win_rate']:.1%}")

        # Also 30-day
        paper30 = paper_simulation(fr_df, prem_df, config, lookback_days=30)
        log(f"    30-day: Return={paper30['return_pct']:+.3f}%  "
            f"Entries={paper30['n_entries']}  Closed={paper30['n_closed']}")

    # Compare to backtest
    log("\n[4/4] Comparison to R106 backtest expectations...")
    paper_best = paper_simulation(fr_df, prem_df, BEST_CONFIG, lookback_days=30)
    comparison = compare_to_backtest(paper_best)
    log(f"  R106 expected: {comparison['r106_per_period_pct']:.4f}% per period")
    log(f"  Paper actual:  {comparison['paper_per_period_pct']:.4f}% per period")
    log(f"  Ratio: {comparison['ratio']:.2f}x  ({comparison['deviation_pct']:+.1f}%)")

    # Save snapshot
    snapshot = {
        "run_time": datetime.utcnow().isoformat(),
        "data_up_to": str(fr_df.timestamp.max()),
        "opportunities_best": current_opportunities(fr_df, BEST_CONFIG),
        "opportunities_alt": current_opportunities(fr_df, ALT_CONFIG),
        "paper_14d_best": paper_simulation(fr_df, prem_df, BEST_CONFIG, lookback_days=14),
        "paper_30d_best": paper_best,
        "comparison": comparison,
    }
    with open(RESULTS_DIR / "r108_paper_snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    # Append to log
    log_entry = {
        "time": datetime.utcnow().isoformat(),
        "n_opps_best": len(snapshot["opportunities_best"]),
        "n_opps_alt": len(snapshot["opportunities_alt"]),
        "paper_14d_ret": snapshot["paper_14d_best"]["return_pct"],
        "paper_30d_ret": paper_best["return_pct"],
        "comparison_ratio": comparison["ratio"],
    }
    with open(RESULTS_DIR / "r108_paper_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # Verdict
    log("\n" + "=" * 70)
    dev = abs(comparison["deviation_pct"])
    if dev <= 30:
        verdict = "PASS"
        log(f"  R108 VERDICT: PASS — paper within {dev:.0f}% of backtest (need ≤30%)")
    elif dev <= 50:
        verdict = "MARGINAL"
        log(f"  R108 VERDICT: MARGINAL — paper {dev:.0f}% deviation (need ≤30%, kill >50%)")
    else:
        verdict = "FAIL"
        log(f"  R108 VERDICT: FAIL — paper {dev:.0f}% deviation from backtest (>50%)")

    log(f"\n⚠️  NOTE: R108 uses historical data as proxy for paper trading.")
    log(f"    For real paper trading, run this script daily after data refresh.")
    log(f"    After 2 weeks of daily runs, check r108_paper_log.jsonl for stability.")

    snapshot["r108_verdict"] = verdict
    with open(RESULTS_DIR / "r108_paper_snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    log("\nDone.")
    return snapshot


if __name__ == "__main__":
    main()
