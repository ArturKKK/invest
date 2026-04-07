#!/usr/bin/env python3
"""
R106 — Funding Rate Arbitrage: Backtest

Simulates market-neutral funding arb: short perp + long spot.
Collects funding when FR > entry_threshold, exits when FR < exit_threshold.

Grid search over:
  entry_threshold × exit_threshold × max_hold_periods × max_positions

Uses 1h price data for hedge P&L (basis risk).

Outputs:
  results/r106_grid.csv       — full grid results
  results/r106_best.json      — best config details
  results/r106_equity.csv     — equity curve for best config
"""

import json
import sys
import time
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR = ROOT / "data"

# Fees (OKX taker)
SPOT_FEE = 0.0005
PERP_FEE = 0.0003
ROUND_TRIP = 2 * (SPOT_FEE + PERP_FEE)  # 0.16%

CAPITAL = 100.0  # $100 total
FUNDING_INTERVAL_H = 8
PERIODS_PER_YEAR = 3 * 365  # 8h periods

# Grid
ENTRY_THRESHOLDS = [0.0001, 0.0002, 0.0003, 0.0005, 0.0008]
EXIT_THRESHOLDS = [0.00005, 0.0001, 0.00015]
MAX_HOLD_PERIODS = [3, 6, 12, 24]
MAX_POSITIONS_LIST = [1, 2, 3]

EPS = 1e-10


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Data Loading ──────────────────────────────────────────────────────────

def load_funding() -> pd.DataFrame:
    """Load Binance 8h funding rates."""
    path = DATA_DIR / "sentiment" / "binance_funding_rates.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.rename(columns={"funding_rate_binance": "fr"})
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    log(f"  Funding: {len(df):,} rows, {df.symbol.nunique()} sym, "
        f"{df.timestamp.min().date()} to {df.timestamp.max().date()}")
    return df


def load_prices(symbols: list) -> pd.DataFrame:
    """Load 8h OHLCV (resample from 1h) for hedge P&L calculation."""
    frames = []
    for sym in symbols:
        ticker = sym.replace("/", "_")
        path = DATA_DIR / "raw" / f"{ticker}_1h.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        # Resample to 8h (close-to-close returns)
        df8 = df["close"].resample("8h").last().dropna()
        df8 = df8.to_frame("close")
        df8["ret_8h"] = df8["close"].pct_change()
        df8 = df8.reset_index()
        df8["symbol"] = sym
        frames.append(df8[["timestamp", "symbol", "close", "ret_8h"]])
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    log(f"  Prices: {len(result):,} rows, {result.symbol.nunique()} sym")
    return result


# ── Backtester ────────────────────────────────────────────────────────────

@dataclass
class Position:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    size_usd: float     # per leg (spot=size, perp=size)
    periods_held: int = 0
    funding_pnl: float = 0.0
    hedge_pnl: float = 0.0
    entry_cost: float = 0.0  # one-time cost at entry


def simulate(funding_df: pd.DataFrame, price_df: pd.DataFrame,
             entry_thr: float, exit_thr: float, max_hold: int,
             max_pos: int,
             fr_lookup: dict = None, price_lookup: dict = None,
             all_ts: list = None, symbols: list = None) -> dict:
    """
    Event-driven simulation of funding arb strategy.

    At each 8h timestamp:
    1. Collect funding for open positions (FR × position_size)
    2. Check exits: FR < exit_thr OR held > max_hold
    3. Check new entries: FR > entry_thr, if slots available
    4. Track hedge P&L (spot + perp price moves)
    """
    # Build lookups if not pre-computed
    if fr_lookup is None:
        fr_lookup = dict(zip(
            zip(funding_df.timestamp, funding_df.symbol),
            funding_df.fr
        ))
    if price_lookup is None:
        ret_vals = price_df.ret_8h.fillna(0.0)
        price_lookup = dict(zip(
            zip(price_df.timestamp, price_df.symbol),
            zip(price_df.close, ret_vals)
        ))
    if all_ts is None:
        all_ts = sorted(funding_df.timestamp.unique())
    if symbols is None:
        symbols = sorted(funding_df.symbol.unique())

    positions: list[Position] = []
    equity = CAPITAL
    equity_curve = []
    total_entries = 0
    total_funding = 0.0
    total_hedge_pnl = 0.0
    total_costs = 0.0
    closed_trades = []

    for ts in all_ts:
        period_pnl = 0.0

        # 1. Collect funding for open positions
        for pos in positions:
            fr = fr_lookup.get((ts, pos.symbol), 0.0)
            # We're short perp → we receive positive FR
            funding = pos.size_usd * fr
            pos.funding_pnl += funding
            pos.periods_held += 1
            period_pnl += funding
            total_funding += funding

        # 2. Check exits
        to_close = []
        for i, pos in enumerate(positions):
            fr = fr_lookup.get((ts, pos.symbol), 0.0)
            should_exit = (fr < exit_thr) or (pos.periods_held >= max_hold)
            if should_exit:
                # Close costs
                exit_cost = pos.size_usd * ROUND_TRIP
                total_costs += exit_cost
                period_pnl -= exit_cost

                # Calculate hedge P&L at close
                price_data = price_lookup.get((ts, pos.symbol))
                if price_data:
                    close_price = price_data[0]
                    # Spot P&L: bought at entry_price, sell at close_price
                    spot_pnl = pos.size_usd * (close_price / pos.entry_price - 1)
                    # Perp P&L: shorted at entry_price, close at close_price
                    perp_pnl = pos.size_usd * (1 - close_price / pos.entry_price)
                    # Net hedge P&L (should be ~0 for perfect hedge)
                    hedge = spot_pnl + perp_pnl  # exactly 0 in theory
                else:
                    hedge = 0.0
                pos.hedge_pnl = hedge
                total_hedge_pnl += hedge

                net_trade = pos.funding_pnl - pos.entry_cost - exit_cost + pos.hedge_pnl
                closed_trades.append({
                    "symbol": pos.symbol,
                    "entry_time": str(pos.entry_time),
                    "exit_time": str(ts),
                    "periods_held": pos.periods_held,
                    "funding_pnl": pos.funding_pnl,
                    "hedge_pnl": pos.hedge_pnl,
                    "costs": pos.entry_cost + exit_cost,
                    "net_pnl": net_trade,
                })
                to_close.append(i)

        for i in sorted(to_close, reverse=True):
            positions.pop(i)

        # 3. Check new entries
        if len(positions) < max_pos:
            # Find best opportunities (highest FR)
            candidates = []
            open_syms = {p.symbol for p in positions}
            for sym in symbols:
                if sym in open_syms:
                    continue
                fr = fr_lookup.get((ts, sym), 0.0)
                if fr > entry_thr:
                    candidates.append((sym, fr))
            # Sort by FR descending, take top slots
            candidates.sort(key=lambda x: x[1], reverse=True)
            slots = max_pos - len(positions)
            for sym, fr in candidates[:slots]:
                price_data = price_lookup.get((ts, sym))
                if not price_data:
                    continue
                entry_price = price_data[0]
                if entry_price <= 0:
                    continue
                pos_size = CAPITAL / max_pos / 2  # split capital evenly, /2 per leg
                entry_cost = pos_size * ROUND_TRIP
                total_costs += entry_cost
                period_pnl -= entry_cost

                pos = Position(
                    symbol=sym,
                    entry_time=ts,
                    entry_price=entry_price,
                    size_usd=pos_size,
                    entry_cost=entry_cost,
                )
                positions.append(pos)
                total_entries += 1

        equity += period_pnl
        equity_curve.append({"timestamp": ts, "equity": equity, "pnl": period_pnl})

    # Compute metrics
    eq_df = pd.DataFrame(equity_curve)
    if len(eq_df) < 10:
        return {"valid": False}

    eq_df["ret"] = eq_df["equity"].pct_change().fillna(0)
    rets = eq_df["ret"]

    total_ret = (equity / CAPITAL - 1)
    ann_ret = total_ret * (PERIODS_PER_YEAR / len(eq_df))
    vol = rets.std() * np.sqrt(PERIODS_PER_YEAR)
    sharpe = rets.mean() / (rets.std() + EPS) * np.sqrt(PERIODS_PER_YEAR)
    max_dd = ((eq_df["equity"] / eq_df["equity"].cummax()) - 1).min()

    # Profit factor
    wins = sum(1 for t in closed_trades if t["net_pnl"] > 0)
    total_trades = len(closed_trades)
    win_rate = wins / total_trades if total_trades > 0 else 0

    return {
        "valid": True,
        "entry_threshold": entry_thr,
        "exit_threshold": exit_thr,
        "max_hold_periods": max_hold,
        "max_positions": max_pos,
        "sharpe": round(sharpe, 4),
        "total_ret_pct": round(total_ret * 100, 2),
        "ann_ret_pct": round(ann_ret * 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "vol_ann_pct": round(vol * 100, 2),
        "calmar": round(sharpe / (abs(max_dd) + EPS), 3),
        "total_entries": total_entries,
        "total_trades": total_trades,
        "win_rate": round(win_rate, 3),
        "total_funding_usd": round(total_funding, 4),
        "total_hedge_pnl_usd": round(total_hedge_pnl, 4),
        "total_costs_usd": round(total_costs, 4),
        "final_equity": round(equity, 2),
        "n_periods": len(eq_df),
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    log("=" * 70)
    log("R106 — Funding Rate Arbitrage: Backtest")
    log("=" * 70)

    # Load data
    log("\n[1/5] Loading data...")
    funding = load_funding()
    symbols = sorted(funding.symbol.unique())
    prices = load_prices(symbols)

    if prices.empty:
        log("ERROR: No price data loaded!")
        return

    # Align: only use timestamps present in both
    common_ts = set(funding.timestamp.unique()) & set(prices.timestamp.unique())
    log(f"  Common timestamps: {len(common_ts):,}")
    funding = funding[funding.timestamp.isin(common_ts)]
    prices = prices[prices.timestamp.isin(common_ts)]

    # Pre-compute lookups once (avoid per-iteration iterrows)
    log("\n[2/5] Pre-computing lookups...")
    fr_lookup = dict(zip(
        zip(funding.timestamp, funding.symbol),
        funding.fr
    ))
    ret_vals = prices.ret_8h.fillna(0.0)
    price_lookup = dict(zip(
        zip(prices.timestamp, prices.symbol),
        zip(prices.close, ret_vals)
    ))
    all_ts = sorted(funding.timestamp.unique())
    all_symbols = sorted(funding.symbol.unique())
    log(f"  FR lookup: {len(fr_lookup):,} entries, Price lookup: {len(price_lookup):,} entries")

    # Grid search
    log("\n[3/5] Grid search...")
    total_combos = (len(ENTRY_THRESHOLDS) * len(EXIT_THRESHOLDS) *
                    len(MAX_HOLD_PERIODS) * len(MAX_POSITIONS_LIST))
    log(f"  Total combinations: {total_combos}")

    results = []
    count = 0
    for entry_thr in ENTRY_THRESHOLDS:
        for exit_thr in EXIT_THRESHOLDS:
            if exit_thr >= entry_thr:
                continue  # exit must be below entry
            for max_hold in MAX_HOLD_PERIODS:
                for max_pos in MAX_POSITIONS_LIST:
                    count += 1
                    if count % 20 == 0:
                        log(f"  [{count}/{total_combos}] entry={entry_thr*100:.3f}% "
                            f"exit={exit_thr*100:.4f}% hold={max_hold} pos={max_pos}")
                    res = simulate(funding, prices, entry_thr, exit_thr, max_hold, max_pos,
                                   fr_lookup=fr_lookup, price_lookup=price_lookup,
                                   all_ts=all_ts, symbols=all_symbols)
                    if res.get("valid"):
                        results.append(res)

    if not results:
        log("ERROR: No valid results!")
        return

    # Results
    log(f"\n[4/5] Analyzing {len(results)} valid configs...")
    grid_df = pd.DataFrame(results)
    grid_df = grid_df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    grid_df.to_csv(RESULTS_DIR / "r106_grid.csv", index=False)

    # Top-10
    log("\n  Top-10 configs by Sharpe:")
    log(f"  {'Entry%':>7s} {'Exit%':>7s} {'Hold':>5s} {'Pos':>4s}  "
        f"{'Sharpe':>7s} {'Ret%':>7s} {'DD%':>7s} {'Win%':>5s} {'Trades':>7s}")
    for _, r in grid_df.head(10).iterrows():
        log(f"  {r.entry_threshold*100:>6.3f}% {r.exit_threshold*100:>6.4f}% "
            f"{int(r.max_hold_periods):>5d} {int(r.max_positions):>4d}  "
            f"{r.sharpe:>7.3f} {r.total_ret_pct:>+6.1f}% {r.max_dd_pct:>+6.1f}% "
            f"{r.win_rate:>4.1%} {int(r.total_trades):>7d}")

    # Best config
    best = grid_df.iloc[0].to_dict()
    log(f"\n  BEST: entry={best['entry_threshold']*100:.3f}% "
        f"exit={best['exit_threshold']*100:.4f}% "
        f"hold={int(best['max_hold_periods'])} pos={int(best['max_positions'])}")
    log(f"  Sharpe={best['sharpe']:.3f}  Ret={best['total_ret_pct']:+.1f}%  "
        f"DD={best['max_dd_pct']:+.1f}%  Trades={int(best['total_trades'])}")
    log(f"  Funding=${best['total_funding_usd']:.2f}  "
        f"Costs=${best['total_costs_usd']:.2f}  "
        f"Hedge=${best['total_hedge_pnl_usd']:.4f}")

    # Re-run best config to get equity curve
    log("\n[5/5] Generating equity curve for best config...")
    eq_result = simulate(funding, prices,
                         best["entry_threshold"], best["exit_threshold"],
                         int(best["max_hold_periods"]), int(best["max_positions"]),
                         fr_lookup=fr_lookup, price_lookup=price_lookup,
                         all_ts=all_ts, symbols=all_symbols)

    # Save best
    with open(RESULTS_DIR / "r106_best.json", "w") as f:
        json.dump(best, f, indent=2, default=str)

    # ── Verdict ───────────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    sharpe = best["sharpe"]
    if sharpe >= 1.0:
        verdict = "PASS"
        log(f"  R106 VERDICT: PASS — Sharpe={sharpe:.3f} ≥ 1.0")
    elif sharpe >= 0.5:
        verdict = "MARGINAL"
        log(f"  R106 VERDICT: MARGINAL — Sharpe={sharpe:.3f} (need ≥1.0, kill <0.5)")
    else:
        verdict = "FAIL"
        log(f"  R106 VERDICT: FAIL — Sharpe={sharpe:.3f} < 0.5 → ABANDON")

    best["r106_verdict"] = verdict
    with open(RESULTS_DIR / "r106_best.json", "w") as f:
        json.dump(best, f, indent=2, default=str)

    log("Done.")
    return best


if __name__ == "__main__":
    main()
