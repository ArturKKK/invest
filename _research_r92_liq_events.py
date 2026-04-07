#!/usr/bin/env python3
"""
R92 — Liquidation Event Mean-Reversion Strategy.

When liquidation spikes happen, the market overshoots → trade mean-reversion.
Event-driven: only trade when liq_zscore > threshold.
Direction: liq_long >> liq_short → longs got liquidated → go long (buy the dip).
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CG_DIR = ROOT / "data" / "raw" / "coinglass"

EPS = 1e-10
PERIODS_PER_YEAR = 2 * 365
ROLL_30 = 30
ROLL_7 = 7

from _research_r68_continuous_wf import (
    load_data, CONTINUOUS_WINDOWS, _cost_for_sym,
)


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def sharpe(rets) -> float:
    if len(rets) < 2:
        return 0.0
    r = np.array(rets, dtype=float)
    eq = np.cumprod(1 + r)
    pct = np.diff(eq) / eq[:-1]
    if len(pct) < 2 or np.std(pct) < EPS:
        return 0.0
    return float(np.mean(pct) / (np.std(pct) + EPS) * np.sqrt(PERIODS_PER_YEAR))


def max_dd(rets) -> float:
    eq = np.cumprod(1 + np.array(rets, dtype=float))
    running_max = np.maximum.accumulate(eq)
    dd = eq / running_max - 1
    return float(np.min(dd))


def load_liq_data() -> pd.DataFrame:
    """Load CG liquidation parquet."""
    path = CG_DIR / "liq.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["cg_date"] = df["timestamp"].dt.normalize()
    df = df.drop_duplicates(subset=["symbol", "cg_date"], keep="last")
    log(f"  Liq data: {len(df):,} rows, {df['symbol'].nunique()} symbols")
    return df


def build_liq_events(liq_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build liquidation event table with zscore and direction.
    Returns: DataFrame with (symbol, cg_date, liq_zscore, liq_direction, liq_total)
    - liq_direction: +1 if longs got liquidated more (go long for mean-rev)
                     -1 if shorts got liquidated more (go short for mean-rev)
    """
    rows = []
    for sym, g in liq_df.groupby("symbol"):
        g = g.sort_values("cg_date").copy()
        total = g["liq_long_usd"] + g["liq_short_usd"]
        roll_mean = total.rolling(ROLL_30, min_periods=ROLL_7).mean()
        roll_std = total.rolling(ROLL_30, min_periods=ROLL_7).std() + EPS
        zscore = (total - roll_mean) / roll_std

        # Direction: if liq_long > liq_short, longs got rekt → buy (mean-rev)
        imb = (g["liq_long_usd"] - g["liq_short_usd"]) / (total + EPS)
        direction = np.where(imb > 0, 1, -1)  # +1 = go long, -1 = go short

        for i in range(len(g)):
            rows.append({
                "symbol": sym,
                "cg_date": g.iloc[i]["cg_date"],
                "liq_zscore": float(zscore.iloc[i]),
                "liq_direction": int(direction[i]),
                "liq_total": float(total.iloc[i]),
                "liq_imb": float(imb.iloc[i]),
            })

    events = pd.DataFrame(rows)
    log(f"  Event table: {len(events):,} rows")
    return events


def simulate_liq_events(
    research_df: pd.DataFrame,
    events: pd.DataFrame,
    windows: list,
    threshold: float = 2.5,
    hold_periods: int = 2,
    n_long: int = 2,
    n_short: int = 2,
    cooldown: int = 1,
) -> pd.DataFrame:
    """
    Event-driven simulation:
    - At each rebalance timestamp, check if any symbols have liq events
    - If event: enter position, hold for H periods, then exit
    - Cooldown: don't re-enter same symbol within C periods
    """
    tz = research_df["timestamp"].dt.tz

    # Filter to test periods
    test_mask = pd.Series(False, index=research_df.index)
    for w in windows:
        ts = pd.Timestamp(w["test_start"], tz=tz)
        te = pd.Timestamp(w["test_end"], tz=tz)
        test_mask |= (research_df["timestamp"] >= ts) & (research_df["timestamp"] <= te)
    df = research_df[test_mask].copy()

    # Shift1 merge: use yesterday's liq data
    df["_cg_date"] = df["timestamp"].dt.normalize() - pd.Timedelta(days=1)
    merged = df.merge(
        events.rename(columns={"cg_date": "_cg_date"}),
        on=["symbol", "_cg_date"],
        how="left",
    )

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}

    # Track active positions: {symbol: (direction, periods_remaining)}
    active_positions = {}
    # Cooldown tracker: {symbol: periods_remaining}
    cooldown_tracker = {}

    all_rets = []

    for ts in timestamps_sorted:
        if ts not in grouped:
            continue
        grp = grouped[ts]

        # Decrement cooldowns
        for sym in list(cooldown_tracker.keys()):
            cooldown_tracker[sym] -= 1
            if cooldown_tracker[sym] <= 0:
                del cooldown_tracker[sym]

        # Check for new events (symbols with zscore > threshold)
        events_now = grp[grp["liq_zscore"] > threshold].copy()

        # Filter out symbols in cooldown
        if len(events_now) > 0:
            events_now = events_now[~events_now["symbol"].isin(cooldown_tracker)]

        # Open new positions from events
        if len(events_now) > 0:
            # Sort by zscore descending (strongest events first)
            events_now = events_now.sort_values("liq_zscore", ascending=False)

            # Separate by direction
            long_events = events_now[events_now["liq_direction"] > 0]
            short_events = events_now[events_now["liq_direction"] < 0]

            # Take top-K per direction
            for _, row in long_events.head(n_long).iterrows():
                sym = row["symbol"]
                if sym not in active_positions:
                    active_positions[sym] = (1, hold_periods)

            for _, row in short_events.head(n_short).iterrows():
                sym = row["symbol"]
                if sym not in active_positions:
                    active_positions[sym] = (-1, hold_periods)

        # Compute returns from active positions
        if active_positions:
            long_rets = []
            short_rets = []
            n_long_act = 0
            n_short_act = 0

            for sym, (direction, periods_left) in list(active_positions.items()):
                sym_row = grp[grp["symbol"] == sym]
                if len(sym_row) == 0:
                    continue
                fwd_ret = float(sym_row.iloc[0]["fwd_ret_12h"])
                if np.isnan(fwd_ret):
                    continue

                if direction > 0:
                    long_rets.append(fwd_ret)
                    n_long_act += 1
                else:
                    short_rets.append(fwd_ret)
                    n_short_act += 1

            total_positions = n_long_act + n_short_act
            if total_positions > 0:
                long_mean = np.mean(long_rets) if long_rets else 0
                short_mean = np.mean(short_rets) if short_rets else 0

                if n_long_act > 0 and n_short_act > 0:
                    gross_ret = 0.5 * long_mean - 0.5 * short_mean
                elif n_short_act > 0:
                    gross_ret = -short_mean
                else:
                    gross_ret = long_mean

                # Costs: opening cost for newly opened positions
                avg_weight = 1.0 / total_positions
                cost = 0.0
                for sym in active_positions:
                    _, pl = active_positions[sym]
                    if pl == hold_periods:  # just opened
                        cost += _cost_for_sym(sym) * avg_weight
                    elif pl <= 0:  # closing
                        cost += _cost_for_sym(sym) * avg_weight

                net_ret = gross_ret - cost
                all_rets.append({
                    "timestamp": ts,
                    "gross_ret": gross_ret,
                    "net_ret": net_ret,
                    "cost": cost,
                    "n_long": n_long_act,
                    "n_short": n_short_act,
                    "n_events": len(events_now) if len(events_now) > 0 else 0,
                })
            else:
                all_rets.append({
                    "timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                    "cost": 0.0, "n_long": 0, "n_short": 0, "n_events": 0,
                })

            # Decrement hold periods and close expired
            for sym in list(active_positions.keys()):
                direction, periods_left = active_positions[sym]
                periods_left -= 1
                if periods_left <= 0:
                    del active_positions[sym]
                    cooldown_tracker[sym] = cooldown
                else:
                    active_positions[sym] = (direction, periods_left)
        else:
            all_rets.append({
                "timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                "cost": 0.0, "n_long": 0, "n_short": 0, "n_events": 0,
            })

    return pd.DataFrame(all_rets) if all_rets else pd.DataFrame()


def main():
    t0 = time.time()
    log("=" * 70)
    log("  R92 — LIQUIDATION EVENT MEAN-REVERSION")
    log("=" * 70)

    # Load data
    log("\n[0] Loading data ...")
    df, regime_df = load_data()
    liq_raw = load_liq_data()

    # Build events
    log("\n[1] Building liquidation events ...")
    events = build_liq_events(liq_raw)

    # Count events per threshold
    for thr in [2.0, 2.5, 3.0]:
        n_events = (events["liq_zscore"] > thr).sum()
        log(f"  Events with zscore > {thr}: {n_events}")

    # Grid search
    log("\n[2] Grid search ...")
    grid = []
    for threshold in [2.0, 2.5, 3.0]:
        for hold in [1, 2, 3]:
            for n_l, n_s in [(1, 1), (2, 1), (2, 2)]:
                for cd in [0, 1, 2]:
                    grid.append({
                        "threshold": threshold,
                        "hold": hold,
                        "n_long": n_l,
                        "n_short": n_s,
                        "cooldown": cd,
                    })

    results = []
    best_sharpe = -999
    best_port = None
    best_cfg = None

    for i, cfg in enumerate(grid):
        port = simulate_liq_events(
            df, events, CONTINUOUS_WINDOWS,
            threshold=cfg["threshold"],
            hold_periods=cfg["hold"],
            n_long=cfg["n_long"],
            n_short=cfg["n_short"],
            cooldown=cfg["cooldown"],
        )

        if len(port) == 0:
            continue

        # Filter out zero-only periods for metrics
        active_periods = port[port["n_long"] + port["n_short"] > 0]
        if len(active_periods) < 10:
            continue

        label = f"thr{cfg['threshold']}_H{cfg['hold']}_K{cfg['n_long']}L{cfg['n_short']}S_cd{cfg['cooldown']}"
        rets = port["net_ret"]
        s = sharpe(rets)
        dd = max_dd(rets)
        n_events_total = int(port["n_events"].sum())
        hit_rate = float((active_periods["net_ret"] > 0).mean())

        m = {
            "label": label,
            "net_sharpe": round(s, 4),
            "max_dd_pct": round(dd * 100, 2),
            "total_ret_pct": round(float((1 + rets).prod() - 1) * 100, 1),
            "hit_rate": round(hit_rate, 3),
            "n_periods": len(rets),
            "n_active": len(active_periods),
            "n_events": n_events_total,
            "config": cfg,
        }
        results.append(m)

        if s > best_sharpe:
            best_sharpe = s
            best_port = port
            best_cfg = cfg

    # Print top configs
    results_sorted = sorted(results, key=lambda x: x["net_sharpe"], reverse=True)

    log(f"\n  TOP 10 CONFIGS:")
    log(f"  {'Label':<40} {'NetSh':>8} {'DD%':>8} {'Ret%':>8} {'Hit':>6} {'Events':>7}")
    log(f"  {'-' * 77}")
    for r in results_sorted[:10]:
        log(f"  {r['label']:<40} {r['net_sharpe']:>8.3f} {r['max_dd_pct']:>7.1f}% "
            f"{r['total_ret_pct']:>7.1f}% {r['hit_rate']:>6.3f} {r['n_events']:>7}")

    # Correlation with R68
    log("\n[3] Correlation with R68 ...")
    r68_equity_path = RESULTS_DIR / "r86_r84_baseline_equity.csv"
    corr_with_r68 = None
    if r68_equity_path.exists() and best_port is not None:
        r68_eq = pd.read_csv(r68_equity_path, parse_dates=["timestamp"])
        r68_rets = r68_eq.set_index("timestamp")["net_ret"]
        r92_rets = best_port.set_index("timestamp")["net_ret"]
        common = r68_rets.index.intersection(r92_rets.index)
        if len(common) > 20:
            corr_with_r68 = float(r68_rets.loc[common].corr(r92_rets.loc[common]))
            log(f"  Corr(R92_best, R68) = {corr_with_r68:.3f}")
    else:
        log("  R68 equity not found")

    # Save
    log("\n[4] Saving ...")
    summary = {
        "script": "r92_liq_events",
        "best_config": best_cfg,
        "best_sharpe": best_sharpe,
        "corr_with_r68": corr_with_r68,
        "n_configs_tested": len(results),
        "grid_results": results_sorted[:20],
        "runtime_sec": round(time.time() - t0, 1),
    }
    (RESULTS_DIR / "r92_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))

    if best_port is not None:
        best_port.to_csv(RESULTS_DIR / "r92_best_equity.csv", index=False)
        log(f"  Saved: r92_best_equity.csv")

    pd.DataFrame(results_sorted).to_csv(RESULTS_DIR / "r92_grid.csv", index=False)

    log(f"\n{'=' * 70}")
    log(f"  R92 COMPLETE — {time.time()-t0:.0f}s")
    log(f"{'=' * 70}")


if __name__ == "__main__":
    main()
