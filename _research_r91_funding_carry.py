#!/usr/bin/env python3
"""
R91 — Funding Carry Strategy (rule-based, NO ML).

Signal: carry_score = -fr_close (shift1).
Long coins with lowest FR (we receive funding), short coins with highest FR.
Uses R68 simulate() with carry_score in place of ML predictions.
"""

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CG_DIR = ROOT / "data" / "raw" / "coinglass"

EPS = 1e-10
PERIODS_PER_YEAR = 2 * 365

from _research_r68_continuous_wf import (
    load_data, CONTINUOUS_WINDOWS, PROD_CFG,
    TIER1_SYMS, TIER2_SYMS, TIER3_SYMS, _cost_for_sym,
)


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def sharpe(rets: pd.Series) -> float:
    if len(rets) < 2:
        return 0.0
    eq = (1 + rets).cumprod()
    r = eq.pct_change().dropna()
    return float(r.mean() / (r.std() + EPS) * np.sqrt(PERIODS_PER_YEAR))


def max_dd(rets: pd.Series) -> float:
    eq = (1 + rets).cumprod()
    return float((eq / eq.cummax() - 1).min())


def portfolio_metrics(port: pd.DataFrame, label: str = "") -> dict:
    rets = port["net_ret"]
    s = sharpe(rets)
    dd = max_dd(rets)
    return {
        "label": label,
        "net_sharpe": round(s, 4),
        "gross_sharpe": round(sharpe(port["gross_ret"]), 4),
        "max_dd_pct": round(dd * 100, 2),
        "calmar": round(s / (abs(dd) + EPS), 3),
        "total_ret_pct": round(float((1 + rets).prod() - 1) * 100, 1),
        "win_rate": round(float((rets > 0).mean()), 3),
        "n_periods": len(rets),
    }


def load_funding_data() -> pd.DataFrame:
    """Load CG funding parquet, shift1."""
    path = CG_DIR / "funding.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["cg_date"] = df["timestamp"].dt.normalize()
    df = df.drop_duplicates(subset=["symbol", "cg_date"], keep="last")
    log(f"  Funding: {len(df):,} rows, {df['symbol'].nunique()} symbols")
    return df[["symbol", "cg_date", "fr_close"]]


def build_carry_signal(research_df: pd.DataFrame, funding_df: pd.DataFrame,
                       windows: list) -> pd.DataFrame:
    """
    Build carry signal: for each (timestamp, symbol) in research frame,
    assign carry_score = -fr_close from shift1 funding data.
    """
    tz = research_df["timestamp"].dt.tz

    # Filter to test periods only
    test_mask = pd.Series(False, index=research_df.index)
    for w in windows:
        ts = pd.Timestamp(w["test_start"], tz=tz)
        te = pd.Timestamp(w["test_end"], tz=tz)
        test_mask |= (research_df["timestamp"] >= ts) & (research_df["timestamp"] <= te)
    df = research_df[test_mask].copy()

    # Shift1 merge key: use yesterday's funding rate
    df["_cg_date"] = df["timestamp"].dt.normalize() - pd.Timedelta(days=1)

    merged = df.merge(
        funding_df.rename(columns={"cg_date": "_cg_date"}),
        on=["symbol", "_cg_date"],
        how="left",
    )

    # carry_score = -fr_close (lower FR = better carry for longs)
    merged["pred"] = -merged["fr_close"].fillna(0)

    # Need fwd_ret for simulation
    merged = merged[["timestamp", "symbol", "pred", "fwd_ret_12h"]].dropna()
    merged = merged.rename(columns={"fwd_ret_12h": "fwd_ret"})

    # Add window labels
    merged["window"] = ""
    for w in windows:
        ts = pd.Timestamp(w["test_start"], tz=tz)
        te = pd.Timestamp(w["test_end"], tz=tz)
        mask = (merged["timestamp"] >= ts) & (merged["timestamp"] <= te)
        merged.loc[mask, "window"] = w["name"]

    # raw_prob placeholder
    merged["raw_prob"] = merged["pred"]

    log(f"  Carry signal: {len(merged):,} rows, "
        f"coverage={merged['pred'].notna().mean():.1%}")
    return merged


def simulate_carry(merged: pd.DataFrame, regime_df: pd.DataFrame,
                   n_long: int, n_short: int, rebal_hours: int = 12) -> pd.DataFrame:
    """
    Simulate carry strategy using same logic as R68 simulate() but:
    - No EMA smoothing, no hysteresis (pure ranking)
    - No trend filter (carry doesn't depend on trend)
    - funding_per_12h = 0 for carry (funding IS the return, don't subtract)
    """
    all_rets = []
    prev_longs: Set[str] = set()
    prev_shorts: Set[str] = set()

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}
    rebal_timestamps = timestamps_sorted[::rebal_hours]

    for ts in rebal_timestamps:
        if ts not in grouped:
            continue
        grp = grouped[ts].copy()
        n = len(grp)
        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl == 0 and ns == 0:
            continue

        grp["pred_rank"] = grp["pred"].rank(ascending=False)

        new_longs = set(grp[grp["pred_rank"] <= nl]["symbol"].tolist()) if nl > 0 else set()
        new_shorts = set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist()) if ns > 0 else set()

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

        # Costs (same as R68 but no funding cost — carry IS the funding)
        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            turnover_cost = sum(_cost_for_sym(sym) * avg_weight for sym in new_opened)
            turnover_cost += sum(_cost_for_sym(sym) * avg_weight for sym in closed)
            total_cost = turnover_cost
        else:
            total_cost = 0.0

        net_ret = gross_ret - total_cost
        prev_longs, prev_shorts = new_longs, new_shorts

        all_rets.append({
            "timestamp": ts, "gross_ret": gross_ret, "net_ret": net_ret,
            "cost": total_cost, "n_long": nl_act, "n_short": ns_act,
            "turnover": len(new_opened) + len(closed),
        })

    return pd.DataFrame(all_rets) if all_rets else pd.DataFrame()


def main():
    t0 = time.time()
    log("=" * 70)
    log("  R91 — FUNDING CARRY STRATEGY")
    log("=" * 70)

    # Load data
    log("\n[0] Loading data ...")
    df, regime_df = load_data()
    funding = load_funding_data()

    # Build carry signal
    log("\n[1] Building carry signal ...")
    carry_merged = build_carry_signal(df, funding, CONTINUOUS_WINDOWS)

    # Grid search
    log("\n[2] Grid search ...")
    configs = [
        {"n_long": 4, "n_short": 2, "rebal": 12, "label": "4L2S_12h"},
        {"n_long": 3, "n_short": 3, "rebal": 12, "label": "3L3S_12h"},
        {"n_long": 2, "n_short": 2, "rebal": 12, "label": "2L2S_12h"},
        {"n_long": 4, "n_short": 2, "rebal": 24, "label": "4L2S_24h"},
        {"n_long": 3, "n_short": 3, "rebal": 24, "label": "3L3S_24h"},
        {"n_long": 2, "n_short": 2, "rebal": 24, "label": "2L2S_24h"},
    ]

    results = []
    best_sharpe = -999
    best_port = None
    best_label = ""

    for cfg in configs:
        port = simulate_carry(
            carry_merged, regime_df,
            n_long=cfg["n_long"], n_short=cfg["n_short"],
            rebal_hours=cfg["rebal"],
        )
        if len(port) == 0:
            log(f"  {cfg['label']}: NO DATA")
            continue
        m = portfolio_metrics(port, label=f"R91_{cfg['label']}")
        results.append(m)
        log(f"  {cfg['label']}: Sharpe={m['net_sharpe']:.3f}  MaxDD={m['max_dd_pct']:.1f}%  "
            f"Ret={m['total_ret_pct']:.1f}%  Win={m['win_rate']:.3f}  N={m['n_periods']}")

        if m["net_sharpe"] > best_sharpe:
            best_sharpe = m["net_sharpe"]
            best_port = port
            best_label = cfg["label"]

    # Correlation with R68
    log("\n[3] Correlation with R68 ...")
    r68_equity_path = RESULTS_DIR / "r86_r84_baseline_equity.csv"
    corr_with_r68 = None
    if r68_equity_path.exists() and best_port is not None:
        r68_eq = pd.read_csv(r68_equity_path, parse_dates=["timestamp"])
        r68_rets = r68_eq.set_index("timestamp")["net_ret"]
        r91_rets = best_port.set_index("timestamp")["net_ret"]
        common = r68_rets.index.intersection(r91_rets.index)
        if len(common) > 20:
            corr_with_r68 = float(r68_rets.loc[common].corr(r91_rets.loc[common]))
            log(f"  Corr(R91_best, R68) = {corr_with_r68:.3f}")
        else:
            log(f"  Not enough common periods ({len(common)})")
    else:
        log("  R68 baseline equity not found, skipping correlation")

    # Save results
    log("\n[4] Saving ...")
    summary = {
        "script": "r91_funding_carry",
        "best_config": best_label,
        "best_sharpe": best_sharpe,
        "corr_with_r68": corr_with_r68,
        "grid_results": results,
        "runtime_sec": round(time.time() - t0, 1),
    }
    (RESULTS_DIR / "r91_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))

    if best_port is not None:
        best_port.to_csv(RESULTS_DIR / "r91_best_equity.csv", index=False)
        log(f"  Saved: r91_best_equity.csv ({best_label})")

    pd.DataFrame(results).to_csv(RESULTS_DIR / "r91_grid.csv", index=False)

    # Summary table
    log(f"\n{'=' * 70}")
    log(f"  R91 RESULTS")
    log(f"{'=' * 70}")
    log(f"  {'Config':<18} {'NetSh':>8} {'GrossSh':>8} {'Ret%':>8} {'DD%':>8} {'Win':>6} {'N':>6}")
    log(f"  {'-' * 62}")
    for r in results:
        log(f"  {r['label']:<18} {r['net_sharpe']:>8.3f} {r['gross_sharpe']:>8.3f} "
            f"{r['total_ret_pct']:>7.1f}% {r['max_dd_pct']:>7.1f}% {r['win_rate']:>6.3f} {r['n_periods']:>6}")

    if corr_with_r68 is not None:
        log(f"\n  Corr(R91_best, R68) = {corr_with_r68:.3f}")
    log(f"  Best: {best_label}, Sharpe={best_sharpe:.3f}")
    log(f"  Runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
