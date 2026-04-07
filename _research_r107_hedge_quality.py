#!/usr/bin/env python3
"""
R107 — Hedge Quality & Basis Risk Analysis

Key question: how much does basis (perp - spot) move during a funding arb position?
If basis moves > funding income → strategy is unprofitable despite positive FR.

Uses:
  - binance_premium_index.parquet (8h premium = perp/spot - 1)
  - binance_funding_rates.parquet (8h FR)

Outputs:
  results/r107_basis_stats.csv        — per-coin basis distribution
  results/r107_worst_case.json        — worst-case basis moves
  results/r107_basis_vs_funding.json  — basis risk vs funding income
  results/r107_revised_backtest.json  — R106 backtest corrected for basis risk
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
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR = ROOT / "data"

# Fees
SPOT_FEE = 0.0005
PERP_FEE = 0.0003
ROUND_TRIP = 2 * (SPOT_FEE + PERP_FEE)  # 0.16%

FUNDING_PERIODS_PER_YEAR = 3 * 365
CAPITAL = 100.0
EPS = 1e-10


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Data Loading ──────────────────────────────────────────────────────────

def load_premium_index() -> pd.DataFrame:
    """Load Binance premium index (basis proxy)."""
    path = DATA_DIR / "sentiment" / "binance_premium_index.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    log(f"  Premium index: {len(df):,} rows, {df.symbol.nunique()} sym, "
        f"{df.timestamp.min().date()} to {df.timestamp.max().date()}")
    return df


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


# ── Analysis Functions ────────────────────────────────────────────────────

def basis_distribution(premium_df: pd.DataFrame) -> pd.DataFrame:
    """Per-coin basis statistics."""
    rows = []
    for sym in sorted(premium_df.symbol.unique()):
        sub = premium_df[premium_df.symbol == sym]["premium_index"].dropna()
        if len(sub) < 100:
            continue
        # Basis change per 8h period
        sub_sorted = premium_df[premium_df.symbol == sym].sort_values("timestamp")
        basis_change = sub_sorted["premium_index"].diff().dropna()

        rows.append({
            "symbol": sym,
            "n_obs": len(sub),
            "basis_mean": sub.mean(),
            "basis_std": sub.std(),
            "basis_p1": sub.quantile(0.01),
            "basis_p5": sub.quantile(0.05),
            "basis_p95": sub.quantile(0.95),
            "basis_p99": sub.quantile(0.99),
            "basis_change_mean": basis_change.mean(),
            "basis_change_std": basis_change.std(),
            "basis_change_p1": basis_change.quantile(0.01),
            "basis_change_p99": basis_change.quantile(0.99),
            "basis_change_abs_mean": basis_change.abs().mean(),
        })
    return pd.DataFrame(rows).sort_values("basis_change_std", ascending=True)


def worst_case_analysis(premium_df: pd.DataFrame) -> dict:
    """Worst-case basis moves over various horizons."""
    results = {}
    for hold_periods in [1, 3, 6, 12, 24]:
        basis_moves = []
        for sym in premium_df.symbol.unique():
            sub = premium_df[premium_df.symbol == sym].sort_values("timestamp")
            basis = sub["premium_index"].values
            if len(basis) < hold_periods + 1:
                continue
            # Rolling basis change over hold_periods
            for i in range(len(basis) - hold_periods):
                delta = basis[i + hold_periods] - basis[i]
                basis_moves.append(delta)

        if not basis_moves:
            continue
        bm = np.array(basis_moves)
        results[f"hold_{hold_periods}_periods"] = {
            "hold_hours": hold_periods * 8,
            "n_observations": len(bm),
            "mean_abs_change_pct": round(float(np.abs(bm).mean()) * 100, 4),
            "std_change_pct": round(float(bm.std()) * 100, 4),
            "p1_pct": round(float(np.percentile(bm, 1)) * 100, 4),
            "p5_pct": round(float(np.percentile(bm, 5)) * 100, 4),
            "p95_pct": round(float(np.percentile(bm, 95)) * 100, 4),
            "p99_pct": round(float(np.percentile(bm, 99)) * 100, 4),
            "worst_pct": round(float(bm.min()) * 100, 4),
            "best_pct": round(float(bm.max()) * 100, 4),
        }
    return results


def basis_vs_funding(premium_df: pd.DataFrame, funding_df: pd.DataFrame) -> dict:
    """Compare basis risk to funding income — the key question."""
    # Merge on timestamp + symbol
    merged = funding_df.merge(premium_df, on=["timestamp", "symbol"], how="inner")
    if merged.empty:
        return {"error": "no overlap"}

    log(f"  Merged: {len(merged):,} rows")

    # For each (symbol, timestamp), compute:
    #   funding_income = fr (what we earn)
    #   basis_change = Δ(premium_index) next period (what we might lose/gain)
    merged = merged.sort_values(["symbol", "timestamp"])
    merged["basis_change"] = merged.groupby("symbol")["premium_index"].diff().shift(-1)
    merged = merged.dropna(subset=["basis_change"])

    # Net P&L per period = funding - basis_change (basis goes against us for short perp)
    # If we're short perp and basis widens (perp goes up relative to spot), we lose
    merged["net_pnl"] = merged["fr"] - merged["basis_change"]

    # Filter to positive FR only (we only enter when FR > threshold)
    for thr in [0.0, 0.0003, 0.0005, 0.0008]:
        sub = merged[merged["fr"] > thr]
        if len(sub) == 0:
            continue
        label = f"fr_gt_{thr:.4f}"
        yield_info = {
            "threshold": thr,
            "n_periods": len(sub),
            "mean_fr_pct": round(sub["fr"].mean() * 100, 4),
            "mean_basis_change_pct": round(sub["basis_change"].mean() * 100, 4),
            "std_basis_change_pct": round(sub["basis_change"].std() * 100, 4),
            "mean_net_pnl_pct": round(sub["net_pnl"].mean() * 100, 4),
            "std_net_pnl_pct": round(sub["net_pnl"].std() * 100, 4),
            "pct_net_positive": round((sub["net_pnl"] > 0).mean() * 100, 2),
            "sharpe_per_period": round(
                sub["net_pnl"].mean() / (sub["net_pnl"].std() + EPS), 4),
            "ann_sharpe": round(
                sub["net_pnl"].mean() / (sub["net_pnl"].std() + EPS) * np.sqrt(FUNDING_PERIODS_PER_YEAR), 2),
            "basis_to_funding_ratio": round(
                sub["basis_change"].std() / (sub["fr"].mean() + EPS), 2),
        }
        yield label, yield_info


def revised_backtest(funding_df: pd.DataFrame, premium_df: pd.DataFrame,
                     entry_thr: float, exit_thr: float, max_hold: int,
                     max_pos: int) -> dict:
    """
    Re-run R106 best config WITH basis risk.
    Net P&L per period = funding_income - basis_change_against_us - costs.
    """
    # Merge funding + premium
    merged = funding_df.merge(premium_df, on=["timestamp", "symbol"], how="inner")
    merged = merged.sort_values(["symbol", "timestamp"])
    merged["basis_change"] = merged.groupby("symbol")["premium_index"].diff()

    # Build lookups
    fr_lookup = dict(zip(zip(merged.timestamp, merged.symbol), merged.fr))
    basis_lookup = dict(zip(zip(merged.timestamp, merged.symbol), merged.basis_change.fillna(0)))
    all_ts = sorted(merged.timestamp.unique())
    symbols = sorted(merged.symbol.unique())

    from dataclasses import dataclass

    @dataclass
    class Pos:
        symbol: str
        entry_time: object
        size_usd: float
        periods_held: int = 0
        funding_pnl: float = 0.0
        basis_pnl: float = 0.0
        entry_cost: float = 0.0

    positions = []
    equity = CAPITAL
    equity_curve = []
    total_entries = 0
    total_funding = 0.0
    total_basis_pnl = 0.0
    total_costs = 0.0
    closed_trades = []

    for ts in all_ts:
        period_pnl = 0.0

        # Collect funding + basis for open positions
        for pos in positions:
            fr = fr_lookup.get((ts, pos.symbol), 0.0)
            bc = basis_lookup.get((ts, pos.symbol), 0.0)
            # Funding income (short perp receives positive FR)
            funding = pos.size_usd * fr
            # Basis risk: if we're short perp & basis widens → loss
            # basis_change = Δ(premium). Short perp loses when premium increases.
            basis_loss = pos.size_usd * bc
            pos.funding_pnl += funding
            pos.basis_pnl -= basis_loss  # negative because short perp
            pos.periods_held += 1
            period_pnl += funding - basis_loss
            total_funding += funding
            total_basis_pnl -= basis_loss

        # Check exits
        to_close = []
        for i, pos in enumerate(positions):
            fr = fr_lookup.get((ts, pos.symbol), 0.0)
            if (fr < exit_thr) or (pos.periods_held >= max_hold):
                exit_cost = pos.size_usd * ROUND_TRIP
                total_costs += exit_cost
                period_pnl -= exit_cost
                net_trade = pos.funding_pnl + pos.basis_pnl - pos.entry_cost - exit_cost
                closed_trades.append({
                    "symbol": pos.symbol,
                    "periods": pos.periods_held,
                    "funding": pos.funding_pnl,
                    "basis": pos.basis_pnl,
                    "costs": pos.entry_cost + exit_cost,
                    "net": net_trade,
                })
                to_close.append(i)
        for i in sorted(to_close, reverse=True):
            positions.pop(i)

        # New entries
        if len(positions) < max_pos:
            open_syms = {p.symbol for p in positions}
            candidates = []
            for sym in symbols:
                if sym in open_syms:
                    continue
                fr = fr_lookup.get((ts, sym), 0.0)
                if fr > entry_thr:
                    candidates.append((sym, fr))
            candidates.sort(key=lambda x: x[1], reverse=True)
            slots = max_pos - len(positions)
            for sym, fr in candidates[:slots]:
                pos_size = CAPITAL / max_pos / 2
                entry_cost = pos_size * ROUND_TRIP
                total_costs += entry_cost
                period_pnl -= entry_cost
                positions.append(Pos(
                    symbol=sym, entry_time=ts, size_usd=pos_size,
                    entry_cost=entry_cost,
                ))
                total_entries += 1

        equity += period_pnl
        equity_curve.append({"timestamp": ts, "equity": equity, "pnl": period_pnl})

    eq_df = pd.DataFrame(equity_curve)
    if len(eq_df) < 10:
        return {"valid": False}

    eq_df["ret"] = eq_df["equity"].pct_change().fillna(0)
    rets = eq_df["ret"]
    total_ret = equity / CAPITAL - 1
    sharpe = rets.mean() / (rets.std() + EPS) * np.sqrt(FUNDING_PERIODS_PER_YEAR)
    max_dd = (eq_df["equity"] / eq_df["equity"].cummax() - 1).min()
    vol = rets.std() * np.sqrt(FUNDING_PERIODS_PER_YEAR)

    wins = sum(1 for t in closed_trades if t["net"] > 0)
    n_trades = len(closed_trades)

    # Per-year equity
    eq_df["timestamp"] = pd.to_datetime([e["timestamp"] for e in equity_curve])
    eq_df["year"] = eq_df["timestamp"].dt.year
    yearly = []
    for y, g in eq_df.groupby("year"):
        yr_ret = (g.iloc[-1]["equity"] / g.iloc[0]["equity"] - 1) * 100
        yearly.append({"year": int(y), "ret_pct": round(yr_ret, 2)})

    return {
        "valid": True,
        "sharpe": round(sharpe, 4),
        "total_ret_pct": round(total_ret * 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "vol_ann_pct": round(vol * 100, 2),
        "total_entries": total_entries,
        "total_trades": n_trades,
        "win_rate": round(wins / max(n_trades, 1), 3),
        "total_funding_usd": round(total_funding, 4),
        "total_basis_pnl_usd": round(total_basis_pnl, 4),
        "total_costs_usd": round(total_costs, 4),
        "final_equity": round(equity, 2),
        "n_periods": len(eq_df),
        "yearly_returns": yearly,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    log("=" * 70)
    log("R107 — Hedge Quality & Basis Risk Analysis")
    log("=" * 70)

    # Load data
    log("\n[1/5] Loading data...")
    premium = load_premium_index()
    funding = load_funding()

    # 1. Basis distribution
    log("\n[2/5] Basis distribution per coin...")
    basis_stats = basis_distribution(premium)
    basis_stats.to_csv(RESULTS_DIR / "r107_basis_stats.csv", index=False)
    log(f"  Coins analyzed: {len(basis_stats)}")
    log(f"  {'Symbol':>12s}  {'Basis µ%':>8s}  {'Basis σ%':>8s}  {'ΔBasis σ%':>10s}  {'|ΔBasis| µ%':>12s}")
    for _, r in basis_stats.head(10).iterrows():
        log(f"  {r.symbol:>12s}  {r.basis_mean*100:>+8.4f}  {r.basis_std*100:>8.4f}  "
            f"{r.basis_change_std*100:>10.4f}  {r.basis_change_abs_mean*100:>12.4f}")

    # 2. Worst-case basis moves
    log("\n[3/5] Worst-case basis moves...")
    worst_case = worst_case_analysis(premium)
    with open(RESULTS_DIR / "r107_worst_case.json", "w") as f:
        json.dump(worst_case, f, indent=2)

    for k, v in worst_case.items():
        log(f"  {k}: mean_abs={v['mean_abs_change_pct']:.4f}%  "
            f"std={v['std_change_pct']:.4f}%  "
            f"p1/p99={v['p1_pct']:.4f}%/{v['p99_pct']:.4f}%  "
            f"worst={v['worst_pct']:.4f}%")

    # 3. Basis vs funding
    log("\n[4/5] Basis risk vs funding income...")
    bvf_results = {}
    for label, info in basis_vs_funding(premium, funding):
        bvf_results[label] = info
        log(f"  {label}: FR_mean={info['mean_fr_pct']:.4f}%  "
            f"basis_σ={info['std_basis_change_pct']:.4f}%  "
            f"net_pnl={info['mean_net_pnl_pct']:+.4f}%  "
            f"ratio={info['basis_to_funding_ratio']:.1f}x  "
            f"ann_sharpe={info['ann_sharpe']:.2f}  "
            f"win={info['pct_net_positive']:.1f}%")

    with open(RESULTS_DIR / "r107_basis_vs_funding.json", "w") as f:
        json.dump(bvf_results, f, indent=2)

    # 4. Revised backtest (R106 best config + basis risk)
    log("\n[5/5] Revised backtest (R106 best + basis risk)...")
    # R106 best: entry=0.0008, exit=0.00005, hold=24, pos=3
    revised = revised_backtest(funding, premium,
                               entry_thr=0.0008, exit_thr=0.00005,
                               max_hold=24, max_pos=3)

    if revised.get("valid"):
        log(f"  REVISED R106 (with basis risk):")
        log(f"    Sharpe={revised['sharpe']:.3f}  (was 6.638 without basis)")
        log(f"    Return={revised['total_ret_pct']:+.2f}%  MaxDD={revised['max_dd_pct']:+.2f}%")
        log(f"    Vol={revised['vol_ann_pct']:.2f}%  (was 0.6% without basis)")
        log(f"    Funding=${revised['total_funding_usd']:.2f}  "
            f"Basis=${revised['total_basis_pnl_usd']:.2f}  "
            f"Costs=${revised['total_costs_usd']:.2f}")
        log(f"    Trades={revised['total_trades']}  Win={revised['win_rate']:.1%}")
        log(f"    Yearly: {revised['yearly_returns']}")

        # Also test R106 #2 config
        log(f"\n  Testing R106 #2 (entry=0.05%, exit=0.01%, hold=24, pos=3)...")
        revised2 = revised_backtest(funding, premium,
                                    entry_thr=0.0005, exit_thr=0.0001,
                                    max_hold=24, max_pos=3)
        if revised2.get("valid"):
            log(f"    Sharpe={revised2['sharpe']:.3f}  Ret={revised2['total_ret_pct']:+.2f}%  "
                f"DD={revised2['max_dd_pct']:+.2f}%  Vol={revised2['vol_ann_pct']:.2f}%")

        # Also test conservative: entry=0.08%, hold=6 (48h), pos=2
        log(f"\n  Testing conservative (entry=0.08%, exit=0.005%, hold=6, pos=2)...")
        revised3 = revised_backtest(funding, premium,
                                    entry_thr=0.0008, exit_thr=0.00005,
                                    max_hold=6, max_pos=2)
        if revised3.get("valid"):
            log(f"    Sharpe={revised3['sharpe']:.3f}  Ret={revised3['total_ret_pct']:+.2f}%  "
                f"DD={revised3['max_dd_pct']:+.2f}%  Vol={revised3['vol_ann_pct']:.2f}%")
    else:
        log("  ERROR: Revised backtest failed!")
        revised = {"valid": False}

    # Save all
    all_results = {
        "revised_r106_best": revised,
        "basis_vs_funding": bvf_results,
        "worst_case_summary": worst_case,
    }
    with open(RESULTS_DIR / "r107_revised_backtest.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── Verdict ───────────────────────────────────────────────────────────
    log("\n" + "=" * 70)

    if not revised.get("valid"):
        verdict = "ERROR"
        log(f"  R107 VERDICT: ERROR")
    else:
        # Kill if basis_vol > 2× avg funding income
        key_bvf = bvf_results.get("fr_gt_0.0008", bvf_results.get("fr_gt_0.0005", {}))
        ratio = key_bvf.get("basis_to_funding_ratio", 999)
        rev_sharpe = revised["sharpe"]

        if ratio > 2.0:
            verdict = "FAIL"
            log(f"  R107 VERDICT: FAIL — basis_vol/funding = {ratio:.1f}x > 2x → basis risk too high")
        elif rev_sharpe < 0.5:
            verdict = "FAIL"
            log(f"  R107 VERDICT: FAIL — revised Sharpe={rev_sharpe:.3f} < 0.5")
        elif rev_sharpe < 1.0:
            verdict = "MARGINAL"
            log(f"  R107 VERDICT: MARGINAL — revised Sharpe={rev_sharpe:.3f}, basis_ratio={ratio:.1f}x")
        else:
            verdict = "PASS"
            log(f"  R107 VERDICT: PASS — revised Sharpe={rev_sharpe:.3f}, basis_ratio={ratio:.1f}x")

    all_results["r107_verdict"] = verdict
    with open(RESULTS_DIR / "r107_revised_backtest.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    log("Done.")
    return all_results


if __name__ == "__main__":
    main()
