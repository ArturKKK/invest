#!/usr/bin/env python3
"""
R105 — Funding Rate Arbitrage: Historical Analysis

Goal: assess the opportunity set for market-neutral funding arbitrage.
Strategy: short perp + long spot = hedge price risk → collect funding payments.

Uses Binance FR (6y history, 8h) as primary, OKX FR (3mo) as cross-validation.

Outputs:
  results/r105_funding_stats.json     — aggregate summary
  results/r105_per_coin_stats.csv     — per-coin FR distribution & carry
  results/r105_opportunity_freq.csv   — opportunities per month at thresholds
  results/r105_autocorr.csv           — FR persistence (autocorrelation)
  results/r105_seasonal.csv           — hour-of-day / day-of-week patterns
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

# Fee structure (OKX taker fees)
SPOT_FEE = 0.0005      # 0.05%
PERP_FEE = 0.0003      # 0.03%
ROUND_TRIP = 2 * (SPOT_FEE + PERP_FEE)  # 0.16%

FUNDING_PERIODS_PER_DAY = 3   # every 8h
FUNDING_PERIODS_PER_YEAR = FUNDING_PERIODS_PER_DAY * 365

THRESHOLDS = [0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.001]  # 0.01% to 0.1%


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Data Loading ──────────────────────────────────────────────────────────

def load_binance_funding() -> pd.DataFrame:
    """Load Binance 8h funding rates (primary — 6y history)."""
    path = ROOT / "data" / "sentiment" / "binance_funding_rates.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.rename(columns={"funding_rate_binance": "fr"})
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    log(f"  Binance FR: {len(df):,} rows, {df.symbol.nunique()} symbols, "
        f"{df.timestamp.min().date()} to {df.timestamp.max().date()}")
    return df[["timestamp", "symbol", "fr"]]


def load_okx_funding() -> pd.DataFrame:
    """Load OKX 8h funding rates (cross-validation)."""
    path = ROOT / "data" / "sentiment" / "funding_rates.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.rename(columns={"funding_rate": "fr"})
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    log(f"  OKX FR:     {len(df):,} rows, {df.symbol.nunique()} symbols, "
        f"{df.timestamp.min().date()} to {df.timestamp.max().date()}")
    return df[["timestamp", "symbol", "fr"]]


# ── Analysis Functions ────────────────────────────────────────────────────

def per_coin_stats(df: pd.DataFrame) -> pd.DataFrame:
    """FR distribution per coin + theoretical carry."""
    rows = []
    for sym in sorted(df.symbol.unique()):
        sub = df[df.symbol == sym]["fr"].dropna()
        if len(sub) < 100:
            continue
        # Positive FR = longs pay shorts → arb opportunity (short perp, long spot)
        pos_fr = sub[sub > 0]
        gross_carry_ann = sub.mean() * FUNDING_PERIODS_PER_YEAR
        net_carry_per_entry = sub.mean() * 3  # avg 3-period hold (24h)
        rows.append({
            "symbol": sym,
            "count": len(sub),
            "mean": sub.mean(),
            "std": sub.std(),
            "p5": sub.quantile(0.05),
            "p25": sub.quantile(0.25),
            "median": sub.median(),
            "p75": sub.quantile(0.75),
            "p95": sub.quantile(0.95),
            "pct_positive": (sub > 0).mean(),
            "pct_gt_002": (sub > 0.0002).mean(),   # >0.02%
            "pct_gt_005": (sub > 0.0005).mean(),   # >0.05%
            "pct_gt_01": (sub > 0.001).mean(),      # >0.1%
            "mean_when_positive": pos_fr.mean() if len(pos_fr) > 0 else 0,
            "gross_carry_ann_pct": gross_carry_ann * 100,
        })
    result = pd.DataFrame(rows).sort_values("gross_carry_ann_pct", ascending=False)
    return result.reset_index(drop=True)


def opportunity_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """How many opportunities per month at each threshold level."""
    df = df.copy()
    df["month"] = df["timestamp"].dt.to_period("M")

    rows = []
    for thr in THRESHOLDS:
        # For each 8h period, count coins with FR > threshold
        hits = df[df.fr > thr].groupby("timestamp")["symbol"].nunique()
        # Monthly stats
        monthly = df[df.fr > thr].groupby("month").agg(
            n_periods=("timestamp", "nunique"),          # unique 8h windows with opportunity
            total_coin_opps=("fr", "count"),            # total coin×opportunity pairs
            best_fr=("fr", "max"),
        ).reset_index()

        n_months = df["month"].nunique()
        rows.append({
            "threshold": thr,
            "threshold_pct": f"{thr*100:.3f}%",
            "avg_opps_per_month": monthly["n_periods"].mean() if len(monthly) > 0 else 0,
            "avg_coin_opps_per_month": monthly["total_coin_opps"].mean() if len(monthly) > 0 else 0,
            "pct_periods_with_opp": (df.groupby("timestamp")["fr"].max() > thr).mean() * 100,
            "avg_coins_per_opp": hits.mean() if len(hits) > 0 else 0,
            "median_fr_when_hit": df[df.fr > thr]["fr"].median() if (df.fr > thr).any() else 0,
        })
    return pd.DataFrame(rows)


def autocorrelation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """FR persistence — autocorrelation at lag 1-6 (8h-48h)."""
    rows = []
    for sym in sorted(df.symbol.unique()):
        sub = df[df.symbol == sym].sort_values("timestamp")["fr"].dropna()
        if len(sub) < 200:
            continue
        row = {"symbol": sym}
        for lag in [1, 2, 3, 6, 12]:
            ac = sub.autocorr(lag=lag)
            row[f"ac_lag{lag}"] = round(ac, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def seasonal_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """FR patterns by funding hour (0, 8, 16 UTC) and day of week."""
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek  # 0=Mon

    # By hour
    hour_stats = df.groupby("hour")["fr"].agg(["mean", "std", "count"]).reset_index()
    hour_stats["period_type"] = "hour"
    hour_stats = hour_stats.rename(columns={"hour": "period_value"})

    # By day of week
    dow_stats = df.groupby("dow")["fr"].agg(["mean", "std", "count"]).reset_index()
    dow_stats["period_type"] = "dow"
    dow_stats = dow_stats.rename(columns={"dow": "period_value"})

    return pd.concat([hour_stats, dow_stats], ignore_index=True)


def regime_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Split data into yearly regimes to check stability."""
    df = df.copy()
    df["year"] = df["timestamp"].dt.year
    rows = []
    for year in sorted(df["year"].unique()):
        sub = df[df["year"] == year]
        if len(sub) < 500:
            continue
        fr = sub["fr"]
        rows.append({
            "year": year,
            "n_obs": len(sub),
            "mean_fr": fr.mean(),
            "std_fr": fr.std(),
            "pct_positive": (fr > 0).mean(),
            "pct_gt_002": (fr > 0.0002).mean(),
            "pct_gt_005": (fr > 0.0005).mean(),
            "gross_carry_ann_pct": fr.mean() * FUNDING_PERIODS_PER_YEAR * 100,
        })
    return pd.DataFrame(rows)


def theoretical_carry_analysis(df: pd.DataFrame) -> dict:
    """
    Compute theoretical carry at various thresholds and hold periods.
    Simulates: enter when FR > threshold, hold for N periods, pay ROUND_TRIP once.
    """
    results = {}
    for thr in [0.0002, 0.0003, 0.0005, 0.0008]:
        for hold_periods in [1, 3, 6, 12]:
            # For each entry point, sum next hold_periods of FR
            entries = []
            for sym in df.symbol.unique():
                sub = df[df.symbol == sym].sort_values("timestamp").reset_index(drop=True)
                fr_arr = sub["fr"].values
                for i in range(len(fr_arr) - hold_periods):
                    if fr_arr[i] > thr:
                        # Sum funding over hold period
                        carry = fr_arr[i:i + hold_periods].sum()
                        net_carry = carry - ROUND_TRIP
                        entries.append(net_carry)

            if len(entries) == 0:
                continue
            entries = np.array(entries)
            key = f"thr_{thr:.4f}_hold_{hold_periods}"
            results[key] = {
                "threshold": thr,
                "hold_periods": hold_periods,
                "hold_hours": hold_periods * 8,
                "n_entries": len(entries),
                "mean_net_carry_pct": round(float(entries.mean()) * 100, 4),
                "median_net_carry_pct": round(float(np.median(entries)) * 100, 4),
                "pct_profitable": round(float((entries > 0).mean()) * 100, 2),
                "worst_carry_pct": round(float(entries.min()) * 100, 4),
                "best_carry_pct": round(float(entries.max()) * 100, 4),
                "monthly_entries": round(len(entries) / (df.timestamp.nunique() / FUNDING_PERIODS_PER_DAY / 30), 1),
            }
    return results


def cross_validate_okx(binance_df: pd.DataFrame, okx_df: pd.DataFrame) -> dict:
    """Compare Binance vs OKX FR for overlapping period."""
    overlap_start = max(binance_df.timestamp.min(), okx_df.timestamp.min())
    overlap_end = min(binance_df.timestamp.max(), okx_df.timestamp.max())

    b = binance_df[(binance_df.timestamp >= overlap_start) & (binance_df.timestamp <= overlap_end)]
    o = okx_df[(okx_df.timestamp >= overlap_start) & (okx_df.timestamp <= overlap_end)]

    # Merge on timestamp + symbol
    merged = b.merge(o, on=["timestamp", "symbol"], suffixes=("_binance", "_okx"), how="inner")

    if len(merged) == 0:
        return {"overlap": 0, "correlation": None}

    corr = merged["fr_binance"].corr(merged["fr_okx"])
    diff = (merged["fr_binance"] - merged["fr_okx"]).abs()

    return {
        "overlap_rows": len(merged),
        "overlap_start": str(overlap_start),
        "overlap_end": str(overlap_end),
        "correlation": round(float(corr), 4),
        "mean_abs_diff": round(float(diff.mean()), 6),
        "mean_binance": round(float(merged["fr_binance"].mean()), 6),
        "mean_okx": round(float(merged["fr_okx"].mean()), 6),
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    log("=" * 70)
    log("R105 — Funding Rate Arbitrage: Historical Analysis")
    log("=" * 70)

    # Load data
    log("\n[1/8] Loading data...")
    binance = load_binance_funding()
    okx = load_okx_funding()

    # Per-coin stats
    log("\n[2/8] Per-coin FR distribution...")
    coin_stats = per_coin_stats(binance)
    coin_stats.to_csv(RESULTS_DIR / "r105_per_coin_stats.csv", index=False)
    log(f"  Top-5 carry coins (ann%):")
    for _, r in coin_stats.head(5).iterrows():
        log(f"    {r.symbol:12s}  mean={r['mean']*100:.4f}%  ann={r.gross_carry_ann_pct:.1f}%  "
            f"pos={r.pct_positive:.1%}  >0.02%={r.pct_gt_002:.1%}")
    log(f"  Bottom-3 carry coins:")
    for _, r in coin_stats.tail(3).iterrows():
        log(f"    {r.symbol:12s}  mean={r['mean']*100:.4f}%  ann={r.gross_carry_ann_pct:.1f}%")

    # Opportunity frequency
    log("\n[3/8] Opportunity frequency at thresholds...")
    opp_freq = opportunity_frequency(binance)
    opp_freq.to_csv(RESULTS_DIR / "r105_opportunity_freq.csv", index=False)
    for _, r in opp_freq.iterrows():
        log(f"  threshold={r.threshold_pct:>8s}  "
            f"opps/month={r.avg_opps_per_month:.0f}  "
            f"coin_opps/month={r.avg_coin_opps_per_month:.0f}  "
            f"pct_periods={r.pct_periods_with_opp:.1f}%  "
            f"avg_coins={r.avg_coins_per_opp:.1f}")

    # Autocorrelation
    log("\n[4/8] FR persistence (autocorrelation)...")
    ac = autocorrelation_analysis(binance)
    ac.to_csv(RESULTS_DIR / "r105_autocorr.csv", index=False)
    mean_ac1 = ac["ac_lag1"].mean()
    mean_ac3 = ac["ac_lag3"].mean()
    log(f"  Mean AC(lag1=8h): {mean_ac1:.3f}")
    log(f"  Mean AC(lag3=24h): {mean_ac3:.3f}")
    log(f"  Mean AC(lag6=48h): {ac['ac_lag6'].mean():.3f}")
    log(f"  Mean AC(lag12=96h): {ac['ac_lag12'].mean():.3f}")
    if mean_ac1 > 0.3:
        log(f"  → FR is PERSISTENT (AC1={mean_ac1:.3f}>0.3) — good for carry hold")
    else:
        log(f"  → FR is MEAN-REVERTING (AC1={mean_ac1:.3f}<0.3) — short hold preferred")

    # Seasonal
    log("\n[5/8] Seasonal patterns...")
    seasonal = seasonal_analysis(binance)
    seasonal.to_csv(RESULTS_DIR / "r105_seasonal.csv", index=False)
    hours = seasonal[seasonal.period_type == "hour"]
    if len(hours) > 0:
        best_hour = hours.loc[hours["mean"].idxmax()]
        worst_hour = hours.loc[hours["mean"].idxmin()]
        log(f"  Best hour (UTC):  {int(best_hour.period_value):02d}:00  mean={best_hour['mean']*100:.4f}%")
        log(f"  Worst hour (UTC): {int(worst_hour.period_value):02d}:00  mean={worst_hour['mean']*100:.4f}%")
    dows = seasonal[seasonal.period_type == "dow"]
    if len(dows) > 0:
        best_dow = dows.loc[dows["mean"].idxmax()]
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        log(f"  Best day: {dow_names[int(best_dow.period_value)]}  mean={best_dow['mean']*100:.4f}%")

    # Regime stability
    log("\n[6/8] Regime stability by year...")
    regimes = regime_analysis(binance)
    for _, r in regimes.iterrows():
        log(f"  {int(r.year)}: mean_fr={r.mean_fr*100:.4f}%  "
            f"ann_carry={r.gross_carry_ann_pct:.1f}%  "
            f"pos={r.pct_positive:.1%}  "
            f">0.02%={r.pct_gt_002:.1%}")

    # Theoretical carry
    log("\n[7/8] Theoretical carry (entry_threshold × hold_period grid)...")
    carry = theoretical_carry_analysis(binance)
    log(f"  {'Threshold':>10s}  {'Hold(h)':>7s}  {'Entries':>8s}  "
        f"{'Net%':>8s}  {'Win%':>6s}  {'Entries/mo':>10s}")
    for k, v in sorted(carry.items()):
        log(f"  {v['threshold']*100:.2f}%    "
            f"{v['hold_hours']:>5d}h  "
            f"{v['n_entries']:>8,d}  "
            f"{v['mean_net_carry_pct']:>+8.4f}  "
            f"{v['pct_profitable']:>5.1f}%  "
            f"{v['monthly_entries']:>10.1f}")

    # Cross-validate OKX
    log("\n[8/8] Cross-validation: Binance vs OKX...")
    xval = cross_validate_okx(binance, okx)
    log(f"  Overlap: {xval.get('overlap_rows', 0):,} rows")
    if xval.get("correlation") is not None:
        log(f"  Correlation: {xval['correlation']:.4f}")
        log(f"  Mean abs diff: {xval['mean_abs_diff']*100:.4f}%")

    # ── Summary ───────────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    # Key decision metrics
    best_threshold = 0.0003  # 0.03%
    opp_at_best = opp_freq[opp_freq.threshold == best_threshold].iloc[0] if (opp_freq.threshold == best_threshold).any() else None

    carry_key = f"thr_{best_threshold:.4f}_hold_3"
    carry_at_best = carry.get(carry_key, {})

    summary = {
        "data_source": "Binance 8h FR",
        "n_symbols": int(binance.symbol.nunique()),
        "date_range": f"{binance.timestamp.min().date()} to {binance.timestamp.max().date()}",
        "total_observations": len(binance),
        "overall_mean_fr_pct": round(binance.fr.mean() * 100, 4),
        "overall_pct_positive": round((binance.fr > 0).mean() * 100, 2),
        "autocorrelation_lag1": round(mean_ac1, 3),
        "autocorrelation_lag3": round(mean_ac3, 3),
        "top5_carry_coins": coin_stats.head(5)["symbol"].tolist(),
        "opportunity_at_003pct": {
            "opps_per_month": round(opp_at_best.avg_opps_per_month, 1) if opp_at_best is not None else None,
            "pct_periods_with_opp": round(opp_at_best.pct_periods_with_opp, 1) if opp_at_best is not None else None,
        },
        "theoretical_carry_003_hold24h": carry_at_best,
        "regime_stability": regimes[["year", "gross_carry_ann_pct"]].to_dict("records"),
        "cross_validation_okx": xval,
        "round_trip_cost_pct": round(ROUND_TRIP * 100, 2),
        "fees": {"spot_taker": SPOT_FEE, "perp_taker": PERP_FEE},
    }

    # Accept/Kill decision
    if opp_at_best is not None:
        opps = opp_at_best.avg_opps_per_month
        if opps >= 5:
            summary["r105_verdict"] = "PASS"
            log(f"\n  R105 VERDICT: PASS — {opps:.0f} opps/month at 0.03% threshold (need ≥5)")
        elif opps >= 2:
            summary["r105_verdict"] = "MARGINAL"
            log(f"\n  R105 VERDICT: MARGINAL — {opps:.0f} opps/month at 0.03% threshold (need ≥5, kill <2)")
        else:
            summary["r105_verdict"] = "FAIL"
            log(f"\n  R105 VERDICT: FAIL — {opps:.0f} opps/month at 0.03% threshold (kill <2)")
    else:
        summary["r105_verdict"] = "ERROR"
        log("\n  R105 VERDICT: ERROR — could not assess at 0.03% threshold")

    # Save
    with open(RESULTS_DIR / "r105_funding_stats.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log(f"\nSaved results to {RESULTS_DIR}/r105_*")
    log("Done.")

    return summary


if __name__ == "__main__":
    main()
