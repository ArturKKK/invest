#!/usr/bin/env python3
"""
R97 — Attribution Analysis: R68 vs R93

WHY R68 Return=179% vs R93 Return=35% when Sharpe is similar (3.78 vs 3.19)?

Uses saved equity CSVs from previous runs:
- results/r86_r84_baseline_equity.csv (R68 4L/2S continuous)
- results/r93_best_equity.csv (R93 4L2S_12h)

Also retrains R68 to save predictions for R100 rank ensemble.

Metrics per strategy:
- n_trading_periods, pct_risk_off
- mean_gross_ret, std_ret, mean_cost
- avg_turnover, avg_positions
- net_ret_per_period, compounded total return
Per-window and per-quarter breakdown.
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

EPS = 1e-10


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def sharpe_ann(rets, ppy):
    if len(rets) < 2:
        return 0.0
    eq = (1 + rets).cumprod()
    r = eq.pct_change().dropna()
    return float(r.mean() / (r.std() + EPS) * np.sqrt(ppy))


def max_dd(rets):
    eq = (1 + rets).cumprod()
    return float((eq / eq.cummax() - 1).min())


def compute_attribution(port: pd.DataFrame, label: str, ppy: int) -> dict:
    """Compute full attribution metrics for a portfolio equity curve."""
    n = len(port)
    gross = port["gross_ret"]
    net = port["net_ret"]
    cost = port["cost"]
    turnover = port["turnover"]
    n_long = port["n_long"]
    n_short = port["n_short"]

    total_ret = float((1 + net).prod() - 1)
    total_ret_gross = float((1 + gross).prod() - 1)

    return {
        "label": label,
        "n_periods": n,
        "mean_gross_ret_bps": round(float(gross.mean()) * 10000, 2),
        "mean_net_ret_bps": round(float(net.mean()) * 10000, 2),
        "std_ret_bps": round(float(net.std()) * 10000, 2),
        "mean_cost_bps": round(float(cost.mean()) * 10000, 2),
        "total_cost_bps": round(float(cost.sum()) * 10000, 1),
        "avg_turnover": round(float(turnover.mean()), 2),
        "avg_positions": round(float((n_long + n_short).mean()), 2),
        "avg_n_long": round(float(n_long.mean()), 2),
        "avg_n_short": round(float(n_short.mean()), 2),
        "total_ret_pct": round(total_ret * 100, 1),
        "total_ret_gross_pct": round(total_ret_gross * 100, 1),
        "max_dd_pct": round(max_dd(net) * 100, 1),
        "sharpe": round(sharpe_ann(net, ppy), 3),
        "gross_sharpe": round(sharpe_ann(gross, ppy), 3),
        # Cost drag
        "cost_drag_ret_pct": round((total_ret_gross - total_ret) * 100, 1),
        # Return per period contribution
        "ret_per_period_bps": round(float(net.mean()) * 10000, 2),
        # Compounding effect
        "n_positive": int((net > 0).sum()),
        "n_negative": int((net <= 0).sum()),
        "win_rate": round(float((net > 0).mean()), 3),
    }


def breakdown_by_column(port: pd.DataFrame, col_name: str, label: str, ppy: int):
    """Break down metrics by a time column (quarter, window, etc.)."""
    groups = sorted(port[col_name].unique())
    rows = []
    for g in groups:
        sub = port[port[col_name] == g]
        net = sub["net_ret"]
        gross = sub["gross_ret"]
        cost = sub["cost"]
        total_ret = float((1 + net).prod() - 1)
        total_ret_gross = float((1 + gross).prod() - 1)
        rows.append({
            "period": str(g),
            "n": len(sub),
            "mean_gross_bps": round(float(gross.mean()) * 10000, 2),
            "mean_net_bps": round(float(net.mean()) * 10000, 2),
            "mean_cost_bps": round(float(cost.mean()) * 10000, 2),
            "std_bps": round(float(net.std()) * 10000, 2),
            "total_ret_pct": round(total_ret * 100, 1),
            "total_ret_gross_pct": round(total_ret_gross * 100, 1),
            "avg_turnover": round(float(sub["turnover"].mean()), 2),
            "sharpe": round(sharpe_ann(net, ppy), 3) if len(net) > 10 else None,
            "max_dd_pct": round(max_dd(net) * 100, 1) if len(net) > 5 else None,
        })
    return rows


def estimate_risk_off_pct(port: pd.DataFrame, rebal_hours: int = 12) -> float:
    """Estimate % of periods skipped (risk-off) due to trend filter."""
    ts = port["timestamp"]
    total_span_hours = (ts.max() - ts.min()).total_seconds() / 3600
    total_possible = int(total_span_hours / rebal_hours)
    actual = len(port)
    if total_possible <= 0:
        return 0.0
    return round(1.0 - actual / total_possible, 3)


def main():
    t0 = time.time()
    log("=" * 70)
    log("  R97 — ATTRIBUTION ANALYSIS: R68 vs R93")
    log("=" * 70)

    # ── Load equity curves ────────────────────────────────────────────────
    log("\n[0] Loading equity curves ...")

    r68_path = RESULTS_DIR / "r86_r84_baseline_equity.csv"
    r93_path = RESULTS_DIR / "r93_best_equity.csv"

    if not r68_path.exists():
        log(f"  ✗ R68 equity not found: {r68_path}")
        log("  Falling back: retraining R68 ...")
        from _research_r68_continuous_wf import (
            load_data, train_ensemble, simulate, CONTINUOUS_WINDOWS,
            SEEDS, PROD_CFG, CHAMPION_FEAT_31, MARKET_LEVEL_FEATURES,
        )
        from _research_r22_models import cs_rank_cols
        df, regime_df = load_data()
        feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
        no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]
        preds_r68 = train_ensemble(df, feats, CONTINUOUS_WINDOWS, seeds=SEEDS,
                                    cs_rank_exclude=no_rank)
        r68_port = simulate(preds_r68, regime_df, 4, 2)
        r68_port.to_csv(r68_path, index=False)
        # Also save R68 predictions for R100
        preds_r68.to_parquet(RESULTS_DIR / "r68_predictions.parquet", index=False)
        log(f"  Saved R68 equity ({len(r68_port)} periods) + predictions")
    else:
        r68_port = pd.read_csv(r68_path, parse_dates=["timestamp"])
        log(f"  R68: {len(r68_port)} periods, {r68_path.name}")

    if not r93_path.exists():
        log(f"  ✗ R93 equity not found: {r93_path}")
        return
    r93_port = pd.read_csv(r93_path, parse_dates=["timestamp"])
    log(f"  R93: {len(r93_port)} periods, {r93_path.name}")

    # Also check if R68 predictions exist (needed for R100)
    r68_pred_path = RESULTS_DIR / "r68_predictions.parquet"
    if not r68_pred_path.exists():
        log("\n  R68 predictions not saved — retraining for R100 ...")
        from _research_r68_continuous_wf import (
            load_data, train_ensemble, CONTINUOUS_WINDOWS,
            SEEDS, CHAMPION_FEAT_31, MARKET_LEVEL_FEATURES,
        )
        from _research_r22_models import cs_rank_cols
        df, regime_df = load_data()
        feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
        no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]
        preds_r68 = train_ensemble(df, feats, CONTINUOUS_WINDOWS, seeds=SEEDS,
                                    cs_rank_exclude=no_rank)
        preds_r68.to_parquet(r68_pred_path, index=False)
        log(f"  Saved R68 predictions: {len(preds_r68):,} rows")
    else:
        log(f"  R68 predictions already saved: {r68_pred_path.name}")

    # ── Attribution ───────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("  [1] OVERALL ATTRIBUTION")
    log("=" * 70)

    # Both use 12h rebalance → periods_per_year = 2*365 = 730
    PPY = 2 * 365  # 12h periods per year

    attr_r68 = compute_attribution(r68_port, "R68_4L2S_12h", PPY)
    attr_r93 = compute_attribution(r93_port, "R93_4L2S_12h", PPY)

    # Risk-off estimation
    roff_r68 = estimate_risk_off_pct(r68_port, 12)
    roff_r93 = estimate_risk_off_pct(r93_port, 12)
    attr_r68["pct_risk_off"] = roff_r68
    attr_r93["pct_risk_off"] = roff_r93

    # Print comparison table
    log(f"\n  {'Metric':<28} {'R68':>12} {'R93':>12} {'Delta':>12}")
    log(f"  {'-' * 64}")
    compare_keys = [
        ("n_periods", ""),
        ("pct_risk_off", ""),
        ("mean_gross_ret_bps", " bps"),
        ("mean_net_ret_bps", " bps"),
        ("std_ret_bps", " bps"),
        ("mean_cost_bps", " bps"),
        ("total_cost_bps", " bps"),
        ("avg_turnover", ""),
        ("avg_positions", ""),
        ("total_ret_pct", "%"),
        ("total_ret_gross_pct", "%"),
        ("cost_drag_ret_pct", "%"),
        ("max_dd_pct", "%"),
        ("sharpe", ""),
        ("gross_sharpe", ""),
        ("win_rate", ""),
        ("n_positive", ""),
        ("n_negative", ""),
    ]
    for key, suffix in compare_keys:
        v68 = attr_r68.get(key, 0)
        v93 = attr_r93.get(key, 0)
        delta = v93 - v68 if isinstance(v68, (int, float)) and isinstance(v93, (int, float)) else ""
        if isinstance(delta, float):
            delta = f"{delta:>+.2f}"
        log(f"  {key:<28} {v68:>12} {v93:>12} {delta:>12}")

    # ── Return decomposition ──────────────────────────────────────────────
    log(f"\n{'=' * 70}")
    log(f"  [2] RETURN DECOMPOSITION")
    log(f"{'=' * 70}")

    # Key decomposition:
    # Total_Return = prod(1 + net_ret_i) - 1
    # net_ret_i = gross_ret_i - cost_i
    # The difference in total return comes from:
    # 1. mean_gross_ret difference (signal quality per period)
    # 2. cost difference
    # 3. compounding effect (higher per-period returns compound more)
    # 4. number of active periods (risk-off skips)

    log(f"\n  R68: {attr_r68['n_periods']} periods × {attr_r68['mean_net_ret_bps']:.1f} bps/period → {attr_r68['total_ret_pct']:.1f}% total")
    log(f"  R93: {attr_r93['n_periods']} periods × {attr_r93['mean_net_ret_bps']:.1f} bps/period → {attr_r93['total_ret_pct']:.1f}% total")

    # Simple linear approx: total_ret ≈ n_periods × mean_ret
    linear_r68 = attr_r68["n_periods"] * attr_r68["mean_net_ret_bps"] / 10000 * 100
    linear_r93 = attr_r93["n_periods"] * attr_r93["mean_net_ret_bps"] / 10000 * 100
    log(f"\n  Linear approx (ignoring compounding):")
    log(f"    R68: {attr_r68['n_periods']} × {attr_r68['mean_net_ret_bps']:.1f}bps = {linear_r68:.1f}%")
    log(f"    R93: {attr_r93['n_periods']} × {attr_r93['mean_net_ret_bps']:.1f}bps = {linear_r93:.1f}%")
    log(f"    Compounding bonus R68: {attr_r68['total_ret_pct'] - linear_r68:.1f}%")
    log(f"    Compounding bonus R93: {attr_r93['total_ret_pct'] - linear_r93:.1f}%")

    # Gross vs net decomposition
    log(f"\n  Gross return (before costs):")
    log(f"    R68: {attr_r68['total_ret_gross_pct']:.1f}%  →  after costs: {attr_r68['total_ret_pct']:.1f}%  (cost drag: {attr_r68['cost_drag_ret_pct']:.1f}%)")
    log(f"    R93: {attr_r93['total_ret_gross_pct']:.1f}%  →  after costs: {attr_r93['total_ret_pct']:.1f}%  (cost drag: {attr_r93['cost_drag_ret_pct']:.1f}%)")

    # Volatility / Sharpe decomposition
    log(f"\n  Return per unit of risk:")
    log(f"    R68: mean={attr_r68['mean_net_ret_bps']:.1f}bps, std={attr_r68['std_ret_bps']:.1f}bps → ratio={attr_r68['mean_net_ret_bps']/max(attr_r68['std_ret_bps'],0.01):.3f}")
    log(f"    R93: mean={attr_r93['mean_net_ret_bps']:.1f}bps, std={attr_r93['std_ret_bps']:.1f}bps → ratio={attr_r93['mean_net_ret_bps']/max(attr_r93['std_ret_bps'],0.01):.3f}")

    # ── Per-window breakdown ──────────────────────────────────────────────
    log(f"\n{'=' * 70}")
    log(f"  [3] PER-WINDOW BREAKDOWN")
    log(f"{'=' * 70}")

    # Assign windows based on CONTINUOUS_WINDOWS date ranges
    from _research_r68_continuous_wf import CONTINUOUS_WINDOWS

    def assign_window(ts):
        for w in CONTINUOUS_WINDOWS:
            ts_start = pd.Timestamp(w["test_start"], tz=ts.tz if hasattr(ts, 'tz') else None)
            ts_end = pd.Timestamp(w["test_end"], tz=ts.tz if hasattr(ts, 'tz') else None)
            if ts_start <= ts <= ts_end:
                return w["name"]
        return "unknown"

    for port, lbl in [(r68_port, "R68"), (r93_port, "R93")]:
        port["window"] = port["timestamp"].apply(assign_window)

    log(f"\n  R68 per window:")
    for row in breakdown_by_column(r68_port, "window", "R68", PPY):
        log(f"    {row['period']:<8}: N={row['n']:>4}  gross={row['mean_gross_bps']:>6.1f}bps  net={row['mean_net_bps']:>6.1f}bps  "
            f"cost={row['mean_cost_bps']:>5.1f}bps  ret={row['total_ret_pct']:>6.1f}%  dd={row['max_dd_pct']}%  sh={row['sharpe']}")

    log(f"\n  R93 per window:")
    for row in breakdown_by_column(r93_port, "window", "R93", PPY):
        log(f"    {row['period']:<8}: N={row['n']:>4}  gross={row['mean_gross_bps']:>6.1f}bps  net={row['mean_net_bps']:>6.1f}bps  "
            f"cost={row['mean_cost_bps']:>5.1f}bps  ret={row['total_ret_pct']:>6.1f}%  dd={row['max_dd_pct']}%  sh={row['sharpe']}")

    # ── Per-quarter breakdown ─────────────────────────────────────────────
    log(f"\n{'=' * 70}")
    log(f"  [4] PER-QUARTER BREAKDOWN")
    log(f"{'=' * 70}")

    for port, lbl in [(r68_port, "R68"), (r93_port, "R93")]:
        port["quarter"] = port["timestamp"].dt.to_period("Q").astype(str)

    log(f"\n  R68 per quarter:")
    for row in breakdown_by_column(r68_port, "quarter", "R68", PPY):
        log(f"    {row['period']:<8}: N={row['n']:>4}  gross={row['mean_gross_bps']:>6.1f}bps  net={row['mean_net_bps']:>6.1f}bps  "
            f"ret={row['total_ret_pct']:>6.1f}%  turnover={row['avg_turnover']:.1f}")

    log(f"\n  R93 per quarter:")
    for row in breakdown_by_column(r93_port, "quarter", "R93", PPY):
        log(f"    {row['period']:<8}: N={row['n']:>4}  gross={row['mean_gross_bps']:>6.1f}bps  net={row['mean_net_bps']:>6.1f}bps  "
            f"ret={row['total_ret_pct']:>6.1f}%  turnover={row['avg_turnover']:.1f}")

    # ── Correlation on aligned timestamps ─────────────────────────────────
    log(f"\n{'=' * 70}")
    log(f"  [5] CORRELATION ON ALIGNED TIMESTAMPS")
    log(f"{'=' * 70}")

    r68_ts = r68_port.set_index("timestamp")["net_ret"]
    r93_ts = r93_port.set_index("timestamp")["net_ret"]
    common = r68_ts.index.intersection(r93_ts.index)
    log(f"\n  Common timestamps: {len(common)}")
    log(f"  R68-only timestamps: {len(r68_ts.index.difference(r93_ts.index))}")
    log(f"  R93-only timestamps: {len(r93_ts.index.difference(r68_ts.index))}")

    if len(common) > 20:
        corr = float(r68_ts.loc[common].corr(r93_ts.loc[common]))
        log(f"  Correlation (net_ret): {corr:.3f}")

        # When one is positive and other negative
        r68c, r93c = r68_ts.loc[common], r93_ts.loc[common]
        both_pos = ((r68c > 0) & (r93c > 0)).sum()
        both_neg = ((r68c <= 0) & (r93c <= 0)).sum()
        r68_pos_r93_neg = ((r68c > 0) & (r93c <= 0)).sum()
        r68_neg_r93_pos = ((r68c <= 0) & (r93c > 0)).sum()
        log(f"  Agreement matrix:")
        log(f"    Both positive:  {both_pos} ({both_pos/len(common)*100:.1f}%)")
        log(f"    Both negative:  {both_neg} ({both_neg/len(common)*100:.1f}%)")
        log(f"    R68+ / R93-:    {r68_pos_r93_neg} ({r68_pos_r93_neg/len(common)*100:.1f}%)")
        log(f"    R68- / R93+:    {r68_neg_r93_pos} ({r68_neg_r93_pos/len(common)*100:.1f}%)")

    # ── Vol-match assessment ──────────────────────────────────────────────
    log(f"\n{'=' * 70}")
    log(f"  [6] VOL-MATCH ASSESSMENT")
    log(f"{'=' * 70}")

    vol_r68 = float(r68_port["net_ret"].std())
    vol_r93 = float(r93_port["net_ret"].std())
    vol_ratio = vol_r68 / (vol_r93 + EPS)
    log(f"\n  Vol R68: {vol_r68*10000:.1f} bps/period")
    log(f"  Vol R93: {vol_r93*10000:.1f} bps/period")
    log(f"  Vol ratio (R68/R93): {vol_ratio:.2f}")
    if abs(vol_ratio - 1.0) > 0.3:
        log(f"  ⚠ Vol mismatch > 30% — vol-scaling recommended for R99 mix")
        vol_match_needed = True
    else:
        log(f"  ✓ Vol similar — no vol-scaling needed")
        vol_match_needed = False

    # ── Summary & Conclusions ─────────────────────────────────────────────
    log(f"\n{'=' * 70}")
    log(f"  [7] SUMMARY & CONCLUSIONS")
    log(f"{'=' * 70}")

    log(f"\n  Return gap: R68 {attr_r68['total_ret_pct']:.1f}% vs R93 {attr_r93['total_ret_pct']:.1f}% (Δ={attr_r68['total_ret_pct'] - attr_r93['total_ret_pct']:.1f}%)")
    log(f"  Sharpe gap: R68 {attr_r68['sharpe']:.3f} vs R93 {attr_r93['sharpe']:.3f} (Δ={attr_r68['sharpe'] - attr_r93['sharpe']:.3f})")
    log(f"  DD gap:     R68 {attr_r68['max_dd_pct']:.1f}% vs R93 {attr_r93['max_dd_pct']:.1f}%")
    log(f"  Vol-match needed: {'YES' if vol_match_needed else 'NO'}")
    log(f"\n  Recommendation for R100 (rank ensemble): vol_match={vol_match_needed}")

    # ── Save results ──────────────────────────────────────────────────────
    result = {
        "script": "r97_attribution",
        "r68": attr_r68,
        "r93": attr_r93,
        "correlation_net_ret": round(corr, 3) if len(common) > 20 else None,
        "common_timestamps": len(common),
        "vol_ratio_r68_over_r93": round(vol_ratio, 3),
        "vol_match_needed": vol_match_needed,
        "runtime_sec": round(time.time() - t0, 1),
    }
    (RESULTS_DIR / "r97_attribution.json").write_text(
        json.dumps(result, indent=2, default=float))
    log(f"\n  Saved: results/r97_attribution.json")
    log(f"  Runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
