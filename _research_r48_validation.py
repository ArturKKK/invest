#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R48 Phase 0 — Validation of cg_taker_imb

Tests:
  0.1  Data consistency (timestamps, uniqueness, W1 coverage)
  0.1b Timestamp sanity (open_time vs close_time, shift correctness)
  0.2  Paired block bootstrap (baseline vs +cg_taker_imb, P(ΔSharpe>0))
  0.3  Monthly stability (monthly Sharpe + monthly IC)
  0.4  Clip/winsorize sensitivity (raw vs clip vs winsorize vs rank)

Usage:
  python _research_r48_validation.py          # full Phase 0
  python _research_r48_validation.py --quick  # BTC/ETH/SOL only
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── project imports ───────────────────────────────────────────

from _research_round7 import WINDOWS, SYM_35
from _research_r30b_fixed import (
    compute_regime_extended,
    eval_with_costs,
    simulate_with_costs,
    train_ensemble,
)
from _research_r33_creative_features import FEAT_28
from _research_r35_new_features import (
    MARKET_LEVEL_FEATURES,
    add_r35_features,
    load_research_frame,
)
from _research_r47_coinglass import (
    CANONICAL_EXEC_CFG,
    CHAMPION_FEAT_30,
    CG_DIR,
    CG_FULL_SYMS,
    add_cg_features,
    compute_cg_features,
    load_cg_daily,
    make_feature_set,
)

# ── config ─────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════
#  0.1 — Data consistency check
# ═══════════════════════════════════════════════════════════════

def check_data_consistency(cg: Dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 70)
    print("  0.1 — Data Consistency Check")
    print("=" * 70)

    for name, df in cg.items():
        print(f"\n  [{name}] {len(df):,} rows, {df['symbol'].nunique()} symbols")

        # Timestamp diff per symbol (should be 1 day)
        for sym in ["BTC", "ETH"]:
            sub = df[df["symbol"] == sym].sort_values("cg_date")
            if len(sub) < 10:
                print(f"    ⚠️  {sym}: only {len(sub)} rows")
                continue
            diffs = sub["cg_date"].diff().dropna()
            vc = diffs.dt.total_seconds().value_counts().head(5)
            print(f"    {sym} timestamp diffs (top-5):")
            for secs, cnt in vc.items():
                print(f"      {secs/3600:.0f}h = {secs/86400:.1f}d → {cnt} occurrences")

        # Uniqueness of (symbol, cg_date)
        dupes = df.duplicated(subset=["symbol", "cg_date"], keep=False).sum()
        print(f"    Duplicates (symbol, cg_date): {dupes}")

    # W1 test period coverage for cg data
    w1_start = pd.Timestamp("2024-10-15", tz="UTC")
    w1_end = pd.Timestamp("2025-01-31", tz="UTC")

    if "taker" in cg:
        tk = cg["taker"]
        w1_data = tk[(tk["cg_date"] >= w1_start) & (tk["cg_date"] <= w1_end)]
        total_sym = tk["symbol"].nunique()
        w1_sym = w1_data["symbol"].nunique()
        w1_days = w1_data["cg_date"].nunique()
        expected_days = (w1_end - w1_start).days + 1
        print(f"\n  W1 test coverage (taker):")
        print(f"    Symbols: {w1_sym}/{total_sym}")
        print(f"    Days: {w1_days}/{expected_days}")
        print(f"    Coverage: {w1_days/expected_days*100:.1f}%")


# ═══════════════════════════════════════════════════════════════
#  0.1b — Timestamp sanity check
# ═══════════════════════════════════════════════════════════════

def check_timestamp_sanity(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  0.1b — Timestamp Sanity Check (open_time vs close_time)")
    print("=" * 70)

    btc = df[df["symbol"] == "BTC/USDT"].sort_values("timestamp").copy()
    if len(btc) == 0:
        print("  ⚠️  No BTC/USDT data")
        return

    # Show 10 rows around midnight to verify that timestamp=open_time
    # If 12h bars: 00:00 and 12:00 UTC
    sample = btc.head(20)[["timestamp", "open", "close", "volume"]].copy()
    sample["hour"] = sample["timestamp"].dt.hour
    sample["date"] = sample["timestamp"].dt.date

    print("\n  First 20 rows of BTC/USDT (timestamp, open, close, volume, hour):")
    for _, r in sample.iterrows():
        print(f"    {r['timestamp']}  O={r['open']:.1f}  C={r['close']:.1f}  "
              f"V={r['volume']:.0f}  hour={r['hour']}")

    # Check if timestamps are at 00:00 and 12:00 (12h bars)
    hours = btc["timestamp"].dt.hour.value_counts().sort_index()
    print(f"\n  Hour distribution (BTC): {dict(hours)}")

    if "cg_taker_imb" in btc.columns:
        # Check the cg_date assignment
        btc["_cg_date"] = btc["timestamp"].dt.normalize() - pd.Timedelta(days=1)
        # Show merge result for midnight rows
        midnight = btc[btc["timestamp"].dt.hour == 0].head(5)
        noon = btc[btc["timestamp"].dt.hour == 12].head(5)
        print(f"\n  Midnight bars (00:00 UTC):")
        for _, r in midnight.iterrows():
            print(f"    ts={r['timestamp']}  cg_date={r['_cg_date'].date()}  "
                  f"cg_taker_imb={r.get('cg_taker_imb', 'N/A')}")
        print(f"\n  Noon bars (12:00 UTC):")
        for _, r in noon.iterrows():
            print(f"    ts={r['timestamp']}  cg_date={r['_cg_date'].date()}  "
                  f"cg_taker_imb={r.get('cg_taker_imb', 'N/A')}")

    # Shift=1 correctness: verify that cg_taker_imb at time T
    # uses data from day T-1 (which covers [D-1, D-1+24h))
    print("\n  ✓ Shift rule: CG(date-1d) at OHLCV bar opening at 00:00 means "
          "we use yesterday's completed daily candle.")
    print("    If OHLCV timestamp = open_time (confirmed R47 QA): shift=1 is correct.")
    print("    If OHLCV timestamp = close_time: shift=1 would be D-2 (too conservative but safe).")


# ═══════════════════════════════════════════════════════════════
#  0.2 — Paired block bootstrap
# ═══════════════════════════════════════════════════════════════

def paired_block_bootstrap(
    baseline_rets: np.ndarray,
    candidate_rets: np.ndarray,
    block_size: int = 14,       # 14 periods × 12h = 7 days
    n_iter: int = 2000,
    seed: int = 42,
) -> Dict:
    """
    Block bootstrap for paired Sharpe difference.
    Both arrays must be same length and time-aligned.
    """
    n = len(baseline_rets)
    assert n == len(candidate_rets), f"Length mismatch: {n} vs {len(candidate_rets)}"

    # Compute observed Sharpe
    ppy = 365 * 2  # 12h bars → 730 per year
    def sharpe(r):
        return np.mean(r) / (np.std(r) + 1e-10) * np.sqrt(ppy)

    obs_base = sharpe(baseline_rets)
    obs_cand = sharpe(candidate_rets)
    obs_delta = obs_cand - obs_base

    rng = np.random.RandomState(seed)
    n_blocks = (n + block_size - 1) // block_size
    boot_deltas = []

    for _ in range(n_iter):
        # Sample block start indices
        starts = rng.randint(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block_size, n)) for s in starts])[:n]
        b_base = baseline_rets[idx]
        b_cand = candidate_rets[idx]
        boot_deltas.append(sharpe(b_cand) - sharpe(b_base))

    boot_deltas = np.array(boot_deltas)
    ci_5 = np.percentile(boot_deltas, 5)
    ci_95 = np.percentile(boot_deltas, 95)
    p_positive = np.mean(boot_deltas > 0) * 100

    return {
        "obs_sharpe_base": round(obs_base, 3),
        "obs_sharpe_cand": round(obs_cand, 3),
        "obs_delta": round(obs_delta, 3),
        "ci_5": round(ci_5, 3),
        "ci_95": round(ci_95, 3),
        "median_delta": round(np.median(boot_deltas), 3),
        "p_positive": round(p_positive, 1),
        "n_iter": n_iter,
        "block_size": block_size,
    }


def run_bootstrap(df: pd.DataFrame, regime_df: pd.DataFrame,
                  mkt_cols: List[str]) -> Dict:
    print("\n" + "=" * 70)
    print("  0.2 — Paired Block Bootstrap (baseline vs +cg_taker_imb)")
    print("=" * 70)

    # Train baseline (30f)
    print("\n  Training baseline (30f) ...")
    feats_base, no_rank_base = make_feature_set([], mkt_cols)
    preds_base = train_ensemble(df, feats_base, WINDOWS, l2=1.0, rolling=False,
                                label="bootstrap_base_30f",
                                cs_rank_exclude=no_rank_base)
    if preds_base is None or preds_base.empty:
        print("  ❌ Baseline training failed")
        return {}

    # Train candidate (31f = +cg_taker_imb)
    print("  Training candidate (31f = +cg_taker_imb) ...")
    feats_cand, no_rank_cand = make_feature_set(["cg_taker_imb"], mkt_cols)
    preds_cand = train_ensemble(df, feats_cand, WINDOWS, l2=1.0, rolling=False,
                                label="bootstrap_cand_31f",
                                cs_rank_exclude=no_rank_cand)
    if preds_cand is None or preds_cand.empty:
        print("  ❌ Candidate training failed")
        return {}

    # Simulate both → get per-period returns
    port_base = simulate_with_costs(preds_base, regime_df, CANONICAL_EXEC_CFG)
    port_cand = simulate_with_costs(preds_cand, regime_df, CANONICAL_EXEC_CFG)

    if port_base.empty or port_cand.empty:
        print("  ❌ Simulation produced empty results")
        return {}

    # Align timestamps (inner join)
    merged = port_base[["timestamp", "portfolio_ret"]].merge(
        port_cand[["timestamp", "portfolio_ret"]],
        on="timestamp", suffixes=("_base", "_cand"),
    )
    print(f"  Aligned periods: {len(merged)}")

    base_rets = merged["portfolio_ret_base"].values
    cand_rets = merged["portfolio_ret_cand"].values

    # Run bootstrap
    result = paired_block_bootstrap(base_rets, cand_rets, block_size=14, n_iter=2000)

    print(f"\n  Observed Sharpe:")
    print(f"    Baseline (30f):              {result['obs_sharpe_base']:+.3f}")
    print(f"    Candidate (+cg_taker_imb):   {result['obs_sharpe_cand']:+.3f}")
    print(f"    Δ Sharpe:                    {result['obs_delta']:+.3f}")
    print(f"\n  Bootstrap (N={result['n_iter']}, block={result['block_size']}×12h=7d):")
    print(f"    P(Δ Sharpe > 0) = {result['p_positive']:.1f}%")
    print(f"    90% CI: [{result['ci_5']:+.3f}, {result['ci_95']:+.3f}]")
    print(f"    Median Δ:  {result['median_delta']:+.3f}")

    if result["p_positive"] >= 90:
        print("    ✅ PASS — improvement is statistically robust")
    elif result["p_positive"] >= 80:
        print("    ⚠️  MARGINAL — moderate evidence of improvement")
    else:
        print("    ❌ FAIL — improvement is NOT statistically robust")

    return result


# ═══════════════════════════════════════════════════════════════
#  0.3 — Monthly stability
# ═══════════════════════════════════════════════════════════════

def run_monthly_stability(df: pd.DataFrame, regime_df: pd.DataFrame,
                          mkt_cols: List[str]) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  0.3 — Monthly Stability (Sharpe + IC)")
    print("=" * 70)

    # Train candidate
    print("\n  Training candidate (31f) for monthly analysis ...")
    feats, no_rank = make_feature_set(["cg_taker_imb"], mkt_cols)
    preds = train_ensemble(df, feats, WINDOWS, l2=1.0, rolling=False,
                           label="monthly_31f", cs_rank_exclude=no_rank)
    if preds is None or preds.empty:
        print("  ❌ Training failed")
        return pd.DataFrame()

    port = simulate_with_costs(preds, regime_df, CANONICAL_EXEC_CFG)
    if port.empty:
        print("  ❌ Empty simulation")
        return pd.DataFrame()

    # Monthly portfolio returns
    port_ts = port.set_index("timestamp")
    monthly_ret = port_ts["portfolio_ret"].resample("ME").sum()
    monthly_cnt = port_ts["portfolio_ret"].resample("ME").count()

    # Monthly IC for cg_taker_imb
    monthly_ic_rows = []
    preds_with_feat = preds.merge(
        df[["timestamp", "symbol", "cg_taker_imb"]],
        on=["timestamp", "symbol"],
        how="left",
    )
    for month, grp in preds_with_feat.groupby(preds_with_feat["timestamp"].dt.to_period("M")):
        sub = grp[["cg_taker_imb", "fwd_ret"]].dropna()
        if len(sub) < 30:
            continue
        ic = stats.spearmanr(sub["cg_taker_imb"], sub["fwd_ret"])[0]
        monthly_ic_rows.append({"month": str(month), "ic": ic})

    monthly_ic = pd.DataFrame(monthly_ic_rows)

    # Print table
    print(f"\n  {'Month':<10} {'Ret':>8} {'n_periods':>10}")
    print(f"  {'─'*10} {'─'*8} {'─'*10}")
    for idx, (ret, cnt) in enumerate(zip(monthly_ret.values, monthly_cnt.values)):
        m = monthly_ret.index[idx].strftime("%Y-%m")
        print(f"  {m:<10} {ret*100:>+7.2f}% {cnt:>10}")

    total_months = len(monthly_ret)
    positive_months = (monthly_ret > 0).sum()
    print(f"\n  Win months: {positive_months}/{total_months} ({positive_months/total_months*100:.0f}%)")

    # Check if improvement comes from 1-2 months
    sorted_rets = monthly_ret.sort_values(ascending=False)
    top2_contrib = sorted_rets.head(2).sum() / sorted_rets.sum() * 100 if sorted_rets.sum() != 0 else 0
    print(f"  Top-2 months contribution: {top2_contrib:.0f}% of total return")
    if top2_contrib > 60:
        print("  ⚠️  WARNING: Improvement concentrated in 1-2 months!")
    else:
        print("  ✅ Returns distributed across months")

    if not monthly_ic.empty:
        print(f"\n  Monthly IC (cg_taker_imb → fwd_ret):")
        print(f"  {'Month':<10} {'IC':>8}")
        print(f"  {'─'*10} {'─'*8}")
        for _, r in monthly_ic.iterrows():
            print(f"  {r['month']:<10} {r['ic']:>+7.3f}")
        mean_ic = monthly_ic["ic"].mean()
        ic_pos = (monthly_ic["ic"] > 0).sum()
        print(f"\n  Mean monthly IC: {mean_ic:+.3f}  ({ic_pos}/{len(monthly_ic)} positive)")

    return monthly_ret


# ═══════════════════════════════════════════════════════════════
#  0.4 — Clip/winsorize sensitivity
# ═══════════════════════════════════════════════════════════════

def apply_clip(df: pd.DataFrame, col: str = "cg_taker_imb") -> pd.DataFrame:
    """Clip to [-0.98, 0.98]."""
    out = df.copy()
    out[col] = out[col].clip(-0.98, 0.98)
    return out


def apply_winsorize(df: pd.DataFrame, col: str = "cg_taker_imb",
                    pct: float = 0.01) -> pd.DataFrame:
    """Winsorize at 1% / 99%."""
    out = df.copy()
    valid = out[col].dropna()
    if len(valid) < 100:
        return out
    lo = valid.quantile(pct)
    hi = valid.quantile(1 - pct)
    out[col] = out[col].clip(lo, hi)
    return out


def apply_gaussian_rank(df: pd.DataFrame, col: str = "cg_taker_imb") -> pd.DataFrame:
    """Rolling Gaussian rank transform per symbol (window=90 bars = 90 days for 1d data)."""
    out = df.copy()

    def _rank_transform(s: pd.Series) -> pd.Series:
        ranked = s.rolling(90, min_periods=30).apply(
            lambda x: stats.rankdata(x)[-1] / len(x), raw=True
        )
        return stats.norm.ppf(ranked.clip(0.01, 0.99))

    out[col] = out.groupby("symbol")[col].transform(_rank_transform)
    return out


def run_clip_sensitivity(df: pd.DataFrame, regime_df: pd.DataFrame,
                         mkt_cols: List[str]) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  0.4 — Clip/Winsorize/Rank Sensitivity")
    print("=" * 70)

    variants = {
        "raw": df,
        "clip_98": apply_clip(df),
        "winsorize_1pct": apply_winsorize(df),
        "gaussian_rank": apply_gaussian_rank(df),
    }

    rows = []
    for vname, vdf in variants.items():
        print(f"\n  [{vname}] Training WF ...")
        feats, no_rank = make_feature_set(["cg_taker_imb"], mkt_cols)
        preds = train_ensemble(vdf, feats, WINDOWS, l2=1.0, rolling=False,
                               label=f"clip_{vname}",
                               cs_rank_exclude=no_rank)
        if preds is None or preds.empty:
            print(f"    ⚠️  {vname}: no predictions")
            continue

        for window in ["W1", "W2", "W3", "ALL"]:
            subset = preds if window == "ALL" else preds[preds["window"] == window]
            port = simulate_with_costs(subset, regime_df, CANONICAL_EXEC_CFG)
            m = eval_with_costs(port, f"{vname}_{window}")
            rows.append({
                "variant": vname,
                "window": window,
                "sharpe": m["sharpe"],
                "sharpe_gross": m["sharpe_gross"],
                "max_dd": m.get("max_dd_pct", 0),
                "cost_pct": m.get("total_cost_pct", 0),
            })

    summary = pd.DataFrame(rows)
    if summary.empty:
        print("  ❌ No results")
        return summary

    # Print pivoted table
    pivot = summary.pivot(index="variant", columns="window", values="sharpe")
    for col in ["W1", "W2", "W3", "ALL"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["W1", "W2", "W3", "ALL"]]

    print(f"\n  {'Variant':<20} {'W1':>7} {'W2':>7} {'W3':>7} {'ALL':>7}")
    print(f"  {'─'*20} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
    for vname in ["raw", "clip_98", "winsorize_1pct", "gaussian_rank"]:
        if vname in pivot.index:
            r = pivot.loc[vname]
            flag = " ✅" if r["ALL"] >= 1.0 else ""
            print(f"  {vname:<20} {r['W1']:>+6.2f} {r['W2']:>+6.2f} {r['W3']:>+6.2f} "
                  f"{r['ALL']:>+6.2f}{flag}")

    raw_all = pivot.loc["raw", "ALL"] if "raw" in pivot.index else 0
    fragile = all(
        (pivot.loc[v, "ALL"] < raw_all * 0.5 if v in pivot.index else True)
        for v in ["clip_98", "winsorize_1pct"]
    )
    if fragile:
        print("\n  ❌ WARNING: signal breaks under clip/winsorize → fragile!")
    else:
        print("\n  ✅ Signal robust to preprocessing variants")

    return summary


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main(quick: bool = False) -> None:
    print("=" * 80)
    print("R48 Phase 0 — VALIDATION OF cg_taker_imb")
    print("=" * 80)

    # ── Load CG daily features ─────────────────────────────────
    print("\n[1] Loading CoinGlass daily data ...")
    cg = load_cg_daily()
    cg_feats_daily = compute_cg_features(cg)
    if cg_feats_daily.empty:
        print("❌ No CG features — aborting")
        return

    # ── 0.1 — Data consistency ─────────────────────────────────
    check_data_consistency(cg)

    # ── Load research frame ─────────────────────────────────────
    print("\n[2] Loading research frame (OHLCV + FEAT_30 + R35) ...")
    df, regime_df = load_research_frame()
    df, _ = add_r35_features(df)
    regime_df = regime_df.sort_index()
    print(f"  Base frame: {len(df):,} rows × {len(df.columns)} cols")

    if quick:
        print("  ⚡ Quick mode: BTC/ETH/SOL only ...")
        df = df[df["symbol"].isin(["BTC/USDT", "ETH/USDT", "SOL/USDT"])].copy()

    # ── Merge CG features ──────────────────────────────────────
    print("\n[3] Merging CG features ...")
    df, per_sym_cols, mkt_cols = add_cg_features(df, cg_feats_daily)
    print(f"  CG features merged: {len(per_sym_cols)} per-sym + {len(mkt_cols)} mkt")
    cg_coverage = df["cg_taker_imb"].notna().mean() * 100
    print(f"  cg_taker_imb coverage: {cg_coverage:.1f}% non-null")

    # ── 0.1b — Timestamp sanity ─────────────────────────────────
    check_timestamp_sanity(df)

    # ── 0.2 — Paired block bootstrap ────────────────────────────
    boot_result = run_bootstrap(df, regime_df, mkt_cols)

    if boot_result and boot_result.get("p_positive", 0) < 80:
        print("\n  ⛔ STOP — cg_taker_imb improvement is NOT statistically robust!")
        print("  Remaining phases may not be worth running.")
        # Still continue to gather info, but flag it

    # ── 0.3 — Monthly stability ─────────────────────────────────
    run_monthly_stability(df, regime_df, mkt_cols)

    # ── 0.4 — Clip sensitivity ──────────────────────────────────
    run_clip_sensitivity(df, regime_df, mkt_cols)

    print("\n" + "=" * 80)
    print("R48 Phase 0 — COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(quick=args.quick)
