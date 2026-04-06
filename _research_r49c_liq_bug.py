#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R49c — Liq-Weighted Bug Check

Diagnoses why simulate_liq_weighted gives ALL=0.53 vs uniform ALL=1.31.

Checks:
  1. Volume data alignment (timestamps, coverage)
  2. Weight distribution (are BTC/ETH always hitting the 2.0 cap?)
  3. Per-symbol signal IC vs log(volume) — is CS-alpha inversely correlated with size?
  4. Realized returns by tier when selected (do large caps underperform given same rank?)
  5. Verify formula: clip(log_v / med, 0.5, 2.0) — correct direction?

Conclusion: bug or genuine signal?
"""

from __future__ import annotations

import warnings
from typing import Set

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

from _research_round7 import WINDOWS, SYM_35
from _research_r30b_fixed import (
    eval_with_costs,
    simulate_with_costs,
    train_ensemble,
)
from _research_r35_new_features import add_r35_features, load_research_frame
from _research_r47_coinglass import (
    CANONICAL_EXEC_CFG,
    add_cg_features,
    compute_cg_features,
    load_cg_daily,
    make_feature_set,
)
from _research_r48_cost import (
    TIER1_SYMS,
    TIER2_SYMS,
    TIER3_SYMS,
    simulate_liq_weighted,
    simulate_with_hybrid_costs,
)

# ─────────────────────────────────────────────────────────────


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════
#  CHECK 1 — Volume data alignment
# ══════════════════════════════════════════════════════════════

def check_volume_alignment(preds: pd.DataFrame, volume_df: pd.DataFrame) -> None:
    section("CHECK 1 — Volume data alignment")

    pred_ts = set(preds["timestamp"].unique())
    vol_ts = set(volume_df["timestamp"].unique())

    overlap = pred_ts & vol_ts
    missing = pred_ts - vol_ts

    print(f"  Pred timestamps  : {len(pred_ts):,}")
    print(f"  Volume timestamps: {len(vol_ts):,}")
    print(f"  Overlap          : {len(overlap):,} ({100*len(overlap)/len(pred_ts):.1f}%)")
    print(f"  Missing in vol   : {len(missing):,}")

    if missing:
        sample = sorted(missing)[:5]
        print(f"  Sample missing   : {sample}")
        print(f"  ⚠️  Timestamps mismatch — weights fallback to 1.0 for {len(missing):,} bars")
        print(f"  → Liq-weighted ≈ equal-weight for those bars")
    else:
        print(f"  ✅ All timestamps aligned")

    # Check symbol coverage
    for ts in sorted(overlap)[:3]:
        vol_grp = volume_df[volume_df["timestamp"] == ts]
        pred_grp = preds[preds["timestamp"] == ts]
        pred_syms = set(pred_grp["symbol"].unique())
        vol_syms = set(vol_grp["symbol"].unique())
        missing_syms = pred_syms - vol_syms
        if missing_syms:
            print(f"  ⚠️  ts={ts}: {len(missing_syms)} symbols missing volume: {list(missing_syms)[:5]}")


# ══════════════════════════════════════════════════════════════
#  CHECK 2 — Weight distribution
# ══════════════════════════════════════════════════════════════

def check_weight_distribution(preds: pd.DataFrame, volume_df: pd.DataFrame) -> None:
    section("CHECK 2 — Weight distribution (do BTC/ETH always hit 2.0 cap?)")

    vol_grouped = {ts: grp for ts, grp in volume_df.groupby("timestamp")}
    grouped = {ts: grp for ts, grp in preds.groupby("timestamp")}

    n_long = CANONICAL_EXEC_CFG.get("n_long", 6)
    n_short = CANONICAL_EXEC_CFG.get("n_short", 3)
    rebal_hours = CANONICAL_EXEC_CFG.get("rebal_hours", 12)

    ts_sorted = sorted(preds["timestamp"].unique())
    rebal_ts = ts_sorted[::rebal_hours]

    weight_records = []

    for ts in rebal_ts[:200]:  # first 200 rebal bars for speed
        if ts not in grouped:
            continue
        grp = grouped[ts].copy()
        n = len(grp)
        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl + ns == 0:
            continue

        grp["pred_rank"] = grp["pred"].rank(ascending=False)
        new_longs = set(grp[grp["pred_rank"] <= nl]["symbol"].tolist())
        new_shorts = set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist())
        selected = new_longs | new_shorts

        vol_grp = vol_grouped.get(ts)
        if vol_grp is None:
            continue
        sym_vol = dict(zip(vol_grp["symbol"], vol_grp["volume"]))

        all_log_vols = [np.log1p(sym_vol.get(s, 1)) for s in selected]
        if not all_log_vols:
            continue
        med = np.median(all_log_vols)

        for sym in selected:
            v = sym_vol.get(sym, 0)
            if v <= 0:
                w = 1.0
            else:
                log_v = np.log1p(v)
                w = float(np.clip(log_v / (med + 1e-10), 0.5, 2.0))

            tier = "T1" if sym in TIER1_SYMS else ("T3" if sym in TIER3_SYMS else "T2")
            side = "long" if sym in new_longs else "short"
            weight_records.append({
                "timestamp": ts,
                "symbol": sym,
                "weight": w,
                "tier": tier,
                "side": side,
                "log_v": np.log1p(v) if v > 0 else 0,
                "median_log_v": med,
            })

    if not weight_records:
        print("  ⚠️  No weight records — volume data missing?")
        return

    wdf = pd.DataFrame(weight_records)

    print(f"\n  Weight statistics by tier:")
    tier_stats = wdf.groupby("tier")["weight"].agg(["mean", "median", "min", "max", "count"])
    print(tier_stats.to_string())

    print(f"\n  Fraction hitting upper cap (2.0): {(wdf['weight'] >= 1.99).mean():.1%}")
    print(f"  Fraction hitting lower cap (0.5): {(wdf['weight'] <= 0.51).mean():.1%}")

    # T1 vs T2/T3 weight ratio
    t1_mean = wdf[wdf["tier"] == "T1"]["weight"].mean()
    t23_mean = wdf[wdf["tier"] != "T1"]["weight"].mean()
    print(f"\n  Avg weight T1   : {t1_mean:.3f}")
    print(f"  Avg weight T2/T3: {t23_mean:.3f}")
    print(f"  T1/T23 ratio     : {t1_mean/t23_mean:.2f}×")

    print(f"\n  → T1 coins appear {t1_mean/t23_mean:.1f}× more in portfolio by weight vs T2/T3")


# ══════════════════════════════════════════════════════════════
#  CHECK 3 — Per-symbol signal IC vs log(volume)
# ══════════════════════════════════════════════════════════════

def check_ic_vs_liquidity(preds: pd.DataFrame, volume_df: pd.DataFrame) -> None:
    section("CHECK 3 — Per-symbol signal IC vs log(volume)")

    # Compute per-symbol IC
    sym_records = []
    for sym, grp in preds.groupby("symbol"):
        grp = grp.dropna(subset=["pred", "fwd_ret"])
        if len(grp) < 20:
            continue
        ic = float(stats.spearmanr(grp["pred"], grp["fwd_ret"])[0])

        # Average volume
        vol_grp = volume_df[volume_df["symbol"] == sym]
        avg_vol = vol_grp["volume"].mean() if len(vol_grp) > 0 else 0
        log_avg_vol = np.log1p(avg_vol) if avg_vol > 0 else 0

        tier = "T1" if sym in TIER1_SYMS else ("T3" if sym in TIER3_SYMS else "T2")
        sym_records.append({
            "symbol": sym,
            "ic": ic,
            "log_avg_vol": log_avg_vol,
            "avg_vol": avg_vol,
            "n_obs": len(grp),
            "tier": tier,
        })

    if not sym_records:
        print("  ⚠️  No IC data")
        return

    sym_df = pd.DataFrame(sym_records).sort_values("ic", ascending=False)

    # Correlation IC vs log_vol
    valid = sym_df[(sym_df["log_avg_vol"] > 0) & sym_df["ic"].notna()]
    if len(valid) >= 5:
        corr, pval = stats.spearmanr(valid["ic"], valid["log_avg_vol"])
        print(f"\n  Spearman corr(IC, log_volume) = {corr:+.3f}  (p={pval:.3f})")
        if corr < -0.2:
            print(f"  ⚠️  NEGATIVE correlation: CS alpha is HIGHER for LESS liquid coins")
            print(f"  → Liq-weighting (upweight liquid) HURTS portfolio alpha")
            print(f"  → This is NOT a bug — it's genuine signal distribution")
        elif corr > 0.2:
            print(f"  ✅ Positive correlation: signal works better for liquid coins")
            print(f"  → Liq-weighting HELPS — if it hurts, there might be a bug")
        else:
            print(f"  → Weak correlation, liq-weighting effect should be neutral")

    print(f"\n  Top-5 IC symbols:")
    print(sym_df[["symbol", "ic", "log_avg_vol", "tier", "n_obs"]].head(5).to_string(index=False))

    print(f"\n  Bottom-5 IC symbols:")
    print(sym_df[["symbol", "ic", "log_avg_vol", "tier", "n_obs"]].tail(5).to_string(index=False))

    print(f"\n  IC by tier:")
    tier_ic = sym_df.groupby("tier")["ic"].agg(["mean", "median", "count"])
    print(tier_ic.to_string())


# ══════════════════════════════════════════════════════════════
#  CHECK 4 — Realized return by tier when selected
# ══════════════════════════════════════════════════════════════

def check_realized_return_by_tier(preds: pd.DataFrame) -> None:
    section("CHECK 4 — Realized fwd_ret by tier when selected as long/short")

    n_long = CANONICAL_EXEC_CFG.get("n_long", 6)
    n_short = CANONICAL_EXEC_CFG.get("n_short", 3)
    rebal_hours = CANONICAL_EXEC_CFG.get("rebal_hours", 12)
    trend_cutoff = CANONICAL_EXEC_CFG.get("trend_cutoff", 0.9)

    ts_sorted = sorted(preds["timestamp"].unique())
    rebal_ts = ts_sorted[::rebal_hours]
    grouped = {ts: grp for ts, grp in preds.groupby("timestamp")}

    selection_records = []

    from _research_r35_new_features import load_research_frame
    _, regime_df = load_research_frame()
    regime_df = regime_df.sort_index()

    for ts in rebal_ts:
        if ts not in grouped:
            continue
        if ts in regime_df.index and regime_df.loc[ts].get("trend_strength", 0) > trend_cutoff:
            continue
        grp = grouped[ts].copy()
        n = len(grp)
        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl + ns == 0:
            continue

        grp["pred_rank"] = grp["pred"].rank(ascending=False)
        longs = grp[grp["pred_rank"] <= nl]
        shorts = grp[grp["pred_rank"] > (n - ns)]

        for _, row in longs.iterrows():
            sym = row["symbol"]
            tier = "T1" if sym in TIER1_SYMS else ("T3" if sym in TIER3_SYMS else "T2")
            selection_records.append({
                "symbol": sym, "tier": tier, "side": "long", "fwd_ret": row["fwd_ret"],
            })
        for _, row in shorts.iterrows():
            sym = row["symbol"]
            tier = "T1" if sym in TIER1_SYMS else ("T3" if sym in TIER3_SYMS else "T2")
            selection_records.append({
                "symbol": sym, "tier": tier, "side": "short", "fwd_ret": row["fwd_ret"],
            })

    if not selection_records:
        print("  ⚠️  No selection data")
        return

    sel_df = pd.DataFrame(selection_records)

    print(f"\n  Long selections: {len(sel_df[sel_df['side']=='long']):,}")
    print(f"  Short selections: {len(sel_df[sel_df['side']=='short']):,}")

    print(f"\n  LONG fwd_ret by tier:")
    long_by_tier = sel_df[sel_df["side"] == "long"].groupby("tier")["fwd_ret"].agg(
        ["mean", "median", "std", "count"]
    )
    print(long_by_tier.to_string())

    print(f"\n  SHORT fwd_ret by tier (want NEGATIVE = short is correct):")
    short_by_tier = sel_df[sel_df["side"] == "short"].groupby("tier")["fwd_ret"].agg(
        ["mean", "median", "std", "count"]
    )
    print(short_by_tier.to_string())

    # Net contribution per tier (long return - short_return directional)
    print(f"\n  INTERPRETATION:")
    for tier in ["T1", "T2", "T3"]:
        long_r = sel_df[(sel_df["tier"] == tier) & (sel_df["side"] == "long")]["fwd_ret"].mean()
        short_r = sel_df[(sel_df["tier"] == tier) & (sel_df["side"] == "short")]["fwd_ret"].mean()
        net = long_r - short_r  # want positive
        n = len(sel_df[sel_df["tier"] == tier])
        if not np.isnan(long_r):
            print(
                f"    {tier}: long_ret={long_r:+.4f}, short_ret={short_r:+.4f}, "
                f"net={net:+.4f}  (n={n})"
            )

    # Conclusion
    t1_long = sel_df[(sel_df["tier"] == "T1") & (sel_df["side"] == "long")]["fwd_ret"].mean()
    t23_long = sel_df[(sel_df["tier"] != "T1") & (sel_df["side"] == "long")]["fwd_ret"].mean()
    if not np.isnan(t1_long) and not np.isnan(t23_long):
        print(f"\n  T1 long mean return : {t1_long:+.4f}")
        print(f"  T2/T3 long mean return: {t23_long:+.4f}")
        if t1_long < t23_long:
            print(f"  ⚠️  T1 (liquid) coins have LOWER mean return when selected as longs")
            print(f"  → Liq-weighting genuinely hurts — upweighting T1 reduces alpha")
        else:
            print(f"  ✅ T1 coins have higher return when selected — liq-weighting should help")


# ══════════════════════════════════════════════════════════════
#  CHECK 5 — Side-by-side simulation with weight trace
# ══════════════════════════════════════════════════════════════

def check_simulation_with_trace(preds: pd.DataFrame, volume_df: pd.DataFrame, regime_df) -> None:
    section("CHECK 5 — Equal vs Liq-weighted simulation (ALL window)")

    port_eq = simulate_with_costs(preds, regime_df, CANONICAL_EXEC_CFG)
    port_lw = simulate_liq_weighted(preds, regime_df, CANONICAL_EXEC_CFG, volume_df)

    m_eq = eval_with_costs(port_eq, "equal")
    m_lw = eval_with_costs(port_lw, "liqwt")

    print(f"\n  Equal-weight Sharpe : {m_eq['sharpe']:+.3f}")
    print(f"  Liq-weighted Sharpe : {m_lw['sharpe']:+.3f}")
    print(f"  Delta Sharpe        : {m_lw['sharpe'] - m_eq['sharpe']:+.3f}")

    if len(port_eq) > 0 and len(port_lw) > 0:
        print(f"\n  Equal  cost%: {m_eq.get('total_cost_pct', 0):.1f}%")
        print(f"  Liqwt  cost%: {m_lw.get('total_cost_pct', 0):.1f}%")

        eq_cum = (1 + port_eq["portfolio_ret"].fillna(0)).cumprod().iloc[-1] - 1
        lw_cum = (1 + port_lw["portfolio_ret"].fillna(0)).cumprod().iloc[-1] - 1
        print(f"\n  Equal cumret: {100*eq_cum:+.1f}%")
        print(f"  Liqwt cumret: {100*lw_cum:+.1f}%")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("R49c — Liq-Weighted Bug Diagnostic")
    print("=" * 70)

    # Load data (same as R48 Phase 3)
    print("\n[1] Loading data ...")
    cg = load_cg_daily()
    cg_feats_daily = compute_cg_features(cg)
    df, _ = load_research_frame()
    df, _ = add_r35_features(df)
    df, per_sym_cols, mkt_cols = add_cg_features(df, cg_feats_daily)

    print("[2] Training champion_31f predictions ...")
    feats, no_rank = make_feature_set(["cg_taker_imb"], mkt_cols)
    preds = train_ensemble(df, feats, WINDOWS, l2=1.0, rolling=False,
                           label="r49c", cs_rank_exclude=no_rank)
    if preds is None or preds.empty:
        print("❌ Training failed")
        return

    print(f"  Predictions: {len(preds):,} rows, {preds['timestamp'].nunique()} timestamps")

    from _research_r35_new_features import load_research_frame as _lrf
    _, regime_df = _lrf()
    regime_df = regime_df.sort_index()
    volume_df = df[["timestamp", "symbol", "volume"]].copy()

    # Run all checks
    check_volume_alignment(preds, volume_df)
    check_weight_distribution(preds, volume_df)
    check_ic_vs_liquidity(preds, volume_df)
    check_realized_return_by_tier(preds)
    check_simulation_with_trace(preds, volume_df, regime_df)

    # ── VERDICT ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print("""
  Based on above checks, one of:

  [BUG] If Check 1 shows timestamp mismatch >50%:
    → weights fall back to 1.0 for most bars → result unreliable
    Fix: align timestamps before passing volume_df

  [BUG] If Check 2 shows T1 consistently at 2.0 cap AND
        Check 4 shows T1 realized return ≥ T2/T3:
    → formula is too aggressive (2.0 cap too high)
    Fix: reduce cap to 1.3 or use log-log scaling

  [GENUINE] If Check 3 shows negative corr(IC, log_volume):
    → CS alpha lives in mid/small caps, liq-weighting kills it
    Conclusion: discard liq-weighted, document as proven useless
    Action: update MEGA_PROMPT.md "proven useless" list

  Recommendation: See output above to determine which case applies.
""")


if __name__ == "__main__":
    main()
