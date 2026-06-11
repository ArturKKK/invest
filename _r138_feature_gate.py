#!/usr/bin/env python3
"""
R138 — Feature-integrity gate on refreshed data.

Motivation: the ABSENCE of this check produced a fake -4.6 Sharpe on MLC in April
2026 (a dead cg_taker_imb after 2026-04-05). After the Jun-10 data refresh some
sources are partial (CoinGlass plan lapsed → cg_taker capped at 2026-05-06).

What this does:
  1. Rebuild full frame via _research_r68_continuous_wf.load_data().
     Assert all 31 CHAMPION_FEAT_31 present; report N/31.
  2. Restrict to SYM_35, OOS window 2026-04-26 -> 2026-06-08. For each of the 31
     features compute % of rows that are non-NaN AND non-zero. Flag >5% dead share
     EXCEPT the 6 dead-by-design CS-ranked market-level features.
  3. Special focus cg_taker_imb: find last date with valid (non-zero post-fill)
     values across symbols. If it dies before 2026-06-08, set usable_oos_end to
     that date so the OOS window is truncated to where ALL critical features live.
  4. gate_pass = (31/31 present) AND (no non-exempt feature >5% dead through
     usable_oos_end).
"""

import sys
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import _research_r68_continuous_wf as wf
from _research_r68_continuous_wf import CHAMPION_FEAT_31
from _research_round7 import SYM_35

# ── OOS window & policy constants ──────────────────────────────
OOS_START = pd.Timestamp("2026-04-26", tz="UTC")
OOS_END_REQUESTED = pd.Timestamp("2026-06-08", tz="UTC")
DEAD_THRESHOLD = 0.05  # >5% dead share flags a feature

# The 6 dead-by-design CS-ranked market-level features that are legitimately
# constant (become 0.0 for all symbols when ranked against self). NOT failures.
EXEMPT_DEAD_BY_DESIGN = {
    "pct_coins_up_12h",
    "pct_coins_up_1h",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
}

CRITICAL_FOCUS = "cg_taker_imb"


def main():
    print("=" * 78)
    print("  R138 FEATURE-INTEGRITY GATE")
    print("=" * 78)

    # ── 1. Rebuild full frame ─────────────────────────────────
    df, _regime = wf.load_data()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    present = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    missing = [f for f in CHAMPION_FEAT_31 if f not in df.columns]
    n_present = len(present)
    print(f"\n  Features present: {n_present}/31")
    if missing:
        print(f"  MISSING: {missing}")
    assert n_present == 31, f"Expected 31 CHAMPION_FEAT_31, found {n_present}; missing={missing}"

    # ── Restrict to SYM_35 ────────────────────────────────────
    df = df[df["symbol"].isin(SYM_35)].copy()

    # The 2 perma-excluded symbols (MATIC/USDT, FTM/USDT) were dropped from CG
    # coverage long ago (CG_FULL_SYMS) — their OHLCV/CG ends in 2024/2025 by design.
    CG_EXCLUDED = {"MATIC/USDT", "FTM/USDT"}

    # ── DIAGNOSIS A: OHLCV freshness per symbol ───────────────
    # The refresh was PARTIAL. A cluster of symbols is frozen well before the OOS
    # end. Any feature for a stale symbol is "dead" past its freeze date — this is
    # a real integrity failure (these are tradable majors, incl. SOL/XRP/LINK).
    ohlcv_last = df.groupby("symbol")["timestamp"].max()
    stale_syms = sorted(
        s for s in ohlcv_last.index
        if s not in CG_EXCLUDED and ohlcv_last[s] < OOS_END_REQUESTED
    )
    fresh_syms = sorted(
        s for s in ohlcv_last.index
        if s not in CG_EXCLUDED and ohlcv_last[s] >= OOS_END_REQUESTED
    )
    print("\n  OHLCV FRESHNESS (excluding perma-dead MATIC/FTM):")
    print(f"    fresh (>= {OOS_END_REQUESTED.date()}) : {len(fresh_syms)} symbols")
    print(f"    stale (<  {OOS_END_REQUESTED.date()}) : {len(stale_syms)} symbols")
    if stale_syms:
        stale_freeze = ohlcv_last[stale_syms].max()
        print(f"    STALE symbols freeze at <= {stale_freeze}:")
        print(f"      {stale_syms}")

    # ── DIAGNOSIS B (instruction #3): cg_taker_imb life ───────
    # cg_taker_imb is a daily CG feature merged left (no ffill). Determine its last
    # alive date among the FRESH-OHLCV symbols (so OHLCV staleness doesn't mask the
    # CoinGlass cap). CoinGlass plan lapsed → taker.parquet capped at cg_date
    # 2026-05-06 → ohlcv frame alive through 2026-05-07.
    cgt = df[df["symbol"].isin(fresh_syms)][["timestamp", "symbol", CRITICAL_FOCUS]].copy()
    cgt["alive"] = cgt[CRITICAL_FOCUS].notna() & (cgt[CRITICAL_FOCUS] != 0.0)
    cg_last_alive_per_sym = cgt[cgt["alive"]].groupby("symbol")["timestamp"].max()
    cg_taker_last_valid_ts = cg_last_alive_per_sym.min()
    print(f"\n  cg_taker_imb last-alive across {len(fresh_syms)} FRESH-OHLCV symbols:")
    print(f"    min(per-symbol last-alive) = {cg_taker_last_valid_ts}")
    print(f"    max(per-symbol last-alive) = {cg_last_alive_per_sym.max()}")

    # ── usable_oos_end: last date where ALL critical features are alive ───
    # Two caps bind: (1) where cg_taker_imb still has data among fresh symbols, and
    # (2) where the STALE OHLCV cluster freezes (past that date those symbols carry
    # zero live features at all). The honest usable_oos_end is the MIN of both —
    # i.e. the last date the FULL SYM_35 trading universe has live critical features.
    caps = {"cg_taker_imb": cg_taker_last_valid_ts}
    if stale_syms:
        caps["ohlcv_stale_cluster"] = ohlcv_last[stale_syms].max()
    binding_name = min(caps, key=lambda k: caps[k])
    usable_oos_end = min(caps.values()).normalize()
    truncated = usable_oos_end < OOS_END_REQUESTED
    print(f"\n  Truncation caps:")
    for k, v in caps.items():
        print(f"    {k:<22} -> {v}")
    print(f"  binding cap        : {binding_name}")
    print(f"  OOS requested end  : {OOS_END_REQUESTED.date()}")
    print(f"  usable_oos_end     : {usable_oos_end.date()}  (truncated={truncated})")

    # ── 2. Per-feature dead-share over the FULL OOS window ────
    # Coverage is measured over the REQUESTED OOS window [2026-04-26 .. 2026-06-08]
    # on the FULL SYM_35 universe (what the system actually trades). This surfaces
    # the true dead share caused by BOTH the stale OHLCV cluster and the cg_taker
    # cap — exactly the integrity failure whose absence faked -4.6 Sharpe.
    oos = df[(df["timestamp"] >= OOS_START) & (df["timestamp"] <= OOS_END_REQUESTED)].copy()
    n_rows = len(oos)
    print(f"\n  COVERAGE WINDOW [{OOS_START.date()} .. {OOS_END_REQUESTED.date()}] "
          f"(full SYM_35): {n_rows:,} rows, {oos['symbol'].nunique()} symbols")

    coverage = []
    any_nonexempt_dead = False
    for feat in CHAMPION_FEAT_31:
        col = oos[feat]
        good = col.notna() & (col != 0.0)
        pct_good = float(good.mean()) if n_rows else 0.0
        pct_dead = 1.0 - pct_good
        exempt = feat in EXEMPT_DEAD_BY_DESIGN
        is_dead = pct_dead > DEAD_THRESHOLD
        # ok = passes the gate: either alive enough OR exempt-by-design
        ok = (not is_dead) or exempt
        if is_dead and not exempt:
            any_nonexempt_dead = True
        coverage.append({
            "feature": feat,
            "pct_nonzero_nonan_in_oos": f"{pct_good * 100:.2f}%",
            "pct_dead": pct_dead,
            "exempt": exempt,
            "ok": ok,
        })

    # Sort: failures first, then exempt, then alive — readable report.
    coverage_sorted = sorted(
        coverage,
        key=lambda r: (r["ok"], -r["pct_dead"]),
    )
    print("\n  COVERAGE TABLE (sorted: failures first)")
    print(f"  {'feature':<24} {'%alive':>9} {'%dead':>8}  {'exempt':>6}  {'ok':>4}")
    print("  " + "-" * 60)
    for r in coverage_sorted:
        print(f"  {r['feature']:<24} {r['pct_nonzero_nonan_in_oos']:>9} "
              f"{r['pct_dead'] * 100:>7.2f}%  {str(r['exempt']):>6}  {str(r['ok']):>4}")

    # ── 4. Gate decision ──────────────────────────────────────
    # gate_pass requires: 31/31 present AND no non-exempt feature dead >5% over the
    # requested OOS window. It FAILS here because (a) cg_taker_imb is capped at
    # 2026-05-07 by the lapsed CoinGlass plan, and (b) 15 tradable symbols have stale
    # OHLCV frozen at 2026-04-25. The requested OOS end (2026-06-08) cannot be
    # honestly covered; usable_oos_end is the last date the full universe is alive.
    gate_pass = (
        (n_present == 31)
        and (not any_nonexempt_dead)
        and (usable_oos_end >= OOS_END_REQUESTED)
    )

    print("\n" + "=" * 78)
    print(f"  features_present : {n_present}/31")
    print(f"  cg_taker last valid : {cg_taker_last_valid_ts.date()}")
    print(f"  usable_oos_end   : {usable_oos_end.date()}")
    print(f"  GATE_PASS        : {gate_pass}")
    print("=" * 78)

    result = {
        "features_present": f"{n_present}/31",
        "missing": missing,
        "cg_taker_last_valid": str(cg_taker_last_valid_ts.date()),
        "usable_oos_end": str(usable_oos_end.date()),
        "truncated": truncated,
        "binding_cap": binding_name,
        "n_stale_ohlcv_symbols": len(stale_syms),
        "stale_ohlcv_symbols": stale_syms,
        "gate_pass": gate_pass,
        "coverage_table": [
            {
                "feature": r["feature"],
                "pct_nonzero_nonan_in_oos": r["pct_nonzero_nonan_in_oos"],
                "ok": r["ok"],
            }
            for r in coverage_sorted
        ],
    }
    print("\nRESULT_JSON_BEGIN")
    print(json.dumps(result))
    print("RESULT_JSON_END")
    return result


if __name__ == "__main__":
    main()
