#!/usr/bin/env python3
"""
R104 — Conservative Tie-Break Ensemble

Use R93 as a tie-breaker within R68's candidate pools, without degrading champion.

Algorithm:
1. By R68 score: select long_pool (top M) and short_pool (bottom N)
2. Within long_pool: rank by R93 raw_prob → final longs = top 4
3. Within short_pool: rank by R93 raw_prob → final shorts = bottom 2
4. Execution and sizing — same as R68.

Grid: M ∈ {6, 8, 10} × N ∈ {3, 4, 5} = 9 configs
Bootstrap: block=10, N=1000
"""

import json, sys, time, warnings
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EPS = 1e-10
PPY = 2 * 365


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sharpe_ann(rets, ppy=PPY):
    if len(rets) < 2:
        return 0.0
    eq = (1 + rets).cumprod()
    r = eq.pct_change().dropna()
    return float(r.mean() / (r.std() + EPS) * np.sqrt(ppy))


def max_dd(rets):
    eq = (1 + rets).cumprod()
    return float((eq / eq.cummax() - 1).min())


def calmar(rets, ppy=PPY):
    s = sharpe_ann(rets, ppy)
    dd = abs(max_dd(rets))
    return s / (dd + EPS) if dd > 0 else 0.0


def block_bootstrap(rets_base, rets_exp, n_boot=1000, block=10, seed=42):
    rng = np.random.default_rng(seed)
    n = min(len(rets_base), len(rets_exp))
    rb = np.array(rets_base[:n], dtype=float)
    re = np.array(rets_exp[:n], dtype=float)
    n_blocks = max(1, n // block)

    def _sh(r):
        if len(r) < 2:
            return 0.0
        eq = (1 + r).cumprod()
        p = np.diff(eq) / eq[:-1]
        if len(p) < 2 or np.std(p) < EPS:
            return 0.0
        return float(np.mean(p) / np.std(p) * np.sqrt(PPY))

    def _calmar(r):
        s = _sh(r)
        eq = (1 + r).cumprod()
        dd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
        return s / (abs(dd) + EPS) if abs(dd) > 0 else 0.0

    sb, se, cb, ce = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, max(1, n - block + 1), size=n_blocks)
        block_idx = np.concatenate([np.arange(i, min(i + block, n)) for i in idx])[:n]
        sb.append(_sh(rb[block_idx]))
        se.append(_sh(re[block_idx]))
        cb.append(_calmar(rb[block_idx]))
        ce.append(_calmar(re[block_idx]))

    sb, se = np.array(sb), np.array(se)
    cb, ce = np.array(cb), np.array(ce)
    ds = se - sb
    dc = ce - cb

    return {
        "p_sharpe_better": round(float((se > sb).mean()), 3),
        "median_delta_sharpe": round(float(np.median(ds)), 4),
        "p5_delta_sharpe": round(float(np.percentile(ds, 5)), 4),
        "p95_delta_sharpe": round(float(np.percentile(ds, 95)), 4),
        "base_sharpe_med": round(float(np.median(sb)), 4),
        "exp_sharpe_med": round(float(np.median(se)), 4),
        "p_calmar_better": round(float((ce > cb).mean()), 3),
        "median_delta_calmar": round(float(np.median(dc)), 4),
    }


# ── Custom simulate with pool+tiebreak selection ─────────────────────────

from _research_r68_continuous_wf import PROD_CFG, _cost_for_sym


def simulate_tiebreak(merged, regime_df, pool_long, pool_short,
                      n_long_final=4, n_short_final=2, cfg=None):
    """
    Simulate with pool+tiebreak selection:
    1. EMA smooth R68 pred (standard R68 behavior)
    2. Select long_pool = top-pool_long by R68 pred
    3. Within long_pool, rank by R93 raw_prob → take top n_long_final
    4. Select short_pool = bottom-pool_short by R68 pred
    5. Within short_pool, rank by R93 raw_prob → take bottom n_short_final
    6. Execution same as R68 (costs, funding, hysteresis at final level)
    """
    if cfg is None:
        cfg = PROD_CFG
    trend_cutoff = cfg["trend_cutoff"]
    rebal_hours = cfg["rebal_hours"]
    ema_alpha = cfg.get("ema_alpha", None)
    hysteresis = cfg.get("hysteresis", 0)
    funding_per_12h = 0.00008

    all_rets = []
    prev_longs: Set[str] = set()
    prev_shorts: Set[str] = set()
    prev_preds: Dict[str, float] = {}

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}
    rebal_timestamps = timestamps_sorted[::rebal_hours]

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        if trend_str > trend_cutoff:
            continue
        grp = grouped[ts].copy()
        n = len(grp)

        nl_pool = min(pool_long, n // 3)
        ns_pool = min(pool_short, n // 3)
        nl_final = min(n_long_final, nl_pool)
        ns_final = min(n_short_final, ns_pool)
        if nl_final == 0 and ns_final == 0:
            continue

        exposure = 1.0
        dyn_threshold = cfg.get("dyn_threshold", 0.5)
        if dyn_threshold is not None and trend_str > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) /
                           (trend_cutoff - dyn_threshold + 1e-10) * 0.5)

        # EMA smooth R68 pred (same as R68)
        if ema_alpha is not None and ema_alpha < 1.0:
            for idx, r in grp.iterrows():
                sym = r["symbol"]
                raw_pred = r["pred_68"]
                smoothed = ema_alpha * raw_pred + (1 - ema_alpha) * prev_preds.get(sym, raw_pred)
                prev_preds[sym] = smoothed
                grp.at[idx, "pred_68"] = smoothed

        # Step 1: rank by EMA'd R68 pred → select pools
        grp["r68_rank"] = grp["pred_68"].rank(ascending=False)

        # Step 2: within pools, rank by R93 raw_prob
        long_pool = grp[grp["r68_rank"] <= nl_pool].copy()
        short_pool = grp[grp["r68_rank"] > (n - ns_pool)].copy()

        if len(long_pool) > 0:
            # Higher R93 prob → more bullish → rank descending → top = best longs
            long_pool["r93_rank"] = long_pool["raw_prob_93"].rank(ascending=False)
            final_longs = set(long_pool[long_pool["r93_rank"] <= nl_final]["symbol"].tolist())
        else:
            final_longs = set()

        if len(short_pool) > 0:
            # Lower R93 prob → more bearish → rank ascending → top = best shorts
            short_pool["r93_rank"] = short_pool["raw_prob_93"].rank(ascending=True)
            final_shorts = set(short_pool[short_pool["r93_rank"] <= ns_final]["symbol"].tolist())
        else:
            final_shorts = set()

        # Apply hysteresis at final selection level
        if hysteresis > 0 and (prev_longs or prev_shorts):
            # Keep incumbents if they're in the pool
            long_pool_syms = set(long_pool["symbol"].tolist())
            short_pool_syms = set(short_pool["symbol"].tolist())

            kept_longs = prev_longs & long_pool_syms
            kept_shorts = prev_shorts & short_pool_syms

            # Merge: keep incumbents that are still in pool, fill remaining from tiebreak
            if len(kept_longs) < nl_final:
                new_from_tb = final_longs - kept_longs
                needed = nl_final - len(kept_longs)
                # Take top by R93 from remaining
                remaining = long_pool[long_pool["symbol"].isin(new_from_tb)].nsmallest(
                    needed, "r93_rank") if len(long_pool) > 0 else pd.DataFrame()
                final_longs = kept_longs | set(remaining["symbol"].tolist()) if len(remaining) > 0 else kept_longs
            else:
                # Too many kept, re-rank by R93 within kept
                kept_df = long_pool[long_pool["symbol"].isin(kept_longs)]
                if len(kept_df) > nl_final:
                    kept_df = kept_df.nsmallest(nl_final, "r93_rank")
                final_longs = set(kept_df["symbol"].tolist())

            if len(kept_shorts) < ns_final:
                new_from_tb = final_shorts - kept_shorts
                needed = ns_final - len(kept_shorts)
                remaining = short_pool[short_pool["symbol"].isin(new_from_tb)].nsmallest(
                    needed, "r93_rank") if len(short_pool) > 0 else pd.DataFrame()
                final_shorts = kept_shorts | set(remaining["symbol"].tolist()) if len(remaining) > 0 else kept_shorts
            else:
                kept_df = short_pool[short_pool["symbol"].isin(kept_shorts)]
                if len(kept_df) > ns_final:
                    kept_df = kept_df.nsmallest(ns_final, "r93_rank")
                final_shorts = set(kept_df["symbol"].tolist())

        new_opened = (final_longs - prev_longs) | (final_shorts - prev_shorts)
        closed = (prev_longs - final_longs) | (prev_shorts - final_shorts)
        total_positions = len(final_longs) + len(final_shorts)

        longs = grp[grp["symbol"].isin(final_longs)]
        shorts = grp[grp["symbol"].isin(final_shorts)]
        long_ret = longs["fwd_ret"].mean() if len(longs) > 0 else 0
        short_ret = shorts["fwd_ret"].mean() if len(shorts) > 0 else 0

        nl_act, ns_act = len(final_longs), len(final_shorts)
        if nl_act > 0 and ns_act > 0:
            gross_ret = 0.5 * long_ret - 0.5 * short_ret
        elif ns_act > 0:
            gross_ret = -short_ret
        else:
            gross_ret = long_ret
        gross_ret *= exposure

        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            turnover_cost = sum(_cost_for_sym(sym) * avg_weight for sym in new_opened)
            turnover_cost += sum(_cost_for_sym(sym) * avg_weight for sym in closed)
            holding_cost = funding_per_12h * (rebal_hours / 12)
            total_cost = turnover_cost + holding_cost
        else:
            total_cost = 0.0

        net_ret = gross_ret - total_cost
        prev_longs, prev_shorts = final_longs, final_shorts

        all_rets.append({
            "timestamp": ts, "gross_ret": gross_ret, "net_ret": net_ret,
            "cost": total_cost, "n_long": nl_act, "n_short": ns_act,
            "turnover": len(new_opened) + len(closed),
        })

    return pd.DataFrame(all_rets) if all_rets else pd.DataFrame()


def main():
    t0 = time.time()
    log("=" * 70)
    log("  R104 — CONSERVATIVE TIE-BREAK ENSEMBLE")
    log("=" * 70)

    from _research_r68_continuous_wf import (
        load_data, simulate, train_ensemble, PROD_CFG,
        CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, MARKET_LEVEL_FEATURES,
    )
    from _research_r22_models import SEEDS

    # ── Load predictions ──────────────────────────────────────────────────
    log("\n[0] Loading predictions ...")
    r68_path = RESULTS_DIR / "r68_predictions.parquet"
    r93_path = RESULTS_DIR / "r93_predictions.parquet"

    if not r93_path.exists():
        log("  ✗ R93 predictions not found!")
        return

    r93_preds = pd.read_parquet(r93_path)
    log(f"  R93: {len(r93_preds):,} rows")

    if not r68_path.exists():
        log("  R68 predictions not cached — retraining ...")
        df, regime_df = load_data()
        feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
        no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]
        r68_preds = train_ensemble(df, feats, CONTINUOUS_WINDOWS,
                                   seeds=SEEDS, cs_rank_exclude=no_rank)
        r68_preds.to_parquet(r68_path, index=False)
        del df
    else:
        r68_preds = pd.read_parquet(r68_path)
        _, regime_df = load_data()

    log(f"  R68: {len(r68_preds):,} rows")

    # ── R68 canonical baseline ────────────────────────────────────────────
    log("\n[1] Computing R68 canonical baseline ...")
    port_r68_canon = simulate(r68_preds, regime_df, 4, 2)
    s_baseline = sharpe_ann(port_r68_canon["net_ret"])
    dd_baseline = max_dd(port_r68_canon["net_ret"])
    ret_baseline = float((1 + port_r68_canon["net_ret"]).prod() - 1) * 100
    log(f"  R68 baseline: Sharpe={s_baseline:.3f}  MaxDD={dd_baseline*100:.1f}%  "
        f"Ret={ret_baseline:.1f}%  N={len(port_r68_canon)}")

    # ── Merge predictions ─────────────────────────────────────────────────
    log("\n[2] Merging predictions ...")

    merged = r68_preds.rename(columns={"pred": "pred_68"}).merge(
        r93_preds[["timestamp", "symbol", "raw_prob"]].rename(
            columns={"raw_prob": "raw_prob_93"}),
        on=["timestamp", "symbol"], how="left"
    )

    # Fill missing R93 with 0.5 (neutral)
    n_missing = merged["raw_prob_93"].isna().sum()
    merged["raw_prob_93"] = merged["raw_prob_93"].fillna(0.5)
    log(f"  Merged: {len(merged):,} rows, R93 coverage: {(1-n_missing/len(merged))*100:.1f}%")

    # ── Grid search: M × N ────────────────────────────────────────────────
    log("\n[3] Grid search: M ∈ {6,8,10} × N ∈ {3,4,5} ...")

    M_values = [6, 8, 10]
    N_values = [3, 4, 5]
    results = []
    best_sharpe = -999
    best_port = None
    best_config = None

    for M in M_values:
        for N in N_values:
            port = simulate_tiebreak(
                merged, regime_df,
                pool_long=M, pool_short=N,
                n_long_final=4, n_short_final=2
            )

            if len(port) == 0:
                log(f"  M={M} N={N}: NO DATA")
                continue

            net = port["net_ret"]
            s = sharpe_ann(net)
            dd = max_dd(net)
            total_ret = float((1 + net).prod() - 1) * 100
            wr = float((net > 0).mean())
            cal = calmar(net)
            avg_turn = port["turnover"].mean()

            tag = ""
            if M == 4 and N == 2:
                tag = " ← equivalent to R68"

            log(f"  M={M:>2} N={N}: Sharpe={s:>7.3f}  MaxDD={dd*100:>6.1f}%  "
                f"Ret={total_ret:>6.1f}%  Calmar={cal:>6.1f}  Turn={avg_turn:.1f}{tag}")

            res = {
                "pool_long": M,
                "pool_short": N,
                "sharpe": round(s, 4),
                "max_dd_pct": round(dd * 100, 2),
                "total_ret_pct": round(total_ret, 1),
                "win_rate": round(wr, 3),
                "n_periods": len(port),
                "calmar": round(cal, 2),
                "avg_turnover": round(avg_turn, 2),
            }
            results.append(res)

            if s > best_sharpe:
                best_sharpe = s
                best_port = port
                best_config = (M, N)

    if best_port is None:
        log("  ✗ No valid configs!")
        return

    log(f"\n  Best: M={best_config[0]} N={best_config[1]}, Sharpe={best_sharpe:.3f}")

    # ── Acceptance check ─────────────────────────────────────────────────
    log("\n[4] Acceptance check ...")
    best_dd = max_dd(best_port["net_ret"])
    dd_improvement = (abs(dd_baseline) - abs(best_dd)) / abs(dd_baseline) * 100

    crit1 = best_sharpe >= s_baseline + 0.05
    crit2 = (dd_improvement >= 15 and best_sharpe >= s_baseline - 0.05)
    log(f"  Criterion 1 (Sharpe >= {s_baseline+0.05:.3f}): "
        f"{'PASS' if crit1 else 'FAIL'} (got {best_sharpe:.3f})")
    log(f"  Criterion 2 (DD↓≥15% + Sharpe≥{s_baseline-0.05:.3f}): "
        f"{'PASS' if crit2 else 'FAIL'} (DD improvement={dd_improvement:.1f}%)")

    # ── Bootstrap ─────────────────────────────────────────────────────────
    log("\n[5] Bootstrap: M={} N={} vs R68 ...".format(*best_config))

    n_common = min(len(port_r68_canon), len(best_port))
    boot = block_bootstrap(
        port_r68_canon["net_ret"].values[:n_common],
        best_port["net_ret"].values[:n_common],
        n_boot=1000, block=10
    )

    log(f"  P(Sharpe better) = {boot['p_sharpe_better']:.3f}")
    log(f"  P(Calmar better) = {boot['p_calmar_better']:.3f}")
    log(f"  ΔSharpe: median={boot['median_delta_sharpe']:.4f}  "
        f"[{boot['p5_delta_sharpe']:.4f}, {boot['p95_delta_sharpe']:.4f}]")

    accept_sharpe = crit1 and boot["p_sharpe_better"] > 0.8
    accept_calmar = crit2 and boot["p_calmar_better"] > 0.8
    accepted = accept_sharpe or accept_calmar

    log(f"\n  Sharpe acceptance: {'PASS ✅' if accept_sharpe else 'FAIL ❌'}")
    log(f"  Calmar acceptance: {'PASS ✅' if accept_calmar else 'FAIL ❌'}")
    log(f"  OVERALL: {'ACCEPTED ✅' if accepted else 'REJECTED ❌'}")

    # ── Save artifacts ────────────────────────────────────────────────────
    log("\n[6] Saving ...")

    summary = {
        "best_pool_long": best_config[0],
        "best_pool_short": best_config[1],
        "best_sharpe": round(best_sharpe, 4),
        "best_maxdd_pct": round(best_dd * 100, 2),
        "best_return_pct": round(float((1 + best_port["net_ret"]).prod() - 1) * 100, 1),
        "baseline_sharpe": round(s_baseline, 4),
        "baseline_maxdd_pct": round(dd_baseline * 100, 2),
        "baseline_return_pct": round(ret_baseline, 1),
        "dd_improvement_pct": round(dd_improvement, 1),
        "accepted": accepted,
        "accept_sharpe": accept_sharpe,
        "accept_calmar": accept_calmar,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(RESULTS_DIR / "r104_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    best_port[["timestamp", "net_ret", "gross_ret", "cost", "turnover",
               "n_long", "n_short"]].to_csv(
        RESULTS_DIR / "r104_equity.csv", index=False)

    with open(RESULTS_DIR / "r104_bootstrap.json", "w") as f:
        json.dump(boot, f, indent=2)

    pd.DataFrame(results).to_csv(RESULTS_DIR / "r104_grid.csv", index=False)

    log(f"\n  Saved: r104_summary.json, r104_equity.csv, r104_bootstrap.json, r104_grid.csv")
    log(f"  Runtime: {time.time() - t0:.0f}s ({(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    main()
