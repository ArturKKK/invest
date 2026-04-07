#!/usr/bin/env python3
"""
R112 — Factor-Mimicking Portfolios (FMP) as new features.

Idea: Build cross-sectional FMP returns from proxy characteristics,
then derive time-series features from FMP returns.

Proxy characteristics (already in R68 feature set):
  - cum_funding_24h   (carry/sentiment)
  - oi_velocity        (flow/positioning)
  - rel_volume_cs      (liquidity/activity)

For each characteristic c at each timestamp t:
  1. z = cross-sectional z-score of c
  2. w = z / sum(|z|)   (dollar-neutral weights)
  3. fmp_ret_t = sum(w_i * fwd_ret_i)

Derived FMP features (per characteristic):
  fmp_level_X       — rolling 24-period cumulative FMP return
  fmp_z120_X        — z-score of FMP return over 120 periods
  fmp_mom_X         — FMP momentum: sum(ret[0:6]) - sum(ret[6:12])
  fmp_tail_X        — rolling skewness of FMP returns (60 periods)

Total: 3 characteristics × 4 features = 12 features.

Pipeline: load_data → build FMP features → IC scan gate → add-only WF if pass
"""

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
RESULTS  = BASE_DIR / "results"
RESULTS.mkdir(exist_ok=True)

from _research_round7 import SYM_35
from _research_r22_models import SEEDS, log
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r68_continuous_wf import (
    CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, PROD_CFG,
    load_data, train_ensemble, simulate, sharpe, analyze,
)

IC_THRESH        = 0.03
STABILITY_THRESH = 2 / 3
COVERAGE_THRESH  = 0.70
REDUND_THRESH    = 0.70

# Proxy characteristics (must exist in df after load_data)
CHARACTERISTICS = ["cum_funding_24h", "oi_velocity", "rel_volume_cs"]

# Derived feature suffixes
FMP_SUFFIXES = ["level", "z120", "mom", "tail"]


def build_fmp_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Build FMP return series from characteristics, then derive features."""
    log("  Building FMP features...")
    df = df.sort_values(["symbol", "timestamp"]).copy()
    timestamps = sorted(df["timestamp"].unique())

    # Use PAST returns (ret_12h) for FMP — NOT fwd_ret_12h to avoid lookahead.
    # ret_12h = close(t)/close(t-12) - 1: fully realized at time t.
    if "ret_12h" not in df.columns:
        df["ret_12h"] = df.groupby("symbol")["close"].transform(
            lambda x: x.pct_change(12)
        )

    # Still need fwd_ret_12h for the IC scan target
    if "fwd_ret_12h" not in df.columns:
        df["fwd_ret_12h"] = df.groupby("symbol")["close"].transform(
            lambda x: x.pct_change(12).shift(-12)
        )

    all_added = []

    for char in CHARACTERISTICS:
        if char not in df.columns or df[char].notna().sum() < 100:
            log(f"    SKIP {char}: not available or insufficient data")
            continue

        log(f"    Processing characteristic: {char}")

        # ── Step 1: Compute FMP returns per timestamp ───────────────
        # Use PAST returns (ret_12h) and LAGGED characteristics (shift 1)
        # to ensure no lookahead.
        fmp_rets = {}
        for ts, grp in df.groupby("timestamp"):
            sub = grp[[char, "ret_12h"]].dropna()
            if len(sub) < 5:
                continue
            z = (sub[char] - sub[char].mean()) / (sub[char].std() + 1e-10)
            w = z / (z.abs().sum() + 1e-10)  # dollar-neutral
            fmp_ret = (w * sub["ret_12h"]).sum()
            fmp_rets[ts] = fmp_ret

        fmp_series = pd.Series(fmp_rets).sort_index()
        log(f"      FMP return series: {len(fmp_series)} timestamps, "
            f"mean={fmp_series.mean():.6f}, std={fmp_series.std():.6f}")

        if len(fmp_series) < 120:
            log(f"      SKIP: too short ({len(fmp_series)} < 120)")
            continue

        # ── Step 2: Derive time-series features ─────────────────────
        fmp_df = fmp_series.reset_index()
        fmp_df.columns = ["timestamp", "fmp_ret"]
        fmp_df = fmp_df.sort_values("timestamp")

        # fmp_level: rolling 24-period cumulative return
        fmp_df[f"fmp_level_{char}"] = fmp_df["fmp_ret"].rolling(24, min_periods=12).sum()

        # fmp_z120: z-score over 120 periods
        roll_mean = fmp_df["fmp_ret"].rolling(120, min_periods=60).mean()
        roll_std = fmp_df["fmp_ret"].rolling(120, min_periods=60).std()
        fmp_df[f"fmp_z120_{char}"] = (fmp_df["fmp_ret"] - roll_mean) / (roll_std + 1e-10)

        # fmp_mom: recent vs older momentum (sum[0:6] - sum[6:12])
        recent = fmp_df["fmp_ret"].rolling(6, min_periods=3).sum()
        older = fmp_df["fmp_ret"].shift(6).rolling(6, min_periods=3).sum()
        fmp_df[f"fmp_mom_{char}"] = recent - older

        # fmp_tail: rolling skewness (60 periods)
        fmp_df[f"fmp_tail_{char}"] = fmp_df["fmp_ret"].rolling(60, min_periods=30).skew()

        # Shift all by 1 — derived features use past FMP rets (built from past returns),
        # shift(1) ensures we only use info fully available at t-1.
        feat_cols = [f"fmp_{suf}_{char}" for suf in FMP_SUFFIXES]
        for col in feat_cols:
            fmp_df[col] = fmp_df[col].shift(1)

        # Merge back to main df (market-level: same for all coins at timestamp t)
        merge_cols = ["timestamp"] + feat_cols
        df = df.merge(fmp_df[merge_cols], on="timestamp", how="left")
        all_added.extend(feat_cols)

    added = [f for f in all_added if f in df.columns and df[f].notna().any()]
    log(f"  Built {len(added)} FMP features")

    return df, added


def ic_scan(
    df: pd.DataFrame,
    feats: List[str],
    existing_feats: List[str],
) -> pd.DataFrame:
    """IC scan with gate."""
    log("\nIC Scan + Gate")
    log("=" * 60)

    tz = df["timestamp"].dt.tz
    rows = []

    for feat in feats:
        if feat not in df.columns:
            rows.append({"feature": feat, "skip": "not_in_frame"})
            continue
        valid = df[[feat, "fwd_ret_12h", "timestamp"]].dropna()
        n_obs = len(valid)
        if n_obs < 100:
            rows.append({"feature": feat, "skip": f"too_few_obs ({n_obs})"})
            continue

        coverage = n_obs / len(df)
        pooled_ic = float(stats.spearmanr(valid[feat], valid["fwd_ret_12h"])[0])

        # Per-window IC
        window_ics = []
        for w in CONTINUOUS_WINDOWS:
            ts_s = pd.Timestamp(w["test_start"], tz=tz)
            ts_e = pd.Timestamp(w["test_end"], tz=tz)
            wdf = valid[(valid["timestamp"] >= ts_s) & (valid["timestamp"] <= ts_e)]
            if len(wdf) < 50:
                window_ics.append(np.nan)
            else:
                wic = float(stats.spearmanr(wdf[feat], wdf["fwd_ret_12h"])[0])
                window_ics.append(wic if not np.isnan(wic) else 0.0)

        stability = sum(
            1 for ic in window_ics if not np.isnan(ic) and abs(ic) >= 0.02
        ) / len(CONTINUOUS_WINDOWS)

        max_corr = 0.0
        max_corr_feat = ""
        for ef in existing_feats:
            if ef not in df.columns:
                continue
            sub = df[[feat, ef]].dropna()
            if len(sub) < 50:
                continue
            c = abs(float(stats.spearmanr(sub[feat], sub[ef])[0]))
            if c > max_corr:
                max_corr = c
                max_corr_feat = ef

        pass_ic   = abs(pooled_ic) >= IC_THRESH
        pass_stab = stability >= STABILITY_THRESH
        pass_cov  = coverage >= COVERAGE_THRESH
        pass_red  = max_corr < REDUND_THRESH
        gate_pass = pass_ic and pass_stab and pass_cov and pass_red
        score = abs(pooled_ic) * stability

        rows.append({
            "feature": feat,
            "pooled_ic": round(pooled_ic, 4),
            "w1_ic": round(window_ics[0], 4) if len(window_ics) > 0 and not np.isnan(window_ics[0]) else None,
            "w2_ic": round(window_ics[1], 4) if len(window_ics) > 1 and not np.isnan(window_ics[1]) else None,
            "w3_ic": round(window_ics[2], 4) if len(window_ics) > 2 and not np.isnan(window_ics[2]) else None,
            "stability": round(stability, 3),
            "coverage": round(coverage, 3),
            "max_corr_existing": round(max_corr, 3),
            "max_corr_feat": max_corr_feat,
            "score": round(score, 4),
            "pass_ic": pass_ic, "pass_stab": pass_stab,
            "pass_cov": pass_cov, "pass_red": pass_red,
            "gate_pass": gate_pass,
            "n_obs": n_obs, "skip": None,
        })

    ic_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    for _, row in ic_df.iterrows():
        if row.get("skip"):
            log(f"  {row['feature']:>30s}: SKIP ({row['skip']})")
            continue
        gp = "✅ PASS" if row["gate_pass"] else "❌ FAIL"
        log(f"  {row['feature']:>30s}: IC={row['pooled_ic']:+.4f}  "
            f"stab={row['stability']:.2f}  cov={row['coverage']:.3f}  "
            f"corr={row['max_corr_existing']:.3f}({row['max_corr_feat']})  {gp}")

    return ic_df


def block_bootstrap_sharpe(
    rets_base: pd.Series, rets_test: pd.Series,
    n_boot: int = 1000, block_size: int = 10, seed: int = 42,
) -> dict:
    rng = np.random.RandomState(seed)
    n = min(len(rets_base), len(rets_test))
    if n < 20:
        return {"p_sharpe_better": 0.5, "p_calmar_better": 0.5}
    rb = rets_base.values[:n]
    rt = rets_test.values[:n]
    sharpe_diffs, calmar_diffs = [], []
    for _ in range(n_boot):
        idx = []
        while len(idx) < n:
            s = rng.randint(0, max(1, n - block_size + 1))
            idx.extend(range(s, min(s + block_size, n)))
        idx = idx[:n]
        rb_b, rt_b = rb[idx], rt[idx]
        sb = rb_b.mean() / (rb_b.std() + 1e-10) * np.sqrt(2 * 365)
        st = rt_b.mean() / (rt_b.std() + 1e-10) * np.sqrt(2 * 365)
        sharpe_diffs.append(st - sb)
        eq_b = np.cumprod(1 + rb_b)
        eq_t = np.cumprod(1 + rt_b)
        dd_b = (eq_b / np.maximum.accumulate(eq_b) - 1).min()
        dd_t = (eq_t / np.maximum.accumulate(eq_t) - 1).min()
        calmar_diffs.append(
            (eq_t[-1] - 1) / (abs(dd_t) + 1e-10) - (eq_b[-1] - 1) / (abs(dd_b) + 1e-10)
        )
    return {
        "p_sharpe_better": round(np.mean(np.array(sharpe_diffs) > 0), 3),
        "p_calmar_better": round(np.mean(np.array(calmar_diffs) > 0), 3),
    }


def main():
    t0 = time.time()
    log("=" * 70)
    log("R112 — Factor-Mimicking Portfolios (FMP)")
    log("=" * 70)

    # ── Load data ───────────────────────────────────────────────────────
    log("\nStep 0: Loading data...")
    df, regime_df = load_data()

    # Compute fwd_ret_12h if missing
    if "fwd_ret_12h" not in df.columns:
        df["fwd_ret_12h"] = df.groupby("symbol")["close"].transform(
            lambda x: x.pct_change(12).shift(-12)
        )

    log(f"  Base: {len(df):,} rows, {df['symbol'].nunique()} symbols")
    log(f"  Characteristics available: "
        f"{[c for c in CHARACTERISTICS if c in df.columns]}")

    # ── Build features ──────────────────────────────────────────────────
    log("\nStep 1: Building FMP features...")
    df, added = build_fmp_features(df)

    if len(added) == 0:
        log("\nERROR: No FMP features could be built. Check data availability.")
        return

    # Coverage
    log("\n  Feature coverage:")
    for f in sorted(added):
        cov = df[f].notna().mean()
        log(f"    {f:>30s}: {cov:.3f}")

    # ── IC Scan ─────────────────────────────────────────────────────────
    log("\nStep 2: IC Scan")
    existing_feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    ic_df = ic_scan(df, added, existing_feats)
    ic_df.to_csv(RESULTS / "r112_ic_report.csv", index=False)
    log(f"\n  Saved IC report → results/r112_ic_report.csv")

    # ── Check gate ──────────────────────────────────────────────────────
    passed = ic_df[ic_df["gate_pass"] == True]
    n_pass = len(passed)

    if n_pass == 0:
        log("\n" + "=" * 70)
        log("RESULT: 0 features pass gate. STOP — no WF test needed.")
        log("=" * 70)
        best_ic = (ic_df[ic_df["pooled_ic"].notna()]["pooled_ic"].abs().max()
                   if len(ic_df[ic_df["pooled_ic"].notna()]) > 0 else 0)
        log(f"\n  Features tested: {len(added)}")
        log(f"  Gate passed: 0")
        log(f"  Best |IC|: {best_ic:.4f}")
        log(f"  VERDICT: ❌ FAIL — no FMP feature has IC ≥ {IC_THRESH}")
    else:
        log(f"\n  {n_pass} features pass gate: {passed['feature'].tolist()}")

        # ── Add-only WF ────────────────────────────────────────────────
        log("\nStep 3: Add-only WF test")

        base_feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
        no_rank_base = [f for f in base_feats if f in MARKET_LEVEL_FEATURES]

        # Baseline
        log("\n  Training R68 baseline...")
        t1 = time.time()
        preds_base = train_ensemble(df, base_feats, CONTINUOUS_WINDOWS,
                                     seeds=SEEDS, cs_rank_exclude=no_rank_base)
        log(f"  Baseline trained in {time.time()-t1:.0f}s")

        cfg_42 = {**PROD_CFG, "n_long": 4, "n_short": 2}
        port_base = simulate(preds_base, regime_df, 4, 2, cfg_42)
        m_base = analyze(port_base, "R68_baseline")

        eq_base = (1 + port_base["net_ret"]).cumprod()
        calmar_base = ((eq_base.iloc[-1] / eq_base.iloc[0] - 1) /
                       (abs((eq_base / eq_base.cummax() - 1).min()) + 1e-10))

        # Test configs: add top features incrementally
        new_feats_ranked = passed.sort_values("score", ascending=False)["feature"].tolist()

        # FMP features are market-level (same for all coins at a timestamp)
        fmp_market_feats = set(added)  # all FMP features are market-level

        ablation_rows = [{"config": "baseline", "feats_added": 0,
                          "net_sharpe": m_base.get("net_sharpe", 0),
                          "max_dd_pct": m_base.get("max_dd_pct", 0),
                          "calmar": round(calmar_base, 3)}]

        for k in range(1, min(len(new_feats_ranked) + 1, 4)):
            add_feats = new_feats_ranked[:k]
            test_feats = base_feats + add_feats
            no_rank_test = list(set(no_rank_base) | (set(add_feats) & fmp_market_feats))

            label = f"+{','.join(add_feats)}"
            log(f"\n  Testing {label}...")
            t2 = time.time()
            preds_test = train_ensemble(df, test_feats, CONTINUOUS_WINDOWS,
                                         seeds=SEEDS, cs_rank_exclude=no_rank_test)
            log(f"  Trained in {time.time()-t2:.0f}s")

            if preds_test is None:
                log(f"  WARNING: no predictions for {label}")
                continue

            port_test = simulate(preds_test, regime_df, 4, 2, cfg_42)
            m_test = analyze(port_test, label)

            eq_test = (1 + port_test["net_ret"]).cumprod()
            dd_test = (eq_test / eq_test.cummax() - 1).min()
            calmar_test = ((eq_test.iloc[-1] / eq_test.iloc[0] - 1) /
                           (abs(dd_test) + 1e-10))

            # Bootstrap
            boot = block_bootstrap_sharpe(port_base["net_ret"], port_test["net_ret"])

            base_sh = m_base.get("net_sharpe", 0)
            test_sh = m_test.get("net_sharpe", 0)
            pass_a = (test_sh >= base_sh + 0.05) and boot["p_sharpe_better"] > 0.80
            pass_b = (test_sh >= base_sh - 0.05 and calmar_test >= calmar_base * 1.05
                      and boot["p_calmar_better"] > 0.80)

            flag = ""
            if pass_a: flag = " ✅ PASS-A"
            elif pass_b: flag = " ✅ PASS-B"

            log(f"    Sh={test_sh:.3f} (Δ={test_sh-base_sh:+.3f})  "
                f"Cal={calmar_test:.2f}  "
                f"P(Sh↑)={boot['p_sharpe_better']:.2f}  "
                f"P(Cal↑)={boot['p_calmar_better']:.2f}{flag}")

            ablation_rows.append({
                "config": label, "feats_added": k,
                "net_sharpe": m_test.get("net_sharpe", 0),
                "max_dd_pct": m_test.get("max_dd_pct", 0),
                "calmar": round(calmar_test, 3),
                "p_sharpe_better": boot["p_sharpe_better"],
                "p_calmar_better": boot["p_calmar_better"],
                "pass_a": pass_a, "pass_b": pass_b,
            })

        abl_df = pd.DataFrame(ablation_rows)
        abl_df.to_csv(RESULTS / "r112_ablation.csv", index=False)
        log(f"\n  Saved ablation → results/r112_ablation.csv")

        # Verdict
        any_pass = any(r.get("pass_a") or r.get("pass_b") for r in ablation_rows)
        if any_pass:
            log("\n  VERDICT: ✅ PASS — FMP features improve R68")
        else:
            log("\n  VERDICT: ❌ FAIL — FMP features do not improve R68")

    # Summary JSON
    summary = {
        "experiment": "R112",
        "features_tested": len(added),
        "gate_passed": n_pass,
        "passed_features": passed["feature"].tolist() if n_pass > 0 else [],
        "best_ic": round(ic_df[ic_df["pooled_ic"].notna()]["pooled_ic"].abs().max(), 4)
            if len(ic_df[ic_df["pooled_ic"].notna()]) > 0 else 0,
    }
    with open(RESULTS / "r112_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0
    log(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    log("Done.")


if __name__ == "__main__":
    main()
