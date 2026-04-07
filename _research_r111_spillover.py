#!/usr/bin/env python3
"""
R111 — Spillover-head: inter-coin lags + market factors as new features.

Features:
  Market-level (same for all coins):
    mkt_ret_12h        — equal-weight universe return (shift1)
    mkt_ret_12h_exBTC  — universe return ex-BTC
    btc_ret_12h_lag1   — BTC 12h return lagged
    eth_ret_12h_lag1   — ETH 12h return lagged
    dispersion_12h     — cross-sectional std of 12h returns
    pc1_ret_lag1       — 1st PCA component of cs rets

  Per-coin:
    beta_btc_60        — rolling 60-period β to BTC (shift1)
    spill_btc          — beta_i * btc_ret_{t-1}
    spill_mkt          — corr_i_to_mkt * mkt_ret_{t-1}

Pipeline: load_data → build features → IC scan gate → add-only WF if pass
"""

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
RESULTS  = BASE_DIR / "results"
RESULTS.mkdir(exist_ok=True)

from _research_round7 import SYM_35
from _research_r22_models import SEEDS, log
from _research_r35_new_features import MARKET_LEVEL_FEATURES, add_r35_features
from _research_r68_continuous_wf import (
    CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, PROD_CFG,
    load_data, train_ensemble, simulate, sharpe, analyze,
)

IC_THRESH       = 0.03
STABILITY_THRESH = 2 / 3
COVERAGE_THRESH  = 0.70
REDUND_THRESH    = 0.70

# All new feature names
MARKET_FEATURES  = [
    "mkt_ret_12h", "mkt_ret_12h_exBTC",
    "btc_ret_12h_lag1", "eth_ret_12h_lag1",
    "dispersion_12h", "pc1_ret_lag1",
]
PERCOIN_FEATURES = [
    "beta_btc_60", "spill_btc", "spill_mkt",
]
ALL_NEW_FEATS = MARKET_FEATURES + PERCOIN_FEATURES


def build_spillover_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Build all spillover features on the hourly frame."""
    log("  Building spillover features...")
    df = df.sort_values(["symbol", "timestamp"]).copy()

    timestamps = sorted(df["timestamp"].unique())
    ts_set = set(timestamps)

    # ── Market-level ────────────────────────────────────────────────
    # Cross-sectional mean return (12h)
    cs_stats = df.groupby("timestamp").agg(
        mkt_ret_12h=("ret_12h", "mean"),
        dispersion_12h=("ret_12h", "std"),
    ).reset_index()

    # Ex-BTC
    non_btc = df[df["symbol"] != "BTC/USDT"].groupby("timestamp")["ret_12h"].mean().reset_index()
    non_btc.columns = ["timestamp", "mkt_ret_12h_exBTC"]
    cs_stats = cs_stats.merge(non_btc, on="timestamp", how="left")

    # BTC and ETH lagged returns
    for leader, col_name in [("BTC/USDT", "btc_ret_12h_lag1"), ("ETH/USDT", "eth_ret_12h_lag1")]:
        leader_df = df[df["symbol"] == leader][["timestamp", "ret_12h"]].copy()
        leader_df = leader_df.sort_values("timestamp")
        leader_df[col_name] = leader_df["ret_12h"].shift(1)  # lag1
        cs_stats = cs_stats.merge(leader_df[["timestamp", col_name]], on="timestamp", how="left")

    # Shift market-level features by 1 period (no lookahead)
    cs_stats = cs_stats.sort_values("timestamp")
    for col in ["mkt_ret_12h", "mkt_ret_12h_exBTC", "dispersion_12h"]:
        cs_stats[col] = cs_stats[col].shift(1)

    # PCA component (rolling)
    log("    Computing PCA(1) on rolling cs return matrix...")
    symbols = sorted(df["symbol"].unique())
    pivot = df.pivot_table(index="timestamp", columns="symbol", values="ret_12h")
    pivot = pivot.sort_index()

    # Rolling PCA with 60 period lookback
    pc1_values = pd.Series(index=pivot.index, dtype=float)
    window = 60
    for i in range(window, len(pivot)):
        chunk = pivot.iloc[i - window:i].dropna(axis=1, how="any")
        if chunk.shape[1] < 5 or chunk.shape[0] < window // 2:
            continue
        # Standardize
        arr = chunk.values
        arr = (arr - arr.mean(axis=0)) / (arr.std(axis=0) + 1e-10)
        try:
            pca = PCA(n_components=1)
            pca.fit(arr)
            # Project last row
            last_std = (pivot.iloc[i].values - arr.mean(axis=0)) / (arr.std(axis=0) + 1e-10)
            # Only use columns that were in chunk
            cols_used = chunk.columns
            last_vals = pivot.iloc[i][cols_used].values
            last_std2 = (last_vals - chunk.mean().values) / (chunk.std().values + 1e-10)
            pc1 = pca.transform(last_std2.reshape(1, -1))[0, 0]
            pc1_values.iloc[i] = pc1
        except Exception:
            pass

    pc1_df = pc1_values.reset_index()
    pc1_df.columns = ["timestamp", "pc1_ret_lag1"]
    pc1_df["pc1_ret_lag1"] = pc1_df["pc1_ret_lag1"].shift(1)  # lag1
    cs_stats = cs_stats.merge(pc1_df, on="timestamp", how="left")

    # Merge market features to df
    df = df.merge(cs_stats, on="timestamp", how="left")

    # ── Per-coin features ───────────────────────────────────────────
    log("    Computing per-coin spillover features...")

    # BTC return for merging
    btc_rets = df[df["symbol"] == "BTC/USDT"][["timestamp", "ret_12h"]].rename(
        columns={"ret_12h": "btc_ret_12h_raw"}
    )
    df = df.merge(btc_rets, on="timestamp", how="left")

    # Rolling beta to BTC (60 period)
    df["beta_btc_60"] = np.nan
    for sym in symbols:
        mask = df["symbol"] == sym
        sym_df = df.loc[mask].copy()
        cov_rb = sym_df["ret_12h"].rolling(60, min_periods=30).cov(sym_df["btc_ret_12h_raw"])
        var_b = sym_df["btc_ret_12h_raw"].rolling(60, min_periods=30).var()
        beta = cov_rb / (var_b + 1e-10)
        df.loc[mask, "beta_btc_60"] = beta.shift(1).values  # shift1

    # Spillover: beta_i * btc_ret_lag1
    df["spill_btc"] = df["beta_btc_60"] * df["btc_ret_12h_lag1"]

    # Rolling corr to market
    df["corr_to_mkt_60"] = np.nan
    for sym in symbols:
        mask = df["symbol"] == sym
        sym_df = df.loc[mask].copy()
        # mkt_ret is already shift1, but we need the non-shifted version for corr
        # Actually mkt_ret_12h is already lagged — use raw ret_12h cross-mean
        raw_mkt = df.groupby("timestamp")["ret_12h"].transform("mean")
        corr = sym_df["ret_12h"].rolling(60, min_periods=30).corr(raw_mkt.loc[mask])
        df.loc[mask, "corr_to_mkt_60"] = corr.shift(1).values  # shift1

    # spill_mkt = corr_i * mkt_ret_lag1
    df["spill_mkt"] = df["corr_to_mkt_60"] * df["mkt_ret_12h"]

    # Cleanup
    df.drop(columns=["btc_ret_12h_raw", "corr_to_mkt_60"], inplace=True, errors="ignore")

    added = [f for f in ALL_NEW_FEATS if f in df.columns and df[f].notna().any()]
    log(f"  Built {len(added)} spillover features")

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
            log(f"  {row['feature']:>25s}: SKIP ({row['skip']})")
            continue
        gp = "✅ PASS" if row["gate_pass"] else "❌ FAIL"
        log(f"  {row['feature']:>25s}: IC={row['pooled_ic']:+.4f}  "
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
    log("R111 — Spillover Features (inter-coin lags + market factors)")
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

    # ── Build features ──────────────────────────────────────────────────
    log("\nStep 1: Building spillover features...")
    df, added = build_spillover_features(df)

    # Coverage
    log("\n  Feature coverage:")
    for f in sorted(added):
        cov = df[f].notna().mean()
        log(f"    {f:>25s}: {cov:.3f}")

    # ── IC Scan ─────────────────────────────────────────────────────────
    log("\nStep 2: IC Scan")
    existing_feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    ic_df = ic_scan(df, added, existing_feats)
    ic_df.to_csv(RESULTS / "r111_ic_report.csv", index=False)
    log(f"\n  Saved IC report → results/r111_ic_report.csv")

    # ── Check gate ──────────────────────────────────────────────────────
    passed = ic_df[ic_df["gate_pass"] == True]
    n_pass = len(passed)

    if n_pass == 0:
        log("\n" + "=" * 70)
        log("RESULT: 0 features pass gate. STOP — no WF test needed.")
        log("=" * 70)
        best_ic = ic_df[ic_df["pooled_ic"].notna()]["pooled_ic"].abs().max() if len(ic_df[ic_df["pooled_ic"].notna()]) > 0 else 0
        log(f"\n  Features tested: {len(added)}")
        log(f"  Gate passed: 0")
        log(f"  Best |IC|: {best_ic:.4f}")
        log(f"  VERDICT: ❌ FAIL — no spillover feature has IC ≥ {IC_THRESH}")
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
        calmar_base = (eq_base.iloc[-1] / eq_base.iloc[0] - 1) / (abs((eq_base / eq_base.cummax() - 1).min()) + 1e-10)

        # Test configs: add top features incrementally
        new_feats_ranked = passed.sort_values("score", ascending=False)["feature"].tolist()

        ablation_rows = [{"config": "baseline", "feats_added": 0,
                          "net_sharpe": m_base.get("net_sharpe", 0),
                          "max_dd_pct": m_base.get("max_dd_pct", 0),
                          "calmar": round(calmar_base, 3)}]

        for k in range(1, min(len(new_feats_ranked) + 1, 4)):  # max +3 features
            add_feats = new_feats_ranked[:k]
            test_feats = base_feats + add_feats
            no_rank_test = list(set(no_rank_base) | (set(add_feats) & set(MARKET_FEATURES)))

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
            calmar_test = (eq_test.iloc[-1] / eq_test.iloc[0] - 1) / (abs(dd_test) + 1e-10)

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
        abl_df.to_csv(RESULTS / "r111_ablation.csv", index=False)
        log(f"\n  Saved ablation → results/r111_ablation.csv")

        # Verdict
        any_pass = any(r.get("pass_a") or r.get("pass_b") for r in ablation_rows)
        if any_pass:
            log("\n  VERDICT: ✅ PASS — spillover features improve R68")
        else:
            log("\n  VERDICT: ❌ FAIL — spillover features do not improve R68")

    # Summary JSON
    summary = {
        "experiment": "R111",
        "features_tested": len(added),
        "gate_passed": n_pass,
        "passed_features": passed["feature"].tolist() if n_pass > 0 else [],
        "best_ic": round(ic_df[ic_df["pooled_ic"].notna()]["pooled_ic"].abs().max(), 4)
            if len(ic_df[ic_df["pooled_ic"].notna()]) > 0 else 0,
    }
    with open(RESULTS / "r111_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0
    log(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    log("Done.")


if __name__ == "__main__":
    main()
