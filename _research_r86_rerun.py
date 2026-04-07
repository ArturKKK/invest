#!/usr/bin/env python3
"""
_research_r86_rerun.py — Clean rerun of R82→R85 using R68's canonical data pipeline.

Fixes the data loading mismatch: uses load_data() from R68 so train sizes match exactly.
"""

import json
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
CG_DIR      = ROOT / "data" / "raw" / "coinglass"

EPS              = 1e-10
PERIODS_PER_YEAR = 2 * 365
ROLL_120         = 120
ROLL_7           = 7
COVERAGE_THR     = 0.95
TEST_START       = "2024-10-15"

# ─── Import R68 canonical pipeline ────────────────────────────────────────────
from _research_r68_continuous_wf import (
    load_data, train_ensemble, simulate, CONTINUOUS_WINDOWS, SEEDS, PROD_CFG,
    CHAMPION_FEAT_31, MARKET_LEVEL_FEATURES, sharpe as r68_sharpe,
)


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def sharpe(rets: pd.Series) -> float:
    if len(rets) < 2:
        return 0.0
    r = (1 + rets).cumprod().pct_change().dropna()
    return float(r.mean() / (r.std() + EPS) * np.sqrt(PERIODS_PER_YEAR))


def max_dd(rets: pd.Series) -> float:
    eq = (1 + rets).cumprod()
    return float((eq / eq.cummax() - 1).min())


def portfolio_metrics(port: pd.DataFrame) -> dict:
    rets = port["net_ret"]
    s = sharpe(rets)
    dd = max_dd(rets)
    return {
        "net_sharpe":    round(s, 4),
        "gross_sharpe":  round(sharpe(port["gross_ret"]), 4),
        "max_dd_pct":    round(dd * 100, 2),
        "calmar":        round(s / (abs(dd) + EPS), 3),
        "total_ret_pct": round(float((1 + rets).prod() - 1) * 100, 1),
        "win_rate":      round(float((rets > 0).mean()), 3),
        "n_periods":     len(rets),
    }


def _zscore_series(s: pd.Series, window: int) -> pd.Series:
    mu  = s.rolling(window, min_periods=ROLL_7).mean()
    std = s.rolling(window, min_periods=ROLL_7).std() + EPS
    return (s - mu) / std


def block_bootstrap_sharpe(rets_base, rets_exp, n_boot=1000, block=10, seed=42):
    rng      = np.random.default_rng(seed)
    n        = min(len(rets_base), len(rets_exp))
    rb, re   = rets_base[:n], rets_exp[:n]
    n_blocks = n // block

    sb_list, se_list = [], []
    for _ in range(n_boot):
        idx       = rng.integers(0, n - block + 1, size=n_blocks)
        block_idx = np.concatenate([np.arange(i, i + block) for i in idx])[:n]

        def _sh(r):
            if len(r) < 2 or r.std() < 1e-10:
                return 0.0
            return float(r.mean() / (r.std() + EPS) * np.sqrt(PERIODS_PER_YEAR))

        sb_list.append(_sh(rb[block_idx]))
        se_list.append(_sh(re[block_idx]))

    sb, se = np.array(sb_list), np.array(se_list)
    delta  = se - sb
    return {
        "p_exp_better":    round(float((se > sb).mean()), 3),
        "median_delta":    round(float(np.median(delta)), 4),
        "mean_delta":      round(float(np.mean(delta)), 4),
        "p5_delta":        round(float(np.percentile(delta, 5)), 4),
        "p95_delta":       round(float(np.percentile(delta, 95)), 4),
        "base_sharpe_med": round(float(np.median(sb)), 4),
        "exp_sharpe_med":  round(float(np.median(se)), 4),
    }


# ─── R82: Build CG z-score features ──────────────────────────────────────────

def build_cg_features(df: pd.DataFrame):
    """Build 13 CG z-score/momentum features, merge shift1 into df."""
    eps = EPS

    def load_cg(name):
        p = CG_DIR / f"{name}.parquet"
        if not p.exists():
            return None
        d = pd.read_parquet(p)
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        d["cg_date"]   = d["timestamp"].dt.normalize()
        d = d.drop_duplicates(subset=["symbol", "cg_date"], keep="last")
        return d

    taker   = load_cg("taker")
    oi_df   = load_cg("oi")
    funding = load_cg("funding")
    liq     = load_cg("liq")
    ls      = load_cg("ls_ratio")

    frames = []
    feat_names = []

    # TAKER
    if taker is not None:
        t = taker[["symbol", "cg_date", "taker_buy_usd", "taker_sell_usd"]].copy()
        total = t["taker_buy_usd"] + t["taker_sell_usd"]
        t["_imb"]  = (t["taker_buy_usd"] - t["taker_sell_usd"]) / (total + eps)
        t["_flow"] = t["taker_buy_usd"] / (t["taker_sell_usd"] + eps)
        rows = []
        for _, g in t.groupby("symbol"):
            g = g.sort_values("cg_date").copy()
            g["cg_taker_imb_z120"]  = _zscore_series(g["_imb"], ROLL_120)
            g["cg_taker_flow_z120"] = _zscore_series(g["_flow"].clip(-10, 10), ROLL_120)
            rows.append(g)
        t = pd.concat(rows)
        cols = ["cg_taker_imb_z120", "cg_taker_flow_z120"]
        frames.append(t[["symbol", "cg_date"] + cols].set_index(["symbol", "cg_date"]))
        feat_names += cols

    # LIQ
    if liq is not None:
        l = liq[["symbol", "cg_date", "liq_long_usd", "liq_short_usd"]].copy()
        total = l["liq_long_usd"] + l["liq_short_usd"]
        l["_imb"] = (l["liq_long_usd"] - l["liq_short_usd"]) / (total + eps)
        l["_log"] = np.log1p(total)
        rows = []
        for _, g in l.groupby("symbol"):
            g = g.sort_values("cg_date").copy()
            g["cg_liq_imb_z120"] = _zscore_series(g["_imb"], ROLL_120)
            g["cg_liq_log_z120"] = _zscore_series(g["_log"], ROLL_120)
            g["cg_liq_spike"]    = (_zscore_series(g["_log"], ROLL_120) > 2.0).astype(float)
            rows.append(g)
        l = pd.concat(rows)
        cols = ["cg_liq_imb_z120", "cg_liq_log_z120", "cg_liq_spike"]
        frames.append(l[["symbol", "cg_date"] + cols].set_index(["symbol", "cg_date"]))
        feat_names += cols

    # OI
    if oi_df is not None:
        o = oi_df[["symbol", "cg_date", "oi_close"]].copy()
        rows = []
        for _, g in o.groupby("symbol"):
            g = g.sort_values("cg_date").copy()
            g["cg_oi_z120"]         = _zscore_series(g["oi_close"], ROLL_120)
            g["cg_oi_notional_chg"] = g["oi_close"].pct_change(1).clip(-1, 1)
            chg_z = _zscore_series(g["cg_oi_notional_chg"].fillna(0), ROLL_120)
            g["cg_oi_surge"]        = (chg_z > 2.0).astype(float)
            rows.append(g)
        o = pd.concat(rows)
        cols = ["cg_oi_z120", "cg_oi_notional_chg", "cg_oi_surge"]
        frames.append(o[["symbol", "cg_date"] + cols].set_index(["symbol", "cg_date"]))
        feat_names += cols

    # FUNDING
    if funding is not None:
        f = funding[["symbol", "cg_date", "fr_close"]].copy()
        rows = []
        for _, g in f.groupby("symbol"):
            g = g.sort_values("cg_date").copy()
            g["cg_fr_z120"]       = _zscore_series(g["fr_close"], ROLL_120)
            g["cg_fr_accel"]      = g["fr_close"].diff(1)
            g["cg_fr_accel_z120"] = _zscore_series(g["cg_fr_accel"].fillna(0), ROLL_120)
            rows.append(g)
        f = pd.concat(rows)
        cols = ["cg_fr_z120", "cg_fr_accel", "cg_fr_accel_z120"]
        frames.append(f[["symbol", "cg_date"] + cols].set_index(["symbol", "cg_date"]))
        feat_names += cols

    # LS RATIO
    if ls is not None:
        s = ls[["symbol", "cg_date", "ls_ratio"]].copy()
        rows = []
        for _, g in s.groupby("symbol"):
            g = g.sort_values("cg_date").copy()
            g["cg_ls_z120"]     = _zscore_series(g["ls_ratio"], ROLL_120)
            g["_ls_chg"]        = g["ls_ratio"].diff(1)
            g["cg_ls_chg_z120"] = _zscore_series(g["_ls_chg"].fillna(0), ROLL_120)
            rows.append(g)
        s = pd.concat(rows)
        cols = ["cg_ls_z120", "cg_ls_chg_z120"]
        frames.append(s[["symbol", "cg_date"] + cols].set_index(["symbol", "cg_date"]))
        feat_names += cols

    cg_all = frames[0].copy()
    for fr in frames[1:]:
        cg_all = cg_all.join(fr, how="outer")
    cg_all = cg_all.reset_index().replace([np.inf, -np.inf], np.nan)
    log(f"  CG feature table: {len(cg_all):,} rows × {len(feat_names)} features")

    # Merge shift1 into research frame
    df2 = df.copy()
    df2["_cg_date"] = df2["timestamp"].dt.normalize() - pd.Timedelta(days=1)
    merged = df2.merge(
        cg_all.rename(columns={"cg_date": "_cg_date"}),
        on=["symbol", "_cg_date"], how="left",
    ).drop(columns=["_cg_date"]).replace([np.inf, -np.inf], np.nan)

    # Coverage gate
    tz = merged["timestamp"].dt.tz
    test_slice = merged[merged["timestamp"] >= pd.Timestamp(TEST_START, tz=tz)]
    passed = []
    for feat in feat_names:
        cov = test_slice[feat].notna().mean()
        status = "✓" if cov >= COVERAGE_THR else "✗ DROPPED"
        log(f"    {status} {feat:<30}: coverage {cov:.1%}")
        if cov >= COVERAGE_THR:
            passed.append(feat)

    log(f"  Features passing coverage gate: {len(passed)}/{len(feat_names)}")
    return merged, passed


# ─── R83: IC Scan ─────────────────────────────────────────────────────────────

def ic_scan(merged, feats, existing_feats):
    """IC scan + redundancy gate."""
    tz = merged["timestamp"].dt.tz
    rows = []
    for feat in feats:
        if feat not in merged.columns:
            continue
        valid = merged[[feat, "fwd_ret_12h", "timestamp"]].dropna()
        if len(valid) < 50:
            continue
        pooled_ic = float(stats.spearmanr(valid[feat], valid["fwd_ret_12h"])[0])

        window_ics = []
        for w in CONTINUOUS_WINDOWS:
            ts_s = pd.Timestamp(w["test_start"], tz=tz)
            ts_e = pd.Timestamp(w["test_end"], tz=tz)
            wdf = valid[(valid["timestamp"] >= ts_s) & (valid["timestamp"] <= ts_e)]
            if len(wdf) < 50:
                window_ics.append(np.nan)
                continue
            wic = float(stats.spearmanr(wdf[feat], wdf["fwd_ret_12h"])[0])
            window_ics.append(wic if not np.isnan(wic) else 0.0)

        stability = sum(1 for ic in window_ics if not np.isnan(ic) and abs(ic) >= 0.02) / 3.0

        max_corr = 0.0
        for ef in existing_feats:
            if ef not in merged.columns:
                continue
            sub = merged[[feat, ef]].dropna()
            if len(sub) < 50:
                continue
            c = abs(float(stats.spearmanr(sub[feat], sub[ef])[0]))
            if c > max_corr:
                max_corr = c

        test_slice = merged[merged["timestamp"] >= pd.Timestamp(TEST_START, tz=tz)]
        coverage = test_slice[feat].notna().mean()
        score = abs(pooled_ic) * stability

        pass_ic = abs(pooled_ic) >= 0.03
        pass_stab = stability >= 2 / 3
        pass_redund = max_corr < 0.7
        pass_cov = coverage >= COVERAGE_THR

        rows.append({
            "feature": feat, "pooled_ic": round(pooled_ic, 4),
            "w1_ic": round(window_ics[0], 4) if not np.isnan(window_ics[0]) else None,
            "w2_ic": round(window_ics[1], 4) if not np.isnan(window_ics[1]) else None,
            "w3_ic": round(window_ics[2], 4) if not np.isnan(window_ics[2]) else None,
            "stability": round(stability, 3), "max_corr_existing": round(max_corr, 3),
            "coverage_test": round(coverage, 3), "score": round(score, 4),
            "pass_ic": pass_ic, "pass_stab": pass_stab,
            "pass_redund": pass_redund, "pass_cov": pass_cov,
            "gate_pass": pass_ic and pass_stab and pass_redund and pass_cov,
        })

    ic_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return ic_df


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    log("=" * 70)
    log("  R86 — CLEAN RERUN (R82→R85) using R68 canonical data pipeline")
    log("=" * 70)

    # ── [0] Load data via R68 ─────────────────────────────────────────────────
    log("\n[0] Loading data via R68 load_data() …")
    df, regime_df = load_data()
    feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]
    log(f"  Frame: {len(df):,} rows, {df['symbol'].nunique()} symbols, {len(feats)} features")

    # ── [1] R82 — CG features ────────────────────────────────────────────────
    log("\n[1] R82 — Building CG z-score features …")
    merged, cg_feats = build_cg_features(df)
    log(f"  {len(cg_feats)} CG features built")

    r82_path = RESULTS_DIR / "r86_r82_feat_list.json"
    r82_path.write_text(json.dumps(cg_feats))

    # ── [2] R83 — IC scan ────────────────────────────────────────────────────
    log("\n[2] R83 — IC scan …")
    ic_df = ic_scan(merged, cg_feats, feats)
    ic_path = RESULTS_DIR / "r86_r83_ic_table.csv"
    ic_df.to_csv(ic_path, index=False)

    log(f"\n  IC SCAN RESULTS:")
    log(f"  {'Feature':<35} {'IC':>8} {'Stab':>6} {'MaxCorr':>8} {'Cov':>7} {'Score':>7} Gate")
    log(f"  {'-' * 75}")
    for _, r in ic_df.iterrows():
        gate = "✅" if r["gate_pass"] else "✗"
        log(f"  {str(r['feature']):<35} {r['pooled_ic']:>8.4f} {r['stability']:>6.2f} "
            f"{r['max_corr_existing']:>8.3f} {r['coverage_test']:>6.1%} {r['score']:>7.4f} {gate}")

    passed = ic_df[ic_df["gate_pass"] == True]
    top_feats = list(passed["feature"].head(2)) if len(passed) > 0 else []

    if not top_feats:
        log("\n  ⚠  No CG features passed gate — R84 CG experiments skipped.")
    else:
        log(f"\n  Top features: {top_feats}")

    # ── [3] R84 — Baseline + experiments ─────────────────────────────────────
    log("\n[3] R84 — Training R68 baseline (canonical) …")
    cfg_4l2s = {**PROD_CFG, "n_long": 4, "n_short": 2}

    preds_base = train_ensemble(df, feats, CONTINUOUS_WINDOWS, seeds=SEEDS,
                                cs_rank_exclude=no_rank)
    base_port = simulate(preds_base, regime_df, n_long=4, n_short=2, cfg=cfg_4l2s)
    base_m = portfolio_metrics(base_port)
    base_m["label"] = "R68_baseline"
    log(f"  Baseline: Sharpe={base_m['net_sharpe']}  MaxDD={base_m['max_dd_pct']}%  "
        f"n_periods={base_m['n_periods']}")

    base_port.to_csv(RESULTS_DIR / "r86_r84_baseline_equity.csv", index=False)
    all_r84 = [base_m]

    # CG experiments (if any passed gate)
    r84_best_port = None
    if len(top_feats) >= 1:
        # Merge CG features into df for training
        cg_cols = [c for c in top_feats if c in merged.columns]
        feat_src = merged[["symbol", "timestamp"] + cg_cols].copy()
        df_exp = df.merge(feat_src, on=["symbol", "timestamp"], how="left", suffixes=("", "_r82"))

        non_rank_patterns = {"_chg", "_accel", "_surge", "_spike", "_accel_z"}
        no_rank_exp = list(set(no_rank) | {
            f for f in cg_cols if any(p in f for p in non_rank_patterns)
            or f in MARKET_LEVEL_FEATURES
        })

        # Exp1: +top1
        top1 = [top_feats[0]]
        all_feats_1 = feats + top1
        log(f"\n  R84 Exp1: +{top1} ({len(all_feats_1)} features) …")
        preds_1 = train_ensemble(df_exp, all_feats_1, CONTINUOUS_WINDOWS, seeds=SEEDS,
                                 cs_rank_exclude=no_rank_exp)
        port_1 = simulate(preds_1, regime_df, n_long=4, n_short=2, cfg=cfg_4l2s)
        m1 = portfolio_metrics(port_1)
        m1["label"] = f"R84_exp1_{top1[0]}"
        all_r84.append(m1)

        sh_delta = m1["net_sharpe"] - base_m["net_sharpe"]
        dd_delta = m1["max_dd_pct"] - base_m["max_dd_pct"]
        dd_improv = -dd_delta / (abs(base_m["max_dd_pct"]) + EPS) * 100
        exp1_ok = (sh_delta >= 0.10) or (dd_improv >= 15 and sh_delta >= -0.05)
        log(f"  Exp1: Sharpe={m1['net_sharpe']} ΔSh={sh_delta:+.3f} DD↓={dd_improv:+.1f}%  "
            f"{'✅' if exp1_ok else '✗'}")

        if exp1_ok:
            r84_best_port = port_1
            port_1.to_csv(RESULTS_DIR / "r86_r84_exp1_equity.csv", index=False)

        # Exp2: +top2 (only if we have 2 features and exp1 passed)
        if len(top_feats) >= 2 and exp1_ok:
            all_feats_2 = feats + top_feats
            log(f"\n  R84 Exp2: +{top_feats} ({len(all_feats_2)} features) …")
            preds_2 = train_ensemble(df_exp, all_feats_2, CONTINUOUS_WINDOWS, seeds=SEEDS,
                                     cs_rank_exclude=no_rank_exp)
            port_2 = simulate(preds_2, regime_df, n_long=4, n_short=2, cfg=cfg_4l2s)
            m2 = portfolio_metrics(port_2)
            m2["label"] = f"R84_exp2_{'+'.join(top_feats)}"
            all_r84.append(m2)
            sh2 = m2["net_sharpe"] - base_m["net_sharpe"]
            dd2_i = -(m2["max_dd_pct"] - base_m["max_dd_pct"]) / (abs(base_m["max_dd_pct"]) + EPS) * 100
            exp2_ok = (sh2 >= 0.10) or (dd2_i >= 15 and sh2 >= -0.05)
            log(f"  Exp2: Sharpe={m2['net_sharpe']} ΔSh={sh2:+.3f} DD↓={dd2_i:+.1f}%  "
                f"{'✅' if exp2_ok else '✗'}")
            if exp2_ok:
                r84_best_port = port_2
                port_2.to_csv(RESULTS_DIR / "r86_r84_exp2_equity.csv", index=False)

    r84_summary = {"script": "r86_r84", "baseline": base_m, "experiments": all_r84[1:]}
    (RESULTS_DIR / "r86_r84_summary.json").write_text(json.dumps(r84_summary, indent=2, default=float))

    log(f"\n  R84 RESULTS:")
    log(f"  {'Label':<40} {'NetSh':>8} {'ΔSh':>8} {'MaxDD%':>8} {'Calmar':>8}")
    log(f"  {'-' * 72}")
    for m in all_r84:
        dsh = m["net_sharpe"] - base_m["net_sharpe"]
        log(f"  {m['label']:<40} {m['net_sharpe']:>8.3f} {dsh:>+8.3f} {m['max_dd_pct']:>7.1f}% {m['calmar']:>8.3f}")

    # ── [4] R85 — Bootstrap ──────────────────────────────────────────────────
    log("\n[4] R85 — Block bootstrap (N=1000, block=10) …")
    rets_base = base_port["net_ret"].values.astype(float)
    r85_results = {}

    # R81 vs R68
    r81_path = RESULTS_DIR / "r81_best_equity.csv"
    if r81_path.exists():
        r81_eq = pd.read_csv(r81_path, parse_dates=["timestamp"])
        rets_r81 = r81_eq["net_ret"].values.astype(float)
        n = min(len(rets_base), len(rets_r81))
        log(f"  R81 vs R68: {n} common periods")
        bsr = block_bootstrap_sharpe(rets_base[:n], rets_r81[:n])
        r85_results["R81_best_vs_R68"] = bsr
        accept = bsr["p_exp_better"] > 0.8 and bsr["median_delta"] > 0.08
        log(f"  R81: P(better)={bsr['p_exp_better']}  medianΔSh={bsr['median_delta']:+.4f}  "
            f"{'✅ ACCEPT' if accept else '✗ REJECT'}")
    else:
        log("  R81 best equity not found — skipping")

    # R84 best vs R68
    if r84_best_port is not None:
        rets_r84 = r84_best_port["net_ret"].values.astype(float)
        n = min(len(rets_base), len(rets_r84))
        bsr2 = block_bootstrap_sharpe(rets_base[:n], rets_r84[:n])
        r85_results["R84_best_vs_R68"] = bsr2
        accept2 = bsr2["p_exp_better"] > 0.8 and bsr2["median_delta"] > 0.08
        log(f"  R84: P(better)={bsr2['p_exp_better']}  medianΔSh={bsr2['median_delta']:+.4f}  "
            f"{'✅ ACCEPT' if accept2 else '✗ REJECT'}")

    r85_out = {"script": "r86_r85_bootstrap", "results": r85_results}
    (RESULTS_DIR / "r86_r85_summary.json").write_text(json.dumps(r85_out, indent=2, default=float))

    # ── Summary ──────────────────────────────────────────────────────────────
    runtime = time.time() - t_start
    log(f"\n{'=' * 70}")
    log(f"  R86 RERUN COMPLETE — {runtime / 60:.1f} min")
    log(f"{'=' * 70}")
    log("  Artifacts:")
    for f in sorted(RESULTS_DIR.glob("r86_*")):
        log(f"    {f.name}")
    log("  DONE.")


if __name__ == "__main__":
    main()
