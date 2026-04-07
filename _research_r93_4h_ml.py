#!/usr/bin/env python3
"""
R93 — 4h ML Engine: LGB+XGB ensemble on fwd_ret_4h target.

Same model architecture as R68 but:
- Target: fwd_ret_4h (instead of fwd_ret_12h)
- Rebalance: every 4h (instead of 12h)
- Grid: K ∈ {4L/2S, 3L/3S, 6L/3S}, with/without trend filter
- Cost model adjusted for 4h holding period
"""

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Set

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EPS = 1e-10
# 4h bars: 6 per day, 365 days
PERIODS_PER_YEAR_4H = 6 * 365

from _research_r68_continuous_wf import (
    load_data, CONTINUOUS_WINDOWS, SEEDS, PROD_CFG,
    CHAMPION_FEAT_31, MARKET_LEVEL_FEATURES,
    LGB_PARAMS, XGB_PARAMS, N_ROUNDS, EARLY_STOP,
    TIER1_SYMS, TIER2_SYMS, TIER3_SYMS, _cost_for_sym,
)
from _research_r22_models import cs_rank_cols


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def sharpe_4h(rets_series, periods_per_year=PERIODS_PER_YEAR_4H):
    if len(rets_series) < 2:
        return 0.0
    eq = (1 + rets_series).cumprod()
    r = eq.pct_change().dropna()
    return float(r.mean() / (r.std() + EPS) * np.sqrt(periods_per_year))


def max_dd(rets_series):
    eq = (1 + rets_series).cumprod()
    return float((eq / eq.cummax() - 1).min())


def portfolio_metrics(port: pd.DataFrame, label: str = "") -> dict:
    rets = port["net_ret"]
    s = sharpe_4h(rets)
    dd = max_dd(rets)
    return {
        "label": label,
        "net_sharpe": round(s, 4),
        "gross_sharpe": round(sharpe_4h(port["gross_ret"]), 4),
        "max_dd_pct": round(dd * 100, 2),
        "calmar": round(s / (abs(dd) + EPS), 3),
        "total_ret_pct": round(float((1 + rets).prod() - 1) * 100, 1),
        "win_rate": round(float((rets > 0).mean()), 3),
        "n_periods": len(rets),
    }


# ── Train ensemble on 4h target ──────────────────────────────────────────────

def train_ensemble_4h(df, feats, windows, seeds=SEEDS, cs_rank_exclude=None):
    """Same as R68 train_ensemble but uses fwd_ret_4h as target."""
    avail = [f for f in feats if f in df.columns]
    rank_exclude = set(cs_rank_exclude or [])
    rank_feats = [f for f in avail if f not in rank_exclude]
    tz = df["timestamp"].dt.tz
    all_lgb, all_xgb = [], []

    for seed in seeds:
        p_lgb = {**LGB_PARAMS, "seed": seed}
        p_xgb = {**XGB_PARAMS, "seed": seed}
        for w in windows:
            te_end = pd.Timestamp(w["test_end"], tz=tz)
            te_start = pd.Timestamp(w["test_start"], tz=tz)
            tr_end = pd.Timestamp(w["train_end"], tz=tz)
            va_start = pd.Timestamp(w["val_start"], tz=tz)
            va_end = pd.Timestamp(w["val_end"], tz=tz)

            train_ = df[df["timestamp"] < tr_end].copy()
            val_ = df[(df["timestamp"] >= va_start) & (df["timestamp"] < va_end)].copy()
            test_ = df[(df["timestamp"] >= te_start) & (df["timestamp"] <= te_end)].copy()
            if len(train_) < 5000 or len(test_) < 200:
                continue
            if rank_feats:
                train_ = cs_rank_cols(train_, rank_feats)
                val_ = cs_rank_cols(val_, rank_feats)
                test_ = cs_rank_cols(test_, rank_feats)

            # 4h target
            for d in [train_, val_, test_]:
                d["target_binary"] = (d["fwd_ret_4h"] > 0).astype(int)

            for col in avail:
                for d in [train_, val_, test_]:
                    if d[col].isna().any():
                        d[col] = d[col].fillna(0)

            tr = train_[avail + ["target_binary"]].dropna()
            va = val_[avail + ["target_binary"]].dropna()
            te = test_[avail + ["target_binary", "timestamp", "symbol"]].dropna()
            fwd = test_[["timestamp", "symbol", "fwd_ret_4h"]].rename(
                columns={"fwd_ret_4h": "fwd_ret"}).dropna()

            for d in [tr, va, te]:
                d.replace([np.inf, -np.inf], np.nan, inplace=True)
            tr, va, te = tr.dropna(), va.dropna(), te.dropna()
            if len(te) == 0:
                continue

            # LGB
            dt = lgb.Dataset(tr[avail], label=tr["target_binary"])
            dv = lgb.Dataset(va[avail], label=va["target_binary"])
            m = lgb.train(p_lgb, dt, num_boost_round=N_ROUNDS,
                          valid_sets=[dv],
                          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                     lgb.log_evaluation(-1)])
            p = m.predict(te[avail])
            rec = te[["timestamp", "symbol"]].copy()
            rec["pred_lgb"] = p
            rec = rec.merge(fwd, on=["timestamp", "symbol"], how="inner")
            rec["window"] = w["name"]
            rec["seed"] = seed
            all_lgb.append(rec)

            # XGB
            dt_x = xgb.DMatrix(tr[avail], label=tr["target_binary"])
            dv_x = xgb.DMatrix(va[avail], label=va["target_binary"])
            m_x = xgb.train(p_xgb, dt_x, num_boost_round=N_ROUNDS,
                             evals=[(dv_x, "val")],
                             early_stopping_rounds=EARLY_STOP, verbose_eval=False)
            p_x = m_x.predict(xgb.DMatrix(te[avail]))
            rec2 = te[["timestamp", "symbol"]].copy()
            rec2["pred_xgb"] = p_x
            rec2 = rec2.merge(fwd, on=["timestamp", "symbol"], how="inner")
            rec2["window"] = w["name"]
            rec2["seed"] = seed
            all_xgb.append(rec2)

            if seed == seeds[0]:
                log(f"  {w['name']}/s{seed}: train={len(tr):,} test={len(te):,}")

    if not all_lgb:
        return None
    lgb_df = pd.concat(all_lgb)
    xgb_df = pd.concat(all_xgb)
    lgb_avg = lgb_df.groupby(["timestamp", "symbol"]).agg(
        pred_lgb=("pred_lgb", "mean"), fwd_ret=("fwd_ret", "first"),
        window=("window", "first")).reset_index()
    xgb_avg = xgb_df.groupby(["timestamp", "symbol"]).agg(
        pred_xgb=("pred_xgb", "mean")).reset_index()
    merged = lgb_avg.merge(xgb_avg, on=["timestamp", "symbol"], how="inner")
    merged["raw_prob"] = 0.5 * merged["pred_lgb"] + 0.5 * merged["pred_xgb"]
    merged["rank_lgb"] = merged.groupby("timestamp")["pred_lgb"].rank(pct=True) - 0.5
    merged["rank_xgb"] = merged.groupby("timestamp")["pred_xgb"].rank(pct=True) - 0.5
    merged["pred"] = 0.5 * merged["rank_lgb"] + 0.5 * merged["rank_xgb"]
    return merged[["timestamp", "symbol", "pred", "raw_prob", "fwd_ret", "window"]]


# ── Simulate 4h strategy ─────────────────────────────────────────────────────

def simulate_4h(merged, regime_df, n_long, n_short, rebal_hours=4, cfg=None):
    """Simulate with 4h rebalance. rebal_hours in 1h bars (so 4 = every 4h)."""
    if cfg is None:
        cfg = PROD_CFG
    trend_cutoff = cfg.get("trend_cutoff", 0.9)
    ema_alpha = cfg.get("ema_alpha", None)
    hysteresis = cfg.get("hysteresis", 0)
    # Funding cost proportional: 4h/12h of the 12h rate
    funding_per_4h = 0.00008 * (4 / 12)

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

        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl == 0 and ns == 0:
            continue

        exposure = 1.0
        dyn_threshold = cfg.get("dyn_threshold", 0.5)
        if dyn_threshold is not None and trend_str > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) /
                           (trend_cutoff - dyn_threshold + 1e-10) * 0.5)

        if ema_alpha is not None and ema_alpha < 1.0:
            for idx, r in grp.iterrows():
                sym = r["symbol"]
                raw_pred = r["pred"]
                smoothed = ema_alpha * raw_pred + (1 - ema_alpha) * prev_preds.get(sym, raw_pred)
                prev_preds[sym] = smoothed
                grp.at[idx, "pred"] = smoothed

        grp["pred_rank"] = grp["pred"].rank(ascending=False)

        if hysteresis > 0 and (prev_longs or prev_shorts):
            new_longs: Set[str] = set()
            new_shorts: Set[str] = set()
            for idx, r in grp.iterrows():
                sym, rank = r["symbol"], r["pred_rank"]
                if sym in prev_longs and rank <= nl + hysteresis:
                    new_longs.add(sym)
                elif sym in prev_shorts and rank > (n - ns - hysteresis):
                    new_shorts.add(sym)
            for _, r in grp[~grp["symbol"].isin(new_longs | new_shorts)].sort_values("pred_rank").head(nl - len(new_longs)).iterrows():
                new_longs.add(r["symbol"])
            for _, r in grp[~grp["symbol"].isin(new_longs | new_shorts)].sort_values("pred_rank", ascending=False).head(ns - len(new_shorts)).iterrows():
                new_shorts.add(r["symbol"])
        else:
            new_longs = set(grp[grp["pred_rank"] <= nl]["symbol"].tolist()) if nl > 0 else set()
            new_shorts = set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist()) if ns > 0 else set()

        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        total_positions = len(new_longs) + len(new_shorts)

        longs = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]
        long_ret = longs["fwd_ret"].mean() if len(longs) > 0 else 0
        short_ret = shorts["fwd_ret"].mean() if len(shorts) > 0 else 0

        nl_act, ns_act = len(new_longs), len(new_shorts)
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
            holding_cost = funding_per_4h * (rebal_hours / 4)
            total_cost = turnover_cost + holding_cost
        else:
            total_cost = 0.0

        net_ret = gross_ret - total_cost
        prev_longs, prev_shorts = new_longs, new_shorts

        all_rets.append({
            "timestamp": ts, "gross_ret": gross_ret, "net_ret": net_ret,
            "cost": total_cost, "n_long": nl_act, "n_short": ns_act,
            "turnover": len(new_opened) + len(closed),
        })

    return pd.DataFrame(all_rets) if all_rets else pd.DataFrame()


# ── Also simulate 4h predictions but with 12h rebalance (subsample) ──────────

def simulate_4h_12h_rebal(merged, regime_df, n_long, n_short, cfg=None):
    """Use 4h model predictions but rebalance every 12h (same as R68 cadence)."""
    return simulate_4h(merged, regime_df, n_long, n_short, rebal_hours=12, cfg=cfg)


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def block_bootstrap_sharpe(rets_base, rets_exp, n_boot=1000, block=10, seed=42,
                           ppy=PERIODS_PER_YEAR_4H):
    rng = np.random.default_rng(seed)
    n = min(len(rets_base), len(rets_exp))
    rb, re = np.array(rets_base[:n], dtype=float), np.array(rets_exp[:n], dtype=float)
    n_blocks = n // block

    def _sh(r):
        if len(r) < 2 or np.std(r) < EPS:
            return 0.0
        return float(np.mean(r) / (np.std(r) + EPS) * np.sqrt(ppy))

    sb_list, se_list = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n - block + 1, size=n_blocks)
        block_idx = np.concatenate([np.arange(i, i + block) for i in idx])[:n]
        sb_list.append(_sh(rb[block_idx]))
        se_list.append(_sh(re[block_idx]))

    sb, se = np.array(sb_list), np.array(se_list)
    delta = se - sb
    return {
        "p_exp_better": round(float((se > sb).mean()), 3),
        "median_delta": round(float(np.median(delta)), 4),
        "mean_delta": round(float(np.mean(delta)), 4),
        "p5_delta": round(float(np.percentile(delta, 5)), 4),
        "p95_delta": round(float(np.percentile(delta, 95)), 4),
        "base_sharpe_med": round(float(np.median(sb)), 4),
        "exp_sharpe_med": round(float(np.median(se)), 4),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("=" * 70)
    log("  R93 — 4h ML ENGINE (LGB+XGB on fwd_ret_4h)")
    log("=" * 70)

    # Load data via R68 canonical pipeline
    log("\n[0] Loading data via R68 load_data() ...")
    df, regime_df = load_data()

    # Verify fwd_ret_4h exists
    if "fwd_ret_4h" not in df.columns:
        log("  ✗ FATAL: fwd_ret_4h not in dataframe!")
        return
    n_valid = df["fwd_ret_4h"].notna().sum()
    log(f"  fwd_ret_4h: {n_valid:,}/{len(df):,} valid ({n_valid/len(df):.1%})")

    feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]
    log(f"  Features: {len(feats)}/31")

    # Train ensemble with 4h target
    log("\n[1] Training 4h ensemble (5 LGB + 5 XGB × 3 windows) ...")
    t1 = time.time()
    preds = train_ensemble_4h(df, feats, CONTINUOUS_WINDOWS, seeds=SEEDS,
                               cs_rank_exclude=no_rank)
    if preds is None:
        log("  ✗ No predictions produced!")
        return
    log(f"  Done in {time.time()-t1:.0f}s, {len(preds):,} predictions")

    # Save predictions
    preds.to_parquet(RESULTS_DIR / "r93_predictions.parquet", index=False)

    # Grid search: different K configs and rebalance cadences
    log("\n[2] Grid search ...")
    configs = [
        # 4h rebalance
        {"n_long": 4, "n_short": 2, "rebal": 4, "label": "4L2S_4h", "use_trend": True},
        {"n_long": 3, "n_short": 3, "rebal": 4, "label": "3L3S_4h", "use_trend": True},
        {"n_long": 6, "n_short": 3, "rebal": 4, "label": "6L3S_4h", "use_trend": True},
        {"n_long": 4, "n_short": 2, "rebal": 4, "label": "4L2S_4h_notrend", "use_trend": False},
        # 8h rebalance
        {"n_long": 4, "n_short": 2, "rebal": 8, "label": "4L2S_8h", "use_trend": True},
        {"n_long": 6, "n_short": 3, "rebal": 8, "label": "6L3S_8h", "use_trend": True},
        # 12h rebalance (same cadence as R68 but 4h model)
        {"n_long": 4, "n_short": 2, "rebal": 12, "label": "4L2S_12h", "use_trend": True},
        {"n_long": 6, "n_short": 3, "rebal": 12, "label": "6L3S_12h", "use_trend": True},
        {"n_long": 4, "n_short": 2, "rebal": 12, "label": "4L2S_12h_notrend", "use_trend": False},
    ]

    results = []
    best_sharpe = -999
    best_port = None
    best_label = ""

    for cfg in configs:
        sim_cfg = {**PROD_CFG}
        if not cfg["use_trend"]:
            sim_cfg["trend_cutoff"] = 99.0  # effectively disable

        port = simulate_4h(preds, regime_df,
                           n_long=cfg["n_long"], n_short=cfg["n_short"],
                           rebal_hours=cfg["rebal"], cfg=sim_cfg)

        if len(port) == 0:
            log(f"  {cfg['label']}: NO DATA")
            continue

        m = portfolio_metrics(port, label=f"R93_{cfg['label']}")
        results.append(m)

        log(f"  {cfg['label']:<22}: Sharpe={m['net_sharpe']:>7.3f}  MaxDD={m['max_dd_pct']:>7.1f}%  "
            f"Ret={m['total_ret_pct']:>7.1f}%  Win={m['win_rate']:.3f}  N={m['n_periods']}")

        if m["net_sharpe"] > best_sharpe:
            best_sharpe = m["net_sharpe"]
            best_port = port
            best_label = cfg["label"]

    # Correlation with R68
    log("\n[3] Correlation with R68 ...")
    r68_equity_path = RESULTS_DIR / "r86_r84_baseline_equity.csv"
    corr_with_r68 = None
    if r68_equity_path.exists() and best_port is not None:
        r68_eq = pd.read_csv(r68_equity_path, parse_dates=["timestamp"])
        r68_rets = r68_eq.set_index("timestamp")["net_ret"]
        r93_rets = best_port.set_index("timestamp")["net_ret"]
        common = r68_rets.index.intersection(r93_rets.index)
        if len(common) > 20:
            corr_with_r68 = float(r68_rets.loc[common].corr(r93_rets.loc[common]))
            log(f"  Corr(R93_best, R68) = {corr_with_r68:.3f}")
        else:
            log(f"  Not enough common periods ({len(common)}), trying daily aggregation ...")
            r68_daily = r68_rets.groupby(r68_rets.index.date).sum()
            r93_daily = r93_rets.groupby(r93_rets.index.date).sum()
            common_d = r68_daily.index.intersection(r93_daily.index)
            if len(common_d) > 20:
                corr_with_r68 = float(r68_daily.loc[common_d].corr(r93_daily.loc[common_d]))
                log(f"  Corr(R93_best, R68) daily = {corr_with_r68:.3f}")
    else:
        log("  R68 equity not found")

    # Bootstrap vs cash
    log("\n[4] Bootstrap: R93 vs cash ...")
    if best_port is not None:
        rets_best = best_port["net_ret"].values.astype(float)

        def bootstrap_vs_cash(rets, n_boot=1000, block=10, seed=42):
            rng = np.random.default_rng(seed)
            n = len(rets)
            n_blocks = n // block
            s_list = []
            for _ in range(n_boot):
                idx = rng.integers(0, n - block + 1, size=n_blocks)
                block_idx = np.concatenate([np.arange(i, i + block) for i in idx])[:n]
                r = rets[block_idx]
                if len(r) < 2 or np.std(r) < EPS:
                    s_list.append(0.0)
                else:
                    s_list.append(float(np.mean(r) / (np.std(r) + EPS) * np.sqrt(PERIODS_PER_YEAR_4H)))
            s_arr = np.array(s_list)
            return {
                "p_positive": round(float((s_arr > 0).mean()), 3),
                "sharpe_med": round(float(np.median(s_arr)), 4),
                "sharpe_p5": round(float(np.percentile(s_arr, 5)), 4),
                "sharpe_p95": round(float(np.percentile(s_arr, 95)), 4),
            }

        bs = bootstrap_vs_cash(rets_best)
        log(f"  P(Sharpe>0)={bs['p_positive']}  MedianSh={bs['sharpe_med']}  "
            f"[{bs['sharpe_p5']}, {bs['sharpe_p95']}]")

    # Save results
    log("\n[5] Saving ...")
    summary = {
        "script": "r93_4h_ml_engine",
        "target": "fwd_ret_4h",
        "model": "5xLGB + 5xXGB (same as R68)",
        "features": len(feats),
        "best_config": best_label,
        "best_sharpe": round(best_sharpe, 4),
        "corr_with_r68": corr_with_r68,
        "bootstrap_vs_cash": bs if best_port is not None else None,
        "grid_results": results,
        "runtime_sec": round(time.time() - t0, 1),
    }
    (RESULTS_DIR / "r93_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))

    if best_port is not None:
        best_port.to_csv(RESULTS_DIR / "r93_best_equity.csv", index=False)
        log(f"  Saved: r93_best_equity.csv ({best_label})")

    pd.DataFrame(results).to_csv(RESULTS_DIR / "r93_grid.csv", index=False)

    # Summary table
    log(f"\n{'=' * 70}")
    log(f"  R93 RESULTS — 4h ML Engine")
    log(f"{'=' * 70}")
    log(f"  {'Config':<24} {'NetSh':>8} {'GrSh':>8} {'Ret%':>8} {'DD%':>8} {'Win':>6} {'N':>6}")
    log(f"  {'-' * 68}")
    for r in sorted(results, key=lambda x: x["net_sharpe"], reverse=True):
        log(f"  {r['label']:<24} {r['net_sharpe']:>8.3f} {r['gross_sharpe']:>8.3f} "
            f"{r['total_ret_pct']:>7.1f}% {r['max_dd_pct']:>7.1f}% {r['win_rate']:>6.3f} {r['n_periods']:>6}")

    if corr_with_r68 is not None:
        log(f"\n  Corr(R93_best, R68) = {corr_with_r68:.3f}")
    log(f"  Best: {best_label}, Sharpe={best_sharpe:.3f}")
    log(f"  Runtime: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    main()
