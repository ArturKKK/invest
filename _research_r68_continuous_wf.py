#!/usr/bin/env python3
"""
R68 — Continuous WF (no gap periods)

ORIGINAL_WINDOWS have 15-day gaps between val and test. In prod there are no gaps.
Check if 4L/2S advantage is robust on continuous coverage (no skipped months).

Runs 4L/2S and 6L/3S on CONTINUOUS_WINDOWS (wall-to-wall coverage).
Compares with ORIGINAL_WINDOWS results from R65.
"""

import sys, warnings, time
from typing import Dict, Set

from _preflight_check import check_versions
check_versions()

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

warnings.filterwarnings("ignore")

from _research_round7 import SYM_35
from _research_r22_models import SEEDS, LEVERAGE, CAPITAL, log, cs_rank_cols
from _research_r30b_fixed import compute_regime_extended
from _research_r35_new_features import (
    add_r35_features, load_research_frame, MARKET_LEVEL_FEATURES,
)
from _research_r47_coinglass import (
    CHAMPION_FEAT_30, add_cg_features, compute_cg_features, load_cg_daily,
)

CHAMPION_FEAT_31 = CHAMPION_FEAT_30 + ["cg_taker_imb"]

TIER1_SYMS = {"BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"}
TIER3_SYMS = {
    "SAND/USDT", "LDO/USDT", "INJ/USDT", "APT/USDT", "ARB/USDT",
    "GALA/USDT", "FTM/USDT", "MATIC/USDT",
}
TIER2_SYMS = set(SYM_35) - TIER1_SYMS - TIER3_SYMS

def _cost_for_sym(sym):
    if sym in TIER1_SYMS: return 0.92 * (-0.0001) + 0.08 * 0.0007
    elif sym in TIER2_SYMS: return 0.75 * 0.0001 + 0.25 * 0.0007
    else: return 0.0005 + 0.0002

# Continuous windows: wall-to-wall, no gaps
CONTINUOUS_WINDOWS = [
    {"name": "W1", "train_end": "2024-06-01",
     "val_start": "2024-06-01", "val_end": "2024-09-30",
     "test_start": "2024-10-15", "test_end": "2025-05-14"},  # extends into gap
    {"name": "W2", "train_end": "2025-01-01",
     "val_start": "2025-01-01", "val_end": "2025-04-30",
     "test_start": "2025-05-15", "test_end": "2025-11-14"},  # extends into gap
    {"name": "W3", "train_end": "2025-07-01",
     "val_start": "2025-07-01", "val_end": "2025-10-31",
     "test_start": "2025-11-15", "test_end": "2026-03-17"},
]

# Also run with ORIGINAL for direct comparison (same seed, same run)
ORIGINAL_WINDOWS = [
    {"name": "W1", "train_end": "2024-06-01",
     "val_start": "2024-06-01", "val_end": "2024-09-30",
     "test_start": "2024-10-15", "test_end": "2025-01-31"},
    {"name": "W2", "train_end": "2025-01-01",
     "val_start": "2025-01-01", "val_end": "2025-04-30",
     "test_start": "2025-05-15", "test_end": "2025-08-31"},
    {"name": "W3", "train_end": "2025-07-01",
     "val_start": "2025-07-01", "val_end": "2025-10-31",
     "test_start": "2025-11-15", "test_end": "2026-03-17"},
]

PROD_CFG = {
    "n_long": 6, "n_short": 3, "rebal_hours": 12,
    "trend_cutoff": 0.9, "dyn_threshold": 0.7,
    "ema_alpha": 0.5, "hysteresis": 3,
    "risk_off_mode": "close",  # "close" = d9019ea (close positions on risk-off), "skip" = cef6e2f (just skip trading)
}

LGB_PARAMS = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.03, "num_leaves": 63,
    "min_child_samples": 100, "subsample": 0.8,
    "colsample_bytree": 0.8, "lambda_l2": 1.0,
    "verbose": -1, "n_jobs": -1,
}
XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "learning_rate": 0.03, "max_depth": 6,
    "min_child_weight": 100, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_lambda": 1.0,
    "n_jobs": -1, "verbosity": 0,
}
N_ROUNDS = 600
EARLY_STOP = 40


def load_data():
    print("=" * 70)
    print("  LOADING DATA")
    print("=" * 70)
    df, regime_df = load_research_frame()
    df, _ = add_r35_features(df)
    cg = load_cg_daily()
    cg_feats = compute_cg_features(cg)
    df, _, _ = add_cg_features(df, cg_feats)
    present = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    missing = [f for f in CHAMPION_FEAT_31 if f not in df.columns]
    if missing:
        print(f"  WARNING: Missing features: {missing}")
        CHAMPION_FEAT_31[:] = present
    print(f"  Frame: {len(df):,} rows, {df['symbol'].nunique()} symbols")
    print(f"  Features: {len(present)}/31")
    return df, regime_df


def train_ensemble(df, feats, windows, seeds=SEEDS, cs_rank_exclude=None):
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
            if len(train_) < 5000 or len(test_) < 200: continue
            if rank_feats:
                train_ = cs_rank_cols(train_, rank_feats)
                val_ = cs_rank_cols(val_, rank_feats)
                test_ = cs_rank_cols(test_, rank_feats)
            for d in [train_, val_, test_]:
                d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)
            for col in avail:
                for d in [train_, val_, test_]:
                    if d[col].isna().any(): d[col] = d[col].fillna(0)

            tr = train_[avail + ["target_binary"]].dropna()
            va = val_[avail + ["target_binary"]].dropna()
            te = test_[avail + ["target_binary", "timestamp", "symbol"]].dropna()
            fwd = test_[["timestamp", "symbol", "fwd_ret_12h"]].rename(
                columns={"fwd_ret_12h": "fwd_ret"}).dropna()
            for d in [tr, va, te]:
                d.replace([np.inf, -np.inf], np.nan, inplace=True)
            tr, va, te = tr.dropna(), va.dropna(), te.dropna()
            if len(te) == 0: continue

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
            rec["window"] = w["name"]; rec["seed"] = seed
            all_lgb.append(rec)

            dt_x = xgb.DMatrix(tr[avail], label=tr["target_binary"])
            dv_x = xgb.DMatrix(va[avail], label=va["target_binary"])
            m_x = xgb.train(p_xgb, dt_x, num_boost_round=N_ROUNDS,
                             evals=[(dv_x, "val")],
                             early_stopping_rounds=EARLY_STOP, verbose_eval=False)
            p_x = m_x.predict(xgb.DMatrix(te[avail]))
            rec2 = te[["timestamp", "symbol"]].copy()
            rec2["pred_xgb"] = p_x
            rec2 = rec2.merge(fwd, on=["timestamp", "symbol"], how="inner")
            rec2["window"] = w["name"]; rec2["seed"] = seed
            all_xgb.append(rec2)

            if seed == seeds[0]:
                log(f"  {w['name']}/s{seed}: train={len(tr):,} test={len(te):,}")

    if not all_lgb: return None
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


def simulate(merged, regime_df, n_long, n_short, cfg=PROD_CFG):
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

    risk_off_mode = cfg.get("risk_off_mode", "close")

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped: continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        if trend_str > trend_cutoff:
            if risk_off_mode == "close":
                # CLOSE mode: record closing cost + 0 return (match live behavior from d9019ea)
                if prev_longs or prev_shorts:
                    n_prev = len(prev_longs) + len(prev_shorts)
                    avg_weight = 1.0 / n_prev if n_prev > 0 else 0
                    close_cost = sum(_cost_for_sym(s) * avg_weight for s in prev_longs | prev_shorts)
                    all_rets.append({
                        "timestamp": ts, "gross_ret": 0.0, "net_ret": -close_cost,
                        "cost": close_cost, "n_long": 0, "n_short": 0,
                        "turnover": n_prev,
                    })
                else:
                    all_rets.append({
                        "timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                        "cost": 0.0, "n_long": 0, "n_short": 0, "turnover": 0,
                    })
                prev_longs, prev_shorts = set(), set()
            elif risk_off_mode == "skip":
                # SKIP mode: just don't open new positions, don't record period (cef6e2f behavior)
                pass
            continue
        grp = grouped[ts].copy()
        n = len(grp)

        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl == 0 and ns == 0: continue

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
        elif ns_act > 0: gross_ret = -short_ret
        else: gross_ret = long_ret
        gross_ret *= exposure

        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            turnover_cost = sum(_cost_for_sym(sym) * avg_weight for sym in new_opened)
            turnover_cost += sum(_cost_for_sym(sym) * avg_weight for sym in closed)
            holding_cost = funding_per_12h * (rebal_hours / 12)
            total_cost = turnover_cost + holding_cost
        else: total_cost = 0.0

        net_ret = gross_ret - total_cost
        prev_longs, prev_shorts = new_longs, new_shorts

        all_rets.append({
            "timestamp": ts, "gross_ret": gross_ret, "net_ret": net_ret,
            "cost": total_cost, "n_long": nl_act, "n_short": ns_act,
            "turnover": len(new_opened) + len(closed),
        })

    return pd.DataFrame(all_rets) if all_rets else pd.DataFrame()


def sharpe(rets_series, periods_per_year=2*365):
    if len(rets_series) < 2: return 0.0
    eq = (1 + rets_series).cumprod()
    r = eq.pct_change().dropna()
    return r.mean() / (r.std() + 1e-10) * np.sqrt(periods_per_year)


def analyze(port, label):
    if port.empty:
        print(f"  {label}: EMPTY")
        return {}
    gs = sharpe(port["gross_ret"])
    ns = sharpe(port["net_ret"])
    eq = (1 + port["net_ret"]).cumprod() * 100
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    maxdd = (eq / eq.cummax() - 1).min()
    wr = (port["net_ret"] > 0).mean() * 100
    avg_pos = (port["n_long"] + port["n_short"]).mean()
    n_periods = len(port)

    print(f"\n  {label}:")
    print(f"    Gross Sharpe: {gs:.3f}  Net Sharpe: {ns:.3f}")
    print(f"    Ret: {total_ret*100:.1f}%  DD: {maxdd*100:.1f}%  WR: {wr:.1f}%  Periods: {n_periods}")

    # Monthly breakdown
    port_c = port.copy()
    port_c["month"] = port_c["timestamp"].dt.to_period("M").astype(str)
    months = sorted(port_c["month"].unique())
    print(f"    {'Month':<10} {'Gross%':>8} {'Net%':>8} {'Cost%':>8} {'Per':>5}")
    for m in months:
        mdf = port_c[port_c["month"] == m]
        gr = ((1 + mdf["gross_ret"]).cumprod().iloc[-1] - 1) * 100
        nr = ((1 + mdf["net_ret"]).cumprod().iloc[-1] - 1) * 100
        mc = mdf["cost"].sum() * 100
        print(f"    {m:<10} {gr:>7.1f}% {nr:>7.1f}% {mc:>7.2f}% {len(mdf):>5}")

    # Quarterly
    port_c["quarter"] = port_c["timestamp"].dt.to_period("Q").astype(str)
    quarters = sorted(port_c["quarter"].unique())
    print(f"    {'Quarter':<10} {'GrossSh':>9} {'NetSh':>9} {'NetRet%':>9}")
    for q in quarters:
        qdf = port_c[port_c["quarter"] == q]
        qgs = sharpe(qdf["gross_ret"])
        qns = sharpe(qdf["net_ret"])
        qnr = ((1 + qdf["net_ret"]).cumprod().iloc[-1] - 1) * 100
        print(f"    {q:<10} {qgs:>9.2f} {qns:>9.2f} {qnr:>8.1f}%")

    return {
        "label": label, "gross_sharpe": round(gs, 3), "net_sharpe": round(ns, 3),
        "total_ret_pct": round(total_ret * 100, 1), "max_dd_pct": round(maxdd * 100, 1),
        "win_rate": round(wr, 1), "n_periods": n_periods,
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("  R68 — CONTINUOUS WF (NO GAP PERIODS)")
    print("=" * 70)

    df, regime_df = load_data()
    feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]

    # Train on CONTINUOUS windows
    print(f"\n  Training ensemble (CONTINUOUS windows)...")
    t1 = time.time()
    preds_cont = train_ensemble(df, feats, CONTINUOUS_WINDOWS, seeds=SEEDS, cs_rank_exclude=no_rank)
    print(f"  Done in {time.time()-t1:.0f}s, {len(preds_cont):,} predictions")

    # Train on ORIGINAL windows (same seeds for fair comparison)
    print(f"\n  Training ensemble (ORIGINAL windows)...")
    t2 = time.time()
    preds_orig = train_ensemble(df, feats, ORIGINAL_WINDOWS, seeds=SEEDS, cs_rank_exclude=no_rank)
    print(f"  Done in {time.time()-t2:.0f}s, {len(preds_orig):,} predictions")

    results = []

    # 4L/2S configs
    print(f"\n  === 4L/2S ===")
    r = analyze(simulate(preds_cont, regime_df, 4, 2), "4L/2S continuous")
    results.append(r)
    r = analyze(simulate(preds_orig, regime_df, 4, 2), "4L/2S original")
    results.append(r)

    # 6L/3S configs
    print(f"\n  === 6L/3S ===")
    r = analyze(simulate(preds_cont, regime_df, 6, 3), "6L/3S continuous")
    results.append(r)
    r = analyze(simulate(preds_orig, regime_df, 6, 3), "6L/3S original")
    results.append(r)

    # Risk-off mode comparison (SKIP vs CLOSE)
    print(f"\n  === RISK-OFF MODE COMPARISON (6L/3S continuous) ===")
    cfg_close = PROD_CFG.copy()
    cfg_close["risk_off_mode"] = "close"  # d9019ea
    cfg_skip = PROD_CFG.copy()
    cfg_skip["risk_off_mode"] = "skip"   # cef6e2f

    port_close = simulate(preds_cont, regime_df, 6, 3, cfg_close)
    port_skip = simulate(preds_cont, regime_df, 6, 3, cfg_skip)

    r_close = analyze(port_close, "6L/3S CLOSE mode (d9019ea)")
    r_skip = analyze(port_skip, "6L/3S SKIP mode (cef6e2f)")
    results.append(r_close)
    results.append(r_skip)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY: CONTINUOUS vs ORIGINAL + RISK-OFF MODES")
    print("=" * 70)
    print(f"  {'Config':<22} {'Gross Sh':>10} {'Net Sh':>10} {'Ret%':>8} {'DD%':>8} {'Periods':>8}")
    print(f"  {'-'*66}")
    for r in results:
        if r:
            print(f"  {r['label']:<22} {r['gross_sharpe']:>10.3f} {r['net_sharpe']:>10.3f} "
                  f"{r['total_ret_pct']:>7.1f}% {r['max_dd_pct']:>7.1f}% {r['n_periods']:>8}")

    pd.DataFrame(results).to_csv("/data/datasets/results_r68_continuous_wf.csv", index=False)
    print(f"\n  Saved: /data/datasets/results_r68_continuous_wf.csv")
    print(f"  Total: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    main()
