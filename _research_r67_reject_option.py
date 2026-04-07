#!/usr/bin/env python3
"""
R67 — Reject Option (Score-Gap Threshold)

Instead of fixed K=4L/2S, only take positions where signal is strong enough.
Long if raw_prob > 0.5 + t, Short if raw_prob < 0.5 - t. Max 4L/2S cap.
Grid: t = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]

Baseline: 4L/2S fixed (R65 Net Sharpe ≈ 2.98).
"""

import sys, warnings, time
from typing import Dict, Set

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
    "n_long": 4, "n_short": 2, "rebal_hours": 12,
    "trend_cutoff": 0.9, "dyn_threshold": 0.7,
    "ema_alpha": 0.5, "hysteresis": 3,
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

THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]


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
            if len(train_) < 5000 or len(test_) < 200:
                continue
            if rank_feats:
                train_ = cs_rank_cols(train_, rank_feats)
                val_ = cs_rank_cols(val_, rank_feats)
                test_ = cs_rank_cols(test_, rank_feats)
            for d in [train_, val_, test_]:
                d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)
            for col in avail:
                for d in [train_, val_, test_]:
                    if d[col].isna().any():
                        d[col] = d[col].fillna(0)

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


def simulate_reject(merged, regime_df, threshold, max_long=4, max_short=2, cfg=PROD_CFG):
    """Simulate with reject option: only take positions where raw_prob > 0.5+t (long) or < 0.5-t (short)."""
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

        exposure = 1.0
        dyn_threshold = cfg.get("dyn_threshold", 0.5)
        if dyn_threshold is not None and trend_str > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) /
                           (trend_cutoff - dyn_threshold + 1e-10) * 0.5)

        # EMA smoothing of pred (for rank-based selection)
        if ema_alpha is not None and ema_alpha < 1.0:
            for idx, r in grp.iterrows():
                sym = r["symbol"]
                raw_pred = r["pred"]
                smoothed = ema_alpha * raw_pred + (1 - ema_alpha) * prev_preds.get(sym, raw_pred)
                prev_preds[sym] = smoothed
                grp.at[idx, "pred"] = smoothed

        # Reject option: filter by raw_prob threshold FIRST
        long_candidates = grp[grp["raw_prob"] > 0.5 + threshold].copy()
        short_candidates = grp[grp["raw_prob"] < 0.5 - threshold].copy()

        # Then rank within candidates, take top max_long / max_short
        if len(long_candidates) > 0:
            long_candidates["pred_rank"] = long_candidates["pred"].rank(ascending=False)
            new_longs = set(long_candidates[long_candidates["pred_rank"] <= max_long]["symbol"].tolist())
        else:
            new_longs = set()

        if len(short_candidates) > 0:
            short_candidates["pred_rank"] = short_candidates["pred"].rank(ascending=True)
            new_shorts = set(short_candidates[short_candidates["pred_rank"] <= max_short]["symbol"].tolist())
        else:
            new_shorts = set()

        # Apply hysteresis for existing positions
        if hysteresis > 0 and (prev_longs or prev_shorts):
            # Keep positions that still pass a relaxed threshold
            relaxed_t = max(0, threshold - 0.005)
            for sym in prev_longs:
                sym_row = grp[grp["symbol"] == sym]
                if len(sym_row) > 0 and sym_row.iloc[0]["raw_prob"] > 0.5 + relaxed_t:
                    if len(new_longs) < max_long:
                        new_longs.add(sym)
            for sym in prev_shorts:
                sym_row = grp[grp["symbol"] == sym]
                if len(sym_row) > 0 and sym_row.iloc[0]["raw_prob"] < 0.5 - relaxed_t:
                    if len(new_shorts) < max_short:
                        new_shorts.add(sym)

        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        total_positions = len(new_longs) + len(new_shorts)

        if total_positions == 0:
            prev_longs = new_longs
            prev_shorts = new_shorts
            continue

        longs = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]

        long_ret = longs["fwd_ret"].mean() if len(longs) > 0 else 0
        short_ret = shorts["fwd_ret"].mean() if len(shorts) > 0 else 0

        nl_act = len(new_longs)
        ns_act = len(new_shorts)
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
        prev_longs = new_longs
        prev_shorts = new_shorts

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


def main():
    t0 = time.time()
    print("=" * 70)
    print("  R67 — REJECT OPTION (SCORE-GAP THRESHOLD)")
    print("=" * 70)

    df, regime_df = load_data()
    feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]

    print(f"\n  Training ensemble...")
    t1 = time.time()
    preds = train_ensemble(df, feats, ORIGINAL_WINDOWS, seeds=SEEDS, cs_rank_exclude=no_rank)
    if preds is None:
        print("  FAILED"); return
    print(f"  Done in {time.time()-t1:.0f}s, {len(preds):,} predictions")

    # Check raw_prob distribution
    print(f"\n  raw_prob stats: mean={preds['raw_prob'].mean():.4f}, "
          f"std={preds['raw_prob'].std():.4f}, "
          f"min={preds['raw_prob'].min():.4f}, max={preds['raw_prob'].max():.4f}")
    for t in THRESHOLDS:
        n_long_cand = (preds["raw_prob"] > 0.5 + t).sum()
        n_short_cand = (preds["raw_prob"] < 0.5 - t).sum()
        pct_l = n_long_cand / len(preds) * 100
        pct_s = n_short_cand / len(preds) * 100
        print(f"  t={t:.2f}: {pct_l:.1f}% long candidates, {pct_s:.1f}% short candidates")

    results = []
    for t in THRESHOLDS:
        label = f"t={t:.2f}" if t > 0 else "baseline_4L2S"
        print(f"\n  Simulating {label}...")
        if t == 0:
            # Use R65-compatible simulate for exact baseline reproduction
            from _research_r65_gross_net import simulate as simulate_r65
            port = simulate_r65(preds, regime_df, n_long=4, n_short=2, cfg=PROD_CFG)
        else:
            port = simulate_reject(preds, regime_df, threshold=t)
        if port.empty:
            print(f"    EMPTY (threshold too high)")
            results.append({"threshold": t, "label": label, "gross_sharpe": 0, "net_sharpe": 0})
            continue

        gs = sharpe(port["gross_ret"])
        ns = sharpe(port["net_ret"])
        eq = (1 + port["net_ret"]).cumprod() * 100
        total_ret = eq.iloc[-1] / eq.iloc[0] - 1
        maxdd = (eq / eq.cummax() - 1).min()
        wr = (port["net_ret"] > 0).mean() * 100
        avg_pos = (port["n_long"] + port["n_short"]).mean()
        avg_turnover = port["turnover"].mean()
        n_periods = len(port)
        n_skip = 450 - n_periods  # approximate skipped periods

        print(f"    Gross Sharpe: {gs:.3f}  Net Sharpe: {ns:.3f}")
        print(f"    Ret: {total_ret*100:.1f}%  DD: {maxdd*100:.1f}%  WR: {wr:.1f}%")
        print(f"    Avg pos: {avg_pos:.1f}  Avg turn: {avg_turnover:.1f}  Periods: {n_periods} (skip {n_skip})")

        # Quarterly
        port_c = port.copy()
        port_c["quarter"] = port_c["timestamp"].dt.to_period("Q").astype(str)
        for q in sorted(port_c["quarter"].unique()):
            qdf = port_c[port_c["quarter"] == q]
            qns = sharpe(qdf["net_ret"])
            qr = ((1 + qdf["net_ret"]).cumprod().iloc[-1] - 1) * 100
            print(f"    {q}: Sharpe={qns:.2f} Ret={qr:.1f}%")

        results.append({
            "threshold": t, "label": label,
            "gross_sharpe": round(gs, 3), "net_sharpe": round(ns, 3),
            "total_ret_pct": round(total_ret * 100, 1),
            "max_dd_pct": round(maxdd * 100, 1),
            "win_rate": round(wr, 1),
            "avg_positions": round(avg_pos, 1),
            "avg_turnover": round(avg_turnover, 1),
            "n_periods": n_periods,
        })

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Threshold':<14} {'Gross Sh':>10} {'Net Sh':>10} {'Ret%':>8} {'DD%':>8} {'WR%':>6} {'AvgPos':>7} {'Periods':>8}")
    print(f"  {'-'*73}")
    for r in results:
        print(f"  {r['label']:<14} {r.get('gross_sharpe',0):>10.3f} {r.get('net_sharpe',0):>10.3f} "
              f"{r.get('total_ret_pct',0):>7.1f}% {r.get('max_dd_pct',0):>7.1f}% "
              f"{r.get('win_rate',0):>5.1f}% {r.get('avg_positions',0):>6.1f} {r.get('n_periods',0):>8}")

    pd.DataFrame(results).to_csv("/data/datasets/results_r67_reject_option.csv", index=False)
    print(f"\n  Saved: /data/datasets/results_r67_reject_option.csv")
    print(f"  Total: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    main()
