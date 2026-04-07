#!/usr/bin/env python3
"""
R65 — Gross vs Net Sharpe: 4L/2S vs 6L/3S

Answers the question: is 4L/2S improvement from alpha or just lower costs?

Outputs:
  1. Gross Sharpe & Net Sharpe for both configs
  2. Quarterly Sharpe breakdown (gross + net)
  3. Monthly returns table
  4. Cost breakdown per config
  5. Stability: re-run with 5 different seed subsets for variance estimate
"""

import sys
import warnings
import time
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


def _cost_for_sym(sym: str) -> float:
    if sym in TIER1_SYMS:
        return 0.92 * (-0.0001) + 0.08 * 0.0007
    elif sym in TIER2_SYMS:
        return 0.75 * 0.0001 + 0.25 * 0.0007
    else:
        return 0.0005 + 0.0002


ORIGINAL_WINDOWS = [
    {"name": "W1",
     "train_end": "2024-06-01",
     "val_start": "2024-06-01", "val_end": "2024-09-30",
     "test_start": "2024-10-15", "test_end": "2025-01-31"},
    {"name": "W2",
     "train_end": "2025-01-01",
     "val_start": "2025-01-01", "val_end": "2025-04-30",
     "test_start": "2025-05-15", "test_end": "2025-08-31"},
    {"name": "W3",
     "train_end": "2025-07-01",
     "val_start": "2025-07-01", "val_end": "2025-10-31",
     "test_start": "2025-11-15", "test_end": "2026-03-17"},
]

PROD_CFG = {
    "n_long": 6, "n_short": 3, "rebal_hours": 12,
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
    print(f"  Dates: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"  Features: {len(present)}/31")
    return df, regime_df


def train_ensemble(df, feats, windows, seeds=SEEDS, cs_rank_exclude=None):
    """Train LGB+XGB ensemble, return per-seed predictions for variance analysis."""
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
            if len(te) == 0:
                continue

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
                ic = stats.spearmanr(p, fwd.set_index(["timestamp", "symbol"]).reindex(
                    te.set_index(["timestamp", "symbol"]).index)["fwd_ret"].values)[0]
                log(f"  {w['name']}/s{seed}: train={len(tr):,} test={len(te):,} IC={ic:.4f}")

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


def simulate(merged, regime_df, n_long, n_short, cfg=PROD_CFG):
    """Simulate with explicit gross/net tracking."""
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

        nl_act = len(new_longs)
        ns_act = len(new_shorts)
        if nl_act > 0 and ns_act > 0:
            gross_port_ret = 0.5 * long_ret - 0.5 * short_ret
        elif ns_act > 0:
            gross_port_ret = -short_ret
        else:
            gross_port_ret = long_ret

        gross_port_ret *= exposure

        # Cost calc
        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            turnover_cost = sum(_cost_for_sym(sym) * avg_weight for sym in new_opened)
            turnover_cost += sum(_cost_for_sym(sym) * avg_weight for sym in closed)
            holding_cost = funding_per_12h * (rebal_hours / 12)
            total_cost = turnover_cost + holding_cost
        else:
            total_cost = 0.0
            turnover_cost = 0.0
            holding_cost = 0.0

        net_port_ret = gross_port_ret - total_cost

        prev_longs = new_longs
        prev_shorts = new_shorts

        all_rets.append({
            "timestamp": ts,
            "gross_ret": gross_port_ret,
            "net_ret": net_port_ret,
            "cost": total_cost,
            "turnover_cost": turnover_cost,
            "holding_cost": holding_cost,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "n_long": nl_act,
            "n_short": ns_act,
            "exposure": exposure,
            "turnover": len(new_opened) + len(closed),
        })

    return pd.DataFrame(all_rets) if all_rets else pd.DataFrame()


def sharpe(rets_series, periods_per_year=2*365):
    """Annualised Sharpe from period returns."""
    if len(rets_series) < 2:
        return 0.0
    eq = (1 + rets_series).cumprod()
    r = eq.pct_change().dropna()
    return r.mean() / (r.std() + 1e-10) * np.sqrt(periods_per_year)


def analyze(port_df, label):
    """Full analysis: gross/net Sharpe, quarterly, monthly."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    if port_df.empty:
        print("  NO DATA")
        return {}

    n_periods = len(port_df)
    gross_sh = sharpe(port_df["gross_ret"])
    net_sh = sharpe(port_df["net_ret"])

    eq_gross = (1 + port_df["gross_ret"]).cumprod() * 100
    eq_net = (1 + port_df["net_ret"]).cumprod() * 100
    total_gross = eq_gross.iloc[-1] / eq_gross.iloc[0] - 1
    total_net = eq_net.iloc[-1] / eq_net.iloc[0] - 1

    maxdd_gross = (eq_gross / eq_gross.cummax() - 1).min()
    maxdd_net = (eq_net / eq_net.cummax() - 1).min()

    wr_gross = (port_df["gross_ret"] > 0).mean() * 100
    wr_net = (port_df["net_ret"] > 0).mean() * 100

    avg_cost_bps = port_df["cost"].mean() * 10000
    total_cost_pct = port_df["cost"].sum() * 100
    avg_turnover = port_df["turnover"].mean()
    avg_pos = (port_df["n_long"] + port_df["n_short"]).mean()

    print(f"\n  {'Metric':<25} {'Gross':>12} {'Net':>12} {'Δ(cost)':>12}")
    print(f"  {'-'*61}")
    print(f"  {'Sharpe':<25} {gross_sh:>12.3f} {net_sh:>12.3f} {gross_sh-net_sh:>12.3f}")
    print(f"  {'Total Return %':<25} {total_gross*100:>11.1f}% {total_net*100:>11.1f}% {(total_gross-total_net)*100:>11.1f}%")
    print(f"  {'Max Drawdown %':<25} {maxdd_gross*100:>11.1f}% {maxdd_net*100:>11.1f}%")
    print(f"  {'Win Rate %':<25} {wr_gross:>11.1f}% {wr_net:>11.1f}%")
    print(f"  {'Avg Cost (bps/period)':<25} {avg_cost_bps:>12.2f}")
    print(f"  {'Total Cost %':<25} {total_cost_pct:>11.1f}%")
    print(f"  {'Avg Turnover/period':<25} {avg_turnover:>12.1f}")
    print(f"  {'Avg Positions':<25} {avg_pos:>12.1f}")
    print(f"  {'N periods':<25} {n_periods:>12}")

    # ── Quarterly breakdown ──
    port_df = port_df.copy()
    port_df["quarter"] = port_df["timestamp"].dt.to_period("Q").astype(str)
    quarters = sorted(port_df["quarter"].unique())

    print(f"\n  {'Quarter':<12} {'Gross Sharpe':>14} {'Net Sharpe':>14} {'Gross Ret%':>12} {'Net Ret%':>12} {'Periods':>8}")
    print(f"  {'-'*72}")
    q_results = []
    for q in quarters:
        qdf = port_df[port_df["quarter"] == q]
        gs = sharpe(qdf["gross_ret"])
        ns = sharpe(qdf["net_ret"])
        gr = ((1 + qdf["gross_ret"]).cumprod().iloc[-1] - 1) * 100 if len(qdf) > 0 else 0
        nr = ((1 + qdf["net_ret"]).cumprod().iloc[-1] - 1) * 100 if len(qdf) > 0 else 0
        print(f"  {q:<12} {gs:>14.2f} {ns:>14.2f} {gr:>11.1f}% {nr:>11.1f}% {len(qdf):>8}")
        q_results.append({"quarter": q, "gross_sharpe": gs, "net_sharpe": ns,
                          "gross_ret_pct": gr, "net_ret_pct": nr, "periods": len(qdf)})

    # ── Monthly returns ──
    port_df["month"] = port_df["timestamp"].dt.to_period("M").astype(str)
    months = sorted(port_df["month"].unique())

    print(f"\n  {'Month':<10} {'Gross Ret%':>12} {'Net Ret%':>12} {'Cost%':>10} {'Periods':>8}")
    print(f"  {'-'*54}")
    for m in months:
        mdf = port_df[port_df["month"] == m]
        gr = ((1 + mdf["gross_ret"]).cumprod().iloc[-1] - 1) * 100 if len(mdf) > 0 else 0
        nr = ((1 + mdf["net_ret"]).cumprod().iloc[-1] - 1) * 100 if len(mdf) > 0 else 0
        mc = mdf["cost"].sum() * 100
        print(f"  {m:<10} {gr:>11.1f}% {nr:>11.1f}% {mc:>9.2f}% {len(mdf):>8}")

    return {
        "gross_sharpe": round(gross_sh, 3),
        "net_sharpe": round(net_sh, 3),
        "delta_sharpe_from_cost": round(gross_sh - net_sh, 3),
        "total_gross_return_pct": round(total_gross * 100, 1),
        "total_net_return_pct": round(total_net * 100, 1),
        "total_cost_pct": round(total_cost_pct, 1),
        "avg_cost_bps": round(avg_cost_bps, 2),
        "avg_positions": round(avg_pos, 1),
        "quarterly": q_results,
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("  R65 — GROSS vs NET SHARPE: 4L/2S vs 6L/3S")
    print("  Question: is improvement from alpha or lower costs?")
    print("=" * 70)

    df, regime_df = load_data()
    feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]

    print(f"\n  Training ensemble (31 feats, {len(SEEDS)} seeds × 3 windows)...")
    t1 = time.time()
    preds = train_ensemble(df, feats, ORIGINAL_WINDOWS, seeds=SEEDS, cs_rank_exclude=no_rank)
    if preds is None:
        print("  ❌ Training failed")
        return
    print(f"  ✅ Training done in {time.time()-t1:.0f}s, {len(preds):,} predictions")

    # ── Run both configs ──
    configs = [
        ("6L/3S (baseline)", 6, 3),
        ("4L/2S (candidate)", 4, 2),
        ("3L/3S", 3, 3),
        ("8L/4S", 8, 4),
    ]

    all_results = {}
    for label, nl, ns in configs:
        print(f"\n  Simulating {label}...")
        port = simulate(preds, regime_df, nl, ns)
        res = analyze(port, label)
        all_results[label] = res

    # ── Summary comparison ──
    print("\n" + "=" * 70)
    print("  SUMMARY: GROSS vs NET COMPARISON")
    print("=" * 70)
    print(f"\n  {'Config':<22} {'Gross Sh':>10} {'Net Sh':>10} {'Δ(cost)':>10} {'AvgCost':>10} {'AvgPos':>8}")
    print(f"  {'-'*70}")
    for label, res in all_results.items():
        if not res:
            continue
        print(f"  {label:<22} {res['gross_sharpe']:>10.3f} {res['net_sharpe']:>10.3f} "
              f"{res['delta_sharpe_from_cost']:>10.3f} {res['avg_cost_bps']:>9.1f}bp {res['avg_positions']:>7.1f}")

    # ── Key answer ──
    r6 = all_results.get("6L/3S (baseline)", {})
    r4 = all_results.get("4L/2S (candidate)", {})
    if r6 and r4:
        gross_diff = r4.get("gross_sharpe", 0) - r6.get("gross_sharpe", 0)
        net_diff = r4.get("net_sharpe", 0) - r6.get("net_sharpe", 0)
        cost_diff = r4.get("avg_cost_bps", 0) - r6.get("avg_cost_bps", 0)
        print(f"\n  ═══ KEY ANSWER ═══")
        print(f"  4L/2S vs 6L/3S:")
        print(f"    Gross Sharpe delta: {gross_diff:+.3f}  {'(better alpha)' if gross_diff > 0.05 else '(similar alpha)' if abs(gross_diff) <= 0.05 else '(worse alpha)'}")
        print(f"    Net Sharpe delta:   {net_diff:+.3f}")
        print(f"    Cost delta (bps):   {cost_diff:+.2f}")
        if abs(gross_diff) <= 0.05 and net_diff > 0.05:
            print(f"    VERDICT: Improvement is from LOWER COSTS, not better alpha")
        elif gross_diff > 0.05:
            print(f"    VERDICT: Improvement is from BETTER ALPHA (top-K selection)")
        else:
            print(f"    VERDICT: No meaningful improvement")

    # ── Save results ──
    rows = []
    for label, res in all_results.items():
        if res:
            rows.append({"config": label, **{k: v for k, v in res.items() if k != "quarterly"}})
    pd.DataFrame(rows).to_csv("/data/datasets/results_r65_gross_net.csv", index=False)
    print(f"\n  Saved: /data/datasets/results_r65_gross_net.csv")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("  DONE")


if __name__ == "__main__":
    main()
