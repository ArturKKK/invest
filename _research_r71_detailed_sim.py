#!/usr/bin/env python3
"""
R71 — Detailed Simulation Breakdown

Detailed per-window, per-month, cumulative analysis for:
  - 4L/2S gapped (original WF)
  - 4L/2S continuous (fill gap periods with last model)
  - 6L/3S gapped (shadow/comparison)

Outputs:
  1. Per-window: Sharpe (gross/net), return, DD, win-rate, #periods
  2. Per-month: return, cumulative return
  3. Summary table: total Sharpe, total return, max DD
  4. Equity curve stats
"""

import sys, warnings, time
from typing import Dict, Set
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy import stats

warnings.filterwarnings("ignore")

from _research_round7 import SYM_35
from _research_r22_models import SEEDS, log, cs_rank_cols
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
    if sym in TIER1_SYMS:
        return 0.92 * (-0.0001) + 0.08 * 0.0007
    elif sym in TIER2_SYMS:
        return 0.75 * 0.0001 + 0.25 * 0.0007
    else:
        return 0.0005 + 0.0002

# ── Walk-forward windows ──
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

# Continuous: extend test of each window to cover gap before next window
CONTINUOUS_WINDOWS = [
    {"name": "W1", "train_end": "2024-06-01",
     "val_start": "2024-06-01", "val_end": "2024-09-30",
     "test_start": "2024-10-15", "test_end": "2025-05-14"},
    {"name": "W2", "train_end": "2025-01-01",
     "val_start": "2025-01-01", "val_end": "2025-04-30",
     "test_start": "2025-05-15", "test_end": "2025-11-14"},
    {"name": "W3", "train_end": "2025-07-01",
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
    print(f"  Dates: {df['timestamp'].min().date()} -> {df['timestamp'].max().date()}")
    print(f"  Features: {len(present)}/31")
    return df, regime_df


def train_ensemble(df, feats, windows, seeds=SEEDS, cs_rank_exclude=None):
    """Train LGB+XGB ensemble, return merged predictions."""
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
    """Simulate with gross/net tracking."""
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
            for _, r in grp.iterrows():
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

        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            turnover_cost = sum(_cost_for_sym(sym) * avg_weight for sym in new_opened)
            turnover_cost += sum(_cost_for_sym(sym) * avg_weight for sym in closed)
            holding_cost = funding_per_12h * (rebal_hours / 12)
            total_cost = turnover_cost + holding_cost
        else:
            total_cost = turnover_cost = holding_cost = 0.0

        net_port_ret = gross_port_ret - total_cost
        prev_longs, prev_shorts = new_longs, new_shorts

        # Find which window this timestamp belongs to
        win_name = merged.loc[merged["timestamp"] == ts, "window"].iloc[0] if ts in merged["timestamp"].values else "?"

        all_rets.append({
            "timestamp": ts, "gross_ret": gross_port_ret, "net_ret": net_port_ret,
            "cost": total_cost, "n_long": nl_act, "n_short": ns_act,
            "exposure": exposure, "turnover": len(new_opened) + len(closed),
            "window": win_name,
            "long_symbols": ",".join(sorted(new_longs)),
            "short_symbols": ",".join(sorted(new_shorts)),
        })

    return pd.DataFrame(all_rets) if all_rets else pd.DataFrame()


def sharpe(rets, periods_per_year=2*365):
    if len(rets) < 2:
        return 0.0
    eq = (1 + rets).cumprod()
    r = eq.pct_change().dropna()
    return r.mean() / (r.std() + 1e-10) * np.sqrt(periods_per_year)


def max_dd(rets):
    eq = (1 + rets).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return dd.min()


def win_rate(rets):
    return (rets > 0).mean() * 100


def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def analyze_results(df_rets, label):
    """Full breakdown for a simulation result."""
    print_section(f"{label} — OVERALL")
    n_periods = len(df_rets)
    gross_sh = sharpe(df_rets["gross_ret"])
    net_sh = sharpe(df_rets["net_ret"])
    total_gross = float((1 + df_rets["gross_ret"]).prod() - 1) * 100
    total_net = float((1 + df_rets["net_ret"]).prod() - 1) * 100
    dd_gross = max_dd(df_rets["gross_ret"]) * 100
    dd_net = max_dd(df_rets["net_ret"]) * 100
    wr = win_rate(df_rets["net_ret"])
    avg_cost = df_rets["cost"].mean() * 100
    avg_turnover = df_rets["turnover"].mean()
    avg_exposure = df_rets["exposure"].mean()

    print(f"  Periods: {n_periods}")
    print(f"  Gross Sharpe: {gross_sh:.3f}   Net Sharpe: {net_sh:.3f}")
    print(f"  Gross Return: {total_gross:+.1f}%   Net Return: {total_net:+.1f}%")
    print(f"  Gross MaxDD: {dd_gross:.1f}%   Net MaxDD: {dd_net:.1f}%")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  Avg Cost/period: {avg_cost:.4f}%   Avg Turnover: {avg_turnover:.1f}")
    print(f"  Avg Exposure: {avg_exposure:.2f}")

    # ── Per Window ──
    print_section(f"{label} — PER WINDOW")
    print(f"  {'Window':<8} {'Periods':>7} {'GrossSh':>8} {'NetSh':>8} {'GrossRet':>9} {'NetRet':>9} {'MaxDD':>7} {'WR%':>6}")
    print(f"  {'-'*65}")
    for wname in sorted(df_rets["window"].unique()):
        w = df_rets[df_rets["window"] == wname]
        g_sh = sharpe(w["gross_ret"])
        n_sh = sharpe(w["net_ret"])
        g_ret = float((1 + w["gross_ret"]).prod() - 1) * 100
        n_ret = float((1 + w["net_ret"]).prod() - 1) * 100
        dd = max_dd(w["net_ret"]) * 100
        wr_w = win_rate(w["net_ret"])
        print(f"  {wname:<8} {len(w):>7} {g_sh:>8.3f} {n_sh:>8.3f} {g_ret:>+8.1f}% {n_ret:>+8.1f}% {dd:>6.1f}% {wr_w:>5.1f}%")

    # ── Per Quarter ──
    print_section(f"{label} — PER QUARTER")
    df_rets = df_rets.copy()
    df_rets["quarter"] = df_rets["timestamp"].dt.to_period("Q").astype(str)
    print(f"  {'Quarter':<10} {'Periods':>7} {'GrossSh':>8} {'NetSh':>8} {'GrossRet':>9} {'NetRet':>9} {'MaxDD':>7} {'WR%':>6}")
    print(f"  {'-'*67}")
    for q in sorted(df_rets["quarter"].unique()):
        qd = df_rets[df_rets["quarter"] == q]
        g_sh = sharpe(qd["gross_ret"])
        n_sh = sharpe(qd["net_ret"])
        g_ret = float((1 + qd["gross_ret"]).prod() - 1) * 100
        n_ret = float((1 + qd["net_ret"]).prod() - 1) * 100
        dd = max_dd(qd["net_ret"]) * 100
        wr_q = win_rate(qd["net_ret"])
        print(f"  {q:<10} {len(qd):>7} {g_sh:>8.3f} {n_sh:>8.3f} {g_ret:>+8.1f}% {n_ret:>+8.1f}% {dd:>6.1f}% {wr_q:>5.1f}%")

    # ── Per Month ──
    print_section(f"{label} — PER MONTH")
    df_rets["month"] = df_rets["timestamp"].dt.to_period("M").astype(str)
    cum_net = 1.0
    print(f"  {'Month':<10} {'Periods':>7} {'GrossRet':>9} {'NetRet':>9} {'CumNet':>9} {'WR%':>6}")
    print(f"  {'-'*53}")
    for m in sorted(df_rets["month"].unique()):
        md = df_rets[df_rets["month"] == m]
        g_ret = float((1 + md["gross_ret"]).prod() - 1) * 100
        n_ret_val = float((1 + md["net_ret"]).prod() - 1)
        cum_net *= (1 + n_ret_val)
        wr_m = win_rate(md["net_ret"])
        print(f"  {m:<10} {len(md):>7} {g_ret:>+8.1f}% {n_ret_val*100:>+8.1f}% {(cum_net-1)*100:>+8.1f}% {wr_m:>5.1f}%")

    # ── Equity curve stats ──
    print_section(f"{label} — EQUITY CURVE STATS")
    eq = (1 + df_rets["net_ret"]).cumprod()
    daily_eq = eq.groupby(df_rets["timestamp"].dt.date).last()
    if len(daily_eq) > 20:
        rolling_30d = daily_eq.pct_change().rolling(30).std() * np.sqrt(365)
        print(f"  Annualized vol (full): {daily_eq.pct_change().std() * np.sqrt(365):.1%}")
        print(f"  Annualized vol (30d rolling, last): {rolling_30d.iloc[-1]:.1%}")
    print(f"  Final equity (starting $1): ${eq.iloc[-1]:.4f}")
    print(f"  Best period: {df_rets['net_ret'].max()*100:+.2f}%")
    print(f"  Worst period: {df_rets['net_ret'].min()*100:+.2f}%")
    print(f"  Median period: {df_rets['net_ret'].median()*100:+.3f}%")
    print(f"  Skewness: {df_rets['net_ret'].skew():.3f}")
    print(f"  Kurtosis: {df_rets['net_ret'].kurtosis():.3f}")

    # ── Top/bottom position frequency ──
    print_section(f"{label} — POSITION FREQUENCY (top 15)")
    all_longs = []
    all_shorts = []
    for _, row in df_rets.iterrows():
        if row["long_symbols"]:
            all_longs.extend(row["long_symbols"].split(","))
        if row["short_symbols"]:
            all_shorts.extend(row["short_symbols"].split(","))
    long_freq = pd.Series(all_longs).value_counts().head(15)
    short_freq = pd.Series(all_shorts).value_counts().head(15)
    print(f"  LONG frequency (% of periods):")
    for sym, cnt in long_freq.items():
        print(f"    {sym:<18} {cnt:>5} ({cnt/n_periods*100:>5.1f}%)")
    print(f"\n  SHORT frequency (% of periods):")
    for sym, cnt in short_freq.items():
        print(f"    {sym:<18} {cnt:>5} ({cnt/n_periods*100:>5.1f}%)")


def main():
    t0 = time.time()
    df, regime_df = load_data()

    # ── GAPPED (original WF) ──
    print_section("TRAINING — GAPPED (original WF)")
    merged_gapped = train_ensemble(
        df, CHAMPION_FEAT_31, ORIGINAL_WINDOWS,
        cs_rank_exclude=set(MARKET_LEVEL_FEATURES) & set(CHAMPION_FEAT_31),
    )
    if merged_gapped is None:
        print("ERROR: No predictions generated for gapped WF")
        return

    # ── CONTINUOUS WF ──
    print_section("TRAINING — CONTINUOUS WF")
    merged_continuous = train_ensemble(
        df, CHAMPION_FEAT_31, CONTINUOUS_WINDOWS,
        cs_rank_exclude=set(MARKET_LEVEL_FEATURES) & set(CHAMPION_FEAT_31),
    )
    if merged_continuous is None:
        print("ERROR: No predictions generated for continuous WF")
        return

    print(f"\n  Gapped predictions: {len(merged_gapped):,} rows, "
          f"{merged_gapped['timestamp'].nunique()} timestamps")
    print(f"  Continuous predictions: {len(merged_continuous):,} rows, "
          f"{merged_continuous['timestamp'].nunique()} timestamps")

    # ── SIMULATE ALL CONFIGS ──
    configs = [
        ("4L/2S GAPPED", merged_gapped, 4, 2),
        ("6L/3S GAPPED", merged_gapped, 6, 3),
        ("4L/2S CONTINUOUS", merged_continuous, 4, 2),
        ("6L/3S CONTINUOUS", merged_continuous, 6, 3),
    ]

    results = {}
    for label, merged, nl, ns in configs:
        cfg = dict(PROD_CFG, n_long=nl, n_short=ns)
        rets = simulate(merged, regime_df, nl, ns, cfg)
        results[label] = rets
        analyze_results(rets, label)

    # ── COMPARISON TABLE ──
    print_section("COMPARISON SUMMARY")
    print(f"  {'Config':<22} {'Periods':>7} {'GrossSh':>8} {'NetSh':>8} "
          f"{'GrossRet':>9} {'NetRet':>9} {'MaxDD':>7} {'WR%':>6}")
    print(f"  {'-'*80}")
    for label, rets in results.items():
        g_sh = sharpe(rets["gross_ret"])
        n_sh = sharpe(rets["net_ret"])
        g_ret = float((1 + rets["gross_ret"]).prod() - 1) * 100
        n_ret = float((1 + rets["net_ret"]).prod() - 1) * 100
        dd = max_dd(rets["net_ret"]) * 100
        wr = win_rate(rets["net_ret"])
        marker = " <-- PROD" if label == "4L/2S CONTINUOUS" else ""
        print(f"  {label:<22} {len(rets):>7} {g_sh:>8.3f} {n_sh:>8.3f} "
              f"{g_ret:>+8.1f}% {n_ret:>+8.1f}% {dd:>6.1f}% {wr:>5.1f}%{marker}")

    # ── 4L/2S vs 6L/3S delta per month (continuous) ──
    print_section("4L/2S vs 6L/3S MONTHLY DELTA (continuous)")
    r4 = results["4L/2S CONTINUOUS"].copy()
    r6 = results["6L/3S CONTINUOUS"].copy()
    r4["month"] = r4["timestamp"].dt.to_period("M").astype(str)
    r6["month"] = r6["timestamp"].dt.to_period("M").astype(str)

    print(f"  {'Month':<10} {'4L2S Net':>9} {'6L3S Net':>9} {'Delta':>8} {'Better':>8}")
    print(f"  {'-'*47}")
    months = sorted(set(r4["month"]) | set(r6["month"]))
    wins_4, wins_6 = 0, 0
    for m in months:
        m4 = r4[r4["month"] == m]
        m6 = r6[r6["month"] == m]
        ret4 = float((1 + m4["net_ret"]).prod() - 1) * 100 if len(m4) > 0 else 0
        ret6 = float((1 + m6["net_ret"]).prod() - 1) * 100 if len(m6) > 0 else 0
        delta = ret4 - ret6
        better = "4L/2S" if delta > 0 else "6L/3S"
        if delta > 0:
            wins_4 += 1
        else:
            wins_6 += 1
        print(f"  {m:<10} {ret4:>+8.1f}% {ret6:>+8.1f}% {delta:>+7.1f}% {better:>8}")
    print(f"\n  4L/2S wins: {wins_4}/{wins_4+wins_6} months ({wins_4/(wins_4+wins_6)*100:.0f}%)")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} min")
    print("  DONE")


if __name__ == "__main__":
    main()
