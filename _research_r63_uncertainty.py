#!/usr/bin/env python3
"""
R63 — Uncertainty Gating via Seed Disagreement

Key idea: instead of averaging 10 seed predictions, we use their STD as
a measure of model uncertainty. High std = seeds disagree = uncertain signal.

We test 3 gating approaches:
  1. uncertainty_filter   — skip trade if p_std > threshold (0.02, 0.03, 0.05)
  2. uncertainty_scaling  — weight_i *= (1 - clip(p_std_i / max_std, 0, 0.7))
  3. agreement_K          — reduce K when top-K positions are uncertain

Uses ORIGINAL_WINDOWS (with gaps) for comparability with baseline Sharpe 1.66.
N_ROUNDS=600, EARLY_STOP=40, 5 seeds.
"""

import sys
import warnings
from typing import Dict, List, Set
import time

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
    CANONICAL_EXEC_CFG, CHAMPION_FEAT_30,
    add_cg_features, compute_cg_features, load_cg_daily,
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
        print(f"  WARNING: Missing {missing}")
    print(f"  Frame: {len(df):,} rows, {df['symbol'].nunique()} symbols")
    print(f"  Dates: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    return df, regime_df


def train_ensemble_with_uncertainty(df, feats, windows, seeds=SEEDS, cs_rank_exclude=None):
    """
    Train LGB+XGB ensemble. Returns predictions with per-seed spread:
      pred      — standard rank-blended score (= prod pipeline)
      p_std     — std across all 10 seed predictions (uncertainty measure)
      p_mean    — mean of all 10 seed probs (before rank-norm)
    """
    avail = [f for f in feats if f in df.columns]
    rank_exclude = set(cs_rank_exclude or [])
    rank_feats = [f for f in avail if f not in rank_exclude]
    tz = df["timestamp"].dt.tz

    # Store per-seed predictions: dict[seed] = {(ts, sym): prob}
    all_seed_preds = []  # list of (seed, model_type, ts, sym, prob, fwd_ret, window)

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

            # LGB
            dt = lgb.Dataset(tr[avail], label=tr["target_binary"])
            dv = lgb.Dataset(va[avail], label=va["target_binary"])
            m = lgb.train(p_lgb, dt, num_boost_round=N_ROUNDS,
                          valid_sets=[dv],
                          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                     lgb.log_evaluation(-1)])
            p_lgb_pred = m.predict(te[avail])
            rec = te[["timestamp", "symbol"]].copy()
            rec["prob"] = p_lgb_pred
            rec = rec.merge(fwd, on=["timestamp", "symbol"], how="inner")
            rec["window"] = w["name"]
            rec["seed"] = seed
            rec["model"] = "lgb"
            all_seed_preds.append(rec)

            # XGB
            dt_x = xgb.DMatrix(tr[avail], label=tr["target_binary"])
            dv_x = xgb.DMatrix(va[avail], label=va["target_binary"])
            m_x = xgb.train(p_xgb, dt_x, num_boost_round=N_ROUNDS,
                             evals=[(dv_x, "val")],
                             early_stopping_rounds=EARLY_STOP, verbose_eval=False)
            p_xgb_pred = m_x.predict(xgb.DMatrix(te[avail]))
            rec2 = te[["timestamp", "symbol"]].copy()
            rec2["prob"] = p_xgb_pred
            rec2 = rec2.merge(fwd, on=["timestamp", "symbol"], how="inner")
            rec2["window"] = w["name"]
            rec2["seed"] = seed
            rec2["model"] = "xgb"
            all_seed_preds.append(rec2)

            if seed == seeds[0]:
                log(f"  {w['name']}/s{seed}: train={len(tr):,} test={len(te):,}")

    if not all_seed_preds:
        return None

    # Merge all per-seed predictions and compute uncertainty
    all_df = pd.concat(all_seed_preds)

    # Aggregate: mean and std across ALL 10 seeds × 2 models = 10 predictions per (ts, sym)
    agg = all_df.groupby(["timestamp", "symbol"]).agg(
        p_mean=("prob", "mean"),
        p_std=("prob", "std"),
        fwd_ret=("fwd_ret", "first"),
        window=("window", "first"),
    ).reset_index()
    agg["p_std"] = agg["p_std"].fillna(0)

    # Standard rank-blended score (reproduce prod pipeline)
    # LGB avg per (ts, sym)
    lgb_avg = all_df[all_df["model"] == "lgb"].groupby(["timestamp", "symbol"]).agg(
        pred_lgb=("prob", "mean")).reset_index()
    xgb_avg = all_df[all_df["model"] == "xgb"].groupby(["timestamp", "symbol"]).agg(
        pred_xgb=("prob", "mean")).reset_index()

    merged = agg.merge(lgb_avg, on=["timestamp", "symbol"], how="inner")
    merged = merged.merge(xgb_avg, on=["timestamp", "symbol"], how="inner")

    merged["rank_lgb"] = merged.groupby("timestamp")["pred_lgb"].rank(pct=True) - 0.5
    merged["rank_xgb"] = merged.groupby("timestamp")["pred_xgb"].rank(pct=True) - 0.5
    merged["pred"] = 0.5 * merged["rank_lgb"] + 0.5 * merged["rank_xgb"]

    # Print uncertainty stats
    print(f"\n  Uncertainty stats (p_std across 10 seeds):")
    print(f"    mean={agg['p_std'].mean():.4f}  std={agg['p_std'].std():.4f}")
    print(f"    p10={agg['p_std'].quantile(0.1):.4f}  p50={agg['p_std'].quantile(0.5):.4f}  "
          f"p90={agg['p_std'].quantile(0.9):.4f}  p99={agg['p_std'].quantile(0.99):.4f}")

    return merged[["timestamp", "symbol", "pred", "p_std", "p_mean", "fwd_ret", "window"]]


def simulate_with_uncertainty(merged, regime_df, cfg, mode="baseline", threshold=None):
    """
    Unified simulator with uncertainty modes:
      baseline           — standard 6L/3S, no uncertainty adjustment
      uncertainty_filter — skip positions with p_std > threshold
      uncertainty_scaling — scale weights by (1 - clip(p_std/max_std, 0, 0.7))
      agreement_K        — reduce K if mean p_std of top-K is high
    """
    n_long = cfg.get("n_long", 6)
    n_short = cfg.get("n_short", 3)
    trend_cutoff = cfg.get("trend_cutoff", 0.9)
    rebal_hours = cfg.get("rebal_hours", 12)
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

    # Global max_std for uncertainty_scaling normalization
    global_max_std = merged["p_std"].quantile(0.95) + 1e-10

    n_filtered_total = 0
    n_total_considered = 0

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        if trend_str > trend_cutoff:
            continue
        grp = grouped[ts].copy()
        n = len(grp)

        nl, ns = min(n_long, n // 3), min(n_short, n // 3)
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

        # Select longs/shorts
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

        n_total_considered += len(new_longs) + len(new_shorts)

        # ── Apply uncertainty gating ────────────────────────
        sym_to_std = dict(zip(grp["symbol"], grp["p_std"]))

        if mode == "uncertainty_filter" and threshold is not None:
            # Remove positions with high uncertainty
            before = len(new_longs) + len(new_shorts)
            new_longs = {s for s in new_longs if sym_to_std.get(s, 0) <= threshold}
            new_shorts = {s for s in new_shorts if sym_to_std.get(s, 0) <= threshold}
            n_filtered_total += before - len(new_longs) - len(new_shorts)
            if not new_longs and not new_shorts:
                prev_longs = set()
                prev_shorts = set()
                continue

        elif mode == "agreement_K":
            # Reduce K if top positions are uncertain
            long_stds = [sym_to_std.get(s, 0) for s in new_longs]
            short_stds = [sym_to_std.get(s, 0) for s in new_shorts]
            mean_long_std = np.mean(long_stds) if long_stds else 0
            mean_short_std = np.mean(short_stds) if short_stds else 0

            # If mean std of top positions > 0.04 (high uncertainty), reduce K
            std_cutoff_hi = 0.04
            std_cutoff_lo = 0.02
            if mean_long_std > std_cutoff_hi and len(new_longs) > 2:
                # Keep only the most certain longs
                new_longs_sorted = sorted(new_longs, key=lambda s: sym_to_std.get(s, 0))
                new_longs = set(new_longs_sorted[:max(2, len(new_longs) - 2)])
            if mean_short_std > std_cutoff_hi and len(new_shorts) > 1:
                new_shorts_sorted = sorted(new_shorts, key=lambda s: sym_to_std.get(s, 0))
                new_shorts = set(new_shorts_sorted[:max(1, len(new_shorts) - 1)])

        # Compute position weights (for uncertainty_scaling)
        longs = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]

        if mode == "uncertainty_scaling" and (len(longs) > 0 or len(shorts) > 0):
            def _unc_weight(sym):
                std = sym_to_std.get(sym, 0)
                return 1.0 - np.clip(std / global_max_std, 0, 0.7)

            if len(longs) > 0:
                long_w = longs["symbol"].map(_unc_weight)
                long_w_sum = long_w.sum()
                long_ret = (longs["fwd_ret"].values * long_w.values / (long_w_sum + 1e-10)).sum() if long_w_sum > 1e-10 else longs["fwd_ret"].mean()
            else:
                long_ret = 0

            if len(shorts) > 0:
                short_w = shorts["symbol"].map(_unc_weight)
                short_w_sum = short_w.sum()
                short_ret = (shorts["fwd_ret"].values * short_w.values / (short_w_sum + 1e-10)).sum() if short_w_sum > 1e-10 else shorts["fwd_ret"].mean()
            else:
                short_ret = 0
        else:
            long_ret = longs["fwd_ret"].mean() if len(longs) > 0 else 0
            short_ret = shorts["fwd_ret"].mean() if len(shorts) > 0 else 0

        # Hybrid cost
        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        total_positions = len(new_longs) + len(new_shorts)

        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            turnover_cost = sum(_cost_for_sym(sym) * avg_weight for sym in new_opened)
            turnover_cost += sum(_cost_for_sym(sym) * avg_weight for sym in closed)
            holding_cost = funding_per_12h * (rebal_hours / 12)
            total_cost = turnover_cost + holding_cost
        else:
            total_cost = 0.0

        nl_act, ns_act = len(new_longs), len(new_shorts)
        if nl_act > 0 and ns_act > 0:
            port_ret = 0.5 * long_ret - 0.5 * short_ret
        elif ns_act > 0:
            port_ret = -short_ret
        else:
            port_ret = long_ret

        port_ret *= exposure
        port_ret -= total_cost

        prev_longs = new_longs
        prev_shorts = new_shorts

        all_rets.append({
            "timestamp": ts,
            "portfolio_ret": port_ret,
            "gross_ret": port_ret + total_cost,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "n_long": nl_act,
            "n_short": ns_act,
            "exposure": exposure,
            "turnover": len(new_opened) + len(closed),
            "cost": total_cost,
        })

    if n_total_considered > 0 and mode in ("uncertainty_filter",):
        print(f"    [{mode} thr={threshold}] filtered {n_filtered_total}/{n_total_considered} "
              f"positions ({100*n_filtered_total/n_total_considered:.1f}%)")

    if not all_rets:
        return pd.DataFrame()
    return pd.DataFrame(all_rets)


def compute_metrics(port_df, capital=100):
    if port_df.empty:
        return {"sharpe": 0, "total_return_pct": 0, "max_dd_pct": 0, "win_rate": 0, "avg_positions": 0}
    eq = (1 + port_df["portfolio_ret"]).cumprod() * capital
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(2 * 365)
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    maxdd = (eq / eq.cummax() - 1).min()
    win_rate = (rets > 0).sum() / len(rets) * 100 if len(rets) > 0 else 0
    avg_pos = (port_df["n_long"] + port_df["n_short"]).mean()
    return {
        "sharpe": round(sharpe, 3),
        "total_return_pct": round(total_ret * 100, 1),
        "max_dd_pct": round(maxdd * 100, 1),
        "win_rate": round(win_rate, 1),
        "avg_positions": round(avg_pos, 1),
    }


def compute_window_metrics(port_df, preds, capital=100):
    results = {}
    for wname in preds["window"].unique():
        wp = preds[preds["window"] == wname]
        ts_min, ts_max = wp["timestamp"].min(), wp["timestamp"].max()
        w_port = port_df[(port_df["timestamp"] >= ts_min) & (port_df["timestamp"] <= ts_max)]
        if len(w_port) < 2:
            results[wname] = 0
            continue
        eq = (1 + w_port["portfolio_ret"]).cumprod()
        rets = eq.pct_change().dropna()
        sh = rets.mean() / (rets.std() + 1e-10) * np.sqrt(2 * 365)
        results[wname] = round(sh, 2)
    return results


def main():
    t0 = time.time()
    print("=" * 70)
    print("  R63 — UNCERTAINTY GATING (SEED DISAGREEMENT)")
    print("  Champion: 31f features, ORIGINAL_WINDOWS + hybrid tiered costs")
    print("=" * 70)

    df, regime_df = load_data()
    feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]

    print(f"\n  Training ensemble with per-seed predictions ({len(SEEDS)} seeds × 2 models)...")
    t1 = time.time()
    preds = train_ensemble_with_uncertainty(
        df, feats, ORIGINAL_WINDOWS, seeds=SEEDS, cs_rank_exclude=no_rank)

    if preds is None:
        print("  ❌ Training failed")
        return
    print(f"  Training complete in {(time.time()-t1)/60:.1f}min")

    # ── Define experiments ──────────────────────────────────
    experiments = [
        ("baseline",             "baseline",             None),
        ("filter_std002",        "uncertainty_filter",   0.02),
        ("filter_std003",        "uncertainty_filter",   0.03),
        ("filter_std005",        "uncertainty_filter",   0.05),
        ("filter_std010",        "uncertainty_filter",   0.10),
        ("scaling",              "uncertainty_scaling",  None),
        ("agreement_K",          "agreement_K",          None),
    ]

    results = []

    print(f"\n  Running {len(experiments)} uncertainty modes...\n")

    for exp_name, mode, threshold in experiments:
        t2 = time.time()
        port = simulate_with_uncertainty(preds, regime_df, PROD_CFG,
                                         mode=mode, threshold=threshold)
        elapsed = time.time() - t2

        if port.empty:
            print(f"  ⚠️  {exp_name}: no returns generated")
            continue

        m = compute_metrics(port)
        wm = compute_window_metrics(port, preds)

        result = {
            "mode": exp_name,
            "threshold": threshold,
            "sharpe": m["sharpe"],
            "total_ret%": m["total_return_pct"],
            "maxDD%": m["max_dd_pct"],
            "win_rate%": m["win_rate"],
            "avg_pos": m["avg_positions"],
            "W1": wm.get("W1", "?"),
            "W2": wm.get("W2", "?"),
            "W3": wm.get("W3", "?"),
        }
        results.append(result)
        print(f"  [{exp_name:22s}] Sharpe={m['sharpe']:5.2f}  Ret={m['total_return_pct']:+6.1f}%  "
              f"DD={m['max_dd_pct']:5.1f}%  AvgPos={m['avg_positions']:.1f}  "
              f"W1={wm.get('W1','?'):5.2f} W2={wm.get('W2','?'):5.2f} W3={wm.get('W3','?'):5.2f}  "
              f"({elapsed:.0f}s)")

    # ── Summary table ──────────────────────────────────────
    print("\n" + "=" * 90)
    print("  R63 RESULTS SUMMARY")
    print("=" * 90)
    print(f"  {'Mode':<24} {'Thr':>5} {'Sharpe':>7} {'Ret%':>7} {'MaxDD%':>7} "
          f"{'WR%':>6} {'AvgPos':>7} {'W1':>6} {'W2':>6} {'W3':>6}")
    print("  " + "-" * 88)
    for r in sorted(results, key=lambda x: x["sharpe"], reverse=True):
        thr_str = f"{r['threshold']:.3f}" if r["threshold"] is not None else "  -  "
        marker = " ← BEST" if r["sharpe"] == max(x["sharpe"] for x in results) else ""
        marker += " ← BASELINE" if r["mode"] == "baseline" else ""
        print(f"  {r['mode']:<24} {thr_str:>5} {r['sharpe']:>7.2f} {r['total_ret%']:>+7.1f} "
              f"{r['maxDD%']:>7.1f} {r['win_rate%']:>6.1f} "
              f"{r['avg_pos']:>7.1f} {r['W1']:>6} {r['W2']:>6} {r['W3']:>6}{marker}")

    print(f"\n  Total elapsed: {(time.time()-t0)/60:.1f}min")
    print("\n  ✅ R63 COMPLETE")

    # Save
    out_path = "/data/datasets/results_r63_uncertainty.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
