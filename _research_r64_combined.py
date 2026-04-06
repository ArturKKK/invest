#!/usr/bin/env python3
"""
R64 — Combined Best: grid_4L2S + uncertainty_filter_std003

Combines the two best findings from R60 and R63:
  - R60 winner: grid_K=4L/2S (from grid_4L2S)
  - R63 winner: uncertainty_filter with threshold=0.03

Tests combinations:
  1. baseline         — 6L/3S, no gating (current)
  2. grid_only        — 4L/2S, no gating
  3. filter_only      — 6L/3S + std<0.03 filter
  4. grid+filter      — 4L/2S + std<0.03 filter  ← main candidate
  5. grid+filter+cg   — 4L/2S + std<0.03 + cg_temporal features

Uses ORIGINAL_WINDOWS for comparability.
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

# Base production config (6L/3S)
BASE_CFG = {
    "n_long": 6, "n_short": 3, "rebal_hours": 12,
    "trend_cutoff": 0.9, "dyn_threshold": 0.7,
    "ema_alpha": 0.5, "hysteresis": 3,
}

# 4L/2S config
K42_CFG = {**BASE_CFG, "n_long": 4, "n_short": 2}

# Temporal cg features (from R61)
CG_TEMPORAL_FEATS = [
    "cg_taker_imb_lag1", "cg_taker_imb_lag2",
    "cg_taker_imb_roll5_mean", "cg_taker_imb_roll5_std",
]

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

STD_FILTER_THRESHOLD = 0.03  # from R63 winner


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cg_taker_imb temporal features per symbol."""
    df = df.copy()

    def per_symbol(grp):
        grp = grp.sort_values("timestamp").copy()
        if "cg_taker_imb" in grp.columns:
            grp["cg_taker_imb_lag1"] = grp["cg_taker_imb"].shift(1)
            grp["cg_taker_imb_lag2"] = grp["cg_taker_imb"].shift(2)
            grp["cg_taker_imb_roll5_mean"] = grp["cg_taker_imb"].rolling(5, min_periods=3).mean()
            grp["cg_taker_imb_roll5_std"] = grp["cg_taker_imb"].rolling(5, min_periods=3).std()
        return grp

    return df.groupby("symbol", group_keys=False).apply(per_symbol)


def load_data():
    print("=" * 70)
    print("  LOADING DATA")
    print("=" * 70)
    df, regime_df = load_research_frame()
    df, _ = add_r35_features(df)
    cg = load_cg_daily()
    cg_feats = compute_cg_features(cg)
    df, _, _ = add_cg_features(df, cg_feats)
    print("  Adding cg_temporal features...")
    df = add_temporal_features(df)
    print(f"  Frame: {len(df):,} rows, {df['symbol'].nunique()} symbols")
    return df, regime_df


def train_ensemble_with_uncertainty(df, feats, windows, seeds=SEEDS, cs_rank_exclude=None):
    """Train ensemble returning per-seed probs for uncertainty estimation."""
    avail = [f for f in feats if f in df.columns]
    rank_exclude = set(cs_rank_exclude or [])
    rank_feats = [f for f in avail if f not in rank_exclude]
    tz = df["timestamp"].dt.tz

    all_records = []

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
            te = test_[avail + ["target_binary", "timestamp", "symbol", "fwd_ret_12h"]].dropna()

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
            p_l = m.predict(te[avail])

            # XGB
            dt_x = xgb.DMatrix(tr[avail], label=tr["target_binary"])
            dv_x = xgb.DMatrix(va[avail], label=va["target_binary"])
            m_x = xgb.train(p_xgb, dt_x, num_boost_round=N_ROUNDS,
                             evals=[(dv_x, "val")],
                             early_stopping_rounds=EARLY_STOP, verbose_eval=False)
            p_x = m_x.predict(xgb.DMatrix(te[avail]))

            rec = te[["timestamp", "symbol", "fwd_ret_12h"]].copy()
            rec["pred_lgb"] = p_l
            rec["pred_xgb"] = p_x
            rec["window"] = w["name"]
            rec["seed"] = seed
            all_records.append(rec)

            if seed == seeds[0]:
                log(f"  {w['name']}/s{seed}: train={len(tr):,} test={len(te):,}")

    if not all_records:
        return None

    big = pd.concat(all_records)

    # Compute mean pred + std across seeds
    agg = big.groupby(["timestamp", "symbol"]).agg(
        pred_lgb_mean=("pred_lgb", "mean"),
        pred_xgb_mean=("pred_xgb", "mean"),
        pred_lgb_std=("pred_lgb", "std"),
        pred_xgb_std=("pred_xgb", "std"),
        fwd_ret=("fwd_ret_12h", "first"),
        window=("window", "first"),
    ).reset_index()

    agg["pred"] = 0.5 * (agg["pred_lgb_mean"].rank(pct=True) - 0.5) + \
                  0.5 * (agg["pred_xgb_mean"].rank(pct=True) - 0.5)
    agg["p_std"] = 0.5 * agg["pred_lgb_std"].fillna(0) + \
                   0.5 * agg["pred_xgb_std"].fillna(0)

    return agg


def simulate_combined(merged, regime_df, cfg, uncertainty_threshold=None):
    """
    Simulate with optional:
      - n_long/n_short from cfg
      - uncertainty gating: skip position if p_std >= threshold
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

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        if trend_str > trend_cutoff:
            continue
        grp = grouped[ts].copy()

        # Uncertainty gating
        if uncertainty_threshold is not None and "p_std" in grp.columns:
            grp = grp[grp["p_std"] < uncertainty_threshold]

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

        longs = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]
        long_ret = longs["fwd_ret"].mean() if len(longs) > 0 else 0
        short_ret = shorts["fwd_ret"].mean() if len(shorts) > 0 else 0

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
            "n_long": nl_act,
            "n_short": ns_act,
            "cost": total_cost,
        })

    return pd.DataFrame(all_rets) if all_rets else pd.DataFrame()


def compute_metrics(port_df):
    if port_df.empty:
        return {"sharpe": 0, "total_return_pct": 0, "max_dd_pct": 0, "win_rate": 0}
    eq = (1 + port_df["portfolio_ret"]).cumprod()
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(2 * 365)
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    maxdd = (eq / eq.cummax() - 1).min()
    win_rate = (rets > 0).sum() / len(rets) * 100
    return {
        "sharpe": round(sharpe, 3),
        "total_return_pct": round(total_ret * 100, 1),
        "max_dd_pct": round(maxdd * 100, 1),
        "win_rate": round(win_rate, 1),
    }


def compute_window_metrics(port_df, preds):
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
    print("  R64 — COMBINED BEST: grid_4L2S + filter_std003 (+cg_temporal)")
    print("  Verifying final configuration from R60/R63/R61 winners")
    print("=" * 70)

    df, regime_df = load_data()
    no_rank_base = [f for f in CHAMPION_FEAT_31 if f in MARKET_LEVEL_FEATURES]
    no_rank_cg = no_rank_base + CG_TEMPORAL_FEATS

    # feat sets
    F31 = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    F35 = [f for f in CHAMPION_FEAT_31 + CG_TEMPORAL_FEATS if f in df.columns]

    experiments = [
        # (name, feats, cfg, unc_threshold, no_rank)
        ("baseline_6L3S",    F31, BASE_CFG,  None,                 no_rank_base),
        ("grid_4L2S",        F31, K42_CFG,   None,                 no_rank_base),
        ("filter_std003",    F31, BASE_CFG,  STD_FILTER_THRESHOLD, no_rank_base),
        ("grid4L2S+filter",  F31, K42_CFG,   STD_FILTER_THRESHOLD, no_rank_base),
        ("grid4L2S+flt+cgt", F35, K42_CFG,   STD_FILTER_THRESHOLD, no_rank_cg),
    ]

    # Train once per feature set
    preds_31 = None
    preds_35 = None

    print("\n  Training 31f ensemble (for baseline/grid/filter)...")
    preds_31 = train_ensemble_with_uncertainty(df, F31, ORIGINAL_WINDOWS,
                                               seeds=SEEDS, cs_rank_exclude=no_rank_base)

    print("\n  Training 35f ensemble (+cg_temporal)...")
    preds_35 = train_ensemble_with_uncertainty(df, F35, ORIGINAL_WINDOWS,
                                               seeds=SEEDS, cs_rank_exclude=no_rank_cg)

    results = []

    print(f"\n  Simulating {len(experiments)} experiment variants...\n")

    for exp_name, feats, cfg, unc_thr, no_rank in experiments:
        preds = preds_31 if feats is F31 else preds_35
        if preds is None:
            continue

        t1 = time.time()
        port = simulate_combined(preds, regime_df, cfg, uncertainty_threshold=unc_thr)
        if port.empty:
            continue

        m = compute_metrics(port)
        wm = compute_window_metrics(port, preds)
        elapsed = time.time() - t1

        result = {
            "experiment": exp_name,
            "n_features": len([f for f in feats if f in df.columns]),
            "n_long": cfg["n_long"],
            "n_short": cfg["n_short"],
            "unc_threshold": unc_thr,
            "sharpe": m["sharpe"],
            "total_ret%": m["total_return_pct"],
            "maxDD%": m["max_dd_pct"],
            "win_rate%": m["win_rate"],
            "W1": wm.get("W1", "?"),
            "W2": wm.get("W2", "?"),
            "W3": wm.get("W3", "?"),
        }
        results.append(result)
        print(f"  [{exp_name:<25}]  Sharpe={m['sharpe']:5.2f}  "
              f"Ret={m['total_return_pct']:+6.1f}%  DD={m['max_dd_pct']:5.1f}%  "
              f"W1={wm.get('W1','?'):5.2f} W2={wm.get('W2','?'):5.2f} W3={wm.get('W3','?'):5.2f}  ({elapsed:.1f}s)")

    # ── Summary ───────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  R64 COMBINED RESULTS SUMMARY")
    print("=" * 100)
    print(f"  {'Experiment':<28} {'K':>4} {'Thr':>6} {'Sharpe':>7} {'Ret%':>7} {'MaxDD%':>7} "
          f"{'WR%':>6} {'W1':>6} {'W2':>6} {'W3':>6}")
    print("  " + "-" * 90)

    baseline_sharpe = next((r["sharpe"] for r in results if r["experiment"] == "baseline_6L3S"), 0)

    for r in sorted(results, key=lambda x: x["sharpe"], reverse=True):
        delta = r["sharpe"] - baseline_sharpe
        k_str = f"{r['n_long']}L{r['n_short']}S"
        thr_str = f"{r['unc_threshold']:.2f}" if r["unc_threshold"] else "none"
        marker = f" (Δ{delta:+.2f})" if r["experiment"] != "baseline_6L3S" else " ← BASELINE"
        print(f"  {r['experiment']:<28} {k_str:>4} {thr_str:>6} {r['sharpe']:>7.2f} "
              f"{r['total_ret%']:>+7.1f} {r['maxDD%']:>7.1f} {r['win_rate%']:>6.1f} "
              f"{r['W1']:>6} {r['W2']:>6} {r['W3']:>6}{marker}")

    print(f"\n  Total elapsed: {(time.time()-t0)/60:.1f}min")
    print("\n  ✅ R64 COMPLETE")

    out_path = "/data/datasets/results_r64_combined.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
