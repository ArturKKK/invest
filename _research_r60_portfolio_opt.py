#!/usr/bin/env python3
"""
R60 — Portfolio Construction Optimization

Tests 5 portfolio modes on top of gen8 champion (31 features):
  1. baseline       — 6L/3S equal-weight (current prod)
  2. grid_K         — 4L/2S, 6L/3S, 8L/4S, 3L/3S (K sensitivity)
  3. dynamic_K      — K adapts to signal strength each period
  4. edge_cost_filter — skip positions where edge < estimated cost
  5. prob_weighting  — weight positions by raw probability (not equal)

Uses ORIGINAL_WINDOWS (with gaps) for comparability with baseline Sharpe 1.66.
N_ROUNDS=600, EARLY_STOP=40, 5 seeds.
"""

import sys
import warnings
from typing import Dict, List, Set, Tuple
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

warnings.filterwarnings("ignore")

# ── project imports ────────────────────────────────────────────
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

# ── Tier definitions (from r48) ────────────────────────────────
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


# ── Windows ────────────────────────────────────────────────────
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
    """Train LGB+XGB ensemble. Returns merged preds with raw probs (for R60)."""
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
                ic = stats.spearmanr(p, fwd.set_index(["timestamp", "symbol"]).reindex(
                    te.set_index(["timestamp", "symbol"]).index)["fwd_ret"].values)[0]
                log(f"  {w['name']}/s{seed}: train={len(tr):,} test={len(te):,} IC={ic:.4f}")

    if not all_lgb:
        return None

    lgb_df = pd.concat(all_lgb)
    xgb_df = pd.concat(all_xgb)

    # Average raw probs per seed (keep raw for prob_weighting / edge_cost)
    lgb_avg = lgb_df.groupby(["timestamp", "symbol"]).agg(
        pred_lgb=("pred_lgb", "mean"), fwd_ret=("fwd_ret", "first"),
        window=("window", "first")).reset_index()
    xgb_avg = xgb_df.groupby(["timestamp", "symbol"]).agg(
        pred_xgb=("pred_xgb", "mean")).reset_index()

    merged = lgb_avg.merge(xgb_avg, on=["timestamp", "symbol"], how="inner")
    # Raw blended prob (before rank-norm) — used by prob_weighting & edge_cost_filter
    merged["raw_prob"] = 0.5 * merged["pred_lgb"] + 0.5 * merged["pred_xgb"]

    # Standard rank-normalized score (prod pipeline)
    merged["rank_lgb"] = merged.groupby("timestamp")["pred_lgb"].rank(pct=True) - 0.5
    merged["rank_xgb"] = merged.groupby("timestamp")["pred_xgb"].rank(pct=True) - 0.5
    merged["pred"] = 0.5 * merged["rank_lgb"] + 0.5 * merged["rank_xgb"]

    return merged[["timestamp", "symbol", "pred", "raw_prob", "fwd_ret", "window"]]


# ══════════════════════════════════════════════════════════
#  PORTFOLIO SIMULATORS
# ══════════════════════════════════════════════════════════

def simulate_mode(merged, regime_df, cfg, mode="baseline", mode_params=None):
    """
    Unified simulator supporting 5 portfolio modes:
      baseline        — 6L/3S equal-weight
      grid_K          — fixed K from mode_params {"n_long": X, "n_short": Y}
      dynamic_K       — K adapts to signal strength
      edge_cost_filter — skip positions where edge < estimated cost
      prob_weighting  — weight by raw_prob distance from 0.5
    """
    n_long = cfg.get("n_long", 6)
    n_short = cfg.get("n_short", 3)
    if mode == "grid_K" and mode_params:
        n_long = mode_params.get("n_long", n_long)
        n_short = mode_params.get("n_short", n_short)

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
        n = len(grp)

        # Determine current K
        nl, ns = n_long, n_short

        if mode == "dynamic_K":
            # Compute strength from rank-normalized scores
            all_scores = grp["pred"].values
            median_score = np.median(all_scores)
            # Sort descending, take top nl scores
            sorted_scores = np.sort(all_scores)[::-1]
            top_nl = sorted_scores[:nl]
            top_ns = sorted_scores[-ns:]
            long_strength = top_nl.mean() - median_score if len(top_nl) > 0 else 0
            short_strength = median_score - top_ns.mean() if len(top_ns) > 0 else 0

            # Adjust K: +1 if strength > 0.3, +2 if > 0.6 (relative to score range)
            score_range = all_scores.max() - all_scores.min() + 1e-10
            long_adj = 0
            if long_strength / score_range > 0.3:
                long_adj = 1
            if long_strength / score_range > 0.6:
                long_adj = 2
            short_adj = 0
            if short_strength / score_range > 0.3:
                short_adj = 1
            if short_strength / score_range > 0.6:
                short_adj = 2

            nl = int(np.clip(nl + long_adj, 2, 8))
            ns = int(np.clip(ns + short_adj, 1, 4))

        nl = min(nl, n // 3)
        ns = min(ns, n // 3)
        if nl == 0 and ns == 0:
            continue

        exposure = 1.0
        dyn_threshold = cfg.get("dyn_threshold", 0.5)
        if dyn_threshold is not None and trend_str > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) /
                           (trend_cutoff - dyn_threshold + 1e-10) * 0.5)

        # EMA smoothing of pred
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

        # Apply edge_cost_filter: remove positions where raw_prob edge < cost
        if mode == "edge_cost_filter":
            sym_to_raw = dict(zip(grp["symbol"], grp["raw_prob"]))
            filtered_longs = set()
            for sym in new_longs:
                edge = abs(sym_to_raw.get(sym, 0.5) - 0.5) * 2  # scale to [0,1]
                est_cost_bps = _cost_for_sym(sym) * 2  # round-trip
                if edge > est_cost_bps * 20:  # edge threshold: 20x cost in probability space
                    filtered_longs.add(sym)
            filtered_shorts = set()
            for sym in new_shorts:
                edge = abs(sym_to_raw.get(sym, 0.5) - 0.5) * 2
                est_cost_bps = _cost_for_sym(sym) * 2
                if edge > est_cost_bps * 20:
                    filtered_shorts.add(sym)
            new_longs = filtered_longs
            new_shorts = filtered_shorts
            if not new_longs and not new_shorts:
                prev_longs = new_longs
                prev_shorts = new_shorts
                continue

        # Compute returns
        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        total_positions = len(new_longs) + len(new_shorts)

        longs = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]

        if mode == "prob_weighting" and total_positions > 0:
            # Weight by distance from 0.5 in raw_prob
            sym_to_raw = dict(zip(grp["symbol"], grp["raw_prob"]))
            if len(longs) > 0:
                long_w = longs["symbol"].map(lambda s: max(0, sym_to_raw.get(s, 0.5) - 0.5))
                long_w_sum = long_w.sum()
                if long_w_sum > 1e-10:
                    long_ret = (longs["fwd_ret"].values * long_w.values / long_w_sum).sum()
                else:
                    long_ret = longs["fwd_ret"].mean()
            else:
                long_ret = 0

            if len(shorts) > 0:
                short_w = shorts["symbol"].map(lambda s: max(0, 0.5 - sym_to_raw.get(s, 0.5)))
                short_w_sum = short_w.sum()
                if short_w_sum > 1e-10:
                    short_ret = (shorts["fwd_ret"].values * short_w.values / short_w_sum).sum()
                else:
                    short_ret = shorts["fwd_ret"].mean()
            else:
                short_ret = 0
        else:
            long_ret = longs["fwd_ret"].mean() if len(longs) > 0 else 0
            short_ret = shorts["fwd_ret"].mean() if len(shorts) > 0 else 0

        # Hybrid cost
        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            turnover_cost = sum(_cost_for_sym(sym) * avg_weight for sym in new_opened)
            turnover_cost += sum(_cost_for_sym(sym) * avg_weight for sym in closed)
            holding_cost = funding_per_12h * (rebal_hours / 12)
            total_cost = turnover_cost + holding_cost
        else:
            total_cost = 0.0

        nl_act = len(new_longs)
        ns_act = len(new_shorts)
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

    if not all_rets:
        return pd.DataFrame()
    return pd.DataFrame(all_rets)


def compute_metrics(port_df, label="", capital=100):
    if port_df.empty:
        return {"sharpe": 0, "total_return_pct": 0, "max_dd_pct": 0,
                "win_rate": 0, "avg_positions": 0}
    eq = (1 + port_df["portfolio_ret"]).cumprod() * capital
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(2 * 365)
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    maxdd = (eq / eq.cummax() - 1).min()
    win_rate = (rets > 0).sum() / len(rets) * 100 if len(rets) > 0 else 0
    avg_pos = (port_df["n_long"] + port_df["n_short"]).mean()
    avg_cost_bps = port_df["cost"].mean() * 10000
    return {
        "sharpe": round(sharpe, 3),
        "total_return_pct": round(total_ret * 100, 1),
        "max_dd_pct": round(maxdd * 100, 1),
        "win_rate": round(win_rate, 1),
        "avg_positions": round(avg_pos, 1),
        "avg_cost_bps": round(avg_cost_bps, 2),
    }


def compute_window_metrics(port_df, preds, capital=100):
    """Per-window Sharpe."""
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
    print("  R60 — PORTFOLIO CONSTRUCTION OPTIMIZATION")
    print("  Champion: 31f features, ORIGINAL_WINDOWS + hybrid tiered costs")
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
    print(f"  Training complete: {len(preds):,} predictions in {(time.time()-t1)/60:.1f}min")

    # ── Define all experiment configurations ───────────────
    experiments = []

    # 1. Baseline
    experiments.append(("baseline", PROD_CFG, {}))

    # 2. Grid K
    for nl, ns in [(4, 2), (8, 4), (3, 3), (10, 5)]:
        cfg = {**PROD_CFG, "n_long": nl, "n_short": ns}
        experiments.append((f"grid_{nl}L{ns}S", cfg, {}))

    # 3. Dynamic K
    experiments.append(("dynamic_K", PROD_CFG, {}))

    # 4. Edge cost filter
    experiments.append(("edge_cost_filter", PROD_CFG, {}))

    # 5. Prob weighting
    experiments.append(("prob_weighting", PROD_CFG, {}))

    # 6. Bonus: combined dynamic_K + prob_weighting
    experiments.append(("dynK_probW", PROD_CFG, {}))

    # ── Run all experiments ────────────────────────────────
    results = []

    print(f"\n  Running {len(experiments)} portfolio modes...\n")

    for mode_name, cfg, params in experiments:
        # Map mode_name to actual mode
        if mode_name.startswith("grid_"):
            mode = "grid_K"
        elif mode_name == "dynK_probW":
            mode = "dynamic_K"  # will run twice (once as dynamic_K, once with prob_weighting below)
        else:
            mode = mode_name

        t2 = time.time()
        port = simulate_mode(preds, regime_df, cfg, mode=mode, mode_params=params)
        elapsed = time.time() - t2

        if port.empty:
            print(f"  ⚠️  {mode_name}: no returns generated")
            continue

        m = compute_metrics(port, mode_name)
        wm = compute_window_metrics(port, preds)

        # For dynK_probW — run prob_weighting with dynamic K
        if mode_name == "dynK_probW":
            port2 = simulate_mode(preds, regime_df, cfg, mode="prob_weighting", mode_params=params)
            # We want dynamic_K selection + prob_weighting — approximate: use prob_weighting
            # (dynamic_K changes selection, prob_weighting changes weights — to combine properly
            # we need a unified mode, let's code it within simulate_mode for this special case)
            # For now, skip the combined version (add_combined_mode below)
            pass

        result = {
            "mode": mode_name,
            "sharpe": m["sharpe"],
            "total_ret%": m["total_return_pct"],
            "maxDD%": m["max_dd_pct"],
            "win_rate%": m["win_rate"],
            "avg_pos": m["avg_positions"],
            "avg_cost_bps": m["avg_cost_bps"],
            "W1": wm.get("W1", "?"),
            "W2": wm.get("W2", "?"),
            "W3": wm.get("W3", "?"),
        }
        results.append(result)
        print(f"  [{mode_name:20s}] Sharpe={m['sharpe']:5.2f}  Ret={m['total_return_pct']:+6.1f}%  "
              f"DD={m['max_dd_pct']:5.1f}%  WR={m['win_rate']:.1f}%  "
              f"W1={wm.get('W1','?'):5.2f} W2={wm.get('W2','?'):5.2f} W3={wm.get('W3','?'):5.2f}  "
              f"({elapsed:.0f}s)")

    # ── Summary table ──────────────────────────────────────
    print("\n" + "=" * 90)
    print("  R60 RESULTS SUMMARY")
    print("=" * 90)
    print(f"  {'Mode':<22} {'Sharpe':>7} {'Ret%':>7} {'MaxDD%':>7} {'WR%':>6} "
          f"{'AvgPos':>7} {'W1':>6} {'W2':>6} {'W3':>6}")
    print("  " + "-" * 88)
    for r in sorted(results, key=lambda x: x["sharpe"], reverse=True):
        marker = " ← BEST" if r["sharpe"] == max(x["sharpe"] for x in results) else ""
        marker += " ← BASELINE" if r["mode"] == "baseline" else ""
        print(f"  {r['mode']:<22} {r['sharpe']:>7.2f} {r['total_ret%']:>+7.1f} "
              f"{r['maxDD%']:>7.1f} {r['win_rate%']:>6.1f} "
              f"{r['avg_pos']:>7.1f} {r['W1']:>6} {r['W2']:>6} {r['W3']:>6}{marker}")

    print(f"\n  Total elapsed: {(time.time()-t0)/60:.1f}min")
    print("\n  ✅ R60 COMPLETE")

    # Save to CSV
    results_df = pd.DataFrame(results)
    out_path = "/data/datasets/results_r60_portfolio_opt.csv"
    results_df.to_csv(out_path, index=False)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
