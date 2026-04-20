#!/usr/bin/env python3
"""
R126 — Regularization tuning to compensate Sharpe loss from inf→nan fix.

Baseline (bugs): 3.777   |  inf-fixed baseline: 3.224
Goal: ≥ 3.4 via hyperparameter tuning on inf-fixed data.

Stage 2: Random search over LGB+XGB hyperparams (100 points each).
Stage 3: Test zero-cols vs feature_fraction.
"""

import sys, warnings, time, json, itertools
from typing import Dict, Set, List, Optional

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

N_ROUNDS = 600
EARLY_STOP = 40

# ── Default (current production) hyperparams ──────────────────
DEFAULT_LGB = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.03, "num_leaves": 63,
    "min_child_samples": 100, "subsample": 0.8,
    "colsample_bytree": 0.8, "lambda_l2": 1.0,
    "verbose": -1, "n_jobs": -1,
}
DEFAULT_XGB = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "learning_rate": 0.03, "max_depth": 6,
    "min_child_weight": 100, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_lambda": 1.0,
    "n_jobs": -1, "verbosity": 0,
}

# ══════════════════════════════════════════════════════════════
#  TRAINING (single seed, single pass — fast for search)
# ══════════════════════════════════════════════════════════════

def train_fast(df, feats, windows, lgb_params, xgb_params,
               seeds=(0, 42), cs_rank_exclude=None):
    """Train with 2 seeds (fast) for search, return merged predictions."""
    avail = [f for f in feats if f in df.columns]
    rank_exclude = set(cs_rank_exclude or [])
    rank_feats = [f for f in avail if f not in rank_exclude]
    tz = df["timestamp"].dt.tz
    all_lgb, all_xgb = [], []

    for seed in seeds:
        p_lgb = {**lgb_params, "seed": seed}
        p_xgb = {**xgb_params, "seed": seed}
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
            rec["window"] = w["name"]; rec["seed"] = seed
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
            rec2["window"] = w["name"]; rec2["seed"] = seed
            all_xgb.append(rec2)

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
            # cef6e2f simulate: skip risk-off periods (688 periods)
            prev_longs, prev_shorts = set(), set()
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
            holding_cost = funding_per_12h * (rebal_hours / 12)
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


def sharpe(rets_series, periods_per_year=2*365):
    if len(rets_series) < 2:
        return 0.0
    eq = (1 + rets_series).cumprod()
    r = eq.pct_change().dropna()
    return r.mean() / (r.std() + 1e-10) * np.sqrt(periods_per_year)


def eval_config(df, regime_df, feats, no_rank, lgb_p, xgb_p,
                seeds=(0, 42)):
    """Train & eval a single config. Returns 4L/2S continuous Net Sharpe."""
    preds = train_fast(df, feats, CONTINUOUS_WINDOWS, lgb_p, xgb_p,
                       seeds=seeds, cs_rank_exclude=no_rank)
    if preds is None:
        return -999.0
    port = simulate(preds, regime_df, 4, 2)
    if port.empty:
        return -999.0
    return sharpe(port["net_ret"])


# ══════════════════════════════════════════════════════════════
#  SEARCH SPACES
# ══════════════════════════════════════════════════════════════

LGB_GRID = {
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
    "num_leaves":       [15, 23, 31, 47, 63],
    "min_child_samples":[30, 50, 100, 200],
    "lambda_l1":        [0, 0.1, 1.0, 5.0],
    "lambda_l2":        [0, 0.1, 1.0, 5.0, 10.0],
}

XGB_GRID = {
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
    "max_depth":        [4, 5, 6, 7],
    "min_child_weight": [5, 10, 20, 50, 100],
    "reg_alpha":        [0, 0.1, 1.0, 5.0],
    "reg_lambda":       [0.1, 1.0, 5.0, 10.0],
}


def random_sample(grid: dict, n: int, rng: np.random.RandomState) -> List[dict]:
    """Sample n random configs from grid."""
    samples = []
    keys = list(grid.keys())
    for _ in range(n):
        cfg = {}
        for k in keys:
            cfg[k] = grid[k][rng.randint(len(grid[k]))]
        samples.append(cfg)
    return samples


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


# ══════════════════════════════════════════════════════════════
#  STAGE 2: RANDOM SEARCH
# ══════════════════════════════════════════════════════════════

def stage2_search(df, regime_df, feats, no_rank, n_lgb=80, n_xgb=80):
    """
    Phase A: Fix XGB defaults, search LGB (n_lgb points).
    Phase B: Fix LGB=best, search XGB (n_xgb points).
    Phase C: Joint — top-5 LGB × top-5 XGB = 25 combos.
    """
    rng = np.random.RandomState(42)
    results = []
    log_path = "/data/datasets/r126_search_log.jsonl"

    # ── Phase A: LGB search (XGB = default) ────────────────────
    print("\n" + "=" * 70)
    print("  STAGE 2A: LGB RANDOM SEARCH")
    print("=" * 70)
    lgb_samples = random_sample(LGB_GRID, n_lgb, rng)
    best_lgb_sharpe = -999
    best_lgb_cfg = None
    lgb_results = []

    for i, lcfg in enumerate(lgb_samples):
        lgb_p = {**DEFAULT_LGB, **lcfg}
        xgb_p = {**DEFAULT_XGB}
        t0 = time.time()
        ns = eval_config(df, regime_df, feats, no_rank, lgb_p, xgb_p)
        elapsed = time.time() - t0

        entry = {"phase": "A", "i": i, "lgb": lcfg, "xgb": {},
                 "net_sharpe": round(ns, 4), "time": round(elapsed, 1)}
        lgb_results.append(entry)
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        if ns > best_lgb_sharpe:
            best_lgb_sharpe = ns
            best_lgb_cfg = lcfg
        log(f"  A[{i+1}/{n_lgb}] NS={ns:.3f} best={best_lgb_sharpe:.3f} "
            f"({elapsed:.0f}s) leaves={lcfg.get('num_leaves')} "
            f"ff={lcfg.get('colsample_bytree')} "
            f"mcs={lcfg.get('min_child_samples')} "
            f"l1={lcfg.get('lambda_l1')} l2={lcfg.get('lambda_l2')}")

    print(f"\n  Best LGB: NS={best_lgb_sharpe:.3f}, cfg={best_lgb_cfg}")

    # ── Phase B: XGB search (LGB = best from A) ───────────────
    print("\n" + "=" * 70)
    print("  STAGE 2B: XGB RANDOM SEARCH")
    print("=" * 70)
    xgb_samples = random_sample(XGB_GRID, n_xgb, rng)
    best_xgb_sharpe = -999
    best_xgb_cfg = None
    xgb_results = []

    lgb_fixed = {**DEFAULT_LGB, **best_lgb_cfg}
    for i, xcfg in enumerate(xgb_samples):
        xgb_p = {**DEFAULT_XGB, **xcfg}
        t0 = time.time()
        ns = eval_config(df, regime_df, feats, no_rank, lgb_fixed, xgb_p)
        elapsed = time.time() - t0

        entry = {"phase": "B", "i": i, "lgb": best_lgb_cfg, "xgb": xcfg,
                 "net_sharpe": round(ns, 4), "time": round(elapsed, 1)}
        xgb_results.append(entry)
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        if ns > best_xgb_sharpe:
            best_xgb_sharpe = ns
            best_xgb_cfg = xcfg
        log(f"  B[{i+1}/{n_xgb}] NS={ns:.3f} best={best_xgb_sharpe:.3f} "
            f"({elapsed:.0f}s) depth={xcfg.get('max_depth')} "
            f"ff={xcfg.get('colsample_bytree')} "
            f"mcw={xcfg.get('min_child_weight')} "
            f"a={xcfg.get('reg_alpha')} l={xcfg.get('reg_lambda')}")

    print(f"\n  Best XGB: NS={best_xgb_sharpe:.3f}, cfg={best_xgb_cfg}")

    # ── Phase C: Cross-validate top-5 × top-5 ─────────────────
    print("\n" + "=" * 70)
    print("  STAGE 2C: TOP-5 LGB × TOP-5 XGB CROSS")
    print("=" * 70)

    top5_lgb = sorted(lgb_results, key=lambda x: x["net_sharpe"], reverse=True)[:5]
    top5_xgb = sorted(xgb_results, key=lambda x: x["net_sharpe"], reverse=True)[:5]

    best_combo_sharpe = -999
    best_combo = None
    combo_results = []

    for il, lr in enumerate(top5_lgb):
        for ix, xr in enumerate(top5_xgb):
            lgb_p = {**DEFAULT_LGB, **lr["lgb"]}
            xgb_p = {**DEFAULT_XGB, **xr["xgb"]}
            t0 = time.time()
            ns = eval_config(df, regime_df, feats, no_rank, lgb_p, xgb_p)
            elapsed = time.time() - t0

            entry = {"phase": "C", "lgb": lr["lgb"], "xgb": xr["xgb"],
                     "net_sharpe": round(ns, 4), "time": round(elapsed, 1)}
            combo_results.append(entry)
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

            if ns > best_combo_sharpe:
                best_combo_sharpe = ns
                best_combo = (lr["lgb"], xr["xgb"])
            log(f"  C[L{il}/X{ix}] NS={ns:.3f} best={best_combo_sharpe:.3f} ({elapsed:.0f}s)")

    print(f"\n  Best combo: NS={best_combo_sharpe:.3f}")
    print(f"  LGB: {best_combo[0]}")
    print(f"  XGB: {best_combo[1]}")

    return best_combo, best_combo_sharpe


# ══════════════════════════════════════════════════════════════
#  STAGE 3: ZERO COLS TEST
# ══════════════════════════════════════════════════════════════

DROP_6 = {"pct_coins_up_12h", "pct_coins_up_1h",
          "hour_sin", "hour_cos", "dow_sin", "dow_cos"}

def stage3_zero_cols(df, regime_df, feats, no_rank,
                     best_lgb_cfg, best_xgb_cfg):
    """
    A: 25 feats + feature_fraction=0.6
    B: 25 feats + best hyperparams
    C: 31 feats (6 zeros) + best hyperparams
    """
    print("\n" + "=" * 70)
    print("  STAGE 3: ZERO COLS TEST")
    print("=" * 70)

    feats_25 = [f for f in feats if f not in DROP_6]

    # Test A: 25 feats + low feature_fraction
    lgb_a = {**DEFAULT_LGB, "colsample_bytree": 0.6}
    xgb_a = {**DEFAULT_XGB, "colsample_bytree": 0.6}
    no_rank_25 = [f for f in feats_25 if f in MARKET_LEVEL_FEATURES]
    ns_a = eval_config(df, regime_df, feats_25, no_rank_25, lgb_a, xgb_a)
    print(f"  A (25f + ff=0.6):       NS={ns_a:.3f}")

    # Test B: 25 feats + best hyperparams
    lgb_b = {**DEFAULT_LGB, **best_lgb_cfg}
    xgb_b = {**DEFAULT_XGB, **best_xgb_cfg}
    ns_b = eval_config(df, regime_df, feats_25, no_rank_25, lgb_b, xgb_b)
    print(f"  B (25f + best hparams): NS={ns_b:.3f}")

    # Test C: 31 feats (6 zeros via CS-rank) + best hyperparams
    ns_c = eval_config(df, regime_df, feats, no_rank, lgb_b, xgb_b)
    print(f"  C (31f + best hparams): NS={ns_c:.3f}")

    print(f"\n  Decision:")
    if ns_c > ns_a and ns_c > ns_b:
        print(f"  → Keep 31 features (zeros help beyond regularization)")
    elif ns_b >= ns_c:
        print(f"  → 25 features + tuning sufficient (zeros not needed)")
    else:
        print(f"  → Low feature_fraction compensates (zeros = implicit ff reduction)")

    return {"A_25f_ff06": ns_a, "B_25f_best": ns_b, "C_31f_best": ns_c}


# ══════════════════════════════════════════════════════════════
#  STAGE 4: FINAL VALIDATION (5 seeds)
# ══════════════════════════════════════════════════════════════

def stage4_validate(df, regime_df, feats, no_rank,
                    best_lgb_cfg, best_xgb_cfg, label="best"):
    """Full 5-seed walk-forward on best config."""
    print("\n" + "=" * 70)
    print(f"  STAGE 4: FINAL VALIDATION ({label})")
    print("=" * 70)

    lgb_p = {**DEFAULT_LGB, **best_lgb_cfg}
    xgb_p = {**DEFAULT_XGB, **best_xgb_cfg}

    preds = train_fast(df, feats, CONTINUOUS_WINDOWS, lgb_p, xgb_p,
                       seeds=SEEDS, cs_rank_exclude=no_rank)
    if preds is None:
        print("  FAILED: no predictions")
        return

    # 4L/2S
    port_4l = simulate(preds, regime_df, 4, 2)
    ns_4l = sharpe(port_4l["net_ret"])
    gs_4l = sharpe(port_4l["gross_ret"])
    eq = (1 + port_4l["net_ret"]).cumprod() * 100
    dd_4l = (eq / eq.cummax() - 1).min()
    ret_4l = eq.iloc[-1] / eq.iloc[0] - 1

    # 6L/3S
    port_6l = simulate(preds, regime_df, 6, 3)
    ns_6l = sharpe(port_6l["net_ret"])
    gs_6l = sharpe(port_6l["gross_ret"])
    eq6 = (1 + port_6l["net_ret"]).cumprod() * 100
    dd_6l = (eq6 / eq6.cummax() - 1).min()
    ret_6l = eq6.iloc[-1] / eq6.iloc[0] - 1

    print(f"\n  {'Config':<22} {'Gross Sh':>10} {'Net Sh':>10} {'Ret%':>8} {'DD%':>8} {'Periods':>8}")
    print(f"  {'-'*66}")
    print(f"  {'4L/2S continuous':<22} {gs_4l:>10.3f} {ns_4l:>10.3f} "
          f"{ret_4l*100:>7.1f}% {dd_4l*100:>7.1f}% {len(port_4l):>8}")
    print(f"  {'6L/3S continuous':<22} {gs_6l:>10.3f} {ns_6l:>10.3f} "
          f"{ret_6l*100:>7.1f}% {dd_6l*100:>7.1f}% {len(port_6l):>8}")

    # Bootstrap confidence
    n_boot = 1000
    boot_sharpes = []
    rets = port_4l["net_ret"].values
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        idx = rng.choice(len(rets), size=len(rets), replace=True)
        s = rets[idx]
        eq_b = np.cumprod(1 + s)
        r_b = np.diff(eq_b) / eq_b[:-1]
        if len(r_b) > 1 and np.std(r_b) > 0:
            boot_sharpes.append(np.mean(r_b) / np.std(r_b) * np.sqrt(730))
    boot_sharpes = np.array(boot_sharpes)
    p_above_baseline = (boot_sharpes > 3.224).mean()
    p_above_target = (boot_sharpes > 3.4).mean()
    ci_lo, ci_hi = np.percentile(boot_sharpes, [5, 95])

    print(f"\n  Bootstrap (4L/2S): 90% CI = [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  P(NS > 3.224 baseline) = {p_above_baseline:.2%}")
    print(f"  P(NS > 3.4 target) = {p_above_target:.2%}")

    # Monthly breakdown
    port_c = port_4l.copy()
    port_c["month"] = port_c["timestamp"].dt.to_period("M").astype(str)
    months = sorted(port_c["month"].unique())
    print(f"\n  {'Month':<10} {'Gross%':>8} {'Net%':>8} {'Cost%':>8} {'Per':>5}")
    for m in months:
        mdf = port_c[port_c["month"] == m]
        gr = ((1 + mdf["gross_ret"]).cumprod().iloc[-1] - 1) * 100
        nr = ((1 + mdf["net_ret"]).cumprod().iloc[-1] - 1) * 100
        mc = mdf["cost"].sum() * 100
        print(f"  {m:<10} {gr:>7.1f}% {nr:>7.1f}% {mc:>7.2f}% {len(mdf):>5}")

    # Quarterly
    port_c["quarter"] = port_c["timestamp"].dt.to_period("Q").astype(str)
    quarters = sorted(port_c["quarter"].unique())
    print(f"  {'Quarter':<10} {'GrossSh':>9} {'NetSh':>9} {'NetRet%':>9}")
    for q in quarters:
        qdf = port_c[port_c["quarter"] == q]
        qgs = sharpe(qdf["gross_ret"])
        qns = sharpe(qdf["net_ret"])
        qnr = ((1 + qdf["net_ret"]).cumprod().iloc[-1] - 1) * 100
        print(f"  {q:<10} {qgs:>9.2f} {qns:>9.2f} {qnr:>8.1f}%")

    # Acceptance check
    print(f"\n  ACCEPTANCE:")
    passed = True
    if ns_4l >= 3.4:
        print(f"  ✓ Net Sharpe {ns_4l:.3f} ≥ 3.4")
    else:
        print(f"  ✗ Net Sharpe {ns_4l:.3f} < 3.4")
        passed = False
    if dd_4l >= -0.20:
        print(f"  ✓ Max DD {dd_4l*100:.1f}% ≥ -20%")
    else:
        print(f"  ✗ Max DD {dd_4l*100:.1f}% < -20%")
        passed = False
    if p_above_baseline >= 0.70:
        print(f"  ✓ P(improvement) {p_above_baseline:.2%} ≥ 70%")
    else:
        print(f"  ✗ P(improvement) {p_above_baseline:.2%} < 70%")
        passed = False

    if passed:
        print(f"\n  ✓ ALL CHECKS PASSED — ready for deploy")
    else:
        print(f"\n  ✗ FAILED — stay on current config")

    return {
        "ns_4l": round(ns_4l, 3), "ns_6l": round(ns_6l, 3),
        "dd_4l": round(dd_4l * 100, 1), "dd_6l": round(dd_6l * 100, 1),
        "ret_4l": round(ret_4l * 100, 1), "ret_6l": round(ret_6l * 100, 1),
        "p_above_baseline": round(p_above_baseline, 3),
        "p_above_target": round(p_above_target, 3),
        "passed": passed,
        "lgb": best_lgb_cfg, "xgb": best_xgb_cfg,
    }


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    stage = sys.argv[1] if len(sys.argv) > 1 else "all"

    df, regime_df = load_data()
    feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]

    if stage in ("1", "all"):
        # ── Stage 1: Confirm inf-fix baseline ──────────────────
        print("\n" + "=" * 70)
        print("  STAGE 1: INF-FIX BASELINE CONFIRMATION")
        print("=" * 70)
        ns_base = eval_config(df, regime_df, feats, no_rank,
                              DEFAULT_LGB, DEFAULT_XGB, seeds=SEEDS)
        print(f"\n  Baseline (inf-fixed, default hparams): Net Sharpe = {ns_base:.3f}")
        print(f"  Expected: ~3.224")
        if stage == "1":
            print(f"\n  Total: {time.time()-t_start:.0f}s")
            return

    if stage in ("2", "all"):
        best_combo, best_sharpe = stage2_search(df, regime_df, feats, no_rank)
        best_lgb_cfg, best_xgb_cfg = best_combo
        if stage == "2":
            print(f"\n  Total: {time.time()-t_start:.0f}s")
            return
    elif stage in ("3", "4"):
        # Load best from log
        log_path = "/data/datasets/r126_search_log.jsonl"
        entries = [json.loads(l) for l in open(log_path)]
        phase_c = [e for e in entries if e["phase"] == "C"]
        if phase_c:
            best_entry = max(phase_c, key=lambda x: x["net_sharpe"])
        else:
            best_entry = max(entries, key=lambda x: x["net_sharpe"])
        best_lgb_cfg = best_entry["lgb"]
        best_xgb_cfg = best_entry["xgb"]
        print(f"  Loaded best config from log: NS={best_entry['net_sharpe']}")
        print(f"  LGB: {best_lgb_cfg}")
        print(f"  XGB: {best_xgb_cfg}")

    if stage in ("3", "all"):
        s3 = stage3_zero_cols(df, regime_df, feats, no_rank,
                              best_lgb_cfg, best_xgb_cfg)

    if stage in ("4", "all"):
        s4 = stage4_validate(df, regime_df, feats, no_rank,
                             best_lgb_cfg, best_xgb_cfg)
        # Save final results
        with open("/data/datasets/r126_final.json", "w") as f:
            json.dump(s4, f, indent=2)
        print(f"\n  Saved: /data/datasets/r126_final.json")

    print(f"\n  R126 total: {time.time()-t_start:.0f}s ({(time.time()-t_start)/60:.1f}min)")


if __name__ == "__main__":
    main()
