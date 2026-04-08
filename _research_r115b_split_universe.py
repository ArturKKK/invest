#!/usr/bin/env python3
"""
R115b — Frozen-normalization split-universe.

Fix for R115 (training on 50 symbols killed the model).

Key idea (both AI consultants):
  - Train model on SYM_35 only (clean, proven signal)
  - Predict on expanded universe (50+ symbols)
  - Select top-4/bottom-2 from larger pool
  - cs_rank on training = among SYM_35 (frozen anchor)
  - cs_rank on prediction = among all available symbols
  - Market-level features computed from SYM_35 anchor only (frozen normalization)

Grid:
  - min_adv_usd ∈ {10M, 20M, 50M}
  - n_long/n_short ∈ {(4, 2), (6, 3)}
  - With R114b champion params: moff=2, mon=0, cutoff=0.9/0.8

Acceptance:
  - Sharpe >= R114b baseline (3.266) or Calmar >= R114b (18.25)
"""
import time, json, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from typing import Set, Dict, List, Optional
import lightgbm as lgb
import xgboost as xgb
warnings.filterwarnings("ignore")

from _research_r22_models import SEEDS, log, cs_rank_cols
from _research_round7 import SYM_35
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal
from _research_r22_models import add_new_features, build_r19_features
from _research_r30b_fixed import add_extra_features_clean, compute_regime_extended
from _research_r33_creative_features import add_r33_features
from _research_r35_new_features import add_r35_features, MARKET_LEVEL_FEATURES
from _research_r47_coinglass import load_cg_daily, compute_cg_features
from _research_r68_continuous_wf import (
    add_cg_features, CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, PROD_CFG,
    sharpe, _cost_for_sym,
    TIER1_SYMS, TIER2_SYMS,
    LGB_PARAMS, XGB_PARAMS, N_ROUNDS, EARLY_STOP,
)
from _research_r113_trend_cutoff_reopt import analyze_config, print_result
from _research_r114b_churn_reduction import simulate_v2b

DATA_DIR = Path("data/raw")
SYM_35_SET = set(SYM_35)


# ─── Frozen-normalization feature builder ────────────────────────

def cs_rank_cols_frozen(df, feats, anchor_symbols):
    """
    Compute cross-sectional percentile ranks using ONLY anchor_symbols
    as the reference distribution. Non-anchor symbols get their rank
    relative to the anchor distribution.
    """
    df = df.copy()
    anchor_mask = df["symbol"].isin(anchor_symbols)

    for f in feats:
        if f not in df.columns:
            continue
        # For each timestamp, rank only among anchor symbols first
        # Then place non-anchor symbols relative to anchor distribution
        def _frozen_rank(grp):
            anchor = grp[grp.index.isin(df.index[anchor_mask])]
            if len(anchor) == 0:
                grp[f] = 0.0
                return grp
            # Rank anchor symbols among themselves
            anchor_vals = anchor[f].dropna()
            if len(anchor_vals) == 0:
                grp[f] = 0.0
                return grp
            # For ALL symbols, compute rank relative to anchor distribution
            # Use searchsorted to find where each value falls in anchor distribution
            sorted_anchor = np.sort(anchor_vals.values)
            n_anchor = len(sorted_anchor)
            # Percentile rank: fraction of anchor values less than each value
            ranks = np.searchsorted(sorted_anchor, grp[f].values, side="right") / n_anchor - 0.5
            grp[f] = ranks
            return grp

        df = df.groupby("timestamp", group_keys=False).apply(_frozen_rank)

    return df


def load_data_expanded_frozen():
    """
    Load ALL symbols but compute market-level features from SYM_35 only
    (frozen anchor normalization).
    """
    log("=" * 70)
    log("  LOADING DATA (frozen split-universe)")
    log("=" * 70)

    # Load ALL OHLCV
    ohlcv = load_ohlcv()
    all_syms = sorted(ohlcv["symbol"].unique().tolist())
    n_total_ohlcv = len(all_syms)
    log(f"  All OHLCV symbols: {n_total_ohlcv}")

    derivs = load_derivatives()

    # Build base features for ALL symbols
    df = build_features_minimal(ohlcv, derivs)
    df = build_r19_features(df)
    df, _ = add_new_features(df)
    df = add_extra_features_clean(df)
    df = add_r33_features(df)

    # ── Frozen market-level features: compute from SYM_35 only ──
    # Save the columns that will be overwritten by add_r35_features
    # (market-level features use groupby("timestamp").transform())
    # We compute them from SYM_35 anchor first, then broadcast to all symbols

    # Step 1: compute r35 features on SYM_35 only → get market-level values
    df_anchor = df[df["symbol"].isin(SYM_35_SET)].copy()
    df_anchor, _ = add_r35_features(df_anchor)

    # Extract market-level features from anchor (one value per timestamp)
    mkt_feats_available = [f for f in MARKET_LEVEL_FEATURES if f in df_anchor.columns]
    mkt_lookup = (df_anchor[["timestamp"] + mkt_feats_available]
                  .drop_duplicates(subset=["timestamp"])
                  .set_index("timestamp"))

    # Step 2: compute r35 features on ALL symbols (for per-symbol features)
    df, _ = add_r35_features(df)

    # Step 3: override market-level features with frozen values from anchor
    for f in mkt_feats_available:
        if f in df.columns:
            df = df.drop(columns=[f])
    df = df.merge(mkt_lookup[mkt_feats_available].reset_index(),
                  on="timestamp", how="left")

    # Regime from full df (BTC-driven, should be stable)
    regime_df = compute_regime_extended(df[df["symbol"].isin(SYM_35_SET)])

    # CoinGlass features (available for ~50 coins)
    try:
        cg = load_cg_daily()
        cg_feats = compute_cg_features(cg)
        df, _, _ = add_cg_features(df, cg_feats)
    except Exception as e:
        log(f"  CoinGlass features skipped: {e}")

    present = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    missing_f = [f for f in CHAMPION_FEAT_31 if f not in df.columns]
    if missing_f:
        log(f"  WARNING: Missing features: {missing_f}")

    n_sym = df["symbol"].nunique()
    n_35 = len([s for s in df["symbol"].unique() if s in SYM_35_SET])
    log(f"  Frame: {len(df):,} rows, {n_sym} symbols ({n_35} anchor + {n_sym - n_35} expanded)")
    log(f"  Features: {len(present)}/{len(CHAMPION_FEAT_31)}")

    # Compute ADV (7-day rolling avg daily $ volume)
    log("  Computing ADV...")
    df["dollar_vol_1h"] = df["close"] * df["volume"]
    df["adv_7d"] = df.groupby("symbol")["dollar_vol_1h"].transform(
        lambda x: x.rolling(168, min_periods=84).mean() * 24)
    log(f"  ADV computed. Median ADV: ${df['adv_7d'].median()/1e6:.1f}M")

    return df, regime_df


# ─── Split-universe train_ensemble ──────────────────────────────

def train_ensemble_split(df_all, feats, windows, seeds=SEEDS,
                         cs_rank_exclude=None, anchor_symbols=None):
    """
    Train on anchor_symbols only, predict on ALL symbols.
    cs_rank on train/val: among anchor only (frozen).
    cs_rank on test: among all available symbols at each timestamp.
    """
    if anchor_symbols is None:
        anchor_symbols = SYM_35_SET

    avail = [f for f in feats if f in df_all.columns]
    rank_exclude = set(cs_rank_exclude or [])
    rank_feats = [f for f in avail if f not in rank_exclude]
    tz = df_all["timestamp"].dt.tz
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

            # TRAIN/VAL: anchor symbols only (SYM_35)
            train_ = df_all[(df_all["timestamp"] < tr_end)
                            & (df_all["symbol"].isin(anchor_symbols))].copy()
            val_ = df_all[(df_all["timestamp"] >= va_start)
                          & (df_all["timestamp"] < va_end)
                          & (df_all["symbol"].isin(anchor_symbols))].copy()

            # TEST: ALL symbols (expanded universe)
            test_ = df_all[(df_all["timestamp"] >= te_start)
                           & (df_all["timestamp"] <= te_end)].copy()

            if len(train_) < 5000 or len(test_) < 200:
                continue

            # cs_rank on train/val: among anchor only
            if rank_feats:
                train_ = cs_rank_cols(train_, rank_feats)
                val_ = cs_rank_cols(val_, rank_feats)
                # cs_rank on test: among ALL symbols (pct=True gives [0,1])
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

            # LightGBM
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

            # XGBoost
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

            n_train_syms = train_["symbol"].nunique()
            n_test_syms = test_["symbol"].nunique()
            if seed == seeds[0]:
                log(f"  {w['name']}/s{seed}: train={len(tr):,} ({n_train_syms} syms) "
                    f"test={len(te):,} ({n_test_syms} syms)")

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


# ─── Volume-filtered simulate with R114b churn params ────────────

def simulate_v2b_volfilter(merged, regime_df, n_long, n_short, cfg,
                           cutoff_on=0.9, cutoff_off=0.8,
                           min_risk_off_periods=2, min_risk_on_periods=0,
                           min_adv_usd=10e6):
    """
    R114b simulate_v2b + point-in-time ADV filter.
    """
    rebal_hours = cfg["rebal_hours"]
    ema_alpha = cfg.get("ema_alpha", None)
    hysteresis = cfg.get("hysteresis", 0)
    dyn_threshold = cfg.get("dyn_threshold", 0.5)
    funding_per_12h = 0.00008

    all_rets = []
    prev_longs: Set[str] = set()
    prev_shorts: Set[str] = set()
    prev_preds: Dict[str, float] = {}
    risk_off = False
    periods_in_off = 0
    periods_in_on = 999

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}
    rebal_timestamps = timestamps_sorted[::rebal_hours]
    universe_sizes = []

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        grp = grouped[ts].copy()

        # ── Volume filter ──
        if "adv_7d" in grp.columns:
            grp = grp[grp["adv_7d"] >= min_adv_usd].copy()
        universe_sizes.append(len(grp))

        if len(grp) == 0:
            all_rets.append({
                "timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                "cost": 0.0, "n_long": 0, "n_short": 0,
                "turnover": 0, "risk_off": True, "n_universe": 0,
            })
            continue

        # ── Update EMA ──
        if ema_alpha is not None and ema_alpha < 1.0:
            for idx, r in grp.iterrows():
                sym = r["symbol"]
                raw_pred = r["pred"]
                smoothed = (ema_alpha * raw_pred
                            + (1 - ema_alpha) * prev_preds.get(sym, raw_pred))
                prev_preds[sym] = smoothed
                grp.at[idx, "pred"] = smoothed

        # ── State machine with timing constraints (R114b) ──
        if cutoff_on is not None:
            if risk_off:
                periods_in_off += 1
                can_exit = (trend_str < cutoff_off
                            and periods_in_off >= min_risk_off_periods)
                if can_exit:
                    risk_off = False
                    periods_in_on = 0
                else:
                    all_rets.append({
                        "timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                        "cost": 0.0, "n_long": 0, "n_short": 0,
                        "turnover": 0, "risk_off": True,
                        "n_universe": len(grp),
                    })
                    continue
            else:
                periods_in_on += 1
                can_enter = (trend_str > cutoff_on
                             and periods_in_on >= min_risk_on_periods)
                if can_enter:
                    risk_off = True
                    periods_in_off = 0
                    periods_in_on = 0
                    if prev_longs or prev_shorts:
                        n_prev = len(prev_longs) + len(prev_shorts)
                        avg_w = 1.0 / n_prev
                        close_cost = sum(_cost_for_sym(s) * avg_w
                                         for s in prev_longs | prev_shorts)
                        all_rets.append({
                            "timestamp": ts, "gross_ret": 0.0,
                            "net_ret": -close_cost, "cost": close_cost,
                            "n_long": 0, "n_short": 0,
                            "turnover": n_prev, "risk_off": True,
                            "n_universe": len(grp),
                        })
                    else:
                        all_rets.append({
                            "timestamp": ts, "gross_ret": 0.0,
                            "net_ret": 0.0, "cost": 0.0,
                            "n_long": 0, "n_short": 0,
                            "turnover": 0, "risk_off": True,
                            "n_universe": len(grp),
                        })
                    prev_longs, prev_shorts = set(), set()
                    continue

        # ── Portfolio construction ──
        n = len(grp)
        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl == 0 and ns == 0:
            all_rets.append({
                "timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                "cost": 0.0, "n_long": 0, "n_short": 0,
                "turnover": 0, "risk_off": False, "n_universe": n,
            })
            continue

        exposure = 1.0
        if (cutoff_on is not None and dyn_threshold is not None
                and trend_str > dyn_threshold):
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) /
                           (cutoff_on - dyn_threshold + 1e-10) * 0.5)

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
            remaining = grp[~grp["symbol"].isin(new_longs | new_shorts)]
            for _, r in remaining.sort_values("pred_rank").head(
                    nl - len(new_longs)).iterrows():
                new_longs.add(r["symbol"])
            for _, r in remaining.sort_values("pred_rank", ascending=False).head(
                    ns - len(new_shorts)).iterrows():
                new_shorts.add(r["symbol"])
        else:
            new_longs = (set(grp[grp["pred_rank"] <= nl]["symbol"].tolist())
                         if nl > 0 else set())
            new_shorts = (set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist())
                          if ns > 0 else set())

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
            turnover_cost = sum(_cost_for_sym(sym) * avg_weight
                                for sym in new_opened)
            turnover_cost += sum(_cost_for_sym(sym) * avg_weight
                                 for sym in closed)
            holding_cost = funding_per_12h * (rebal_hours / 12)
            total_cost = turnover_cost + holding_cost
        else:
            total_cost = 0.0

        net_ret = gross_ret - total_cost
        prev_longs, prev_shorts = new_longs, new_shorts

        all_rets.append({
            "timestamp": ts, "gross_ret": gross_ret, "net_ret": net_ret,
            "cost": total_cost, "n_long": nl_act, "n_short": ns_act,
            "turnover": len(new_opened) + len(closed), "risk_off": False,
            "n_universe": n,
        })

    port = pd.DataFrame(all_rets) if all_rets else pd.DataFrame()
    if universe_sizes:
        port.attrs["avg_universe"] = np.mean(universe_sizes)
        port.attrs["min_universe"] = min(universe_sizes)
        port.attrs["max_universe"] = max(universe_sizes)
    return port


# ─── Main ────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("=" * 70)
    log("R115b — Frozen Split-Universe (train SYM_35, predict ALL)")
    log("=" * 70)

    os.makedirs("results", exist_ok=True)

    # ── Load data (frozen normalization) ──
    log("\nPhase 1: Load expanded data with frozen market-level features")
    df, regime_df = load_data_expanded_frozen()
    n_total = df["symbol"].nunique()
    n_anchor = df[df["symbol"].isin(SYM_35_SET)]["symbol"].nunique()
    n_expanded = n_total - n_anchor
    log(f"  Universe: {n_total} total ({n_anchor} anchor + {n_expanded} expanded)")

    base_feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in base_feats if f in MARKET_LEVEL_FEATURES]

    # ── Phase 2: Train (SYM_35), predict (ALL) ──
    log("\nPhase 2: Training split-universe ensemble...")
    log("  Train/val: SYM_35 only | Test: ALL symbols")
    t1 = time.time()
    preds = train_ensemble_split(df, base_feats, CONTINUOUS_WINDOWS,
                                 seeds=SEEDS, cs_rank_exclude=no_rank,
                                 anchor_symbols=SYM_35_SET)
    train_time = time.time() - t1
    log(f"  Trained in {train_time:.0f}s")

    if preds is None or len(preds) == 0:
        log("  ERROR: No predictions generated!")
        return

    n_pred_syms = preds["symbol"].nunique()
    n_pred_anchor = preds[preds["symbol"].isin(SYM_35_SET)]["symbol"].nunique()
    log(f"  Predictions: {len(preds):,} rows, {n_pred_syms} symbols "
        f"({n_pred_anchor} anchor + {n_pred_syms - n_pred_anchor} expanded)")

    # ── Merge ADV into preds ──
    adv_lookup = (df[["timestamp", "symbol", "adv_7d"]]
                  .drop_duplicates(subset=["timestamp", "symbol"]))
    preds = preds.merge(adv_lookup, on=["timestamp", "symbol"], how="left")

    # ── Phase 3: R114b baseline (SYM_35, same training) ──
    log("\n" + "=" * 70)
    log("R114b baseline (SYM_35 only, 4L/2S, moff=2)")
    log("=" * 70)

    preds_35 = preds[preds["symbol"].isin(SYM_35_SET)].copy()
    cfg = dict(PROD_CFG)
    port_base = simulate_v2b(preds_35, regime_df, 4, 2, cfg,
                             cutoff_on=0.9, cutoff_off=0.8,
                             min_risk_off_periods=2, min_risk_on_periods=0)
    m_base = analyze_config(port_base, "R114b_SYM35_4L2S")
    print_result(m_base)

    # ── Phase 4: Grid search ──
    log("\n" + "=" * 70)
    log("R115b Grid: split-universe with volume filter")
    log("=" * 70)

    MIN_ADV_GRID = [10e6, 20e6, 50e6]
    NL_NS_GRID = [(4, 2), (6, 3)]

    results = [m_base]

    for min_adv in MIN_ADV_GRID:
        for nl, ns in NL_NS_GRID:
            label = f"split_adv{int(min_adv/1e6)}M_{nl}L{ns}S"
            log(f"\n  {label}...")

            port = simulate_v2b_volfilter(
                preds, regime_df, nl, ns, cfg,
                cutoff_on=0.9, cutoff_off=0.8,
                min_risk_off_periods=2, min_risk_on_periods=0,
                min_adv_usd=min_adv)

            m = analyze_config(port, label)
            m["min_adv_usd"] = int(min_adv)
            m["n_long_cfg"] = nl
            m["n_short_cfg"] = ns
            if hasattr(port, 'attrs'):
                m["avg_universe"] = round(port.attrs.get("avg_universe", 0), 1)
                m["min_universe"] = int(port.attrs.get("min_universe", 0))
                m["max_universe"] = int(port.attrs.get("max_universe", 0))
            else:
                m["avg_universe"] = 0
                m["min_universe"] = 0
                m["max_universe"] = 0
            print_result(m)
            log(f"    Universe: avg={m['avg_universe']:.0f}, "
                f"min={m['min_universe']}, max={m['max_universe']}")

            # Show what % of selected positions are outside SYM_35
            # (diagnostic: are we actually picking expanded coins?)
            results.append(m)

    # ── Also test: baseline model quality on SYM_35 subset of expanded preds ──
    log("\n" + "=" * 70)
    log("Diagnostic: model quality check (train 35, predict 35 vs predict all)")
    log("=" * 70)

    # Same model, predict SYM_35 only
    port_35only = simulate_v2b(preds_35, regime_df, 4, 2, cfg,
                               cutoff_on=0.9, cutoff_off=0.8,
                               min_risk_off_periods=2, min_risk_on_periods=0)
    m_35 = analyze_config(port_35only, "split_predict35_4L2S")
    print_result(m_35)
    log(f"  → If this matches R114b baseline closely, model didn't degrade from split-universe")

    # ── Results table ──
    log("\n" + "=" * 70)
    log("R115b RESULTS")
    log("=" * 70)

    hdr = (f"  {'Config':<30} {'NetSh':>7} {'GrSh':>7} {'Ret%':>7} "
           f"{'DD%':>7} {'Calmar':>7} {'%flat':>6} {'Cost%':>6} "
           f"{'AvgN':>5}")
    sep = (f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} "
           f"{'-'*6} {'-'*6} {'-'*5}")
    log(hdr)
    log(sep)

    for m in results:
        avg_n = m.get("avg_universe", "n/a")
        avg_n_str = f"{avg_n:>5.0f}" if isinstance(avg_n, (int, float)) else f"{avg_n:>5}"
        log(f"  {m['label']:<30} {m['net_sharpe']:>7.3f} "
            f"{m['gross_sharpe']:>7.3f} {m['total_ret_pct']:>7.1f} "
            f"{m['max_dd_pct']:>7.1f} {m['calmar']:>7.2f} "
            f"{m['pct_flat']:>5.1f}% {m['total_cost_pct']:>6.2f} {avg_n_str}")

    # ── Best config ──
    expanded = [m for m in results[1:] if m["net_sharpe"] > 0]
    if expanded:
        best = max(expanded, key=lambda x: x["calmar"])
        log(f"\n  Best expanded: {best['label']}")
        log(f"  vs R114b baseline:")
        for metric in ['net_sharpe', 'calmar', 'max_dd_pct', 'total_ret_pct']:
            v0 = m_base[metric]
            v1 = best[metric]
            log(f"    {metric}: {v0:.3f} → {v1:.3f}  Δ={v1-v0:+.3f}")

        if best["net_sharpe"] >= m_base["net_sharpe"]:
            log(f"\n  >>> RESULT: WIN — split-universe improves selection")
        elif best["net_sharpe"] >= m_base["net_sharpe"] - 0.1:
            log(f"\n  >>> RESULT: MARGINAL — similar performance, larger universe")
        else:
            log(f"\n  >>> RESULT: FAIL — expanding universe doesn't help")
    else:
        log(f"\n  >>> RESULT: FAIL — no valid expanded configs")

    # ── Save ──
    df_res = pd.DataFrame(results)
    df_res.to_csv("results/r115b_grid.csv", index=False)
    best_to_save = best if expanded else m_base
    with open("results/r115b_best.json", "w") as f:
        json.dump(best_to_save, f, indent=2, default=str)

    log(f"\nSaved: results/r115b_grid.csv, r115b_best.json")
    log(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
