#!/usr/bin/env python3
"""
R122 — Directional BTC Model During Risk-Off Periods
======================================================

Currently 36.6% of time the bot sits in cash (risk-off when trend_strength > 0.9).
This experiment tests whether a simple directional BTC/ETH strategy can capture
returns during those flat periods.

Plan:
  Step 1: Extract risk-off periods, compute BTC forward returns during them
  Step 2: Naive baseline — long BTC when trend UP, short when DOWN
  Step 3: LGB model on BTC-specific features during risk-off
  Step 4: Walk-forward validation on same windows
  Step 5: Combine with main model, compare total Sharpe/DD/Calmar

Acceptance:
  - Combined Sharpe > 2.83 (S6 prod_blended baseline)
  - Risk-off model Sharpe > 0 standalone
  - Combined DD < -15%
"""

import time, json, os, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
warnings.filterwarnings("ignore")

from _research_r22_models import SEEDS, log
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r68_continuous_wf import (
    CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, PROD_CFG,
    load_data, train_ensemble, sharpe,
)
from _research_r113_trend_cutoff_reopt import analyze_config, print_result
from _research_r121_realistic_costs import (
    simulate_r121, cost_prod_blended, R114B_CFG,
    per_window_metrics, COST_MODELS,
)


# ─── Config ──────────────────────────────────────────────────

CUTOFF_ON = 0.9
CUTOFF_OFF = 0.8
MIN_OFF_PERIODS = 2
REBAL_HOURS = 12

# BTC directional cost: single asset, taker on entry + exit
# OKX BTC taker = 5bp fee + 1bp spread = 6bp per side = 12bp round-trip
BTC_COST_PER_TRADE = 0.0006   # 6 bps per side
BTC_FUNDING_PER_12H = 0.00012  # 1.2bp per 12h hold

# Position sizing for risk-off directional trades (fraction of normal)
RISKOFF_EXPOSURE = 0.50  # 50% of normal capital initially


# ─── Step 1: Extract risk-off periods ────────────────────────

def extract_riskoff_periods(regime_df, rebal_hours=12):
    """Identify all risk-off timestamps using the same state machine as R121.

    Returns:
        riskoff_ts: set of timestamps where portfolio is flat (risk-off)
        riskoff_entries: list of (entry_ts, exit_ts) tuples for each spell
    """
    ts_sorted = sorted(regime_df.index)
    rebal_ts = ts_sorted[::rebal_hours]

    risk_off = False
    periods_in_off = 0
    periods_in_on = 999
    riskoff_ts = set()
    spells = []
    current_spell_start = None

    for ts in rebal_ts:
        if ts not in regime_df.index:
            continue
        trend_str = regime_df.loc[ts].get("trend_strength", 0)

        if risk_off:
            periods_in_off += 1
            can_exit = (trend_str < CUTOFF_OFF and periods_in_off >= MIN_OFF_PERIODS)
            if can_exit:
                risk_off = False
                periods_in_on = 0
                if current_spell_start is not None:
                    spells.append((current_spell_start, ts))
                    current_spell_start = None
            else:
                riskoff_ts.add(ts)
        else:
            periods_in_on += 1
            can_enter = (trend_str > CUTOFF_ON and periods_in_on >= 0)
            if can_enter:
                risk_off = True
                periods_in_off = 0
                periods_in_on = 0
                riskoff_ts.add(ts)
                current_spell_start = ts

    if current_spell_start is not None:
        spells.append((current_spell_start, rebal_ts[-1]))

    return riskoff_ts, spells


def build_btc_features(df, regime_df):
    """Build BTC-specific features for directional model.

    Uses only BTC/USDT rows + regime data. Returns a dataframe indexed by timestamp.
    """
    btc = df[df["symbol"] == "BTC/USDT"].copy()
    if btc.empty:
        return pd.DataFrame()

    btc = btc.sort_values("timestamp").set_index("timestamp")

    # Price-based momentum features
    close = btc["close"] if "close" in btc.columns else None
    ret_cols = [c for c in btc.columns if c.startswith("ret_") or c == "fwd_ret_12h"]

    feat = pd.DataFrame(index=btc.index)

    # Forward return (target)
    if "fwd_ret_12h" in btc.columns:
        feat["btc_fwd_ret"] = btc["fwd_ret_12h"]

    # Existing features that are useful for BTC directional
    useful_feats = [
        "ret_12h", "ret_24h", "ret_48h", "ret_168h",
        "atr_14", "gk_vol_24h",
        "vol_ratio_24h", "dist_from_high_24h",
        "adx", "mfi_14",
        "ret_skew_24h", "ret_kurt_24h",
        "vwap_dev_24h", "obv_ma_ratio_24",
    ]
    for f in useful_feats:
        if f in btc.columns:
            feat[f] = btc[f]

    # Add regime features
    regime_feats = ["trend_strength", "trend_direction", "btc_ret_24h",
                    "vol_regime", "btc_vol_7d"]
    for f in regime_feats:
        if f in regime_df.columns:
            feat[f] = regime_df[f]

    # Derivative features (BTC-specific)
    deriv_feats = ["btc_dvol", "iv_rv_spread", "dvol_zscore",
                   "premium_zscore_12h", "oi_velocity", "taker_imb_z"]
    for f in deriv_feats:
        if f in btc.columns:
            feat[f] = btc[f]

    # Fear & Greed
    for f in ["fng_value", "fng_zscore"]:
        if f in btc.columns:
            feat[f] = btc[f]

    # Macro
    for f in ["vix_close", "vix_zscore", "dxy_ret_7d"]:
        if f in btc.columns:
            feat[f] = btc[f]

    feat = feat.dropna(subset=["btc_fwd_ret"])
    return feat


# ─── Step 2: Naive baseline ─────────────────────────────────

def naive_baseline(btc_feat, riskoff_ts, regime_df):
    """Simple rule: long BTC when trend UP, short when DOWN during risk-off.

    Returns: pd.DataFrame with columns [timestamp, gross_ret, net_ret, cost, signal]
    """
    results = []
    prev_signal = 0  # 0 = flat

    for ts in sorted(riskoff_ts):
        if ts not in btc_feat.index:
            continue
        row = btc_feat.loc[ts]
        fwd_ret = row["btc_fwd_ret"]

        # Trend direction from regime_df
        if ts in regime_df.index:
            trend_dir = regime_df.loc[ts].get("trend_direction", 0)
        else:
            trend_dir = 0

        # Signal: +1 long, -1 short based on trend direction
        signal = 1.0 if trend_dir > 0 else -1.0

        # Scale by exposure
        gross_ret = signal * fwd_ret * RISKOFF_EXPOSURE

        # Cost: only pay when signal changes
        turnover_cost = 0.0
        if signal != prev_signal:
            # Close old + open new
            if prev_signal != 0:
                turnover_cost += BTC_COST_PER_TRADE  # close old
            turnover_cost += BTC_COST_PER_TRADE  # open new
        holding_cost = BTC_FUNDING_PER_12H
        total_cost = (turnover_cost + holding_cost) * RISKOFF_EXPOSURE

        net_ret = gross_ret - total_cost
        prev_signal = signal

        results.append({
            "timestamp": ts,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "cost": total_cost,
            "signal": signal,
            "btc_fwd_ret": fwd_ret,
        })

    return pd.DataFrame(results)


# ─── Step 3: LGB directional model ──────────────────────────

def train_btc_directional(btc_feat, riskoff_ts, windows, regime_df):
    """Train LGB on BTC features, predict direction during risk-off.

    Target: sign(btc_fwd_ret_12h) → binary classification.
    Only trains on risk-off period data.

    Returns: pd.DataFrame with predictions for each risk-off timestamp.
    """
    # Feature columns (exclude target)
    feat_cols = [c for c in btc_feat.columns if c != "btc_fwd_ret"]
    feat_cols = [c for c in feat_cols if btc_feat[c].notna().mean() > 0.5]

    if not feat_cols:
        log("  ERROR: No valid feature columns")
        return pd.DataFrame()

    log(f"  BTC directional features ({len(feat_cols)}): {feat_cols}")

    tz = btc_feat.index[0].tz if hasattr(btc_feat.index[0], 'tz') else None

    LGB_PARAMS = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "verbose": -1,
    }

    all_preds = []

    for seed in SEEDS:
        params = {**LGB_PARAMS, "seed": seed}

        for w in windows:
            te_end = pd.Timestamp(w["test_end"], tz=tz)
            te_start = pd.Timestamp(w["test_start"], tz=tz)
            tr_end = pd.Timestamp(w["train_end"], tz=tz)
            va_start = pd.Timestamp(w["val_start"], tz=tz)
            va_end = pd.Timestamp(w["val_end"], tz=tz)

            # Split — use ALL data for training (not just risk-off)
            # Risk-off periods are rare, so we train on all data
            # but only predict/evaluate on risk-off timestamps
            train_ = btc_feat[btc_feat.index < tr_end].copy()
            val_ = btc_feat[(btc_feat.index >= va_start) & (btc_feat.index < va_end)].copy()
            test_ = btc_feat[(btc_feat.index >= te_start) & (btc_feat.index <= te_end)].copy()

            if len(train_) < 500 or len(test_) < 50:
                continue

            # Target: binary (positive BTC return)
            for d in [train_, val_, test_]:
                d["target"] = (d["btc_fwd_ret"] > 0).astype(int)

            # Fill NaN
            for col in feat_cols:
                for d in [train_, val_, test_]:
                    if d[col].isna().any():
                        d[col] = d[col].fillna(0)

            # Remove inf
            for d in [train_, val_, test_]:
                d[feat_cols] = d[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

            tr_clean = train_[feat_cols + ["target"]].dropna()
            va_clean = val_[feat_cols + ["target"]].dropna()
            te_clean = test_[feat_cols + ["target", "btc_fwd_ret"]].dropna()

            if len(te_clean) < 10:
                continue

            # Train
            dt = lgb.Dataset(tr_clean[feat_cols], label=tr_clean["target"])
            dv = lgb.Dataset(va_clean[feat_cols], label=va_clean["target"])
            m = lgb.train(params, dt, num_boost_round=500,
                          valid_sets=[dv],
                          callbacks=[lgb.early_stopping(30, verbose=False),
                                     lgb.log_evaluation(-1)])

            # Predict
            p = m.predict(te_clean[feat_cols])
            rec = pd.DataFrame({
                "timestamp": te_clean.index,
                "pred_prob": p,
                "btc_fwd_ret": te_clean["btc_fwd_ret"].values,
                "window": w["name"],
                "seed": seed,
            })
            all_preds.append(rec)

            if seed == SEEDS[0]:
                log(f"    {w['name']}/s{seed}: train={len(tr_clean):,} "
                    f"test={len(te_clean):,} (riskoff={sum(1 for t in te_clean.index if t in riskoff_ts)})")

    if not all_preds:
        return pd.DataFrame()

    preds = pd.concat(all_preds)
    # Average across seeds
    avg = preds.groupby(["timestamp"]).agg(
        pred_prob=("pred_prob", "mean"),
        btc_fwd_ret=("btc_fwd_ret", "first"),
        window=("window", "first"),
    ).reset_index()

    return avg


def simulate_btc_directional(preds, riskoff_ts):
    """Simulate BTC directional strategy during risk-off periods.

    Signal: pred_prob > 0.5 → long, else short.
    Only trades during risk-off timestamps.
    """
    results = []
    prev_signal = 0

    for _, row in preds.iterrows():
        ts = row["timestamp"]
        if ts not in riskoff_ts:
            continue

        fwd_ret = row["btc_fwd_ret"]
        prob = row["pred_prob"]

        # Signal: continuous sizing based on confidence
        signal = 1.0 if prob > 0.5 else -1.0
        # Scale by confidence (0.5 = neutral → 0 exposure, 1.0 = full long)
        confidence = abs(prob - 0.5) * 2  # 0 to 1
        position = signal * confidence * RISKOFF_EXPOSURE

        gross_ret = position * fwd_ret

        # Cost
        turnover_cost = 0.0
        if signal != prev_signal:
            if prev_signal != 0:
                turnover_cost += BTC_COST_PER_TRADE
            turnover_cost += BTC_COST_PER_TRADE
        holding_cost = BTC_FUNDING_PER_12H
        total_cost = (turnover_cost + holding_cost) * abs(position)

        net_ret = gross_ret - total_cost
        prev_signal = signal

        results.append({
            "timestamp": ts,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "cost": total_cost,
            "signal": signal,
            "confidence": confidence,
            "position": position,
            "btc_fwd_ret": fwd_ret,
        })

    return pd.DataFrame(results)


# ─── Step 5: Combine main + risk-off ────────────────────────

def combine_strategies(main_port, riskoff_port):
    """Combine main L/S model with risk-off BTC directional.

    main_port: portfolio returns from R121 simulate (includes risk_off=True flat periods)
    riskoff_port: BTC directional returns during risk-off periods
    """
    combined = main_port.copy()

    # For risk-off periods, replace 0% return with directional model return
    riskoff_idx = {}
    for _, row in riskoff_port.iterrows():
        riskoff_idx[row["timestamp"]] = row

    replaced = 0
    for i, row in combined.iterrows():
        if row.get("risk_off", False) and row["timestamp"] in riskoff_idx:
            ro_row = riskoff_idx[row["timestamp"]]
            combined.at[i, "gross_ret"] = ro_row["gross_ret"]
            combined.at[i, "net_ret"] = ro_row["net_ret"]
            combined.at[i, "cost"] = ro_row["cost"]
            replaced += 1

    return combined, replaced


# ─── Main ────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("=" * 70)
    log("R122 — Directional BTC Model During Risk-Off Periods")
    log("=" * 70)

    os.makedirs("results", exist_ok=True)

    # ── Load data ──
    log("\nStep 0: Loading data...")
    df, regime_df = load_data()
    base_feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in base_feats if f in MARKET_LEVEL_FEATURES]

    # ── Build BTC feature set ──
    log("\nStep 1: Building BTC features & extracting risk-off periods...")
    btc_feat = build_btc_features(df, regime_df)
    log(f"  BTC feature rows: {len(btc_feat):,}")
    log(f"  BTC feature cols: {btc_feat.shape[1]} "
        f"({', '.join(c for c in btc_feat.columns if c != 'btc_fwd_ret')})")

    riskoff_ts, spells = extract_riskoff_periods(regime_df)
    log(f"  Risk-off timestamps: {len(riskoff_ts)}")
    log(f"  Risk-off spells: {len(spells)}")
    if spells:
        durations = [(s[1] - s[0]).total_seconds() / 3600 for s in spells]
        log(f"  Spell durations (hours): min={min(durations):.0f}, "
            f"max={max(durations):.0f}, mean={np.mean(durations):.0f}")

    # Stats on BTC returns during risk-off
    riskoff_rets = btc_feat.loc[btc_feat.index.isin(riskoff_ts), "btc_fwd_ret"]
    if len(riskoff_rets) > 0:
        log(f"\n  BTC returns during risk-off ({len(riskoff_rets)} periods):")
        log(f"    Mean:  {riskoff_rets.mean()*100:.3f}%")
        log(f"    Std:   {riskoff_rets.std()*100:.3f}%")
        log(f"    Skew:  {riskoff_rets.skew():.2f}")
        pct_positive = (riskoff_rets > 0).mean() * 100
        log(f"    Positive: {pct_positive:.1f}%")
        log(f"    If always long:  Sharpe≈{riskoff_rets.mean() / (riskoff_rets.std() + 1e-10) * np.sqrt(730):.2f}")

        # By trend direction
        trend_dirs = regime_df.loc[regime_df.index.isin(riskoff_ts), "trend_direction"]
        up_mask = trend_dirs > 0
        down_mask = trend_dirs <= 0
        up_ts = set(trend_dirs[up_mask].index)
        down_ts = set(trend_dirs[down_mask].index)
        up_rets = btc_feat.loc[btc_feat.index.isin(up_ts), "btc_fwd_ret"]
        down_rets = btc_feat.loc[btc_feat.index.isin(down_ts), "btc_fwd_ret"]
        log(f"\n    Trend UP periods ({len(up_rets)}):   mean={up_rets.mean()*100:.3f}%")
        log(f"    Trend DOWN periods ({len(down_rets)}): mean={down_rets.mean()*100:.3f}%")

    # ══════════════════════════════════════════════════════════
    # Step 2: Naive baseline
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("Step 2: Naive Baseline (long when UP, short when DOWN)")
    log("=" * 70)

    naive_port = naive_baseline(btc_feat, riskoff_ts, regime_df)
    if not naive_port.empty:
        naive_sh = sharpe(naive_port["net_ret"])
        naive_gross_sh = sharpe(naive_port["gross_ret"])
        naive_eq = (1 + naive_port["net_ret"]).cumprod()
        naive_ret = (naive_eq.iloc[-1] / naive_eq.iloc[0] - 1) * 100
        naive_dd = ((naive_eq / naive_eq.cummax()) - 1).min() * 100
        log(f"  Naive baseline (risk-off only):")
        log(f"    Periods:     {len(naive_port)}")
        log(f"    Gross Sharpe: {naive_gross_sh:.3f}")
        log(f"    Net Sharpe:   {naive_sh:.3f}")
        log(f"    Return:       {naive_ret:.1f}%")
        log(f"    MaxDD:        {naive_dd:.1f}%")
        log(f"    Win rate:     {(naive_port['net_ret'] > 0).mean()*100:.1f}%")
        log(f"    Avg signal:   {naive_port['signal'].mean():.2f} "
            f"(1=always long, -1=always short)")

        # Break down by direction
        long_mask = naive_port["signal"] > 0
        short_mask = naive_port["signal"] < 0
        if long_mask.any():
            log(f"    Long periods ({long_mask.sum()}): "
                f"mean ret={naive_port.loc[long_mask, 'net_ret'].mean()*100:.3f}%")
        if short_mask.any():
            log(f"    Short periods ({short_mask.sum()}): "
                f"mean ret={naive_port.loc[short_mask, 'net_ret'].mean()*100:.3f}%")
    else:
        naive_sh = 0
        log("  ERROR: No naive baseline results")

    # ══════════════════════════════════════════════════════════
    # Step 3: Train main model + LGB directional
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("Step 3: Training main ensemble + BTC directional model")
    log("=" * 70)

    # Re-use main model training
    log("\n  Training main ensemble (same as R121)...")
    t1 = time.time()
    preds = train_ensemble(df, base_feats, CONTINUOUS_WINDOWS,
                           seeds=SEEDS, cs_rank_exclude=no_rank)
    log(f"  Main ensemble trained in {time.time()-t1:.0f}s")

    # Train BTC directional model
    log("\n  Training BTC directional LGB...")
    t2 = time.time()
    btc_preds = train_btc_directional(btc_feat, riskoff_ts, CONTINUOUS_WINDOWS, regime_df)
    log(f"  BTC model trained in {time.time()-t2:.0f}s")

    if btc_preds.empty:
        log("  ERROR: No BTC predictions generated")
        return

    # Model accuracy on risk-off periods
    riskoff_preds = btc_preds[btc_preds["timestamp"].isin(riskoff_ts)]
    if not riskoff_preds.empty:
        correct = ((riskoff_preds["pred_prob"] > 0.5) ==
                    (riskoff_preds["btc_fwd_ret"] > 0)).mean()
        log(f"  BTC model accuracy (risk-off): {correct*100:.1f}%")
        log(f"  Mean pred_prob: {riskoff_preds['pred_prob'].mean():.3f}")

    # ══════════════════════════════════════════════════════════
    # Step 4: Simulate and combine
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("Step 4: Simulate & Combine")
    log("=" * 70)

    cfg = dict(R114B_CFG)
    cost_fn, funding = COST_MODELS["prod_blended"]

    # S6 baseline (main model only, risk-off = flat)
    main_port = simulate_r121(preds, regime_df, 4, 2, cfg,
                              cutoff_on=0.9, cutoff_off=0.8,
                              min_risk_off_periods=2,
                              cost_fn=cost_fn,
                              funding_per_12h=funding,
                              exec_delay_penalty=0.0003)
    base_metrics = analyze_config(main_port, "S6_baseline")
    log(f"\n  S6 Baseline (main model, risk-off=flat):")
    print_result(base_metrics)

    # Naive combined
    naive_combined, naive_replaced = combine_strategies(main_port, naive_port)
    naive_combined_metrics = analyze_config(naive_combined, "Naive_combined")
    log(f"\n  Naive Combined (main + naive BTC directional, replaced {naive_replaced} periods):")
    print_result(naive_combined_metrics)

    # LGB directional during risk-off
    btc_riskoff_port = simulate_btc_directional(btc_preds, riskoff_ts)
    if not btc_riskoff_port.empty:
        btc_standalone_sh = sharpe(btc_riskoff_port["net_ret"])
        log(f"\n  LGB BTC standalone (risk-off only):")
        log(f"    Periods: {len(btc_riskoff_port)}")
        log(f"    Sharpe:  {btc_standalone_sh:.3f}")
        log(f"    Mean confidence: {btc_riskoff_port['confidence'].mean():.3f}")

        # LGB combined
        lgb_combined, lgb_replaced = combine_strategies(main_port, btc_riskoff_port)
        lgb_combined_metrics = analyze_config(lgb_combined, "LGB_combined")
        log(f"\n  LGB Combined (main + LGB BTC directional, replaced {lgb_replaced} periods):")
        print_result(lgb_combined_metrics)

        # Per-window
        pw_base = per_window_metrics(main_port, preds)
        pw_naive = per_window_metrics(naive_combined, preds)
        pw_lgb = per_window_metrics(lgb_combined, preds)
    else:
        lgb_combined_metrics = None
        pw_base = per_window_metrics(main_port, preds)
        pw_naive = per_window_metrics(naive_combined, preds)
        pw_lgb = {}

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("R122 — COMPARISON")
    log("=" * 70)

    hdr = (f"  {'Strategy':<30} {'NetSh':>7} {'GrSh':>7} {'Ret%':>7} "
           f"{'DD%':>7} {'Calmar':>7} {'W1sh':>6} {'W2sh':>6} {'W3sh':>6}")
    log(hdr)
    log(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*6}")

    for label, m, pw in [("S6_baseline", base_metrics, pw_base),
                          ("Naive_combined", naive_combined_metrics, pw_naive),
                          ("LGB_combined", lgb_combined_metrics, pw_lgb)]:
        if m is None:
            continue
        w1 = pw.get("W1", {}).get("sharpe", 0)
        w2 = pw.get("W2", {}).get("sharpe", 0)
        w3 = pw.get("W3", {}).get("sharpe", 0)
        log(f"  {m['label']:<30} {m['net_sharpe']:>7.3f} {m['gross_sharpe']:>7.3f} "
            f"{m['total_ret_pct']:>7.1f} {m['max_dd_pct']:>7.1f} {m['calmar']:>7.2f} "
            f"{w1:>6.3f} {w2:>6.3f} {w3:>6.3f}")

    # Deltas
    log(f"\n  {'Strategy':<30} {'ΔSharpe':>8} {'ΔRet%':>8} {'ΔDD%':>8}")
    log(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    for label, m in [("Naive_combined", naive_combined_metrics),
                      ("LGB_combined", lgb_combined_metrics)]:
        if m is None:
            continue
        log(f"  {m['label']:<30} "
            f"{m['net_sharpe']-base_metrics['net_sharpe']:>+8.3f} "
            f"{m['total_ret_pct']-base_metrics['total_ret_pct']:>+8.1f} "
            f"{m['max_dd_pct']-base_metrics['max_dd_pct']:>+8.1f}")

    # ── Verdict ──
    log("\n" + "=" * 70)
    log("VERDICT")
    log("=" * 70)

    best = lgb_combined_metrics or naive_combined_metrics
    if best:
        delta_sh = best["net_sharpe"] - base_metrics["net_sharpe"]
        dd_ok = best["max_dd_pct"] > -15.0
        improved = delta_sh > 0

        if improved and dd_ok:
            log(f"  ✅ PASS — Combined Sharpe {best['net_sharpe']:.3f} "
                f"(+{delta_sh:.3f} vs baseline {base_metrics['net_sharpe']:.3f})")
            log(f"    DD {best['max_dd_pct']:.1f}% (limit: -15.0%)")
        else:
            reasons = []
            if not improved:
                reasons.append(f"ΔSharpe={delta_sh:+.3f}")
            if not dd_ok:
                reasons.append(f"DD={best['max_dd_pct']:.1f}%>-15%")
            log(f"  ❌ FAIL — {', '.join(reasons)}")
            log(f"    Combined: Sharpe={best['net_sharpe']:.3f}, DD={best['max_dd_pct']:.1f}%")
            log(f"    Baseline: Sharpe={base_metrics['net_sharpe']:.3f}, DD={base_metrics['max_dd_pct']:.1f}%")

    # ── Save results ──
    save = {
        "baseline_sharpe": base_metrics["net_sharpe"],
        "naive_sharpe": naive_combined_metrics["net_sharpe"] if naive_combined_metrics else None,
        "lgb_sharpe": lgb_combined_metrics["net_sharpe"] if lgb_combined_metrics else None,
        "naive_riskoff_sharpe": round(naive_sh, 3),
        "riskoff_periods": len(riskoff_ts),
        "riskoff_spells": len(spells),
        "riskoff_pct": round(len(riskoff_ts) / max(1, len(main_port)) * 100, 1),
    }
    with open("results/r122_riskoff_btc.json", "w") as f:
        json.dump(save, f, indent=2)

    log(f"\nSaved: results/r122_riskoff_btc.json")
    log(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
