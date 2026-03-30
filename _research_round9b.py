#!/usr/bin/env python3
"""
Research Round 9B — Targeted follow-up combos from R9.

Best individual findings from R9:
  - LightGBM: higher IC (0.060-0.072 vs Ridge 0.013-0.027), Sharpe +0.53, 11/13 WM
  - EMA=3: worst month -4.2% (vs -6.4%), at cost of -$222 equity
  - shrink=0.05: minimal equity cost (-$6), reduces worst month
  - shrink=0.15: best Sharpe (3.62), reduces worst month

Goal: find combos that beat R7 on BOTH Sharpe AND equity.
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings("ignore")

from _research_round7 import (
    SYM_35, WINDOWS, FEATURES as FEATURES_14, cs_rank,
    compute_regime, simulate, eval_config, show,
    train_and_predict_multi,
)
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal

LEVERAGE = 5
CAPITAL  = 100

CFG_BASE = {
    "n_long": 6, "n_short": 3,
    "trend_cutoff": 0.8, "dyn_threshold": 0.5,
    "eq_mom_boost": True, "kelly_sizing": True,
    "strategy_momentum": True, "strat_mom_lookback": 48,
    "regime_asym": True, "vol_scaling": True,
    "signal_ema": 2, "rebal_hours": 12,
}


def run(preds, regime_df, cfg, label):
    sub = simulate(preds, regime_df, 12, cfg)
    r = eval_config(sub, 12, label, LEVERAGE, CAPITAL)
    show(r)
    return r


def header(title):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")


def train_lgb(df, feats):
    import lightgbm as lgb
    feat_r = [f"{f}_r" for f in feats if f in df.columns]
    feats  = [f for f in feats if f in df.columns]
    fwd_col = "fwd_ret_12h"
    all_preds = []

    for w in WINDOWS:
        train = df[df["timestamp"] <  pd.Timestamp(w["train_end"], tz="UTC")].copy()
        val   = df[(df["timestamp"] >= pd.Timestamp(w["val_start"],  tz="UTC")) &
                   (df["timestamp"] <  pd.Timestamp(w["val_end"],    tz="UTC"))].copy()
        test  = df[(df["timestamp"] >= pd.Timestamp(w["test_start"], tz="UTC")) &
                   (df["timestamp"] <= pd.Timestamp(w["test_end"],   tz="UTC"))].copy()
        if len(train) < 5000 or len(test) < 200:
            continue
        for d in [train, val, test]:
            for feat in feats:
                d[f"{feat}_r"] = cs_rank(d, feat)
            d["target_rank"] = d.groupby("timestamp")[fwd_col].rank(pct=True) - 0.5
        train_c = train[feat_r + ["target_rank"]].dropna()
        val_c   = val[feat_r + ["target_rank"]].dropna()
        test_c  = test[feat_r + ["target_rank", "timestamp", "symbol"]].dropna()
        dtrain = lgb.Dataset(train_c[feat_r], label=train_c["target_rank"])
        dval   = lgb.Dataset(val_c[feat_r],   label=val_c["target_rank"])
        params = {"objective": "regression", "metric": "mse",
                  "learning_rate": 0.05, "num_leaves": 31,
                  "min_child_samples": 100, "subsample": 0.8,
                  "colsample_bytree": 0.8, "verbose": -1, "n_jobs": -1}
        model = lgb.train(params, dtrain, num_boost_round=300,
                          valid_sets=[dval],
                          callbacks=[lgb.early_stopping(30, verbose=False),
                                     lgb.log_evaluation(-1)])
        test_c = test_c.copy()
        test_c["pred"] = model.predict(test_c[feat_r])
        ic = stats.spearmanr(test_c["pred"], test_c["target_rank"])[0]
        print(f"    {w['name']}: trees={model.best_iteration}  IC={ic:.4f}")
        fwd_data = test[["timestamp", "symbol", fwd_col]].rename(
            columns={fwd_col: "fwd_ret"}).dropna()
        merged = test_c[["timestamp", "symbol", "pred"]].merge(
            fwd_data, on=["timestamp", "symbol"], how="inner")
        all_preds.append(merged)
    return pd.concat(all_preds, ignore_index=True) if all_preds else None


def main():
    print("=" * 70)
    print("  RESEARCH ROUND 9B: Targeted Follow-up Combos")
    print("=" * 70)

    print("\n📊 Loading data...")
    ohlcv  = load_ohlcv()
    ohlcv  = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df     = build_features_minimal(ohlcv, derivs)
    feats  = [f for f in FEATURES_14 if f in df.columns]
    regime_df = compute_regime(df)
    print(f"   {df.shape}, {df['symbol'].nunique()} syms")

    # ── Baselines ──────────────────────────────────────────────
    print("\n🏋️  Training Ridge (baseline)...")
    p12_ridge = train_and_predict_multi(df, feats, horizons=[12])[12]
    r_base = run(p12_ridge, regime_df, CFG_BASE, "BASELINE Ridge EMA=2 (R7 PROD)")

    print("\n🏋️  Training LightGBM...")
    p12_lgb = train_lgb(df, feats)
    r_lgb = run(p12_lgb, regime_df, CFG_BASE, "LightGBM EMA=2 (R9B baseline)")

    # ── LightGBM combos ────────────────────────────────────────
    header("LightGBM + Signal Tweaks")
    lgb_combos = [
        ("LGB EMA=None", {**CFG_BASE, "signal_ema": None}),
        ("LGB EMA=3",    {**CFG_BASE, "signal_ema": 3}),
        ("LGB EMA=4",    {**CFG_BASE, "signal_ema": 4}),
        ("LGB shrink=0.05 EMA=2",  {**CFG_BASE, "pred_shrinkage": 0.05}),
        ("LGB shrink=0.1  EMA=2",  {**CFG_BASE, "pred_shrinkage": 0.1}),
        ("LGB shrink=0.15 EMA=2",  {**CFG_BASE, "pred_shrinkage": 0.15}),
        ("LGB shrink=0.05 EMA=3",  {**CFG_BASE, "pred_shrinkage": 0.05, "signal_ema": 3}),
        ("LGB shrink=0.1  EMA=3",  {**CFG_BASE, "pred_shrinkage": 0.1,  "signal_ema": 3}),
        ("LGB shrink=0.15 EMA=3",  {**CFG_BASE, "pred_shrinkage": 0.15, "signal_ema": 3}),
        ("LGB conviction EMA=2",   {**CFG_BASE, "conviction_weight": True}),
        ("LGB conviction EMA=3",   {**CFG_BASE, "conviction_weight": True, "signal_ema": 3}),
    ]
    lgb_results = []
    for label, cfg in lgb_combos:
        r = run(p12_lgb, regime_df, cfg, label)
        lgb_results.append((label, r))

    # ── Ridge combos ─────────────────────────────────────────
    header("Ridge + Best Signal Tweaks")
    ridge_combos = [
        ("Ridge shrink=0.05 EMA=2", {**CFG_BASE, "pred_shrinkage": 0.05}),
        ("Ridge shrink=0.05 EMA=3", {**CFG_BASE, "pred_shrinkage": 0.05, "signal_ema": 3}),
        ("Ridge shrink=0.1  EMA=3", {**CFG_BASE, "pred_shrinkage": 0.1,  "signal_ema": 3}),
        ("Ridge shrink=0.15 EMA=3", {**CFG_BASE, "pred_shrinkage": 0.15, "signal_ema": 3}),
        ("Ridge EMA=3 conv",        {**CFG_BASE, "signal_ema": 3, "conviction_weight": True}),
    ]
    ridge_results = []
    for label, cfg in ridge_combos:
        r = run(p12_ridge, regime_df, cfg, label)
        ridge_results.append((label, r))

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY vs R7 BASELINE")
    print("=" * 70)
    print(f"\n  {'Config':<42}  Eq      ΔEq    Sh    ΔSh    Wr       WM")
    print("  " + "─" * 78)

    all_r = [("BASELINE", r_base), ("LGB baseline", r_lgb)] + lgb_results + ridge_results
    all_r.sort(key=lambda x: x[1]["equity"] if x[1] else 0, reverse=True)

    for label, r in all_r:
        if r is None:
            continue
        eq_d = r["equity"]  - r_base["equity"]
        sh_d = r["sharpe"]  - r_base["sharpe"]
        wr   = r["worst_m"] * 100
        mark = "✅" if eq_d >= 0 and sh_d >= 0 else (
               "⚠️ " if eq_d >= 0 or sh_d >= 0 else "❌")
        print(f"  {mark} {label:<42}  "
              f"${r['equity']:>5.0f}  {eq_d:>+5.0f}  {r['sharpe']:.2f}  "
              f"{sh_d:>+.2f}  {wr:>+.1f}%  {r['win_months']}/{r['total_months']}")

    # ── Best combos spotlight ─────────────────────────────────
    print("\n  Best by Sharpe (top 5):")
    top_sh = sorted(all_r, key=lambda x: x[1]["sharpe"] if x[1] else 0, reverse=True)[:5]
    for label, r in top_sh:
        if r:
            print(f"    Sh={r['sharpe']:.2f}  Eq=${r['equity']:>5.0f}  Wr={r['worst_m']*100:>+.1f}%  [{label}]")

    print("\n  Best equity with Sharpe >= R7 (3.59):")
    good_sh = [(lb, r) for lb, r in all_r if r and r["sharpe"] >= 3.59]
    good_sh.sort(key=lambda x: x[1]["equity"], reverse=True)
    for label, r in good_sh[:5]:
        print(f"    Eq=${r['equity']:>5.0f}  Sh={r['sharpe']:.2f}  Wr={r['worst_m']*100:>+.1f}%  [{label}]")

    print("\n  Safest (worst month >= -5.0% AND Sharpe >= 3.0):")
    safe = [(lb, r) for lb, r in all_r if r and r["worst_m"] >= -0.05 and r["sharpe"] >= 3.0]
    safe.sort(key=lambda x: x[1]["equity"], reverse=True)
    for label, r in safe[:5]:
        print(f"    Eq=${r['equity']:>5.0f}  Sh={r['sharpe']:.2f}  Wr={r['worst_m']*100:>+.1f}%  [{label}]")


if __name__ == "__main__":
    main()
