#!/usr/bin/env python3
"""
Research Round 9 — Four independent improvement directions, all vs R7 baseline.

Conditions: same WINDOWS, leverage=5, capital=$100, R7 winner cfg as baseline.

PHASE A: Multi-horizon blend (4h / 12h / 24h / combos)
PHASE B: LightGBM vs Ridge (drop-in model replacement)
PHASE C: Position count sweep (4L/2S … 10L/5S) + rebalancing freq (6h/8h/12h/24h)
PHASE D: Signal refinements (conviction weight, shrinkage sweep, EMA sweep)
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
    train_and_predict_multi, blend_predictions, load_data,
)
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal

LEVERAGE = 5
CAPITAL  = 100

# R7 winner config (PRODUCTION baseline)
CFG_BASE = {
    "n_long": 6, "n_short": 3,
    "trend_cutoff": 0.8, "dyn_threshold": 0.5,
    "eq_mom_boost": True, "kelly_sizing": True,
    "strategy_momentum": True, "strat_mom_lookback": 48,
    "regime_asym": True, "vol_scaling": True,
    "signal_ema": 2, "rebal_hours": 12,
}


# ──────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────
def run(preds, regime_df, cfg, label):
    sub = simulate(preds, regime_df, 12, cfg)
    r   = eval_config(sub, 12, label, LEVERAGE, CAPITAL)
    show(r)
    return r


def header(title):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")


def summary_row(label, r14, r):
    if r is None:
        return
    eq_d  = r["equity"]   - r14["equity"]
    sh_d  = r["sharpe"]   - r14["sharpe"]
    wr_d  = (r["worst_m"] - r14["worst_m"]) * 100
    mark  = "✅" if r["equity"] > r14["equity"] and r["sharpe"] > r14["sharpe"] else (
            "⚠️ " if r["equity"] > r14["equity"] or r["sharpe"] > r14["sharpe"] else "❌")
    print(f"  {mark} {label:<50}  "
          f"Eq=${r['equity']:>5.0f} (Δ{eq_d:>+5.0f})  "
          f"Sh={r['sharpe']:.2f} (Δ{sh_d:>+.2f})  "
          f"Wr={r['worst_m']*100:>+.1f}% (Δ{wr_d:>+.1f})")


# ──────────────────────────────────────────────────────────────────
#  LGB TRAINING (drop-in replacement for Ridge)
# ──────────────────────────────────────────────────────────────────
def train_lgb(df, feats):
    """Train LightGBM per walk-forward window, return preds identical format to Ridge."""
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

        params = {
            "objective": "regression", "metric": "mse",
            "learning_rate": 0.05, "num_leaves": 31,
            "min_child_samples": 100, "subsample": 0.8,
            "colsample_bytree": 0.8, "verbose": -1,
            "n_jobs": -1,
        }
        model = lgb.train(
            params, dtrain, num_boost_round=300,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )

        test_c = test_c.copy()
        test_c["pred"] = model.predict(test_c[feat_r])

        ic = stats.spearmanr(test_c["pred"], test_c["target_rank"])[0]
        print(f"    12h {w['name']}: n_trees={model.best_iteration}  "
              f"test_IC={ic:.4f}  ({len(test_c):,} obs)")

        fwd_data = test[["timestamp", "symbol", fwd_col]].rename(
            columns={fwd_col: "fwd_ret"}).dropna()
        merged = test_c[["timestamp", "symbol", "pred"]].merge(
            fwd_data, on=["timestamp", "symbol"], how="inner")
        all_preds.append(merged)

    if not all_preds:
        return None
    return pd.concat(all_preds, ignore_index=True)


# ──────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  RESEARCH ROUND 9: Four Improvement Directions")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────
    print("\n📊 Loading data...")
    ohlcv   = load_ohlcv()
    ohlcv   = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs  = load_derivatives()
    df      = build_features_minimal(ohlcv, derivs)
    print(f"   Shape: {df.shape}, symbols: {df['symbol'].nunique()}")

    feats = [f for f in FEATURES_14 if f in df.columns]
    regime_df = compute_regime(df)

    # ── Baseline ──────────────────────────────────────────────────
    print("\n🏋️  Training baseline (Ridge, 12h)...")
    preds_multi = train_and_predict_multi(df, feats, horizons=[4, 12, 24])
    p12 = preds_multi[12]

    r_base = run(p12, regime_df, CFG_BASE, "BASELINE Ridge 12h (R7 PROD)")

    print(f"\n  Baseline: Eq=${r_base['equity']:.0f}  Sh={r_base['sharpe']:.2f}  "
          f"Wr={r_base['worst_m']*100:.1f}%")

    results = []

    # ══════════════════════════════════════════════════════════════
    #  PHASE A: Multi-horizon blend
    # ══════════════════════════════════════════════════════════════
    header("PHASE A: Multi-Horizon Blend")

    # Single horizons
    p4  = preds_multi[4]
    p24 = preds_multi[24]

    blend_configs = [
        ("4h only",           preds_multi, {4: 1.0, 12: 0.0, 24: 0.0}),
        ("24h only",          preds_multi, {4: 0.0, 12: 0.0, 24: 1.0}),
        ("4h+12h equal",      preds_multi, {4: 0.5, 12: 0.5, 24: 0.0}),
        ("12h+24h equal",     preds_multi, {4: 0.0, 12: 0.5, 24: 0.5}),
        ("4h+12h+24h equal",  preds_multi, {4: 1/3, 12: 1/3, 24: 1/3}),
        ("4h+12h+24h 1:2:1",  preds_multi, {4: 0.25, 12: 0.50, 24: 0.25}),
        ("4h+12h+24h 1:3:1",  preds_multi, {4: 0.2,  12: 0.6,  24: 0.2}),
    ]

    for label, pd_dict, weights in blend_configs:
        # Build preds with only the horizons we need (non-zero weights)
        active = {h: pd_dict[h] for h in pd_dict if weights.get(h, 0) > 0}
        blended = blend_predictions(active, weights={h: weights[h] for h in active})
        r = run(blended, regime_df, CFG_BASE, f"A: {label}")
        results.append(("A", label, r))
        summary_row(label, r_base, r)

    # ══════════════════════════════════════════════════════════════
    #  PHASE B: LightGBM model
    # ══════════════════════════════════════════════════════════════
    header("PHASE B: LightGBM (non-linear, same 14 features)")
    print("  Training LightGBM...")
    preds_lgb = train_lgb(df, feats)

    if preds_lgb is not None:
        r = run(preds_lgb, regime_df, CFG_BASE, "B: LightGBM (n_leaves=31, lr=0.05)")
        results.append(("B", "LightGBM", r))
        summary_row("LightGBM", r_base, r)

        # Also try LGB + multi-horizon blend (train new LGB for 4h, 24h)
        # Too slow — skip for now, just test 12h LGB
    else:
        print("  ⚠️  LGB training failed")

    # ══════════════════════════════════════════════════════════════
    #  PHASE C: Position count + rebalancing frequency
    # ══════════════════════════════════════════════════════════════
    header("PHASE C: Position Count & Rebalancing Frequency")

    # C1: Position count (fix rebal=12h, vary L/S)
    print("\n  C1: Position count sweep (rebal_hours=12):")
    position_configs = [
        (4, 2), (5, 2), (5, 3), (6, 2), (7, 3), (8, 3), (8, 4), (9, 3), (10, 4), (10, 5)
    ]
    for nl, ns in position_configs:
        cfg = {**CFG_BASE, "n_long": nl, "n_short": ns}
        label = f"{nl}L/{ns}S"
        r = run(p12, regime_df, cfg, f"C1: {label}")
        results.append(("C1", label, r))
        summary_row(label, r_base, r)

    # C2: Rebalancing frequency (fix 6L/3S, vary rebal_hours)
    print("\n  C2: Rebalancing frequency sweep (6L/3S):")
    for rebal in [4, 6, 8, 24, 48]:
        cfg = {**CFG_BASE, "rebal_hours": rebal}
        label = f"rebal={rebal}h"
        r = run(p12, regime_df, cfg, f"C2: {label}")
        results.append(("C2", label, r))
        summary_row(label, r_base, r)

    # ══════════════════════════════════════════════════════════════
    #  PHASE D: Signal refinements
    # ══════════════════════════════════════════════════════════════
    header("PHASE D: Signal Refinements")

    # D1: EMA sweep (current=2)
    print("\n  D1: Signal EMA span (current=2):")
    for ema in [None, 1, 3, 4, 5, 8]:
        cfg = {**CFG_BASE, "signal_ema": ema}
        label = f"EMA={ema}"
        r = run(p12, regime_df, cfg, f"D1: {label}")
        results.append(("D1", label, r))
        summary_row(label, r_base, r)

    # D2: Conviction weighting (current=False)
    print("\n  D2: Conviction weighting:")
    for cv in [True, False]:
        cfg = {**CFG_BASE, "conviction_weight": cv}
        label = f"conviction={cv}"
        r = run(p12, regime_df, cfg, f"D2: {label}")
        results.append(("D2", label, r))
        summary_row(label, r_base, r)

    # D3: Prediction shrinkage (current=None)
    print("\n  D3: Prediction shrinkage (current=None):")
    for shrink in [None, 0.05, 0.1, 0.15, 0.2, 0.3]:
        cfg = {**CFG_BASE, "pred_shrinkage": shrink}
        label = f"shrink={shrink}"
        r = run(p12, regime_df, cfg, f"D3: {label}")
        results.append(("D3", label, r))
        summary_row(label, r_base, r)

    # D4: vol_target sweep (current not set → defaults internally)
    print("\n  D4: Vol target (for vol_scaling):")
    for vt in [0.01, 0.015, 0.02, 0.03]:
        cfg = {**CFG_BASE, "vol_target": vt}
        label = f"vol_target={vt}"
        r = run(p12, regime_df, cfg, f"D4: {label}")
        results.append(("D4", label, r))
        summary_row(label, r_base, r)

    # ══════════════════════════════════════════════════════════════
    #  BEST COMBINATIONS
    # ══════════════════════════════════════════════════════════════
    header("BEST COMBINATIONS")

    # Pick top improvers from each phase
    # Sort all results by equity (safe: worst > -15%)
    safe = [(ph, lb, r) for ph, lb, r in results
            if r is not None and r["worst_m"] > -0.15]
    safe.sort(key=lambda x: x[2]["equity"] * (x[2]["sharpe"] ** 0.5), reverse=True)

    print("\n  Top 10 configs by equity × √Sharpe (worst_m > -15%):")
    for i, (ph, lb, r) in enumerate(safe[:10], 1):
        print(f"  #{i:2d} [{ph}] {lb:<40}  "
              f"Eq=${r['equity']:>5.0f}  Sh={r['sharpe']:.2f}  "
              f"Wr={r['worst_m']*100:>+.1f}%  Cal={r['calmar']:.1f}")

    # Best combo attempt: take top result per phase and combine
    print("\n  Trying Best-of-Phase combination...")

    best_per_phase = {}
    for phase in ["A", "B", "C1", "C2", "D1", "D2", "D3"]:
        phase_res = [(lb, r) for ph, lb, r in results
                     if ph == phase and r is not None and r["worst_m"] > -0.15]
        if phase_res:
            phase_res.sort(key=lambda x: x[1]["equity"], reverse=True)
            best_per_phase[phase] = phase_res[0]

    for ph, (lb, r) in best_per_phase.items():
        print(f"    Best [{ph}]: {lb} → Eq=${r['equity']:.0f}  Sh={r['sharpe']:.2f}")

    # Build combo cfg from best A, C1, D1, D3
    combo_cfg = {**CFG_BASE}
    combo_preds = p12  # start with 12h baseline

    # Best A (horizon blend)?
    best_a = best_per_phase.get("A")
    if best_a and best_a[1]["equity"] > r_base["equity"] * 1.05:
        # rebuild blended preds for best A blend
        for label, pd_dict, weights in blend_configs:
            if label == best_a[0]:
                active = {h: pd_dict[h] for h in pd_dict if weights.get(h, 0) > 0}
                combo_preds = blend_predictions(active, {h: weights[h] for h in active})
                print(f"    Using blend: {label}")
                break

    # Best C1 (position count)
    best_c1 = best_per_phase.get("C1")
    if best_c1:
        nl, ns = int(best_c1[0].split("L/")[0]), int(best_c1[0].split("/")[1].replace("S", ""))
        combo_cfg["n_long"]  = nl
        combo_cfg["n_short"] = ns
        print(f"    Using positions: {nl}L/{ns}S")

    # Best D3 (shrinkage)
    best_d3 = best_per_phase.get("D3")
    if best_d3 and best_d3[1]["equity"] > r_base["equity"]:
        shrink_val = None if best_d3[0] == "shrink=None" else float(best_d3[0].split("=")[1])
        combo_cfg["pred_shrinkage"] = shrink_val
        print(f"    Using shrinkage: {shrink_val}")

    # Best D1 (EMA)
    best_d1 = best_per_phase.get("D1")
    if best_d1 and best_d1[1]["equity"] > r_base["equity"]:
        ema_str = best_d1[0].split("=")[1]
        ema_val = None if ema_str == "None" else int(ema_str)
        combo_cfg["signal_ema"] = ema_val
        print(f"    Using signal EMA: {ema_val}")

    r_combo = run(combo_preds, regime_df, combo_cfg, "COMBO: best-of-phase combination")
    summary_row("COMBO", r_base, r_combo)

    # LGB combo if LGB was better
    if preds_lgb is not None:
        best_lgb_cfg = {**combo_cfg}
        r_lgb_combo = run(preds_lgb, regime_df, best_lgb_cfg, "COMBO + LightGBM")
        summary_row("COMBO + LightGBM", r_base, r_lgb_combo)

    # ══════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════
    header("FINAL SUMMARY — R9 vs Baseline")
    print(f"\n  BASELINE (R7 prod):  Eq=${r_base['equity']:.0f}  "
          f"Sh={r_base['sharpe']:.2f}  Wr={r_base['worst_m']*100:.1f}%  "
          f"Cal={r_base['calmar']:.1f}")
    print()
    print(f"  {'Config':<52}  Eq      ΔEq    Sh    ΔSh    Wr       ΔWr")
    print("  " + "-" * 85)
    all_results = [(ph, lb, r) for ph, lb, r in results if r is not None]
    all_results.sort(key=lambda x: x[2]["equity"], reverse=True)
    for ph, lb, r in all_results[:20]:
        eq_d = r["equity"] - r_base["equity"]
        sh_d = r["sharpe"] - r_base["sharpe"]
        wr_d = (r["worst_m"] - r_base["worst_m"]) * 100
        mark = "✅" if eq_d > 0 and sh_d > 0 else ("⚠️ " if eq_d > 0 or sh_d > 0 else "❌")
        print(f"  {mark} [{ph}] {lb:<44}  "
              f"${r['equity']:>5.0f}  {eq_d:>+5.0f}  {r['sharpe']:.2f}  {sh_d:>+.2f}  "
              f"{r['worst_m']*100:>+.1f}%  {wr_d:>+.1f}%")


if __name__ == "__main__":
    main()
