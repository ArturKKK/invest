#!/usr/bin/env python3
"""
R20 — Deep Research Round.

Six experiments on top of R19 winner (LGB-23f, Sh=2.50):

  EXP-A: Funding carry signal (correctly loaded from binance_funding_rates.parquet)
  EXP-B: Position count sweep: n_long × n_short (4 combos)
  EXP-C: Regime threshold sweep: trend_cutoff 0.5 → 1.5 (to reduce Nov24 -29.6%)
  EXP-D: Multi-horizon training: 4h, 24h, ensemble of 12h+24h
  EXP-E: Permutation test of R19 winner (Sh=2.50 — is it lucky?)
  EXP-F: Rebalance interval: 12h vs 24h vs 6h

All built on R19 winner features (FEATURES_23).
Confirmed no leakage: vol features backward-looking, IC scan on train only.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from pathlib import Path
import warnings, time, sys, random
warnings.filterwarnings("ignore")

from _research_round7 import (
    SYM_35, WINDOWS, cs_rank,
    compute_regime, simulate, eval_config, show,
)
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal

PROJECT = Path(__file__).parent
DATA_DIR = PROJECT / "data"
SENT_DIR = DATA_DIR / "sentiment"

# R19 winner feature set
FEATURES_23 = [
    # 12f baseline
    "ret_12h", "ret_24h", "ret_48h",
    "residual_12h", "residual_24h",
    "mom_z_24h",
    "oi_chg_12h", "oi_chg_24h", "oi_zscore",
    "taker_cvd_12h", "taker_cvd_24h",
    "ls_divergence",
    # +5 vol (R18)
    "atr_14", "rvol_12h", "gk_vol_24h", "rvol_24h", "iv_rv_spread",
    # +6 breadth+season (R19)
    "pct_coins_up_12h", "pct_coins_up_1h",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

SEEDS = [0, 7, 13, 42, 99]
LEVERAGE = 5
CAPITAL  = 100

CFG_REGIME = {
    "n_long": 6, "n_short": 3, "trend_cutoff": 0.8,
    "dyn_threshold": 0.5, "kelly_sizing": False,
    "vol_scaling": False, "regime_asym": False, "rebal_hours": 12,
}
CFG_BARE = {**CFG_REGIME, "trend_cutoff": 999, "dyn_threshold": None}


def log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE BUILDERS (identical to R19)
# ═══════════════════════════════════════════════════════════════════════════════

def build_r19_features(df):
    """Apply all R19 feature enrichment."""
    # TA vol
    try:
        ta = pd.read_parquet(DATA_DIR / "features" / "crypto_features_1h.parquet",
                             columns=["timestamp", "symbol", "atr_14", "gk_vol_24h",
                                      "rsi_14", "bb_pband_20"])
        ta["timestamp"] = pd.to_datetime(ta["timestamp"], utc=True)
        df = df.merge(ta, on=["timestamp", "symbol"], how="left")
        log("  [TA] atr_14, gk_vol_24h, rsi_14, bb_pband_20")
    except Exception as e:
        log(f"  [TA] Error: {e}")

    # IV-RV spread
    try:
        dv = pd.read_parquet(SENT_DIR / "deribit_dvol.parquet")
        dv["timestamp"] = pd.to_datetime(dv["timestamp"], utc=True)
        btc_dv = (dv[dv["currency"] == "BTC"][["timestamp", "dvol_close"]]
                  .sort_values("timestamp")
                  .rename(columns={"dvol_close": "btc_dvol"}))
        btc_dv = btc_dv.set_index("timestamp").resample("1h").ffill().reset_index()
        df = df.merge(btc_dv, on="timestamp", how="left")
        df["btc_dvol"] = df["btc_dvol"].ffill()
        if "rvol_24h" in df.columns:
            df["iv_rv_spread"] = (df["btc_dvol"] / 100
                                  - df.groupby("symbol")["rvol_24h"]
                                  .transform(lambda x: x * np.sqrt(24 * 365)))
        df["dvol_zscore"] = ((df["btc_dvol"]
                              - df["btc_dvol"].rolling(168 * 4, min_periods=168).mean())
                             / (df["btc_dvol"].rolling(168 * 4, min_periods=168).std() + 1e-10))
        log("  [DVOL] btc_dvol, iv_rv_spread, dvol_zscore")
    except Exception as e:
        log(f"  [DVOL] Error: {e}")

    # Market breadth
    if "ret_12h" in df.columns:
        breadth = (df.groupby("timestamp")[["ret_12h", "ret_1h"]]
                   .agg(pct_coins_up_12h=("ret_12h", lambda x: (x > 0).mean()),
                        pct_coins_up_1h =("ret_1h",  lambda x: (x > 0).mean()))
                   .reset_index())
        df = df.merge(breadth, on="timestamp", how="left")
        btc_r = df.loc[df["symbol"] == "BTC/USDT", ["timestamp", "ret_12h"]].rename(
            columns={"ret_12h": "btc_ret12_ts"})
        df = df.merge(btc_r, on="timestamp", how="left")
        df["btc_outperform"] = df["ret_12h"] - df["btc_ret12_ts"]
        df.drop(columns=["btc_ret12_ts"], inplace=True, errors="ignore")
        log("  [BREADTH] pct_coins_up_12h/1h, btc_outperform")

    # Seasonality
    df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
    log("  [SEASON] hour_sin/cos, dow_sin/cos")

    return df


def add_funding_carry(df):
    """
    Funding carry features — already computed in build_features_minimal:
      funding_rate_binance, cum_funding_24h, cum_funding_72h, cum_funding_168h,
      funding_zscore, funding_x_mom_12h, funding_x_mom_24h
    Just confirm they exist and report coverage.
    """
    fund_cols = [c for c in df.columns if "fund" in c.lower()]
    if fund_cols:
        non_null = df[fund_cols[0]].notna().mean()
        log(f"  [FUNDING] Pre-existing: {fund_cols} (non-null={non_null:.1%})")
    else:
        log("  [FUNDING] No funding cols found in df")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def cs_rank_cols(df, feats):
    df = df.copy()
    for f in feats:
        if f in df.columns:
            df[f] = df.groupby("timestamp")[f].rank(pct=True) - 0.5
    return df


def train_lgb(df, feats, seeds=SEEDS, target_col="fwd_ret_12h", params_override=None):
    avail = [f for f in feats if f in df.columns]
    all_preds = []
    tz = df["timestamp"].dt.tz
    for seed in seeds:
        seed_preds = []
        for w in WINDOWS:
            train = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz=tz)].copy()
            val   = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz=tz)) &
                       (df["timestamp"] <  pd.Timestamp(w["val_end"],   tz=tz))].copy()
            test  = df[(df["timestamp"] >= pd.Timestamp(w["test_start"], tz=tz)) &
                       (df["timestamp"] <= pd.Timestamp(w["test_end"],   tz=tz))].copy()
            if len(train) < 5000 or len(test) < 200:
                continue
            train = cs_rank_cols(train, avail)
            val   = cs_rank_cols(val,   avail)
            test  = cs_rank_cols(test,  avail)
            for d in [train, val, test]:
                if target_col in d.columns:
                    d["target_rank"] = d.groupby("timestamp")[target_col].rank(pct=True) - 0.5
            train_c = train[avail + ["target_rank"]].dropna()
            val_c   = val[avail + ["target_rank"]].dropna()
            params = {
                "objective": "regression", "metric": "mse",
                "learning_rate": 0.03, "num_leaves": 63,
                "min_child_samples": 100, "subsample": 0.8,
                "colsample_bytree": 0.8, "lambda_l2": 1.0,
                "seed": seed, "verbose": -1, "n_jobs": -1,
            }
            if params_override:
                params.update(params_override)
            dtrain = lgb.Dataset(train_c[avail], label=train_c["target_rank"])
            dval   = lgb.Dataset(val_c[avail],   label=val_c["target_rank"])
            model  = lgb.train(params, dtrain, num_boost_round=600,
                               valid_sets=[dval],
                               callbacks=[lgb.early_stopping(40, verbose=False),
                                          lgb.log_evaluation(-1)])
            test_c = test[avail + ["target_rank", "timestamp", "symbol"]].dropna()
            if len(test_c) == 0:
                continue
            preds  = model.predict(test_c[avail])
            fwd    = test[["timestamp", "symbol", target_col]].rename(
                     columns={target_col: "fwd_ret"}).dropna()
            merged = test_c[["timestamp", "symbol"]].copy()
            merged["pred"] = preds
            merged = merged.merge(fwd, on=["timestamp", "symbol"], how="inner")
            merged["window"] = w["name"]
            seed_preds.append(merged)
        if seed_preds:
            all_preds.append(pd.concat(seed_preds, ignore_index=True))
    if not all_preds:
        return None
    combined = pd.concat(all_preds, ignore_index=True)
    return (combined.groupby(["timestamp", "symbol"])
            .agg(pred=("pred", "mean"), fwd_ret=("fwd_ret", "first"),
                 window=("window", "first"))
            .reset_index())


def ic_quick(preds, label):
    ics = []
    for wname in ["W1", "W2", "W3"]:
        sub = preds[preds["window"] == wname]
        w_ics = []
        for ts, grp in sub.groupby("timestamp"):
            if len(grp) >= 10:
                ic = stats.spearmanr(grp["pred"], grp["fwd_ret"])[0]
                if not np.isnan(ic):
                    w_ics.append(ic)
        if w_ics:
            a = np.array(w_ics)
            ics.extend(w_ics)
            log(f"    {wname}: IC={a.mean():.4f} IC>0={( a>0).mean()*100:.0f}%")
    if ics:
        a = np.array(ics)
        log(f"    ALL: IC={a.mean():.4f} +/-{a.std():.4f}")


def run_eval(preds, regime_df, label, cfgs=None, verbose_months=False):
    if preds is None:
        log(f"  ⚠  {label}: no predictions")
        return []
    if cfgs is None:
        cfgs = [("bare", CFG_BARE), ("regime", CFG_REGIME)]
    ic_quick(preds, label)
    results = []
    for cfg_name, cfg in cfgs:
        port = simulate(preds, regime_df, 12, cfg)
        if port is None:
            continue
        # Override horizon in eval if needed
        r = eval_config(port, 12, f"{label} [{cfg_name}]", LEVERAGE, CAPITAL)
        if r:
            show(r)
            if verbose_months:
                for m in r.get("month_data", []):
                    log(f"       {m['month']}   {m['ret']*100:+.1f}%  eq=${m['equity']:>8.0f}")
            results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT A: Funding carry
# ═══════════════════════════════════════════════════════════════════════════════

def exp_a(df, regime_df, base_preds):
    log("\n" + "═" * 80)
    log("  EXP-A: Funding Carry Signal")
    log("═" * 80)

    df = add_funding_carry(df)
    # funding_rate_binance and cum_funding_* already in df from build_features_minimal
    fund_feats = [f for f in ["cum_funding_24h", "funding_zscore",
                               "funding_x_mom_12h", "funding_x_mom_24h"]
                  if f in df.columns]
    if not fund_feats:
        log("  No funding features available, skip")
        return [], df

    log(f"  Using existing funding features: {fund_feats}")
    feats = list(dict.fromkeys(FEATURES_23 + fund_feats))  # dedupe
    feats = [f for f in feats if f in df.columns]
    log(f"\n  A1: LGB 23f + funding ({len(feats)}f)...")
    preds = train_lgb(df, feats)
    results = run_eval(preds, regime_df, f"LGB-{len(feats)}f+funding")
    return results, df


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT B: Position count sweep
# ═══════════════════════════════════════════════════════════════════════════════

def exp_b(preds, regime_df):
    log("\n" + "═" * 80)
    log("  EXP-B: Position Count Sweep (n_long × n_short)")
    log("═" * 80)
    log("  R19 winner: 6L/3S. Testing alternatives...")

    combos = [
        (4, 2, "4L/2S"),
        (5, 2, "5L/2S"),
        (6, 3, "6L/3S (baseline)"),
        (8, 4, "8L/4S"),
        (10, 5, "10L/5S"),
        (6, 2, "6L/2S"),
        (8, 3, "8L/3S"),
    ]
    results = []
    for nl, ns, name in combos:
        cfg = {**CFG_REGIME, "n_long": nl, "n_short": ns}
        port = simulate(preds, regime_df, 12, cfg)
        if port is None:
            continue
        r = eval_config(port, 12, f"B-{name}", LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT C: Regime threshold sweep
# ═══════════════════════════════════════════════════════════════════════════════

def exp_c(preds, regime_df):
    log("\n" + "═" * 80)
    log("  EXP-C: Regime Threshold Sweep (trend_cutoff)")
    log("═" * 80)
    log("  Goal: reduce Nov24 -29.6% without destroying upside")

    cutoffs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 999]
    results = []
    for tc in cutoffs:
        cfg = {**CFG_REGIME, "trend_cutoff": tc,
               "dyn_threshold": tc * 0.625 if tc < 900 else None}
        port = simulate(preds, regime_df, 12, cfg)
        if port is None:
            continue
        r = eval_config(port, 12, f"C-cutoff={tc}", LEVERAGE, CAPITAL)
        if r:
            covered = len(port) / len(preds["timestamp"].unique()) if preds is not None else 0
            flag = "✅" if r["worst_m"] > -0.25 else ("⚠️" if r["worst_m"] > -0.35 else "❌")
            log(f"  {flag} cutoff={tc:<5} Sh={r['sharpe']:+.2f} "
                f"WM={r['win_months']}/{r['total_months']} "
                f"Wr={r['worst_m']*100:+.1f}% Eq=${r['equity']:.0f}")
            results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT D: Multi-horizon training
# ═══════════════════════════════════════════════════════════════════════════════

def exp_d(df, regime_df):
    log("\n" + "═" * 80)
    log("  EXP-D: Multi-Horizon Target Training (4h, 24h, ensemble)")
    log("═" * 80)
    feats = [f for f in FEATURES_23 if f in df.columns]
    all_results = []

    for h, tgt in [(4, "fwd_ret_4h"), (24, "fwd_ret_24h")]:
        if tgt not in df.columns:
            log(f"  {tgt} not found, skip")
            continue
        log(f"\n  D{h//4}: LGB trained on {h}h target...")
        # Need preds aligned to fwd_ret_12h for simulate() to work
        # Solution: train on h-target, but simulate still uses fwd_ret from 12h column
        # We need to pass the right fwd_ret column
        df_copy = df.rename(columns={tgt: "_target_h", "fwd_ret_12h": "_fwd_12h"})
        df_copy["fwd_ret_12h"] = df_copy["_target_h"]  # temporarily swap target
        preds_h = train_lgb(df_copy, feats, target_col="fwd_ret_12h")
        # Restore fwd_ret to 12h for simulation
        if preds_h is not None:
            preds_h_fwd = preds_h.merge(
                df[["timestamp", "symbol", "fwd_ret_12h"]].rename(
                    columns={"fwd_ret_12h": "fwd_ret_true"}),
                on=["timestamp", "symbol"], how="left")
            preds_h_fwd["fwd_ret"] = preds_h_fwd["fwd_ret_true"]
            preds_h_fwd = preds_h_fwd.drop(columns=["fwd_ret_true"])
            all_results += run_eval(preds_h_fwd, regime_df, f"LGB-{h}h-target",
                                    cfgs=[("regime", CFG_REGIME)])

    # Ensemble: train 12h + 24h, average predictions
    if "fwd_ret_24h" in df.columns:
        log("\n  D_ens: Train separately on 12h and 24h, ensemble predictions...")
        preds_12h = train_lgb(df, feats)
        preds_24h = train_lgb(df, feats, target_col="fwd_ret_24h")
        if preds_12h is not None and preds_24h is not None:
            ens = preds_12h.merge(
                preds_24h[["timestamp", "symbol", "pred"]].rename(
                    columns={"pred": "pred_24h"}),
                on=["timestamp", "symbol"], how="inner")
            # Normalize each to [-0.5, 0.5] range by rank then average
            for ts, grp in ens.groupby("timestamp"):
                ens.loc[grp.index, "pred"] = (
                    0.5 * (grp["pred"].rank(pct=True) - 0.5) +
                    0.5 * (grp["pred_24h"].rank(pct=True) - 0.5))
            all_results += run_eval(ens, regime_df, "LGB-12h+24h-ens",
                                    cfgs=[("regime", CFG_REGIME)])

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT E: Permutation test for R19 winner (Sh=2.50)
# ═══════════════════════════════════════════════════════════════════════════════

def exp_e(preds, regime_df, n_perms=200):
    log("\n" + "═" * 80)
    log(f"  EXP-E: Permutation Test for R19 winner (n={n_perms})")
    log("═" * 80)
    log("  Shuffling predictions within each timestamp and re-running simulate()...")

    # Real Sharpe
    port_real = simulate(preds, regime_df, 12, CFG_REGIME)
    r_real = eval_config(port_real, 12, "real", LEVERAGE, CAPITAL)
    sh_real = r_real["sharpe"] if r_real else 0
    log(f"  Real Sharpe: {sh_real:.4f}")

    # Permutations
    rng = np.random.default_rng(42)
    null_sharpes = []
    for i in range(n_perms):
        perm = preds.copy()
        perm["pred"] = perm.groupby("timestamp")["pred"].transform(
            lambda x: rng.permutation(x.values))
        port_perm = simulate(perm, regime_df, 12, CFG_REGIME)
        r_perm = eval_config(port_perm, 12, "perm", LEVERAGE, CAPITAL)
        if r_perm:
            null_sharpes.append(r_perm["sharpe"])

    null = np.array(null_sharpes)
    z_score = (sh_real - null.mean()) / (null.std() + 1e-10)
    p_value = (null >= sh_real).mean()
    log(f"  Null  mean={null.mean():.3f} ± {null.std():.3f}")
    log(f"  Real  Sh={sh_real:.3f}")
    log(f"  z={z_score:.2f}, p={p_value:.4f} (fraction of perms >= real)")
    if p_value < 0.05:
        log(f"  ✅ SIGNIFICANT at 5% level — Sh=2.50 is NOT luck (p={p_value:.4f})")
    else:
        log(f"  ❌ NOT significant (p={p_value:.4f}) — potentially overfitted!")
    return z_score, p_value


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT F: Rebalance interval
# ═══════════════════════════════════════════════════════════════════════════════

def exp_f(preds, regime_df):
    log("\n" + "═" * 80)
    log("  EXP-F: Rebalance Interval (6h vs 12h vs 24h)")
    log("═" * 80)
    log("  Currently 12h. Testing alternatives...")

    results = []
    for rh, name in [(6, "6h-rebal"), (12, "12h-rebal (baseline)"), (24, "24h-rebal")]:
        cfg = {**CFG_REGIME, "rebal_hours": rh}
        port = simulate(preds, regime_df, 12, cfg)
        if port is None:
            continue
        r = eval_config(port, 12, f"F-{name}", LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 80)
    log("  R20 — DEEP RESEARCH (6 experiments)")
    log("=" * 80)

    # ── Load data + build R19 features ───────────────────────────────────────
    log("\n  Loading data...")
    ohlcv   = load_ohlcv()
    ohlcv   = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs  = load_derivatives()
    df      = build_features_minimal(ohlcv, derivs)
    regime_df = compute_regime(df)
    log(f"  Base: {len(df):,} rows, {df['symbol'].nunique()} symbols")

    log("\n  Building R19 features...")
    df = build_r19_features(df)

    avail_23 = [f for f in FEATURES_23 if f in df.columns]
    log(f"  FEATURES_23 availability: {len(avail_23)}/{len(FEATURES_23)}")

    # ── Train R19 winner (control) ────────────────────────────────────────────
    log("\n" + "═" * 80)
    log("  CONTROL: R19 winner (LGB-23f, regime filter)")
    log("═" * 80)
    log("\n  Training R19 winner for all exps (reused across B/C/E/F)...")
    preds_19 = train_lgb(df, avail_23)
    log("")
    ctrl_results = run_eval(preds_19, regime_df, "R19-winner-ctrl",
                            cfgs=[("regime", CFG_REGIME)], verbose_months=True)

    all_results = list(ctrl_results)

    # ── Experiments ──────────────────────────────────────────────────────────
    # A: Funding carry
    res_a, df = exp_a(df, regime_df, preds_19)
    all_results.extend(res_a)

    # Train updated predictions with funding (for passing to later exps if useful)
    fund_feats = [f for f in ["cum_funding_24h", "funding_zscore",
                               "funding_x_mom_12h", "funding_x_mom_24h"]
                  if f in df.columns]
    preds_fund = None
    if fund_feats:
        log(f"\n  Training with funding carry for B/C/E/F reference...")
        feats_with_fund = list(dict.fromkeys(avail_23 + fund_feats))
        preds_fund = train_lgb(df, feats_with_fund)

    # B: Position count sweep (on R19 winner and fund version)
    all_results.extend(exp_b(preds_19, regime_df))
    if preds_fund is not None:
        log("\n  B (with funding):")
        all_results.extend(exp_b(preds_fund, regime_df))

    # C: Regime threshold sweep
    all_results.extend(exp_c(preds_19, regime_df))

    # D: Multi-horizon  
    all_results.extend(exp_d(df, regime_df))

    # E: Permutation test
    z, p = exp_e(preds_19, regime_df, n_perms=300)

    # F: Rebalance interval
    all_results.extend(exp_f(preds_19, regime_df))

    # ── Final ranking ─────────────────────────────────────────────────────────
    log("\n" + "═" * 80)
    log("  FINAL RANKINGS — ALL R20 EXPERIMENTS")
    log("═" * 80)

    # Only configs that are direct comparisons (not sweeps C/F which have many)
    main_results = [r for r in all_results
                    if not r["name"].startswith("C-") and not r["name"].startswith("F-")
                    and not r["name"].startswith("B-")]
    if main_results:
        ranked = sorted(main_results, key=lambda r: -r["sharpe"])
        log(f"\n  Main configs ({len(ranked)}):")
        for i, r in enumerate(ranked, 1):
            flag = "✅" if r["sharpe"] >= 2.5 else ("⚠️ " if r["sharpe"] >= 1.8 else "❌")
            log(f"    #{i:2d} {flag} {r['name']:<60s} "
                f"Sh={r['sharpe']:+.2f} WM={r['win_months']}/{r['total_months']} "
                f"Wr={r['worst_m']*100:.1f}% Eq=${r['equity']:.0f}")

    # Best C (regime threshold)
    c_results = [r for r in all_results if r["name"].startswith("C-")]
    if c_results:
        best_c = max(c_results, key=lambda r: r["sharpe"])
        safe_c = max(c_results, key=lambda r: r["sharpe"] - abs(r["worst_m"]))
        log(f"\n  Best regime cutoff by Sharpe: {best_c['name']} "
            f"Sh={best_c['sharpe']:.2f} Wr={best_c['worst_m']*100:.1f}%")
        log(f"  Best regime cutoff by risk-adj: {safe_c['name']} "
            f"Sh={safe_c['sharpe']:.2f} Wr={safe_c['worst_m']*100:.1f}%")

    # Best B (position counts)
    b_results = [r for r in all_results if r["name"].startswith("B-")]
    if b_results:
        best_b = max(b_results, key=lambda r: r["sharpe"])
        log(f"\n  Best position count: {best_b['name']} "
            f"Sh={best_b['sharpe']:.2f} Wr={best_b['worst_m']*100:.1f}%")

    log(f"\n  Permutation test summary: z={z:.2f}, p={p:.4f}")

    elapsed = time.time() - t0
    log(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
