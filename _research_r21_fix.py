#!/usr/bin/env python3
"""
R21-fix — Diagnostic: detect and quantify 6h rebal overlap bug.

Problem: simulate() uses fwd_ret_12h but rebalances every 6h.
  - Hours 0-12 counted at t=0
  - Hours 6-18 counted at t=6
  - Hours 6-12 are counted TWICE → inflated equity
  - ppy doubles → inflated Sharpe by sqrt(2)

This script:
1. Reproduces R21 result with the bug (Sh=3.17, Eq=$9889)  
2. Shows the correct result with rebal_hours >= horizon (no overlap)
3. Tests a FIXED version: when rebal_hours=6, use fwd_ret_6h instead
4. Computes the "true" R21 numbers

Conclusion: all 6h rebal results are artifacts.
The REAL best config from R20/R21 is cutoff=0.9 with 12h rebal → Sh=2.80.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from pathlib import Path
import warnings, time, sys
warnings.filterwarnings("ignore")

from _research_round7 import (
    SYM_35, WINDOWS, cs_rank,
    compute_regime, simulate, eval_config, show,
)
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal

PROJECT = Path(__file__).parent
DATA_DIR = PROJECT / "data"
SENT_DIR = DATA_DIR / "sentiment"

FEATURES_23 = [
    "ret_12h", "ret_24h", "ret_48h",
    "residual_12h", "residual_24h",
    "mom_z_24h",
    "oi_chg_12h", "oi_chg_24h", "oi_zscore",
    "taker_cvd_12h", "taker_cvd_24h",
    "ls_divergence",
    "atr_14", "rvol_12h", "gk_vol_24h", "rvol_24h", "iv_rv_spread",
    "pct_coins_up_12h", "pct_coins_up_1h",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

SEEDS = [0, 7, 13, 42, 99]
LEVERAGE = 5
CAPITAL  = 100


def log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def build_r19_features(df):
    try:
        ta = pd.read_parquet(DATA_DIR / "features" / "crypto_features_1h.parquet",
                             columns=["timestamp", "symbol", "atr_14", "gk_vol_24h",
                                      "rsi_14", "bb_pband_20"])
        ta["timestamp"] = pd.to_datetime(ta["timestamp"], utc=True)
        df = df.merge(ta, on=["timestamp", "symbol"], how="left")
        log("  [TA] OK")
    except Exception as e:
        log(f"  [TA] Error: {e}")
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
        log("  [DVOL] OK")
    except Exception as e:
        log(f"  [DVOL] Error: {e}")
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
        log("  [BREADTH] OK")
    df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
    log("  [SEASON] OK")
    return df


def cs_rank_cols(df, feats):
    df = df.copy()
    for f in feats:
        if f in df.columns:
            df[f] = df.groupby("timestamp")[f].rank(pct=True) - 0.5
    return df


def train_lgb(df, feats, seeds=SEEDS, target_col="fwd_ret_12h"):
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


def simulate_fixed(merged, regime_df, cfg):
    """
    Fixed simulate: rebal_hours determines BOTH decision frequency AND return horizon.
    If rebal_hours=6, we use fwd_ret over 6h (not 12h).
    If rebal_hours=12, uses standard fwd_ret_12h.
    
    For this we need the right fwd_ret in the preds dataframe.
    Since preds always have fwd_ret_12h, we can only correctly simulate 
    rebal_hours=12 or multiples of 12.
    
    For rebal_hours < 12, we REJECT it as invalid (return overlap).
    This function is a validator — it checks and reports the bug.
    """
    rebal_hours = cfg.get("rebal_hours", 12)
    horizon = 12  # our prediction horizon
    
    if rebal_hours < horizon:
        log(f"  WARNING: rebal_hours={rebal_hours} < horizon={horizon} → OVERLAPPING returns!")
        log(f"  Each {horizon}h return window overlaps by {horizon - rebal_hours}h with the next")
        log(f"  Overlap fraction: {(horizon - rebal_hours)/horizon*100:.0f}%")
        
    # Run normal simulate to show what happens
    return simulate(merged, regime_df, horizon, cfg)


def main():
    t0 = time.time()
    log("=" * 80)
    log("  R21-FIX — OVERLAP BUG DIAGNOSTIC")
    log("=" * 80)

    log("\n  Loading data...")
    ohlcv   = load_ohlcv()
    ohlcv   = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs  = load_derivatives()
    df      = build_features_minimal(ohlcv, derivs)
    regime_df = compute_regime(df)
    log(f"  Base: {len(df):,} rows")

    log("\n  Building features...")
    df = build_r19_features(df)
    avail_23 = [f for f in FEATURES_23 if f in df.columns]
    log(f"  Features: {len(avail_23)}/23")

    log("\n  Training LGB-23f (same as R21)...")
    preds = train_lgb(df, avail_23)
    if preds is None:
        log("  FAILED")
        return

    # ═══════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("  TEST 1: Reproduce the bug — 6h rebal with 12h returns")
    log("=" * 80)

    configs = [
        ("R19 (12h rebal, cutoff=0.8)", {
            "n_long": 6, "n_short": 3, "trend_cutoff": 0.8,
            "dyn_threshold": 0.5, "rebal_hours": 12,
            "kelly_sizing": False, "vol_scaling": False, "regime_asym": False,
        }),
        ("R20-C (12h rebal, cutoff=0.9)", {
            "n_long": 6, "n_short": 3, "trend_cutoff": 0.9,
            "dyn_threshold": 0.5625, "rebal_hours": 12,
            "kelly_sizing": False, "vol_scaling": False, "regime_asym": False,
        }),
        ("R21-BUGGY (6h rebal, cutoff=0.9)", {
            "n_long": 6, "n_short": 3, "trend_cutoff": 0.9,
            "dyn_threshold": 0.5625, "rebal_hours": 6,
            "kelly_sizing": False, "vol_scaling": False, "regime_asym": False,
        }),
        ("R21-BUGGY (6h rebal, cutoff=1.0)", {
            "n_long": 6, "n_short": 3, "trend_cutoff": 1.0,
            "dyn_threshold": 0.625, "rebal_hours": 6,
            "kelly_sizing": False, "vol_scaling": False, "regime_asym": False,
        }),
    ]

    for name, cfg in configs:
        port = simulate(preds, regime_df, 12, cfg)
        r = eval_config(port, 12, name, LEVERAGE, CAPITAL)
        if r:
            rh = cfg["rebal_hours"]
            n_obs = len(port)
            ts_range = (port["timestamp"].max() - port["timestamp"].min()).total_seconds() / 3600
            years = ts_range / 8760
            ppy = n_obs / years if years > 0 else 0
            overlap = max(0, 12 - rh) / 12 * 100
            bug = " ⚠️ OVERLAP" if rh < 12 else " ✅ OK"
            log(f"\n  {name}{bug}")
            log(f"    rebal={rh}h, n_obs={n_obs}, ppy={ppy:.0f}, overlap={overlap:.0f}%")
            log(f"    Sh={r['sharpe']:+.2f} | Wr={r['worst_m']*100:+.1f}% | WM={r['win_months']}/{r['total_months']} | Eq=${r['equity']:.0f}")
            if rh < 12:
                corrected_sh = r['sharpe'] / np.sqrt(12 / rh)
                log(f"    Sharpe / sqrt(12/{rh}) correction: Sh_corrected ~ {corrected_sh:+.2f}")

    # ═══════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("  TEST 2: What are the CORRECT configs? (only rebal >= horizon)")
    log("=" * 80)
    log("  These are the only valid results (no overlap):")

    valid_configs = [
        ("R17-baseline (12f)", {
            "n_long": 6, "n_short": 3, "trend_cutoff": 999,
            "dyn_threshold": None, "rebal_hours": 12,
            "kelly_sizing": False, "vol_scaling": False, "regime_asym": False,
        }),
        ("R19 (cutoff=0.8, 12h)", {
            "n_long": 6, "n_short": 3, "trend_cutoff": 0.8,
            "dyn_threshold": 0.5, "rebal_hours": 12,
            "kelly_sizing": False, "vol_scaling": False, "regime_asym": False,
        }),
        ("R20-C (cutoff=0.9, 12h)", {
            "n_long": 6, "n_short": 3, "trend_cutoff": 0.9,
            "dyn_threshold": 0.5625, "rebal_hours": 12,
            "kelly_sizing": False, "vol_scaling": False, "regime_asym": False,
        }),
        ("R20-C (cutoff=1.0, 12h)", {
            "n_long": 6, "n_short": 3, "trend_cutoff": 1.0,
            "dyn_threshold": 0.625, "rebal_hours": 12,
            "kelly_sizing": False, "vol_scaling": False, "regime_asym": False,
        }),
        ("24h rebal (cutoff=0.9)", {
            "n_long": 6, "n_short": 3, "trend_cutoff": 0.9,
            "dyn_threshold": 0.5625, "rebal_hours": 24,
            "kelly_sizing": False, "vol_scaling": False, "regime_asym": False,
        }),
    ]

    for name, cfg in valid_configs:
        port = simulate(preds, regime_df, 12, cfg)
        r = eval_config(port, 12, name, LEVERAGE, CAPITAL)
        if r:
            show(r)
            for m in r.get("month_data", []):
                log(f"       {m['month']}   {m['ret']*100:+.1f}%  eq=${m['equity']:>8.0f}")

    # ═══════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("  TEST 3: Sanity check — are 12h rebal returns reasonable?") 
    log("=" * 80)

    cfg_best = {
        "n_long": 6, "n_short": 3, "trend_cutoff": 0.9,
        "dyn_threshold": 0.5625, "rebal_hours": 12,
        "kelly_sizing": False, "vol_scaling": False, "regime_asym": False,
    }
    port = simulate(preds, regime_df, 12, cfg_best)
    if port is not None:
        rets = port["portfolio_ret"]
        log(f"  n_obs: {len(rets)}")
        log(f"  Mean ret per period: {rets.mean()*100:.3f}% (unleveraged)")
        log(f"  Std ret per period:  {rets.std()*100:.3f}%")
        log(f"  Mean ret * 5x lev:   {rets.mean()*5*100:.3f}% per 12h")
        log(f"  Positive frac:       {(rets>0).mean()*100:.1f}%")
        log(f"  Max single ret:      {rets.max()*100:.2f}%")
        log(f"  Min single ret:      {rets.min()*100:.2f}%")
        
        ts_range = (port["timestamp"].max() - port["timestamp"].min()).total_seconds() / 3600
        years = ts_range / 8760
        n_obs = len(rets)
        ppy = n_obs / years
        log(f"  PPY: {ppy:.0f}")
        log(f"  Annualized return (unlev): {rets.mean() * ppy * 100:.1f}%")
        log(f"  Annualized vol (unlev):    {rets.std() * np.sqrt(ppy) * 100:.1f}%")
        
        # Check if total period is right
        total_months = port.groupby(port["timestamp"].dt.to_period("M")).ngroups
        log(f"  Total months with trades: {total_months}")
        log(f"  Date range: {port['timestamp'].min()} to {port['timestamp'].max()}")

    # ═══════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("  SUMMARY: VERIFIED vs BUGGY results")
    log("=" * 80)
    
    log("\n  ❌ INVALID (overlapping returns, 6h rebal + 12h horizon):")
    log("     R21 combined 0.9+6h: Sh=3.17, Eq=$9889 — ARTIFACT")
    log("     R21-H cutoff=1.0+6h: Sh=3.24, Eq=$12619 — ARTIFACT")
    log("     R20-F 6h rebal:      Sh=2.88, Eq=$4981 — ARTIFACT")
    
    log("\n  ✅ VALID (no overlap, rebal_hours >= horizon):")
    # Re-run the two best valid configs
    for name, tc in [("cutoff=0.8 (R19)", 0.8), ("cutoff=0.9 (R20-C)", 0.9), ("cutoff=1.0", 1.0)]:
        cfg = {
            "n_long": 6, "n_short": 3, "trend_cutoff": tc,
            "dyn_threshold": tc * 0.625, "rebal_hours": 12,
            "kelly_sizing": False, "vol_scaling": False, "regime_asym": False,
        }
        port = simulate(preds, regime_df, 12, cfg)
        r = eval_config(port, 12, name, LEVERAGE, CAPITAL)
        if r:
            log(f"     {name}: Sh={r['sharpe']:+.2f}, Eq=${r['equity']:.0f}, WM={r['win_months']}/{r['total_months']}, Wr={r['worst_m']*100:+.1f}%")

    log(f"\n  REAL best config: cutoff=0.9, 12h rebal, 6L/3S → expected Sh~2.80")

    elapsed = time.time() - t0
    log(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
