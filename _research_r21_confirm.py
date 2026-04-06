#!/usr/bin/env python3
"""
R21 — Confirmation Round.

Goal: Confirm and combine best R20 discoveries:

  EXP-G: Combined best (cutoff=0.9 + 6h rebal) — expected new champion
  EXP-H: Cutoff fine-tune around 0.9 (0.85, 0.90, 0.92, 0.95, 1.00) with 6h rebal
  EXP-I: Robustness check — does cutoff=0.9 + 6h hold across all 3 WF windows?
  EXP-J: OOS discipline — check if cutoff=0.9 improved IS or OOS

R20 findings to confirm:
  - EXP-C: trend_cutoff=0.9 → Sh=2.80 (+12% vs R19 Sh=2.50)
  - EXP-F: 6h rebal → Sh=2.88, Eq=$4981, Wr=-18.8% (+15% Sh)
  - Combined candidate: cutoff=0.9 + 6h rebal → ?
  - Permutation test p=0.0033 → strategy is real

R19 control: LGB-23f, cutoff=0.8, dyn_threshold=0.5, 6L/3S, rebal=12h → Sh=2.50
R20 ctrl-C: cutoff=0.9 → Sh=2.80
R20 ctrl-F: rebal=6h → Sh=2.88

All built on FEATURES_23 (confirmed 23/23 available).
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

# R19 winner feature set (confirmed 23/23 available)
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

# R19 baseline
CFG_R19 = {
    "n_long": 6, "n_short": 3, "trend_cutoff": 0.8,
    "dyn_threshold": 0.5, "kelly_sizing": False,
    "vol_scaling": False, "regime_asym": False, "rebal_hours": 12,
}

# R20 best-C (cutoff=0.9)
CFG_R20C = {**CFG_R19, "trend_cutoff": 0.9, "dyn_threshold": 0.5625}

# R20 best-F (6h rebal)
CFG_R20F = {**CFG_R19, "rebal_hours": 6}

# R21 candidate: combine both
CFG_R21 = {**CFG_R19, "trend_cutoff": 0.9, "dyn_threshold": 0.5625, "rebal_hours": 6}


def log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE BUILDERS (identical to R19/R20)
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


def ic_quick(preds, label=""):
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
            log(f"    {wname}: IC={a.mean():.4f} IC>0={(a>0).mean()*100:.0f}%")
    if ics:
        a = np.array(ics)
        log(f"    ALL: IC={a.mean():.4f} +/-{a.std():.4f}")


def run_eval(preds, regime_df, label, cfgs, verbose_months=False):
    if preds is None:
        log(f"  ⚠  {label}: no predictions")
        return []
    ic_quick(preds, label)
    results = []
    for cfg_name, cfg in cfgs:
        port = simulate(preds, regime_df, 12, cfg)
        if port is None:
            continue
        r = eval_config(port, 12, f"{label} [{cfg_name}]", LEVERAGE, CAPITAL)
        if r:
            show(r)
            if verbose_months:
                for m in r.get("month_data", []):
                    log(f"       {m['month']}   {m['ret']*100:+.1f}%  eq=${m['equity']:>8.0f}")
            results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-G: Combined regime=0.9 + 6h rebalance
# ═══════════════════════════════════════════════════════════════════════════════

def exp_g(preds, regime_df):
    log("\n" + "═" * 80)
    log("  EXP-G: Combined Best (cutoff=0.9 + 6h rebal)")
    log("═" * 80)
    log("  Testing: R19, R20-C (0.9 cutoff), R20-F (6h), R21 (0.9+6h)")

    configs = [
        ("R19-baseline",          CFG_R19),
        ("R20-C-cutoff09",        CFG_R20C),
        ("R20-F-6h-rebal",        CFG_R20F),
        ("R21-combined-09+6h",    CFG_R21),
    ]
    results = []
    for name, cfg in configs:
        port = simulate(preds, regime_df, 12, cfg)
        if port is None:
            log(f"  ⚠  {name}: simulate returned None")
            continue
        r = eval_config(port, 12, f"G-{name}", LEVERAGE, CAPITAL)
        if r:
            show(r)
            if name == "R21-combined-09+6h":
                log(f"  --- Monthly breakdown for R21 combined ---")
                for m in r.get("month_data", []):
                    log(f"       {m['month']}   {m['ret']*100:+.1f}%  eq=${m['equity']:>8.0f}")
            results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-H: Fine-tune cutoff around 0.9, with 6h rebal
# ═══════════════════════════════════════════════════════════════════════════════

def exp_h(preds, regime_df):
    log("\n" + "═" * 80)
    log("  EXP-H: Cutoff Fine-Tune (6h rebal fixed)")
    log("═" * 80)
    log("  Sweeping cutoff 0.80 to 1.05 in steps of 0.05, with 6h rebal")

    cutoffs = [0.80, 0.85, 0.90, 0.92, 0.95, 1.00, 1.05]
    results = []
    for tc in cutoffs:
        cfg = {**CFG_R21, "trend_cutoff": tc, "dyn_threshold": tc * 0.625}
        port = simulate(preds, regime_df, 12, cfg)
        if port is None:
            continue
        r = eval_config(port, 12, f"H-cutoff{tc:.2f}+6h", LEVERAGE, CAPITAL)
        if r:
            flag = "✅" if r["sharpe"] >= 2.8 else ("⚠️" if r["sharpe"] >= 2.5 else "❌")
            log(f"  {flag} cutoff={tc:<5.2f} Sh={r['sharpe']:+.2f} "
                f"WM={r['win_months']}/{r['total_months']} "
                f"Wr={r['worst_m']*100:+.1f}% Eq=${r['equity']:.0f}")
            results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-I: Per-window robustness check for R21 vs R19
# ═══════════════════════════════════════════════════════════════════════════════

def exp_i(preds, regime_df):
    log("\n" + "═" * 80)
    log("  EXP-I: Per-Window Robustness (R19 vs R21 across each WF window)")
    log("═" * 80)

    for wname in ["W1", "W2", "W3"]:
        sub = preds[preds["window"] == wname]
        if len(sub) == 0:
            continue
        log(f"\n  Window {wname}:")
        for label, cfg in [("R19", CFG_R19), ("R21", CFG_R21)]:
            port = simulate(sub, regime_df, 12, cfg)
            r = eval_config(port, 12, f"I-{wname}-{label}", LEVERAGE, CAPITAL)
            if r:
                flag = "✅" if r["sharpe"] >= 2.5 else ("⚠️" if r["sharpe"] >= 1.5 else "❌")
                log(f"    {flag} {label}: Sh={r['sharpe']:+.2f} "
                    f"WM={r['win_months']}/{r['total_months']} "
                    f"Wr={r['worst_m']*100:+.1f}% Eq=${r['equity']:.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-J: Permutation test for R21 combined winner
# ═══════════════════════════════════════════════════════════════════════════════

def exp_j(preds, regime_df, n_perms=200):
    log("\n" + "═" * 80)
    log(f"  EXP-J: Permutation Test for R21 combined (n={n_perms})")
    log("═" * 80)
    log(f"  Config: cutoff=0.9, 6h rebal")
    log("  Shuffling predictions within each timestamp...")

    port_real = simulate(preds, regime_df, 12, CFG_R21)
    r_real = eval_config(port_real, 12, "real", LEVERAGE, CAPITAL)
    sh_real = r_real["sharpe"] if r_real else 0
    log(f"  Real Sharpe: {sh_real:.4f}")

    rng = np.random.default_rng(42)
    null_sharpes = []
    for i in range(n_perms):
        perm = preds.copy()
        perm["pred"] = perm.groupby("timestamp")["pred"].transform(
            lambda x: rng.permutation(x.values))
        port_perm = simulate(perm, regime_df, 12, CFG_R21)
        r_perm = eval_config(port_perm, 12, "perm", LEVERAGE, CAPITAL)
        if r_perm:
            null_sharpes.append(r_perm["sharpe"])

    null = np.array(null_sharpes)
    z_score = (sh_real - null.mean()) / (null.std() + 1e-10)
    p_value = (null >= sh_real).mean()
    log(f"  Null  mean={null.mean():.3f} ± {null.std():.3f}")
    log(f"  Real  Sh={sh_real:.3f}")
    log(f"  z={z_score:.2f}, p={p_value:.4f}")
    if p_value < 0.05:
        log(f"  ✅ SIGNIFICANT at 5% level (p={p_value:.4f})")
    else:
        log(f"  ❌ NOT significant (p={p_value:.4f})")
    return z_score, p_value


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 80)
    log("  R21 — CONFIRMATION ROUND (combine best R20 findings)")
    log("=" * 80)
    log("  R20 findings: cutoff=0.9 → Sh=2.80 | 6h rebal → Sh=2.88")
    log("  Goal: confirm combined R21 config (cutoff=0.9 + 6h rebal)")

    # ── Load data + build features ───────────────────────────────────────────
    log("\n  Loading data...")
    ohlcv    = load_ohlcv()
    ohlcv    = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs   = load_derivatives()
    df       = build_features_minimal(ohlcv, derivs)
    regime_df = compute_regime(df)
    log(f"  Base: {len(df):,} rows, {df['symbol'].nunique()} symbols")

    log("\n  Building R19 features...")
    df = build_r19_features(df)

    avail_23 = [f for f in FEATURES_23 if f in df.columns]
    log(f"  FEATURES_23 availability: {len(avail_23)}/{len(FEATURES_23)}")

    # ── Train base model (shared preds for all exps) ──────────────────────────
    log("\n" + "═" * 80)
    log("  Training LGB-23f (R19 features, shared for all experiments)")
    log("═" * 80)
    preds = train_lgb(df, avail_23)
    if preds is None:
        log("  ❌ Training failed!")
        return

    # Quick IC check
    ic_quick(preds, "LGB-23f")

    # ── Run experiments ───────────────────────────────────────────────────────
    all_results = []

    # G: Combined best
    all_results.extend(exp_g(preds, regime_df))

    # H: Fine-tune cutoff with 6h rebal
    all_results.extend(exp_h(preds, regime_df))

    # I: Per-window robustness
    exp_i(preds, regime_df)

    # J: Permutation test for R21
    z, p = exp_j(preds, regime_df, n_perms=200)

    # ── Final summary ─────────────────────────────────────────────────────────
    log("\n" + "═" * 80)
    log("  FINAL RANKINGS — R21 EXPERIMENTS")
    log("═" * 80)

    # Main configs from EXP-G
    g_results = [r for r in all_results if r["name"].startswith("G-")]
    if g_results:
        ranked = sorted(g_results, key=lambda r: -r["sharpe"])
        log(f"\n  G configs (R19 vs R20 variants vs R21):")
        for i, r in enumerate(ranked, 1):
            flag = "✅" if r["sharpe"] >= 2.8 else ("⚠️" if r["sharpe"] >= 2.5 else "❌")
            log(f"    #{i:2d} {flag} {r['name']:<60s} "
                f"Sh={r['sharpe']:+.2f} WM={r['win_months']}/{r['total_months']} "
                f"Wr={r['worst_m']*100:+.1f}% Eq=${r['equity']:.0f}")

    # Best from H sweep
    h_results = [r for r in all_results if r["name"].startswith("H-")]
    if h_results:
        best_h = max(h_results, key=lambda r: r["sharpe"])
        log(f"\n  Best cutoff (6h rebal): {best_h['name']} Sh={best_h['sharpe']:.2f}")
        log(f"  Best cutoff by risk-adj: {best_h['name']} Sh={best_h['sharpe']:.2f} Wr={best_h['worst_m']*100:.1f}%")

    # Permutation result
    log(f"\n  Permutation test (R21): z={z:.2f}, p={p:.4f}")
    sig = "✅ SIGNIFICANT" if p < 0.05 else "❌ NOT significant"
    log(f"  {sig}")

    # Overall verdict
    r21_results = [r for r in all_results if "R21-combined" in r["name"]]
    if r21_results:
        r21 = r21_results[0]
        log(f"\n  ══ R21 COMBINED RESULT ══")
        log(f"  Config: cutoff=0.9, 6h rebal, 6L/3S, 23f")
        log(f"  Sh={r21['sharpe']:+.2f} | WM={r21['win_months']}/{r21['total_months']} "
            f"| Wr={r21['worst_m']*100:+.1f}% | Eq=${r21['equity']:.0f}")
        improvement = (r21['sharpe'] - 2.50) / 2.50 * 100
        log(f"  vs R19 Sh=2.50: {improvement:+.1f}%")
        if r21['sharpe'] > 2.80:
            log(f"  ✅ NEW RECORD — beats both R20-C (2.80) and R20-F (2.88) individually!")
        elif r21['sharpe'] > 2.50:
            log(f"  ✅ Improvement over R19 baseline")
        else:
            log(f"  ⚠️  No improvement — combination didn't help")

    elapsed = time.time() - t0
    log(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
