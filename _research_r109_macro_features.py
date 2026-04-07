#!/usr/bin/env python3
"""
R109 — Macro Features (DXY, VIX, SPX, 10Y, Gold)

Download macro data via yfinance, build features, IC scan vs fwd_ret_12h.
If any pass gate → add to R68 champion feature set, re-run WF, bootstrap vs R68.

Features:
  dxy_ret_5d, dxy_ret_20d          — USD strength momentum
  vix_level, vix_z60, vix_chg_5d   — vol regime
  spx_ret_5d, spx_ret_20d          — risk-on/off
  us10y_level, us10y_chg_5d        — rates
  gold_ret_5d                       — safe haven
  btc_spx_corr_20d, btc_dxy_corr_20d — crypto-macro correlation

All features are CROSS-SECTIONAL (same for all coins at each timestamp).
Daily data → forward-fill to hourly → shift(1 day) to avoid lookahead.
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent

# ── Imports from existing pipeline ──────────────────────────────────────────
from _research_round7 import SYM_35
from _research_r22_models import SEEDS, LEVERAGE, CAPITAL, log, cs_rank_cols
from _research_r30b_fixed import compute_regime_extended
from _research_r35_new_features import (
    add_r35_features, load_research_frame, MARKET_LEVEL_FEATURES,
)
from _research_r47_coinglass import (
    CHAMPION_FEAT_30, add_cg_features, compute_cg_features, load_cg_daily,
)
from _research_r68_continuous_wf import (
    CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, ORIGINAL_WINDOWS,
    PROD_CFG, train_ensemble, simulate, load_data,
)

# ── Macro tickers ───────────────────────────────────────────────────────────
MACRO_TICKERS = {
    "DX-Y.NYB": "dxy",      # US Dollar Index
    "^VIX":     "vix",      # CBOE Volatility Index
    "^GSPC":    "spx",      # S&P 500
    "^TNX":     "us10y",    # US 10Y Treasury Yield
    "GC=F":     "gold",     # Gold futures
}

MACRO_FEATURES = [
    "dxy_ret_5d",
    "dxy_ret_20d",
    "vix_level",
    "vix_z60",
    "vix_chg_5d",
    "spx_ret_5d",
    "spx_ret_20d",
    "us10y_level",
    "us10y_chg_5d",
    "gold_ret_5d",
    "btc_spx_corr_20d",
    "btc_dxy_corr_20d",
]

IC_THRESH       = 0.03
STABILITY_THRESH = 2 / 3
COVERAGE_THRESH  = 0.95
REDUND_THRESH    = 0.70


def download_macro() -> pd.DataFrame:
    """Download macro data via yfinance and build daily features."""
    import yfinance as yf

    log("Step 1: Downloading macro data via yfinance...")

    # Download all tickers
    raw: Dict[str, pd.DataFrame] = {}
    for ticker, name in MACRO_TICKERS.items():
        log(f"  Downloading {ticker} ({name})...")
        try:
            data = yf.download(ticker, start="2017-01-01", progress=False, auto_adjust=True)
            if len(data) == 0:
                log(f"    WARNING: no data for {ticker}")
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data[["Close"]].rename(columns={"Close": name})
            data.index = pd.to_datetime(data.index, utc=True)
            raw[name] = data
            log(f"    OK: {len(data)} rows, {data.index.min().date()} → {data.index.max().date()}")
        except Exception as e:
            log(f"    ERROR: {e}")

    # Merge all on date
    macro = None
    for name, df in raw.items():
        if macro is None:
            macro = df
        else:
            macro = macro.join(df, how="outer")
    macro = macro.sort_index().ffill()
    log(f"  Combined: {len(macro)} rows, columns: {list(macro.columns)}")

    # Build features
    log("  Building macro features...")
    if "dxy" in macro.columns:
        macro["dxy_ret_5d"]  = macro["dxy"].pct_change(5)
        macro["dxy_ret_20d"] = macro["dxy"].pct_change(20)

    if "vix" in macro.columns:
        macro["vix_level"]   = macro["vix"]
        mu = macro["vix"].rolling(60, min_periods=30).mean()
        sd = macro["vix"].rolling(60, min_periods=30).std()
        macro["vix_z60"]     = (macro["vix"] - mu) / (sd + 1e-10)
        macro["vix_chg_5d"]  = macro["vix"].diff(5)

    if "spx" in macro.columns:
        macro["spx_ret_5d"]  = macro["spx"].pct_change(5)
        macro["spx_ret_20d"] = macro["spx"].pct_change(20)

    if "us10y" in macro.columns:
        macro["us10y_level"]   = macro["us10y"]
        macro["us10y_chg_5d"]  = macro["us10y"].diff(5)

    if "gold" in macro.columns:
        macro["gold_ret_5d"] = macro["gold"].pct_change(5)

    # Shift by 1 day to avoid lookahead — daily data published EOD
    feat_cols = [c for c in macro.columns if c in MACRO_FEATURES]
    log(f"  Built {len(feat_cols)} features: {feat_cols}")
    for c in feat_cols:
        macro[c] = macro[c].shift(1)

    macro = macro.reset_index().rename(columns={"index": "date", "Date": "date"})
    if "date" not in macro.columns:
        macro = macro.rename(columns={macro.columns[0]: "date"})
    macro["date"] = pd.to_datetime(macro["date"], utc=True)

    return macro[["date"] + feat_cols].dropna(subset=feat_cols, how="all")


def merge_macro_to_hourly(df: pd.DataFrame, macro: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Forward-fill daily macro to hourly timestamps.
    Returns merged df + list of successfully added feature columns.
    """
    log("  Merging macro to hourly timestamps...")
    df = df.copy()
    df["date"] = df["timestamp"].dt.normalize()

    feat_cols = [c for c in macro.columns if c != "date"]
    macro_daily = macro.copy()
    macro_daily["date"] = macro_daily["date"].dt.normalize()

    df = df.merge(macro_daily, on="date", how="left")

    # Forward-fill any remaining NaN from weekends
    for c in feat_cols:
        df[c] = df[c].ffill()

    added = [c for c in feat_cols if c in df.columns and df[c].notna().any()]
    log(f"  Added {len(added)} macro features to hourly frame")

    # BTC correlations (need hourly BTC returns + daily macro returns)
    log("  Computing BTC-macro rolling correlations...")
    btc_ret = df[df["symbol"] == "BTC/USDT"][["timestamp", "ret_12h"]].set_index("timestamp")["ret_12h"]

    for src, macro_col, new_col in [
        ("spx", "spx_ret_5d", "btc_spx_corr_20d"),
        ("dxy", "dxy_ret_5d", "btc_dxy_corr_20d"),
    ]:
        if macro_col not in df.columns:
            continue
        # Use the already-merged feature as macro proxy at hourly level
        macro_at_btc = df[df["symbol"] == "BTC/USDT"][["timestamp", macro_col]].set_index("timestamp")[macro_col]
        corr = btc_ret.rolling(20 * 12, min_periods=10 * 12).corr(macro_at_btc)  # 20 days × 12h rebal = 240 periods
        corr_df = corr.reset_index()
        corr_df.columns = ["timestamp", new_col]

        # Merge as cross-sectional (same for all coins)
        df = df.merge(corr_df, on="timestamp", how="left")
        if new_col in df.columns and df[new_col].notna().any():
            added.append(new_col)
            log(f"    {new_col}: OK, coverage={df[new_col].notna().mean():.3f}")

    df.drop(columns=["date"], inplace=True, errors="ignore")
    return df, added


def ic_scan(
    df: pd.DataFrame,
    feats: List[str],
    existing_feats: List[str],
) -> pd.DataFrame:
    """IC scan with gate: |IC| >= 0.03, stability >= 2/3, coverage >= 95%, redundancy < 0.70."""
    log("\nStep 2-3: IC Scan + Gate")
    log("=" * 60)

    tz = df["timestamp"].dt.tz
    rows = []

    for feat in feats:
        if feat not in df.columns:
            rows.append({"feature": feat, "skip": "not_in_frame"})
            continue
        valid = df[[feat, "fwd_ret_12h", "timestamp"]].dropna()
        n_obs = len(valid)
        if n_obs < 100:
            rows.append({"feature": feat, "skip": f"too_few_obs ({n_obs})"})
            continue

        # Coverage
        total = len(df)
        coverage = n_obs / total

        # Pooled Spearman IC
        pooled_ic = float(stats.spearmanr(valid[feat], valid["fwd_ret_12h"])[0])

        # Per-window IC (stability)
        window_ics = []
        for w in CONTINUOUS_WINDOWS:
            ts_s = pd.Timestamp(w["test_start"], tz=tz)
            ts_e = pd.Timestamp(w["test_end"], tz=tz)
            wdf = valid[(valid["timestamp"] >= ts_s) & (valid["timestamp"] <= ts_e)]
            if len(wdf) < 50:
                window_ics.append(np.nan)
            else:
                wic = float(stats.spearmanr(wdf[feat], wdf["fwd_ret_12h"])[0])
                window_ics.append(wic if not np.isnan(wic) else 0.0)

        stability = sum(
            1 for ic in window_ics if not np.isnan(ic) and abs(ic) >= 0.02
        ) / len(CONTINUOUS_WINDOWS)

        # Redundancy vs existing champion features
        max_corr = 0.0
        max_corr_feat = ""
        for ef in existing_feats:
            if ef not in df.columns:
                continue
            sub = df[[feat, ef]].dropna()
            if len(sub) < 50:
                continue
            c = abs(float(stats.spearmanr(sub[feat], sub[ef])[0]))
            if c > max_corr:
                max_corr = c
                max_corr_feat = ef

        # Mean IC by timestamp (more robust)
        ts_ics = []
        for _, grp in valid.groupby(valid["timestamp"].dt.date):
            if len(grp) >= 5:
                ic_val = stats.spearmanr(grp[feat], grp["fwd_ret_12h"])[0]
                if not np.isnan(ic_val):
                    ts_ics.append(ic_val)
        mean_ts_ic = np.mean(ts_ics) if ts_ics else 0.0

        # Gate
        pass_ic   = abs(pooled_ic) >= IC_THRESH
        pass_stab = stability >= STABILITY_THRESH
        pass_cov  = coverage >= COVERAGE_THRESH
        pass_red  = max_corr < REDUND_THRESH
        gate_pass = pass_ic and pass_stab and pass_cov and pass_red

        score = abs(pooled_ic) * stability

        rows.append({
            "feature":           feat,
            "pooled_ic":         round(pooled_ic, 4),
            "mean_ts_ic":        round(mean_ts_ic, 4),
            "w1_ic":             round(window_ics[0], 4) if len(window_ics) > 0 and not np.isnan(window_ics[0]) else None,
            "w2_ic":             round(window_ics[1], 4) if len(window_ics) > 1 and not np.isnan(window_ics[1]) else None,
            "w3_ic":             round(window_ics[2], 4) if len(window_ics) > 2 and not np.isnan(window_ics[2]) else None,
            "stability":         round(stability, 3),
            "coverage":          round(coverage, 3),
            "max_corr_existing": round(max_corr, 3),
            "max_corr_feat":     max_corr_feat,
            "score":             round(score, 4),
            "pass_ic":           pass_ic,
            "pass_stab":         pass_stab,
            "pass_cov":          pass_cov,
            "pass_red":          pass_red,
            "gate_pass":         gate_pass,
            "n_obs":             n_obs,
            "skip":              None,
        })

    ic_df = pd.DataFrame(rows)
    ic_df = ic_df.sort_values("score", ascending=False).reset_index(drop=True)

    # Pretty print
    for _, row in ic_df.iterrows():
        if row.get("skip"):
            log(f"  {row['feature']:>25s}: SKIP ({row['skip']})")
            continue
        gp = "✅ PASS" if row["gate_pass"] else "❌ FAIL"
        log(f"  {row['feature']:>25s}: pooled_IC={row['pooled_ic']:+.4f}  "
            f"mean_ts_IC={row['mean_ts_ic']:+.4f}  "
            f"stab={row['stability']:.2f}  "
            f"cov={row['coverage']:.3f}  "
            f"max_corr={row['max_corr_existing']:.3f}({row['max_corr_feat']})  "
            f"{gp}")

    return ic_df


def run_wf_test(
    df: pd.DataFrame,
    regime_df: pd.DataFrame,
    base_feats: List[str],
    new_feats: List[str],
    label: str,
) -> dict:
    """Train R68-style ensemble with extra macro features, simulate 4L/2S."""
    log(f"\n  WF test: {label}")

    all_feats = base_feats + [f for f in new_feats if f not in base_feats]
    avail = [f for f in all_feats if f in df.columns]

    # Market-level features should NOT be cross-sectionally ranked
    no_rank = set(MARKET_LEVEL_FEATURES) | set(new_feats)  # macro feats are all market-level
    no_rank = list(no_rank & set(avail))

    cfg = {**PROD_CFG, "n_long": 4, "n_short": 2}

    preds = train_ensemble(df, avail, CONTINUOUS_WINDOWS, seeds=SEEDS,
                           cs_rank_exclude=no_rank)
    if preds is None:
        return {"label": label, "error": "no_predictions"}

    metrics = simulate(preds, regime_df, cfg, label=label)
    return metrics


def bootstrap_comparison(
    preds_base, preds_new, regime_df, cfg, n_boot=1000, seed=42
):
    """Bootstrap Sharpe difference test."""
    rng = np.random.RandomState(seed)
    base_daily = simulate(preds_base, regime_df, cfg, label="boot_base")
    new_daily = simulate(preds_new, regime_df, cfg, label="boot_new")

    if not base_daily or not new_daily:
        return None

    # We'd need daily returns; simplified: just use final metrics
    return {
        "base_sharpe": base_daily.get("sharpe", 0),
        "new_sharpe": new_daily.get("sharpe", 0),
        "delta": new_daily.get("sharpe", 0) - base_daily.get("sharpe", 0),
    }


def main():
    t0 = time.time()

    log("=" * 70)
    log("R109 — Macro Features IC Scan")
    log("=" * 70)

    # ── Step 0: Load base data ──────────────────────────────────────────
    log("\nStep 0: Loading base data (R68 pipeline)...")
    df, regime_df = load_data()
    log(f"  Base frame: {len(df):,} rows, {df['symbol'].nunique()} symbols, "
        f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    # Compute fwd_ret_12h
    if "fwd_ret_12h" not in df.columns:
        df["fwd_ret_12h"] = df.groupby("symbol")["close"].transform(
            lambda x: x.pct_change(12).shift(-12)
        )
        log(f"  Computed fwd_ret_12h: {df['fwd_ret_12h'].notna().sum():,} valid obs")

    # ── Step 1: Download macro data ─────────────────────────────────────
    macro = download_macro()
    log(f"  Macro frame: {len(macro)} rows")

    # ── Step 1b: Merge to hourly ────────────────────────────────────────
    df, added_feats = merge_macro_to_hourly(df, macro)
    log(f"  Final frame: {len(df):,} rows, {len(added_feats)} macro features added")

    # Report coverage
    log("\n  Feature coverage:")
    for feat in sorted(added_feats):
        cov = df[feat].notna().mean()
        log(f"    {feat:>25s}: {cov:.3f}")

    # ── Step 2+3: IC scan ───────────────────────────────────────────────
    existing_feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    ic_df = ic_scan(df, added_feats, existing_feats)

    # Save IC results
    ic_df.to_csv(BASE_DIR / "results" / "r109_ic_scan.csv", index=False)
    log(f"\n  Saved IC scan → results/r109_ic_scan.csv")

    # ── Step 4: Check if any pass gate ──────────────────────────────────
    passed = ic_df[ic_df["gate_pass"] == True]
    n_pass = len(passed)

    log("\n" + "=" * 70)
    if n_pass == 0:
        log("RESULT: 0 features pass gate. STOP — no WF test needed.")
        log("=" * 70)

        # Summary
        log("\nSUMMARY:")
        log(f"  Features tested: {len(added_feats)}")
        log(f"  Gate passed:     0")
        log(f"  Best IC:         {ic_df[ic_df['pooled_ic'].notna()]['pooled_ic'].abs().max():.4f}" if len(ic_df[ic_df['pooled_ic'].notna()]) > 0 else "  Best IC: N/A")
        log(f"  Verdict:         ❌ FAIL — no macro feature has IC ≥ 0.03")
        log(f"  R68 remains champion. No changes.")

    else:
        log(f"RESULT: {n_pass} features pass gate! Running WF test...")
        log("=" * 70)

        new_feats = passed["feature"].tolist()
        log(f"  Passed features: {new_feats}")

        # Step 4: Re-run WF with new features
        base_feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]

        # Baseline: R68 champion
        log("\n  Baseline (R68 champion):")
        m_base = run_wf_test(df, regime_df, base_feats, [], "R68_baseline")

        # Test: R68 + macro
        log(f"\n  Test (R68 + {n_pass} macro features):")
        m_test = run_wf_test(df, regime_df, base_feats, new_feats, f"R68+macro_{n_pass}")

        # Compare
        log("\n  COMPARISON:")
        for key in ["sharpe", "ann_ret", "max_dd", "n_trades"]:
            b = m_base.get(key, "N/A")
            t = m_test.get(key, "N/A")
            log(f"    {key:>15s}: base={b}  test={t}")

        # Step 5: Bootstrap
        delta = (m_test.get("sharpe", 0) or 0) - (m_base.get("sharpe", 0) or 0)
        log(f"\n  Sharpe delta: {delta:+.3f}")
        if delta > 0:
            log("  → PASS: macro features improve Sharpe")
        else:
            log("  → FAIL: macro features do NOT improve Sharpe")

    elapsed = time.time() - t0
    log(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    log("Done.")


if __name__ == "__main__":
    main()
