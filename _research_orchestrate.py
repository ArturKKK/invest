#!/usr/bin/env python3
"""
_research_orchestrate.py — Master runner: R82 → R83 → R84 → R85

Waits for R81 to finish, then runs the remaining phases of DeepResearch v3
in sequence without human intervention.

Usage:
  nohup /data/datasets/.venv/bin/python _research_orchestrate.py \
      > /data/datasets/orchestrate.log 2>&1 &

Phase summary:
  Phase 1  R81  Vol Overlay grid          — already running (wait for it)
  Phase 2  R82  CG Feature Factory        — build z-scores / momentum
  Phase 3  R83  IC Scan + Redundancy Gate — pick top-2 features
  Phase 4  R84  Add-only WF test          — retrain ensemble + compare
  Phase 5  R85  Bootstrap significance    — block bootstrap

Checkpointing: each phase writes a result file; if it already exists the
phase is skipped and the script resumes from the next one.
"""

import json
import sys
import time
import traceback
import warnings
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
FEAT_DIR    = ROOT / "data" / "features"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FEAT_DIR.mkdir(parents=True, exist_ok=True)

STATUS_FILE = RESULTS_DIR / "orchestrate_status.json"
CG_DIR      = ROOT / "data" / "raw" / "coinglass"

EPS              = 1e-10
PERIODS_PER_YEAR = 2 * 365
ROLL_120         = 120
ROLL_7           = 7
COVERAGE_THR     = 0.95

# R68 test windows
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
TEST_START = "2024-10-15"


# ─── Utilities ────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def update_status(phase: str, state: str, extra: dict = None) -> None:
    data = {}
    if STATUS_FILE.exists():
        try:
            data = json.loads(STATUS_FILE.read_text())
        except Exception:
            pass
    data[phase] = {"state": state, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {})}
    STATUS_FILE.write_text(json.dumps(data, indent=2, default=float))


def sharpe(rets: pd.Series) -> float:
    if len(rets) < 2:
        return 0.0
    r = (1 + rets).cumprod().pct_change().dropna()
    return float(r.mean() / (r.std() + EPS) * np.sqrt(PERIODS_PER_YEAR))


def max_dd(rets: pd.Series) -> float:
    eq = (1 + rets).cumprod()
    return float((eq / eq.cummax() - 1).min())


def calmar(rets: pd.Series) -> float:
    s  = sharpe(rets)
    dd = abs(max_dd(rets))
    return float(s / dd) if dd > EPS else 0.0


def total_ret(rets: pd.Series) -> float:
    return float((1 + rets).prod() - 1)


def win_rate(rets: pd.Series) -> float:
    return float((rets > 0).mean())


def portfolio_metrics(port: pd.DataFrame, ret_col: str = "net_ret") -> dict:
    rets = port[ret_col]
    return {
        "net_sharpe":    round(sharpe(rets), 4),
        "gross_sharpe":  round(sharpe(port["gross_ret"]), 4),
        "max_dd_pct":    round(max_dd(rets) * 100, 2),
        "calmar":        round(calmar(rets), 3),
        "total_ret_pct": round(total_ret(rets) * 100, 1),
        "win_rate":      round(win_rate(rets), 3),
        "n_periods":     len(rets),
    }


def monthly_csv(port: pd.DataFrame, path: Path, ret_col: str = "net_ret") -> None:
    p = port.copy()
    p["month"] = p["timestamp"].dt.to_period("M").astype(str)
    rows = []
    for m, g in p.groupby("month"):
        rows.append({
            "month": m,
            "net_ret_pct":   round(total_ret(g[ret_col]) * 100, 2),
            "gross_ret_pct": round(total_ret(g["gross_ret"]) * 100, 2),
            "net_sharpe":    round(sharpe(g[ret_col]), 3),
            "max_dd_pct":    round(max_dd(g[ret_col]) * 100, 2),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def quarterly_csv(port: pd.DataFrame, path: Path, ret_col: str = "net_ret") -> None:
    p = port.copy()
    p["quarter"] = p["timestamp"].dt.to_period("Q").astype(str)
    rows = []
    for q, g in p.groupby("quarter"):
        rows.append({
            "quarter": q,
            "net_ret_pct":   round(total_ret(g[ret_col]) * 100, 2),
            "gross_ret_pct": round(total_ret(g["gross_ret"]) * 100, 2),
            "net_sharpe":    round(sharpe(g[ret_col]), 3),
            "max_dd_pct":    round(max_dd(g[ret_col]) * 100, 2),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


# ─── Phase: wait for R81 ──────────────────────────────────────────────────────

def wait_for_r81(timeout_min: int = 60) -> dict:
    """Block until r81_summary.json appears or timeout."""
    path    = RESULTS_DIR / "r81_summary.json"
    log_p   = Path("/data/datasets/r81.log")
    deadline = time.time() + timeout_min * 60
    log(f"Waiting for R81 (timeout {timeout_min} min) …")
    while time.time() < deadline:
        if path.exists():
            data = json.loads(path.read_text())
            log("R81 complete!")
            return data
        # Show last line of r81.log to indicate activity
        if log_p.exists():
            try:
                last = log_p.read_text().splitlines()[-1]
                log(f"  R81: {last[:120]}")
            except Exception:
                pass
        time.sleep(30)
    raise TimeoutError(f"R81 did not finish within {timeout_min} minutes")


# ─── Phase 2: R82 — CG Feature Factory ───────────────────────────────────────

def _zscore_series(s: pd.Series, window: int) -> pd.Series:
    mu  = s.rolling(window, min_periods=ROLL_7).mean()
    std = s.rolling(window, min_periods=ROLL_7).std() + EPS
    return (s - mu) / std


def build_cg_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build rolling z-score / momentum features from the 5 CG parquets.
    All features keyed by (symbol, cg_date) with shift1 alignment.

    Returns:
      merged  — research frame + CG features
      feats   — list of new feature column names
    """
    log("  Loading CG parquets …")
    eps = EPS

    def load_normalize(name: str) -> Optional[pd.DataFrame]:
        p = CG_DIR / f"{name}.parquet"
        if not p.exists():
            log(f"  WARNING: {p} not found")
            return None
        d = pd.read_parquet(p)
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        d["cg_date"]   = d["timestamp"].dt.normalize()
        d = d.drop_duplicates(subset=["symbol", "cg_date"], keep="last")
        return d

    taker   = load_normalize("taker")
    oi_df   = load_normalize("oi")
    funding = load_normalize("funding")
    liq     = load_normalize("liq")
    ls      = load_normalize("ls_ratio")

    feature_frames = []
    feat_names: List[str] = []

    # ── TAKER ────────────────────────────────────────────────────────────────
    if taker is not None:
        log("  Building TAKER features …")
        t = taker[["symbol", "cg_date", "taker_buy_usd", "taker_sell_usd"]].copy()
        total = t["taker_buy_usd"] + t["taker_sell_usd"]
        t["_imb"]  = (t["taker_buy_usd"] - t["taker_sell_usd"]) / (total + eps)
        t["_flow"] = t["taker_buy_usd"] / (t["taker_sell_usd"] + eps)

        t = t.sort_values(["symbol", "cg_date"])
        rows = []
        for sym, g in t.groupby("symbol"):
            g = g.copy()
            g["cg_taker_imb_z120"]  = _zscore_series(g["_imb"],  ROLL_120)
            g["cg_taker_flow_z120"] = _zscore_series(g["_flow"].clip(-10, 10), ROLL_120)
            rows.append(g)
        t = pd.concat(rows)

        new_cols = ["cg_taker_imb_z120", "cg_taker_flow_z120"]
        feature_frames.append(
            t[["symbol", "cg_date"] + new_cols].set_index(["symbol", "cg_date"])
        )
        feat_names += new_cols
        log(f"  TAKER: {new_cols}")

    # ── LIQUIDATIONS ─────────────────────────────────────────────────────────
    if liq is not None:
        log("  Building LIQ features …")
        l = liq[["symbol", "cg_date", "liq_long_usd", "liq_short_usd"]].copy()
        total = l["liq_long_usd"] + l["liq_short_usd"]
        l["_imb"] = (l["liq_long_usd"] - l["liq_short_usd"]) / (total + eps)
        l["_log"] = np.log1p(total)

        l = l.sort_values(["symbol", "cg_date"])
        rows = []
        for sym, g in l.groupby("symbol"):
            g = g.copy()
            g["cg_liq_imb_z120"] = _zscore_series(g["_imb"], ROLL_120)
            g["cg_liq_log_z120"] = _zscore_series(g["_log"], ROLL_120)
            # spike: liq_total z-score > 2 (binary flag)
            liq_tot_z = _zscore_series(g["_log"], ROLL_120)
            g["cg_liq_spike"]    = (liq_tot_z > 2.0).astype(float)
            rows.append(g)
        l = pd.concat(rows)

        new_cols = ["cg_liq_imb_z120", "cg_liq_log_z120", "cg_liq_spike"]
        feature_frames.append(
            l[["symbol", "cg_date"] + new_cols].set_index(["symbol", "cg_date"])
        )
        feat_names += new_cols
        log(f"  LIQ: {new_cols}")

    # ── OI ───────────────────────────────────────────────────────────────────
    if oi_df is not None:
        log("  Building OI features …")
        o = oi_df[["symbol", "cg_date", "oi_close"]].copy()
        o = o.sort_values(["symbol", "cg_date"])
        rows = []
        for sym, g in o.groupby("symbol"):
            g = g.copy()
            g["_oi"]               = g["oi_close"]
            g["cg_oi_z120"]        = _zscore_series(g["_oi"], ROLL_120)
            g["cg_oi_notional_chg"] = g["_oi"].pct_change(1).clip(-1, 1)
            # surge: single-period OI change z-score > 2
            chg_z = _zscore_series(g["cg_oi_notional_chg"].fillna(0), ROLL_120)
            g["cg_oi_surge"] = (chg_z > 2.0).astype(float)
            rows.append(g)
        o = pd.concat(rows)

        new_cols = ["cg_oi_z120", "cg_oi_notional_chg", "cg_oi_surge"]
        feature_frames.append(
            o[["symbol", "cg_date"] + new_cols].set_index(["symbol", "cg_date"])
        )
        feat_names += new_cols
        log(f"  OI: {new_cols}")

    # ── FUNDING RATE ─────────────────────────────────────────────────────────
    if funding is not None:
        log("  Building FUNDING features …")
        f = funding[["symbol", "cg_date", "fr_close"]].copy()
        f = f.sort_values(["symbol", "cg_date"])
        rows = []
        for sym, g in f.groupby("symbol"):
            g = g.copy()
            g["cg_fr_z120"]       = _zscore_series(g["fr_close"], ROLL_120)
            g["cg_fr_accel"]      = g["fr_close"].diff(1)
            g["cg_fr_accel_z120"] = _zscore_series(g["cg_fr_accel"].fillna(0), ROLL_120)
            rows.append(g)
        f = pd.concat(rows)

        new_cols = ["cg_fr_z120", "cg_fr_accel", "cg_fr_accel_z120"]
        feature_frames.append(
            f[["symbol", "cg_date"] + new_cols].set_index(["symbol", "cg_date"])
        )
        feat_names += new_cols
        log(f"  FUNDING: {new_cols}")

    # ── LS RATIO ─────────────────────────────────────────────────────────────
    if ls is not None:
        log("  Building LS RATIO features …")
        s = ls[["symbol", "cg_date", "ls_ratio"]].copy()
        s = s.sort_values(["symbol", "cg_date"])
        rows = []
        for sym, g in s.groupby("symbol"):
            g = g.copy()
            g["cg_ls_z120"]     = _zscore_series(g["ls_ratio"], ROLL_120)
            g["_ls_chg"]        = g["ls_ratio"].diff(1)
            g["cg_ls_chg_z120"] = _zscore_series(g["_ls_chg"].fillna(0), ROLL_120)
            rows.append(g)
        s = pd.concat(rows)

        new_cols = ["cg_ls_z120", "cg_ls_chg_z120"]
        feature_frames.append(
            s[["symbol", "cg_date"] + new_cols].set_index(["symbol", "cg_date"])
        )
        feat_names += new_cols
        log(f"  LS: {new_cols}")

    if not feature_frames:
        raise RuntimeError("No CG feature frames built")

    # Join all feature frames
    cg_all = feature_frames[0].copy()
    for fr in feature_frames[1:]:
        cg_all = cg_all.join(fr, how="outer")
    cg_all = cg_all.reset_index()
    cg_all = cg_all.replace([np.inf, -np.inf], np.nan)
    log(f"  CG feature table: {len(cg_all):,} rows × {len(feat_names)} features")

    # Merge into research frame (shift1)
    log("  Merging into 12h frame (shift1) …")
    df2 = df.copy()
    df2["_cg_date"] = df2["timestamp"].dt.normalize() - pd.Timedelta(days=1)
    merged = df2.merge(
        cg_all.rename(columns={"cg_date": "_cg_date"}),
        on=["symbol", "_cg_date"],
        how="left",
    ).drop(columns=["_cg_date"])
    merged = merged.replace([np.inf, -np.inf], np.nan)

    # ── Coverage gate ────────────────────────────────────────────────────────
    tz         = merged["timestamp"].dt.tz
    test_slice = merged[merged["timestamp"] >= pd.Timestamp(TEST_START, tz=tz)]
    total_rows = len(test_slice)
    passed = []
    for feat in feat_names:
        if feat not in merged.columns:
            continue
        cov = test_slice[feat].notna().mean()
        if cov >= COVERAGE_THR:
            passed.append(feat)
            log(f"    ✓ {feat:<30}: coverage {cov:.1%}")
        else:
            log(f"    ✗ DROPPED {feat:<26}: coverage {cov:.1%} < {COVERAGE_THR:.0%}")

    log(f"  Features passing coverage gate: {len(passed)}/{len(feat_names)}")
    return merged, passed


# ─── Phase 3: R83 — IC Scan + Redundancy Gate ─────────────────────────────────

def ic_scan_features(
    merged: pd.DataFrame,
    feats: List[str],
    existing_feats: List[str],
    ic_thresh: float = 0.03,
    stability_thresh: float = 2 / 3,
    redund_thresh: float = 0.7,
) -> pd.DataFrame:
    """
    Compute IC for each feature, apply gate, return ranked table.
    """
    tz  = merged["timestamp"].dt.tz
    rows = []

    for feat in feats:
        if feat not in merged.columns:
            continue
        valid = merged[[feat, "fwd_ret_12h", "timestamp"]].dropna()
        n_obs = len(valid)
        if n_obs < 50:
            rows.append({"feature": feat, "skip": "too_few_obs"})
            continue

        # Pooled IC
        pooled_ic = float(stats.spearmanr(valid[feat], valid["fwd_ret_12h"])[0])

        # Per-window IC
        window_ics = []
        for w in CONTINUOUS_WINDOWS:
            ts_s = pd.Timestamp(w["test_start"], tz=tz)
            ts_e = pd.Timestamp(w["test_end"],   tz=tz)
            wdf  = valid[(valid["timestamp"] >= ts_s) & (valid["timestamp"] <= ts_e)]
            if len(wdf) < 50:
                window_ics.append(np.nan)
                continue
            wic = float(stats.spearmanr(wdf[feat], wdf["fwd_ret_12h"])[0])
            window_ics.append(wic if not np.isnan(wic) else 0.0)

        stability = sum(1 for ic in window_ics if not np.isnan(ic) and abs(ic) >= 0.02) / 3.0

        # Redundancy check vs existing features
        max_corr = 0.0
        for ef in existing_feats:
            if ef not in merged.columns:
                continue
            sub = merged[[feat, ef]].dropna()
            if len(sub) < 50:
                continue
            c = abs(float(stats.spearmanr(sub[feat], sub[ef])[0]))
            if c > max_corr:
                max_corr = c

        # Coverage on test period
        tz2        = merged["timestamp"].dt.tz
        test_slice = merged[merged["timestamp"] >= pd.Timestamp(TEST_START, tz=tz2)]
        coverage   = test_slice[feat].notna().mean() if feat in test_slice.columns else 0.0

        # Score
        score = abs(pooled_ic) * stability

        rows.append({
            "feature":     feat,
            "pooled_ic":   round(pooled_ic, 4),
            "w1_ic":       round(window_ics[0], 4) if not np.isnan(window_ics[0]) else None,
            "w2_ic":       round(window_ics[1], 4) if not np.isnan(window_ics[1]) else None,
            "w3_ic":       round(window_ics[2], 4) if not np.isnan(window_ics[2]) else None,
            "stability":   round(stability, 3),
            "max_corr_existing": round(max_corr, 3),
            "coverage_test": round(coverage, 3),
            "score":       round(score, 4),
            "pass_ic":     abs(pooled_ic) >= ic_thresh,
            "pass_stab":   stability >= stability_thresh,
            "pass_redund": max_corr < redund_thresh,
            "pass_cov":    coverage >= COVERAGE_THR,
            "skip":        None,
        })

    ic_df = pd.DataFrame(rows)
    ic_df["gate_pass"] = (
        ic_df["pass_ic"].fillna(False) &
        ic_df["pass_stab"].fillna(False) &
        ic_df["pass_redund"].fillna(False) &
        ic_df["pass_cov"].fillna(False)
    )
    return ic_df.sort_values("score", ascending=False).reset_index(drop=True)


# ─── Phase 4: R84 — Add-only WF test ─────────────────────────────────────────

def run_r84_experiment(
    df: pd.DataFrame,
    regime_df,
    base_feats: List[str],
    new_feats: List[str],
    label: str,
    no_rank_base: List[str],
) -> dict:
    """Train ensemble with base_feats + new_feats, simulate 4L/2S, return metrics."""
    from _research_r68_continuous_wf import train_ensemble, simulate, SEEDS, PROD_CFG
    from _research_r35_new_features import MARKET_LEVEL_FEATURES

    all_feats = base_feats + [f for f in new_feats if f not in base_feats]
    avail     = [f for f in all_feats if f in df.columns]

    # Features that should NOT be CS-ranked (market-level + raw change/accel/surge/spike)
    non_rank_patterns = {"_chg", "_accel", "_surge", "_spike", "_accel_z"}
    no_rank = list(set(no_rank_base) | {
        f for f in avail
        if any(p in f for p in non_rank_patterns)
        or f in MARKET_LEVEL_FEATURES
    })

    log(f"  Training {label}: {len(avail)} features ({len(new_feats)} new) …")
    cfg = {**PROD_CFG, "n_long": 4, "n_short": 2}

    preds = train_ensemble(df, avail, CONTINUOUS_WINDOWS, seeds=SEEDS,
                           cs_rank_exclude=no_rank)
    if preds is None:
        log(f"  WARNING: {label} — no predictions returned")
        return {"label": label, "net_sharpe": None}

    port = simulate(preds, regime_df, n_long=4, n_short=2, cfg=cfg)
    m    = portfolio_metrics(port)
    m["label"] = label
    return {"port": port, "metrics": m}


# ─── Phase 5: R85 — Bootstrap ─────────────────────────────────────────────────

def block_bootstrap_sharpe(
    rets_base: np.ndarray,
    rets_exp: np.ndarray,
    n_boot:  int = 1000,
    block:   int = 10,
    seed:    int = 42,
) -> dict:
    """
    Block bootstrap: resample blocks of length `block` from both series,
    compute Sharpe for each resample, return P(Sharpe_exp > Sharpe_base).
    """
    rng     = np.random.default_rng(seed)
    n       = min(len(rets_base), len(rets_exp))
    rb      = rets_base[:n]
    re      = rets_exp[:n]
    n_blocks = n // block

    sharpe_base_boot = []
    sharpe_exp_boot  = []

    for _ in range(n_boot):
        idx       = rng.integers(0, n - block + 1, size=n_blocks)
        block_idx = np.concatenate([np.arange(i, i + block) for i in idx])[:n]
        rb_s      = rb[block_idx]
        re_s      = re[block_idx]

        def _sh(r):
            if len(r) < 2 or r.std() < 1e-10:
                return 0.0
            return float(r.mean() / (r.std() + EPS) * np.sqrt(PERIODS_PER_YEAR))

        sharpe_base_boot.append(_sh(rb_s))
        sharpe_exp_boot.append(_sh(re_s))

    sb  = np.array(sharpe_base_boot)
    se  = np.array(sharpe_exp_boot)
    delta = se - sb

    return {
        "p_exp_better":    round(float((se > sb).mean()), 3),
        "median_delta":    round(float(np.median(delta)), 4),
        "mean_delta":      round(float(np.mean(delta)), 4),
        "p5_delta":        round(float(np.percentile(delta, 5)), 4),
        "p95_delta":       round(float(np.percentile(delta, 95)), 4),
        "base_sharpe_med": round(float(np.median(sb)), 4),
        "exp_sharpe_med":  round(float(np.median(se)), 4),
    }


# ─── MAIN ORCHESTRATION ───────────────────────────────────────────────────────

def main():
    t_start = time.time()
    log("=" * 70)
    log("  ORCHESTRATE: DeepResearch v3  R82 → R83 → R84 → R85")
    log("=" * 70)
    update_status("orchestrator", "started")

    # ─── [0] Wait for R81 ────────────────────────────────────────────────────
    r81_done = (RESULTS_DIR / "r81_summary.json").exists()
    if not r81_done:
        log("\n[0] Waiting for R81 to finish …")
        update_status("R81", "waiting")
        try:
            r81_data = wait_for_r81(timeout_min=60)
            update_status("R81", "done", r81_data.get("best_config", {}))
        except TimeoutError as e:
            log(f"  ERROR: {e}")
            log("  Continuing with R82+ without R81 best config.")
            r81_data = {}
            update_status("R81", "timeout")
    else:
        log("\n[0] R81 already done — skipping wait.")
        r81_data = json.loads((RESULTS_DIR / "r81_summary.json").read_text())

    # Print R81 outcome
    baseline_sh = None
    baseline_dd = None
    if r81_data:
        b = r81_data.get("baseline", {})
        baseline_sh = b.get("net_sharpe")
        baseline_dd = b.get("max_dd_pct")
        best = r81_data.get("best_config")
        n_acc = r81_data.get("n_accepted", 0)
        log(f"  R81 baseline: Sharpe={baseline_sh}  MaxDD={baseline_dd}%")
        if best:
            log(f"  R81 best: {best.get('label')}  ΔDD={best.get('dd_improv_pct')}%"
                f"  ΔSh={best.get('sh_delta')}")
            log(f"  R81 accepted configs: {n_acc}")
        else:
            log("  R81: no config passed acceptance criteria")

    # ─── [1] Load research frame ──────────────────────────────────────────────
    r82_path = FEAT_DIR / "frame_12h_with_cg_features.parquet"

    if r82_path.exists():
        log("\n[1] R82 frame already exists — loading …")
        merged = pd.read_parquet(r82_path)
        # Reload feat list from a separate file
        feat_list_path = RESULTS_DIR / "r82_feat_list.json"
        if feat_list_path.exists():
            cg_feats = json.loads(feat_list_path.read_text())
        else:
            cg_feats = [c for c in merged.columns if c.startswith("cg_") and c not in
                        ["cg_taker_imb", "cg_liq_intensity", "cg_liq_total",
                         "cg_liq_imbalance", "cg_liq_zscore", "cg_liq_accel",
                         "cg_taker_imb_z", "cg_ls_ratio", "cg_ls_zscore",
                         "mkt_cg_liq_total", "mkt_cg_liq_log", "mkt_cg_liq_imb"]]
        log(f"  Loaded {len(merged):,} rows, {len(cg_feats)} CG features")
        update_status("R82", "loaded_existing")
    else:
        log("\n[1] R82 — Loading base research frame …")
        update_status("R82", "running")

        from _research_r35_new_features import load_research_frame, add_r35_features
        df_base, _regime = load_research_frame()
        df_base, _       = add_r35_features(df_base)

        from _research_r47_coinglass import load_cg_daily, compute_cg_features, add_cg_features
        cg_raw   = load_cg_daily()
        cg_feats_existing = compute_cg_features(cg_raw)
        df_base, _, _ = add_cg_features(df_base, cg_feats_existing)

        log("\n[2] R82 — Building z-score / momentum features …")
        merged, cg_feats = build_cg_features(df_base)

        log(f"\n  Saving enriched frame → {r82_path}")
        merged.to_parquet(r82_path, index=False)

        feat_list_path = RESULTS_DIR / "r82_feat_list.json"
        feat_list_path.write_text(json.dumps(cg_feats))
        log(f"  Saved feature list → {feat_list_path}")
        update_status("R82", "done", {"n_feats": len(cg_feats), "feats": cg_feats})

    # ─── [2] R83 — IC Scan + Redundancy Gate ─────────────────────────────────
    r83_path = RESULTS_DIR / "r83_ic_table.csv"
    if r83_path.exists():
        log("\n[3] R83 IC table already exists — loading …")
        ic_df = pd.read_csv(r83_path)
        update_status("R83", "loaded_existing")
    else:
        log("\n[3] R83 — IC scan …")
        update_status("R83", "running")

        # Existing CG production features for redundancy check
        existing_prod = ["cg_taker_imb", "cum_funding_24h"]

        ic_df = ic_scan_features(merged, cg_feats, existing_feats=existing_prod)
        ic_df.to_csv(r83_path, index=False)
        log(f"  Saved IC table → {r83_path}")

    # Print IC table
    log("\n  IC SCAN RESULTS:")
    log(f"  {'Feature':<30} {'IC':>8} {'Stab':>7} {'MaxCorr':>8} {'Cov':>7} {'Score':>7} {'Gate':>6}")
    log(f"  {'-'*68}")
    for _, r in ic_df.iterrows():
        if r.get("skip"):
            continue
        gate = "✅" if r.get("gate_pass") else "✗"
        log(f"  {str(r['feature']):<30} {r['pooled_ic']:>8.4f} "
            f"{r['stability']:>7.2f} {r['max_corr_existing']:>8.3f} "
            f"{r['coverage_test']:>7.1%} {r['score']:>7.4f}  {gate}")

    update_status("R83", "done")

    # Pick top-2 features that pass the gate
    passed = ic_df[ic_df["gate_pass"] == True].sort_values("score", ascending=False)
    if len(passed) == 0:
        top_feats = []
        log("\n  ⚠  No CG features passed the gate. Skipping R84/R85 feature additions.")
    else:
        top_feats = list(passed["feature"].head(2))
        log(f"\n  Top features passing gate: {top_feats}")

    # ─── [3] Load base frame for R84 ─────────────────────────────────────────
    log("\n[4] R84 — Loading data for ensemble re-training …")

    from _research_r35_new_features import load_research_frame, add_r35_features, MARKET_LEVEL_FEATURES
    from _research_r47_coinglass import load_cg_daily, compute_cg_features, add_cg_features
    from _research_r68_continuous_wf import CHAMPION_FEAT_31, simulate, SEEDS, PROD_CFG

    df_base, regime_df = load_research_frame()
    df_base, _         = add_r35_features(df_base)
    cg_raw_data        = load_cg_daily()
    cg_feat_table      = compute_cg_features(cg_raw_data)
    df_base, _, _      = add_cg_features(df_base, cg_feat_table)

    base_feats = [f for f in CHAMPION_FEAT_31 if f in df_base.columns]
    no_rank_base = [f for f in base_feats if f in MARKET_LEVEL_FEATURES]

    # ─── Baseline (R68 re-run for single reference equity) ───────────────────
    r84_baseline_path = RESULTS_DIR / "r84_baseline_equity.csv"
    if r84_baseline_path.exists():
        log("  R84 baseline equity already exists — loading …")
        base_port = pd.read_csv(r84_baseline_path, parse_dates=["timestamp"])
        base_port["timestamp"] = pd.to_datetime(base_port["timestamp"], utc=True)
        base_m = portfolio_metrics(base_port)
        base_m["label"] = "R68_baseline"
        baseline_sh = base_m["net_sharpe"]
        baseline_dd = base_m["max_dd_pct"]
    else:
        log("  Running R84 baseline (R68 4L/2S) …")
        base_result = run_r84_experiment(
            df_base, regime_df,
            base_feats=base_feats,
            new_feats=[],
            label="R68_baseline",
            no_rank_base=no_rank_base,
        )
        base_port    = base_result["port"]
        base_m       = base_result["metrics"]
        baseline_sh  = base_m["net_sharpe"]
        baseline_dd  = base_m["max_dd_pct"]
        base_port.to_csv(r84_baseline_path, index=False)
        monthly_csv(base_port,    RESULTS_DIR / "r84_baseline_monthly.csv")
        quarterly_csv(base_port,  RESULTS_DIR / "r84_baseline_quarterly.csv")
        log(f"  Baseline: Sharpe={baseline_sh}  MaxDD={baseline_dd}%")

    # ─── R84 experiments ─────────────────────────────────────────────────────
    all_r84 = [base_m]

    def merge_cg_feats_into_df(df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
        """Merge R82 features into df by (symbol, timestamp) shift1."""
        feat_src = merged[["symbol", "timestamp"] + [c for c in feat_cols if c in merged.columns]].copy()
        return df.merge(feat_src, on=["symbol", "timestamp"], how="left", suffixes=("", "_r82"))

    r84_best_exp = None
    r84_best_port = None

    if len(top_feats) >= 1:
        top1 = [top_feats[0]]
        r84_exp1_path = RESULTS_DIR / "r84_exp1_equity.csv"

        if r84_exp1_path.exists():
            log(f"\n  R84 Exp1 already exists — loading …")
            exp1_port = pd.read_csv(r84_exp1_path, parse_dates=["timestamp"])
            exp1_port["timestamp"] = pd.to_datetime(exp1_port["timestamp"], utc=True)
            exp1_m = portfolio_metrics(exp1_port)
            exp1_m["label"] = f"R84_exp1_{top1[0]}"
        else:
            log(f"\n  R84 Exp1: +{top1} …")
            update_status("R84_exp1", "running", {"feats": top1})

            df_r84 = merge_cg_feats_into_df(df_base, top1)
            exp1_result = run_r84_experiment(
                df_r84, regime_df,
                base_feats=base_feats,
                new_feats=top1,
                label=f"R84_exp1_{top1[0]}",
                no_rank_base=no_rank_base,
            )
            exp1_port = exp1_result["port"]
            exp1_m    = exp1_result["metrics"]
            exp1_port.to_csv(r84_exp1_path, index=False)
            monthly_csv(exp1_port,   RESULTS_DIR / "r84_exp1_monthly.csv")
            quarterly_csv(exp1_port, RESULTS_DIR / "r84_exp1_quarterly.csv")

        all_r84.append(exp1_m)
        sh_delta  = exp1_m["net_sharpe"] - baseline_sh
        dd_delta  = exp1_m["max_dd_pct"] - baseline_dd  # negative = improvement
        dd_improv = -dd_delta / abs(baseline_dd + EPS) * 100
        exp1_ok   = (sh_delta >= 0.10) or (dd_improv >= 15 and sh_delta >= -0.05)

        log(f"\n  R84 Exp1 result: Sharpe={exp1_m['net_sharpe']}  ΔSh={sh_delta:+.3f}  "
            f"ΔDD={dd_improv:+.1f}%  {'✅ PASS' if exp1_ok else '✗ FAIL'}")
        update_status("R84_exp1", "done", {
            "net_sharpe": exp1_m["net_sharpe"], "sh_delta": sh_delta,
            "dd_improv_pct": dd_improv, "accepted": exp1_ok,
        })

        if exp1_ok:
            r84_best_exp  = exp1_m
            r84_best_port = exp1_port

            # Run Exp2 (top-2) only if top-1 passed
            if len(top_feats) >= 2:
                top2 = top_feats[:2]
                r84_exp2_path = RESULTS_DIR / "r84_exp2_equity.csv"

                if r84_exp2_path.exists():
                    log(f"\n  R84 Exp2 already exists — loading …")
                    exp2_port = pd.read_csv(r84_exp2_path, parse_dates=["timestamp"])
                    exp2_port["timestamp"] = pd.to_datetime(exp2_port["timestamp"], utc=True)
                    exp2_m = portfolio_metrics(exp2_port)
                    exp2_m["label"] = f"R84_exp2_{top2[0]}_{top2[1]}"
                else:
                    log(f"\n  R84 Exp2: +{top2} …")
                    update_status("R84_exp2", "running", {"feats": top2})
                    df_r84_2 = merge_cg_feats_into_df(df_base, top2)
                    exp2_result = run_r84_experiment(
                        df_r84_2, regime_df,
                        base_feats=base_feats,
                        new_feats=top2,
                        label=f"R84_exp2_{top2[0]}_{top2[1]}",
                        no_rank_base=no_rank_base,
                    )
                    exp2_port = exp2_result["port"]
                    exp2_m    = exp2_result["metrics"]
                    exp2_port.to_csv(r84_exp2_path, index=False)
                    monthly_csv(exp2_port,   RESULTS_DIR / "r84_exp2_monthly.csv")
                    quarterly_csv(exp2_port, RESULTS_DIR / "r84_exp2_quarterly.csv")

                all_r84.append(exp2_m)
                sh2 = exp2_m["net_sharpe"] - baseline_sh
                log(f"\n  R84 Exp2: Sharpe={exp2_m['net_sharpe']}  ΔSh={sh2:+.3f}")
                update_status("R84_exp2", "done",
                              {"net_sharpe": exp2_m["net_sharpe"], "sh_delta": sh2})

                # Best exp is whichever gave higher Sharpe
                if exp2_m["net_sharpe"] > exp1_m["net_sharpe"]:
                    r84_best_exp  = exp2_m
                    r84_best_port = exp2_port
                    log(f"  Best R84: Exp2 ({r84_best_exp['label']})")
                else:
                    log(f"  Best R84: Exp1 ({r84_best_exp['label']})")

    # Save R84 summary
    r84_summary = {
        "script":      "r84_cg_addonly_wf",
        "baseline":    base_m,
        "experiments": all_r84[1:],
        "best_exp":    r84_best_exp,
    }
    r84_sum_path = RESULTS_DIR / "r84_summary.json"
    r84_sum_path.write_text(json.dumps(r84_summary, indent=2, default=float))
    log(f"\n  Saved R84 summary → {r84_sum_path}")

    # ── R84 table ─────────────────────────────────────────────────────────────
    log("\n  R84 RESULTS:")
    log(f"  {'Label':<40} {'NetSh':>8} {'ΔSh':>8} {'MaxDD%':>8} {'Calmar':>8}")
    log(f"  {'-'*70}")
    for m in all_r84:
        delta_sh = m["net_sharpe"] - baseline_sh if m["label"] != "R68_baseline" else 0.0
        log(f"  {m['label']:<40} {m['net_sharpe']:>8.3f} {delta_sh:>+8.3f} "
            f"{m['max_dd_pct']:>7.1f}% {m['calmar']:>8.3f}")

    # ─── [4] R85 — Bootstrap ─────────────────────────────────────────────────
    r85_path = RESULTS_DIR / "r85_summary.json"
    if r85_path.exists():
        log("\n[5] R85 already done — loading …")
        r85_data = json.loads(r85_path.read_text())
        update_status("R85", "loaded_existing")
    else:
        r85_results = {}
        log("\n[5] R85 — Block bootstrap significance …")
        update_status("R85", "running")

        rets_base = base_port["net_ret"].values.astype(float)

        # R85a: R81 best vs R68
        r81_best_path = RESULTS_DIR / "r81_best_equity.csv"
        if r81_best_path.exists():
            log("  Bootstrap: R68 vs R81_best …")
            r81_eq = pd.read_csv(r81_best_path, parse_dates=["timestamp"])
            rets_r81 = r81_eq["net_ret"].values.astype(float)
            n = min(len(rets_base), len(rets_r81))
            bsr = block_bootstrap_sharpe(rets_base[:n], rets_r81[:n])
            r85_results["R81_best_vs_R68"] = bsr
            accept_r81 = bsr["p_exp_better"] > 0.8 and bsr["median_delta"] > 0.08
            log(f"  R81: P(better)={bsr['p_exp_better']}  medianΔ={bsr['median_delta']:+.4f}"
                f"  {'✅ ACCEPT' if accept_r81 else '✗ REJECT'}")
        else:
            log("  R81 best equity not found — skipping R81 bootstrap")

        # R85b: R84 best vs R68
        if r84_best_port is not None:
            log("  Bootstrap: R68 vs R84_best …")
            rets_r84 = r84_best_port["net_ret"].values.astype(float)
            n = min(len(rets_base), len(rets_r84))
            bsr2 = block_bootstrap_sharpe(rets_base[:n], rets_r84[:n])
            r85_results["R84_best_vs_R68"] = bsr2
            accept_r84 = bsr2["p_exp_better"] > 0.8 and bsr2["median_delta"] > 0.08
            log(f"  R84: P(better)={bsr2['p_exp_better']}  medianΔ={bsr2['median_delta']:+.4f}"
                f"  {'✅ ACCEPT' if accept_r84 else '✗ REJECT'}")
        else:
            log("  No R84 best port — skipping R84 bootstrap")

        r85_summary = {"script": "r85_bootstrap", "results": r85_results}
        r85_path.write_text(json.dumps(r85_summary, indent=2, default=float))
        log(f"  Saved R85 → {r85_path}")
        update_status("R85", "done")

    # ─── Final summary ────────────────────────────────────────────────────────
    runtime = time.time() - t_start
    log("\n" + "=" * 70)
    log("  ORCHESTRATION COMPLETE")
    log("=" * 70)
    log(f"  Runtime: {runtime/60:.1f} min")
    log(f"  Artifacts in {RESULTS_DIR}:")
    for f in sorted(RESULTS_DIR.glob("r8*.{csv,json}")):
        log(f"    {f.name}")

    update_status("orchestrator", "complete", {"runtime_min": round(runtime / 60, 1)})
    log("  DONE.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL ERROR:")
        traceback.print_exc()
        update_status("orchestrator", "error")
        sys.exit(1)
