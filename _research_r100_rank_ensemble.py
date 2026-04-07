#!/usr/bin/env python3
"""
R100 — Rank Ensemble: combine R68 (12h) and R93 (4h) predictions.

r_combined = α * rank_cs(p_12h) + (1-α) * rank_cs(p_4h)

Grid: α ∈ {0.0, 0.25, 0.50, 0.75, 1.0}
Selection: TOP 4 / BOTTOM 2 from combined_rank → simulate with R68 costs, 12h rebalance.
Bootstrap best α vs R68 (α=1.0).

Needs:
- results/r68_predictions.parquet (saved by R97 or generated here)
- results/r93_predictions.parquet (saved by R93)
"""

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Set, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EPS = 1e-10
PPY = 2 * 365  # 12h periods per year


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def sharpe_ann(rets, ppy=PPY):
    if len(rets) < 2:
        return 0.0
    eq = (1 + rets).cumprod()
    r = eq.pct_change().dropna()
    return float(r.mean() / (r.std() + EPS) * np.sqrt(ppy))


def max_dd(rets):
    eq = (1 + rets).cumprod()
    return float((eq / eq.cummax() - 1).min())


# ── Simulate (same logic as R68) ─────────────────────────────────────────────

from _research_r68_continuous_wf import (
    PROD_CFG, _cost_for_sym, CONTINUOUS_WINDOWS,
)


def simulate_rank_ensemble(merged, regime_df, n_long=4, n_short=2, cfg=None):
    """Simulate portfolio from combined rank predictions. 12h rebalance."""
    if cfg is None:
        cfg = PROD_CFG
    trend_cutoff = cfg.get("trend_cutoff", 0.9)
    rebal_hours = cfg.get("rebal_hours", 12)
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
            remaining = grp[~grp["symbol"].isin(new_longs | new_shorts)]
            for _, r in remaining.sort_values("pred_rank").head(max(0, nl - len(new_longs))).iterrows():
                new_longs.add(r["symbol"])
            remaining2 = grp[~grp["symbol"].isin(new_longs | new_shorts)]
            for _, r in remaining2.sort_values("pred_rank", ascending=False).head(max(0, ns - len(new_shorts))).iterrows():
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


# ── Block bootstrap ───────────────────────────────────────────────────────────

def block_bootstrap(rets_base, rets_exp, n_boot=1000, block=10, seed=42):
    rng = np.random.default_rng(seed)
    n = min(len(rets_base), len(rets_exp))
    rb = np.array(rets_base[:n], dtype=float)
    re = np.array(rets_exp[:n], dtype=float)
    n_blocks = n // block

    def _sh(r):
        if len(r) < 2 or np.std(r) < EPS:
            return 0.0
        return float(np.mean(r) / (np.std(r) + EPS) * np.sqrt(PPY))

    sb_list, se_list = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n - block + 1, size=n_blocks)
        block_idx = np.concatenate([np.arange(i, i + block) for i in idx])[:n]
        sb_list.append(_sh(rb[block_idx]))
        se_list.append(_sh(re[block_idx]))

    sb, se = np.array(sb_list), np.array(se_list)
    delta = se - sb
    return {
        "p_exp_better": round(float((se > sb).mean()), 3),
        "median_delta": round(float(np.median(delta)), 4),
        "mean_delta": round(float(np.mean(delta)), 4),
        "p5_delta": round(float(np.percentile(delta, 5)), 4),
        "p95_delta": round(float(np.percentile(delta, 95)), 4),
        "base_sharpe_med": round(float(np.median(sb)), 4),
        "exp_sharpe_med": round(float(np.median(se)), 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("=" * 70)
    log("  R100 — RANK ENSEMBLE (R68 12h + R93 4h)")
    log("=" * 70)

    # ── Load predictions ──────────────────────────────────────────────────
    log("\n[0] Loading predictions ...")

    r93_pred_path = RESULTS_DIR / "r93_predictions.parquet"
    r68_pred_path = RESULTS_DIR / "r68_predictions.parquet"

    if not r93_pred_path.exists():
        log(f"  ✗ R93 predictions not found: {r93_pred_path}")
        return

    r93_preds = pd.read_parquet(r93_pred_path)
    log(f"  R93 preds: {len(r93_preds):,} rows, cols={list(r93_preds.columns)}")

    # Get R68 predictions — retrain if needed
    if r68_pred_path.exists():
        r68_preds = pd.read_parquet(r68_pred_path)
        log(f"  R68 preds: {len(r68_preds):,} rows (from cache)")
    else:
        log("  R68 predictions not cached — retraining ...")
        from _research_r68_continuous_wf import (
            load_data, train_ensemble, CONTINUOUS_WINDOWS,
            SEEDS, CHAMPION_FEAT_31, MARKET_LEVEL_FEATURES,
        )
        from _research_r22_models import cs_rank_cols
        df, regime_df_data = load_data()
        feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
        no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]
        r68_preds = train_ensemble(df, feats, CONTINUOUS_WINDOWS, seeds=SEEDS,
                                    cs_rank_exclude=no_rank)
        r68_preds.to_parquet(r68_pred_path, index=False)
        log(f"  R68 preds: {len(r68_preds):,} rows (retrained & saved)")

    # Load regime_df for simulation
    log("  Loading regime_df ...")
    from _research_r68_continuous_wf import load_data
    df_full, regime_df = load_data()
    del df_full  # free memory

    # ── Align predictions ─────────────────────────────────────────────────
    log("\n[1] Aligning predictions on common timestamps ...")

    # R68 preds: timestamp, symbol, pred, raw_prob, fwd_ret, window
    # R93 preds: same columns but fwd_ret is fwd_ret_4h
    # Both have 'pred' column (rank-based combined score)

    # We need: for each (timestamp, symbol) → pred_12h from R68, pred_4h from R93
    r68_slim = r68_preds[["timestamp", "symbol", "pred", "fwd_ret"]].rename(
        columns={"pred": "pred_12h", "fwd_ret": "fwd_ret_12h"})
    r93_slim = r93_preds[["timestamp", "symbol", "pred"]].rename(
        columns={"pred": "pred_4h"})

    merged = r68_slim.merge(r93_slim, on=["timestamp", "symbol"], how="inner")
    log(f"  R68 timestamps: {r68_preds['timestamp'].nunique()}")
    log(f"  R93 timestamps: {r93_preds['timestamp'].nunique()}")
    log(f"  Common: {merged['timestamp'].nunique()} timestamps, {len(merged):,} rows")

    if len(merged) == 0:
        log("  ✗ No common timestamps! Cannot proceed.")
        log("  Checking timestamp ranges ...")
        log(f"  R68: {r68_preds['timestamp'].min()} ... {r68_preds['timestamp'].max()}")
        log(f"  R93: {r93_preds['timestamp'].min()} ... {r93_preds['timestamp'].max()}")
        return

    # ── Grid search over α ────────────────────────────────────────────────
    log("\n[2] Grid search: α ∈ {0.0, 0.25, 0.50, 0.75, 1.0} ...")

    alphas = [0.0, 0.25, 0.50, 0.75, 1.0]
    results = []
    best_sharpe = -999
    best_port = None
    best_alpha = None

    for alpha in alphas:
        # Compute cross-sectional ranks within each timestamp
        m = merged.copy()
        m["rank_12h"] = m.groupby("timestamp")["pred_12h"].rank(pct=True)
        m["rank_4h"] = m.groupby("timestamp")["pred_4h"].rank(pct=True)
        m["pred"] = alpha * m["rank_12h"] + (1 - alpha) * m["rank_4h"]
        m["fwd_ret"] = m["fwd_ret_12h"]  # simulate on 12h forward returns

        # Simulate
        port = simulate_rank_ensemble(
            m[["timestamp", "symbol", "pred", "fwd_ret"]],
            regime_df, n_long=4, n_short=2)

        if len(port) == 0:
            log(f"  α={alpha:.2f}: NO DATA")
            continue

        net = port["net_ret"]
        s = sharpe_ann(net)
        dd = max_dd(net)
        total_ret = float((1 + net).prod() - 1) * 100
        wr = float((net > 0).mean())

        res = {
            "alpha": alpha,
            "sharpe": round(s, 3),
            "max_dd_pct": round(dd * 100, 1),
            "total_ret_pct": round(total_ret, 1),
            "win_rate": round(wr, 3),
            "n_periods": len(port),
            "calmar": round(s / (abs(dd) + EPS), 2),
        }
        results.append(res)

        tag = ""
        if alpha == 1.0:
            tag = " ← pure R68"
        elif alpha == 0.0:
            tag = " ← pure R93"

        log(f"  α={alpha:.2f}: Sharpe={s:>7.3f}  MaxDD={dd*100:>6.1f}%  "
            f"Ret={total_ret:>6.1f}%  Win={wr:.3f}  N={len(port)}{tag}")

        if s > best_sharpe:
            best_sharpe = s
            best_port = port
            best_alpha = alpha

    if best_port is None:
        log("  ✗ No valid configs!")
        return

    log(f"\n  Best: α={best_alpha:.2f}, Sharpe={best_sharpe:.3f}")

    # ── Bootstrap: best vs R68 baseline (α=1.0) ──────────────────────────
    log("\n[3] Bootstrap: best α vs R68 (α=1.0) ...")

    # Get R68-only portfolio (α=1.0)
    m_r68 = merged.copy()
    m_r68["rank_12h"] = m_r68.groupby("timestamp")["pred_12h"].rank(pct=True)
    m_r68["pred"] = m_r68["rank_12h"]
    m_r68["fwd_ret"] = m_r68["fwd_ret_12h"]
    port_r68 = simulate_rank_ensemble(
        m_r68[["timestamp", "symbol", "pred", "fwd_ret"]],
        regime_df, n_long=4, n_short=2)

    if best_alpha != 1.0 and len(port_r68) > 0:
        # Align by timestamp
        r68_rets = port_r68.set_index("timestamp")["net_ret"]
        best_rets = best_port.set_index("timestamp")["net_ret"]
        common = r68_rets.index.intersection(best_rets.index)
        log(f"  Common periods for bootstrap: {len(common)}")

        if len(common) > 50:
            bs = block_bootstrap(
                r68_rets.loc[common].values,
                best_rets.loc[common].values,
                n_boot=1000, block=10)
            log(f"  P(best > R68) = {bs['p_exp_better']}")
            log(f"  ΔSharpe: median={bs['median_delta']:.3f}, [{bs['p5_delta']:.3f}, {bs['p95_delta']:.3f}]")
            log(f"  Base(R68) Sharpe median: {bs['base_sharpe_med']:.3f}")
            log(f"  Best Sharpe median: {bs['exp_sharpe_med']:.3f}")
        else:
            bs = None
            log("  Not enough common periods for bootstrap")
    else:
        bs = None
        if best_alpha == 1.0:
            log("  Best = R68 pure, no bootstrap needed (ensemble didn't help)")

    # ── Also try with vol-scaling R93 preds ───────────────────────────────
    log("\n[4] Vol-adjusted rank ensemble ...")

    # Read R97 results for vol info
    r97_path = RESULTS_DIR / "r97_attribution.json"
    vol_match_needed = False
    if r97_path.exists():
        r97 = json.loads(r97_path.read_text())
        vol_match_needed = r97.get("vol_match_needed", False)
        log(f"  R97 says vol_match_needed: {vol_match_needed}")

    if vol_match_needed:
        # Scale R93 predictions to match R68 prediction variance
        for alpha in [0.25, 0.50]:
            m = merged.copy()
            # Normalize predictions to zero mean, unit variance per timestamp
            for col in ["pred_12h", "pred_4h"]:
                g_mean = m.groupby("timestamp")[col].transform("mean")
                g_std = m.groupby("timestamp")[col].transform("std")
                m[f"{col}_norm"] = (m[col] - g_mean) / (g_std + EPS)
            m["pred"] = alpha * m["pred_12h_norm"] + (1 - alpha) * m["pred_4h_norm"]
            m["fwd_ret"] = m["fwd_ret_12h"]
            port = simulate_rank_ensemble(
                m[["timestamp", "symbol", "pred", "fwd_ret"]],
                regime_df, n_long=4, n_short=2)
            if len(port) == 0:
                continue
            net = port["net_ret"]
            s = sharpe_ann(net)
            dd = max_dd(net)
            total_ret = float((1 + net).prod() - 1) * 100
            log(f"  α={alpha:.2f} (vol-adj): Sharpe={s:.3f}  DD={dd*100:.1f}%  Ret={total_ret:.1f}%")
    else:
        log("  Vol-match not needed, skipping vol-adjusted search")

    # ── Save results ──────────────────────────────────────────────────────
    log("\n[5] Saving ...")

    summary = {
        "script": "r100_rank_ensemble",
        "best_alpha": best_alpha,
        "best_sharpe": round(best_sharpe, 4),
        "grid_results": results,
        "bootstrap_vs_r68": bs,
        "common_timestamps": merged["timestamp"].nunique(),
        "common_rows": len(merged),
        "vol_match_needed": vol_match_needed,
        "runtime_sec": round(time.time() - t0, 1),
    }
    (RESULTS_DIR / "r100_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))

    if best_port is not None:
        best_port.to_csv(RESULTS_DIR / "r100_best_equity.csv", index=False)

    log(f"\n  Saved: r100_summary.json, r100_best_equity.csv")
    log(f"  Runtime: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    main()
