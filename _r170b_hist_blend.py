#!/usr/bin/env python3
"""R170b — HIST transformer leg, blend half. CPU VM ONLY.

Takes the GPU-trained HIST preds (cache/r170_hist_{std,alt}_preds.parquet,
copy from /data/datasets/ if shipped via S3), subsets to the champion
universe, re-ranks per timestamp (centered pct-rank), and runs the same
pre-registered gate as R168/R169 against the frozen stack:
    final = champ_s10 + 0.5*spec_s10 + k * hist_rank,  k in {0.25, 0.5}
    adopt iff std P>=0.85 AND alt delta>0 for the SAME k.
PRE-GATE: standalone rank-IC (vs 12h fwd) t_NW12 >= 2 within our universe.
Diagnostic: corr(champ.pred, hist_rank) — HIST is a different representation,
expectation is corr well below the dead legs (CatBoost 0.74, h24).
"""
from _preflight_check import check_versions
check_versions()

import json
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _research_r68_continuous_wf as r68
from _research_r68_continuous_wf import sharpe
from _research_r121_realistic_costs import R114B_CFG
from src.costs import cost_prod_blended
from _r136_s6_retest import simulate_r136, A1_FROZEN, L_FROZEN, Q_FROZEN
import _r129_persistence_gate as r129

KGRID = [0.25, 0.5]


def _nw_tstat(x, lags=12):
    x = np.asarray(x, dtype=float); n = len(x)
    if n < 50: return np.nan
    d = x - x.mean(); var = d @ d / n
    for k in range(1, min(lags, n - 1) + 1):
        var += 2.0 * (1.0 - k / (lags + 1.0)) * (d[:-k] @ d[k:]) / n
    return x.mean() / (np.sqrt(max(var, 1e-18) / n) + 1e-18)


def boot_paired(a, b, n_boot=1000, block=14, seed=170):
    m = a[["timestamp", "net_ret"]].rename(columns={"net_ret": "x"}).merge(
        b[["timestamp", "net_ret"]].rename(columns={"net_ret": "y"}), on="timestamp")
    x, y = m["x"].values, m["y"].values; n = len(x)
    rng_ = np.random.RandomState(seed); wins = 0
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, min(s + block, n))
                              for s in rng_.randint(0, n - block, size=n // block + 1)])[:n]
        sx = (x[idx].sum() / (x[idx].std() + 1e-12)) / np.sqrt(len(idx))
        sy = (y[idx].sum() / (y[idx].std() + 1e-12)) / np.sqrt(len(idx))
        wins += (sx > sy)
    return wins / n_boot


print("Loading regime + caches...")
df, regime_df = r68.load_data()
if "timestamp" in regime_df.columns:
    regime_df = regime_df.set_index("timestamp")
regime_aug = r129.add_persistence(regime_df, lookback=L_FROZEN)
thr = r129.expanding_quantile_threshold(regime_aug[f"td_persist_{L_FROZEN}h"], Q_FROZEN, min_periods=720)
gate = (regime_aug[f"td_persist_{L_FROZEN}h"] < thr)
del df


def run_gated(preds, label):
    port = simulate_r136(preds, regime_aug, 4, 2, dict(R114B_CFG),
                         cutoff_on=0.9, cutoff_off=0.8, min_risk_off_periods=2,
                         cost_fn=cost_prod_blended, funding_per_12h=0.00012,
                         exec_delay_penalty=0.0003, a1_cfg=A1_FROZEN, gate_series=gate)
    ns = sharpe(port["net_ret"])
    print(f"  {label:40s} Net={ns:+.3f}  n={len(port)}", flush=True)
    return ns, port


champ = pd.read_parquet("cache/r167_champ30_s10_w23_preds.parquet")
spec = pd.read_parquet("cache/r166_spec_venue5_s10_preds.parquet")
base = champ.merge(spec[["timestamp", "symbol", "pred"]].rename(columns={"pred": "spred"}),
                   on=["timestamp", "symbol"], how="left")
base["spred"] = base["spred"].fillna(0.0)
base["pred"] = base["pred"] + 0.5 * base["spred"]
ns_base, p_base = run_gated(base, "STACK s10 (frozen base)")

results = {"stack_base": round(float(ns_base), 3)}
for tag in ("std", "alt"):
    path = f"cache/r170_hist_{tag}_preds.parquet"
    if not os.path.exists(path):
        s3 = f"/data/datasets/r170_hist_{tag}_preds.parquet"
        if os.path.exists(s3):
            import shutil; shutil.copy(s3, path)
        else:
            print(f"MISSING {path} (and not in /data/datasets) — train half not shipped yet")
            continue
    hist = pd.read_parquet(path)
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True)
    # subset to champion universe, then centered re-rank within it
    uni = base[["timestamp", "symbol"]]
    hist = uni.merge(hist[["timestamp", "symbol", "pred_hist"]], on=["timestamp", "symbol"], how="inner")
    hist["hp"] = hist.groupby("timestamp")["pred_hist"].rank(pct=True) - 0.5
    cov = len(hist) / len(uni) * 100
    print(f"\n=== HIST leg {tag.upper()} (coverage {cov:.1f}% of stack rows) ===")
    mg = base.merge(hist[["timestamp", "symbol", "hp"]], on=["timestamp", "symbol"], how="left")
    mg["hp"] = mg["hp"].fillna(0.0)
    ev = mg.dropna(subset=["fwd_ret"])
    ics = [spearmanr(g["hp"], g["fwd_ret"]).correlation
           for _, g in ev.groupby("timestamp") if g["hp"].nunique() > 2]
    ics = pd.Series([i for i in ics if not np.isnan(i)])
    t_nw = _nw_tstat(ics.values)
    cc = champ.merge(hist[["timestamp", "symbol", "hp"]], on=["timestamp", "symbol"], how="inner")
    corr = cc[["pred", "hp"]].corr().iloc[0, 1]
    print(f"  [hist {tag}] standalone IC(12h fwd)={ics.mean():+.4f} t_NW12={t_nw:+.2f} corr(champ)={corr:+.3f}")
    results[f"pregate_{tag}"] = {"t": round(float(t_nw), 2), "corr": round(float(corr), 3),
                                 "cov_pct": round(cov, 1)}
    if t_nw < 2:
        print(f"  [hist {tag}] PRE-GATE FAIL — skip blends")
        continue
    for k in KGRID:
        bl = mg.copy()
        bl["pred"] = bl["pred"] + k * bl["hp"]
        ns_b, p_b = run_gated(bl, f"stack + {k}*hist {tag}")
        pwin = boot_paired(p_b, p_base)
        print(f"     -> delta {ns_b - ns_base:+.3f}, P(4leg>stack) = {pwin:.3f}")
        results[f"hist_{tag}_k{k}"] = {"ns": round(float(ns_b), 3),
                                       "delta": round(float(ns_b - ns_base), 3),
                                       "p": round(float(pwin), 3)}

with open("results_r170b_hist_blend.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nGATE (pre-registered): adopt iff same k has std P>=0.85 AND alt delta>0.")
print("R170b done.")
