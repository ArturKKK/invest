#!/usr/bin/env python3
"""R175 — third independent 10-seed draw (batch C) → s30 stack. VM ONLY.

R172 exposed the seed lottery on the LEVEL: draw A = 3.080, draw B = 2.397,
s20 = 2.596. Batch C triangulates the central estimate (mean of 3 independent
draws + the s30 variance-reduced stack). Reporting/deploy artifact — seed
count is not a tunable signal.
Needs caches from R167 (champ A), R166 (spec A), R172 (champ/spec B).
"""
from _preflight_check import check_versions
check_versions()

import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import _research_r68_continuous_wf as r68
from _research_r68_continuous_wf import CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, sharpe, train_ensemble
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r121_realistic_costs import R114B_CFG
from src.costs import cost_prod_blended
from _r136_s6_retest import simulate_r136, A1_FROZEN, L_FROZEN, Q_FROZEN
import _r129_persistence_gate as r129

SEEDS_C = [4, 11, 17, 46, 103, 5, 12, 18, 47, 104]
VENUE = ["okx_binance_basis_z168", "okx_binance_basis_mom24",
         "coinbase_premium_z168", "coinbase_premium_mom24", "basis_range_z168"]
SPEC_START = pd.Timestamp("2023-07-01", tz="UTC")
W23 = CONTINUOUS_WINDOWS[1:]


def zscore(p, w):
    return (p - p.rolling(w, min_periods=w // 2).mean()) / (p.rolling(w, min_periods=w // 2).std() + 1e-12)


print("Loading frame + building venue features...")
df, regime_df = r68.load_data()
if "timestamp" in regime_df.columns:
    regime_df = regime_df.set_index("timestamp")
bclose = df.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="first")
bclose.columns = [c.replace("/", "") for c in bclose.columns]
grid = bclose.index
panels = {}
oc = pd.read_parquet("data/raw/okx/okx_candles_1h.parquet")
oc["sym"] = oc["instId"].str.replace("-USDT-SWAP", "", regex=False) + "USDT"
oc["ts"] = pd.to_datetime(pd.to_numeric(oc["ts"]), unit="ms", utc=True)
okxp = oc.pivot_table(index="ts", columns="sym", values="close", aggfunc="first").astype(float).reindex(grid)
com = [c for c in okxp.columns if c in bclose.columns]
vb = okxp[com] / bclose[com] - 1
panels["okx_binance_basis_z168"] = zscore(vb, 168)
panels["okx_binance_basis_mom24"] = vb - vb.shift(24)
del oc, okxp, vb
cb = pd.read_parquet("data/raw/coinbase/coinbase_candles_1h.parquet")
cb["sym"] = cb["product"].str.replace("-USD", "", regex=False) + "USDT"
cb["tsx"] = pd.to_datetime(pd.to_numeric(cb["ts"], errors="coerce"), unit="s", utc=True)
cbp = cb.pivot_table(index="tsx", columns="sym", values="close", aggfunc="first").astype(float).reindex(grid)
com = [c for c in cbp.columns if c in bclose.columns]
prem = cbp[com] / bclose[com] - 1
panels["coinbase_premium_z168"] = zscore(prem, 168)
panels["coinbase_premium_mom24"] = prem - prem.shift(24)
del cb, cbp, prem
pr = pd.read_parquet("data/raw/basis/premium_index_klines_1h.parquet")
pr["timestamp"] = pd.to_datetime(pr["timestamp"], utc=True)
rng = (pr.pivot_table(index="timestamp", columns="symbol", values="high", aggfunc="first")
       - pr.pivot_table(index="timestamp", columns="symbol", values="low", aggfunc="first")).reindex(grid)
panels["basis_range_z168"] = zscore(rng, 168)
del pr, rng
df["bsym"] = df["symbol"].str.replace("/", "", regex=False)
for name, p in panels.items():
    out = p.astype("float32").reset_index()
    idc = out.columns[0]
    out = out.melt(id_vars=idc, var_name="bsym", value_name=name).rename(columns={idc: "timestamp"})
    df = df.merge(out, on=["timestamp", "bsym"], how="left")
panels.clear()

feats30 = [f for f in CHAMPION_FEAT_31 if f in df.columns and f != "cg_taker_imb"]
df_spec = df[df["timestamp"] >= SPEC_START].copy()
regime_aug = r129.add_persistence(regime_df, lookback=L_FROZEN)
thr = r129.expanding_quantile_threshold(regime_aug[f"td_persist_{L_FROZEN}h"], Q_FROZEN, min_periods=720)
gate = (regime_aug[f"td_persist_{L_FROZEN}h"] < thr)


def run_gated(preds, label):
    port = simulate_r136(preds, regime_aug, 4, 2, dict(R114B_CFG),
                         cutoff_on=0.9, cutoff_off=0.8, min_risk_off_periods=2,
                         cost_fn=cost_prod_blended, funding_per_12h=0.00012,
                         exec_delay_penalty=0.0003, a1_cfg=A1_FROZEN, gate_series=gate)
    ns = sharpe(port["net_ret"])
    ret = ((1 + port["net_ret"]).prod() - 1) * 100
    dd = ((1 + port["net_ret"]).cumprod() / (1 + port["net_ret"]).cumprod().cummax() - 1).min() * 100
    print(f"  {label:38s} Net={ns:+.3f}  Ret={ret:+.1f}%  DD={dd:+.1f}%  n={len(port)}", flush=True)
    return ns, port


print("\n=== TRAIN batch C ===", flush=True)
champ_c = train_ensemble(df, feats30, W23, seeds=SEEDS_C,
                         cs_rank_exclude=[f for f in feats30 if f in MARKET_LEVEL_FEATURES])
champ_c.to_parquet("cache/r175_champ30_s10c_w23_preds.parquet", index=False)
spec_c = train_ensemble(df_spec, VENUE, W23, seeds=SEEDS_C, cs_rank_exclude=[])
spec_c.to_parquet("cache/r175_spec_venue5_s10c_preds.parquet", index=False)
print("batch C cached", flush=True)

champ_a = pd.read_parquet("cache/r167_champ30_s10_w23_preds.parquet")
spec_a = pd.read_parquet("cache/r166_spec_venue5_s10_preds.parquet")
champ_b = pd.read_parquet("cache/r172_champ30_s10b_w23_preds.parquet")
spec_b = pd.read_parquet("cache/r172_spec_venue5_s10b_preds.parquet")


def avg_preds(frames):
    m = frames[0][["timestamp", "symbol", "pred", "fwd_ret"]].rename(columns={"pred": "p0"})
    for i, f in enumerate(frames[1:], 1):
        m = m.merge(f[["timestamp", "symbol", "pred"]].rename(columns={"pred": f"p{i}"}),
                    on=["timestamp", "symbol"], how="inner")
    cols = [c for c in m.columns if c.startswith("p")]
    m["pred"] = m[cols].mean(axis=1)
    return m[["timestamp", "symbol", "pred", "fwd_ret"]]


def stack(champ, spec):
    mg = champ.merge(spec[["timestamp", "symbol", "pred"]].rename(columns={"pred": "spred"}),
                     on=["timestamp", "symbol"], how="left")
    mg["spred"] = mg["spred"].fillna(0.0)
    mg["pred"] = mg["pred"] + 0.5 * mg["spred"]
    return mg


results = {}
print("\n=== LEVEL TRIANGULATION ===")
for tag, ch, sp in (("A", champ_a, spec_a), ("B", champ_b, spec_b), ("C", champ_c, spec_c)):
    ns, _ = run_gated(stack(ch, sp), f"STACK s10-{tag}")
    results[f"s10_{tag}"] = round(float(ns), 3)
ns30, _ = run_gated(stack(avg_preds([champ_a, champ_b, champ_c]),
                          avg_preds([spec_a, spec_b, spec_c])), "STACK s30 (A+B+C)")
results["s30"] = round(float(ns30), 3)
draws = [results["s10_A"], results["s10_B"], results["s10_C"]]
results["draw_mean"] = round(float(np.mean(draws)), 3)
results["draw_std"] = round(float(np.std(draws)), 3)
print(f"\n  draws mean={results['draw_mean']} std={results['draw_std']}  s30={results['s30']}")

with open("results_r175_seeds30.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("R175 done.")
