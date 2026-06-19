#!/usr/bin/env python3
"""R191 — sanity validation (Jan-cutoff model, monthly OOS) + deploy retrain. VM CPU.

Two models, both via the proven memory-safe train_leg from R190:
  A) JAN model: train < 2026-01-01 (val 2025-11-01..2026-01-01) → results_*_jan/
     Forward-evaluated Jan→now with a MONTH-BY-MONTH breakdown (stack 1x &
     3.5x+VT, BTC monthly return for context, monthly IC) so we can SEE
     whether a flat/crash regime — not a code bug — explains the weak return.
  B) DEPLOY model: train < DEPLOY_TRAIN_END (val → DEPLOY_VAL_END) →
     results_cls_prod/ + results_spec_prod/  (this is what goes live).

Run: python3.11 _r191_validate_and_deploy.py
"""
from _preflight_check import check_versions
check_versions()

import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _research_r68_continuous_wf as r68
from _research_r68_continuous_wf import CHAMPION_FEAT_31, sharpe
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r121_realistic_costs import R114B_CFG
from src.costs import cost_prod_blended
from _r136_s6_retest import simulate_r136, A1_FROZEN, L_FROZEN, Q_FROZEN
import _r129_persistence_gate as r129
from _r190_deploy_retrain import train_leg, predict_leg, zscore, _nw_tstat, SEEDS30, VENUE, SPEC_START

PPY = 2 * 365


def vt_scale_series(net_ret):
    """R179 de-risk-only VT: trailing-30 std vs expanding median, clip[0.5,1.0],
    shifted 1 period (no lookahead). Returns a per-period scale aligned to net_ret."""
    s = pd.Series(np.asarray(net_ret, dtype=float))
    vol = s.rolling(30, min_periods=30).std()
    ref = vol.expanding(min_periods=60).median()
    return (ref / vol).clip(0.5, 1.0).shift(1).fillna(1.0).values


def monthly_report(port, frame, label):
    """Month-by-month: stack 1x & 3.5x+VT returns, BTC return, n, IC context."""
    p = port.copy()
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True)
    p = p.sort_values("timestamp").reset_index(drop=True)
    sc = vt_scale_series(p["net_ret"].values)
    p["m"] = p["timestamp"].dt.strftime("%Y-%m")
    # BTC monthly close-to-close from the frame
    btc = frame[frame["symbol"] == "BTC/USDT"][["timestamp", "close"]].copy()
    btc["timestamp"] = pd.to_datetime(btc["timestamp"], utc=True)
    btc = btc.sort_values("timestamp")
    btc["m"] = btc["timestamp"].dt.strftime("%Y-%m")
    btc_m = btc.groupby("m")["close"].agg(["first", "last"])
    btc_m["btc_ret"] = (btc_m["last"] / btc_m["first"] - 1) * 100
    print(f"\n=== {label}: monthly OOS ===")
    print(f"{'month':8s} | {'1x':>8s} | {'3.5x+VT':>9s} | {'BTC':>8s} | {'n':>4s}")
    print("-" * 50)
    p["sc"] = sc
    for m, g in p.groupby("m"):
        r1 = ((1 + g["net_ret"]).prod() - 1) * 100
        r35 = ((1 + 3.5 * g["sc"] * g["net_ret"]).prod() - 1) * 100
        b = btc_m["btc_ret"].get(m, float("nan"))
        print(f"{m:8s} | {r1:+7.1f}% | {r35:+8.1f}% | {b:+7.1f}% | {len(g):>4d}")
    print("-" * 50)
    tot1 = ((1 + p["net_ret"]).prod() - 1) * 100
    eq35 = np.cumprod(1 + 3.5 * p["sc"].values * p["net_ret"].values)
    tot35 = (eq35[-1] - 1) * 100
    dd35 = ((eq35 / np.maximum.accumulate(eq35)) - 1).min() * 100
    months = p["m"].nunique()
    print(f"TOTAL 1x: {tot1:+.1f}%  (Sharpe {sharpe(p['net_ret']):+.2f}, ann {((1+tot1/100)**(12/months)-1)*100:+.0f}%)")
    print(f"TOTAL 3.5x+VT: {tot35:+.1f}%  (maxDD {dd35:+.1f}%)")
    return {"total_1x": round(float(tot1), 1), "sharpe_1x": round(float(sharpe(p["net_ret"])), 3),
            "total_35vt": round(float(tot35), 1), "dd_35vt": round(float(dd35), 1), "n": int(len(p))}


def build_frame():
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
    return df, regime_df


def stack_forward(df, df_spec, feats30, champ_rank_excl, c_l, c_x, s_l, s_x, lo, hi):
    champ = predict_leg(df, feats30, champ_rank_excl, c_l, c_x, lo, hi)
    spec = predict_leg(df_spec, VENUE, [], s_l, s_x, lo, hi)
    mg = champ.merge(spec[["timestamp", "symbol", "pred"]].rename(columns={"pred": "spred"}),
                     on=["timestamp", "symbol"], how="left")
    mg["spred"] = mg["spred"].fillna(0.0)
    mg["pred"] = mg["pred"] + 0.5 * mg["spred"]
    return mg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jan-seeds", type=int, default=15)
    ap.add_argument("--deploy-seeds", type=int, default=30)
    args = ap.parse_args()

    print("Loading frame + venue...", flush=True)
    df, regime_df = build_frame()
    feats30 = [f for f in CHAMPION_FEAT_31 if f in df.columns and f != "cg_taker_imb"]
    champ_rank_excl = [f for f in feats30 if f in MARKET_LEVEL_FEATURES]
    df_spec = df[df["timestamp"] >= SPEC_START].copy()
    regime_aug = r129.add_persistence(regime_df, lookback=L_FROZEN)
    thr = r129.expanding_quantile_threshold(regime_aug[f"td_persist_{L_FROZEN}h"], Q_FROZEN, min_periods=720)
    gate = (regime_aug[f"td_persist_{L_FROZEN}h"] < thr)

    def run_sim(mg):
        return simulate_r136(mg, regime_aug, 4, 2, dict(R114B_CFG),
                             cutoff_on=0.9, cutoff_off=0.8, min_risk_off_periods=2,
                             cost_fn=cost_prod_blended, funding_per_12h=0.00012,
                             exec_delay_penalty=0.0003, a1_cfg=A1_FROZEN, gate_series=gate)

    results = {}

    # ── A) JAN model: train<2026-01-01, OOS Jan→now monthly ──
    js = SEEDS30[:args.jan_seeds]
    print(f"\n########## JAN MODEL (train<2025-11-01, val→2026-01-01, {len(js)} seeds) ##########", flush=True)
    # train<2025-11-01, val [2025-11-01, 2026-01-01] → Jan→now stays fully OOS
    cjl, cjx, _ = train_leg(df, feats30, champ_rank_excl, js, "results_cls_jan", "cls",
                            "2025-11-01", "2026-01-01")
    sjl, sjx, _ = train_leg(df_spec, VENUE, [], js, "results_spec_jan", "spec",
                            "2025-11-01", "2026-01-01")
    mg = stack_forward(df, df_spec, feats30, champ_rank_excl, cjl, cjx, sjl, sjx,
                       "2026-01-01", "2026-06-18")
    port = run_sim(mg)
    results["jan"] = monthly_report(port, df, "JAN model OOS")
    ev = mg.dropna(subset=["fwd_ret"])
    ics = pd.Series({ts: spearmanr(g["pred"], g["fwd_ret"]).correlation
                     for ts, g in ev.groupby("timestamp") if g["pred"].nunique() > 2}).dropna()
    print(f"JAN model OOS IC = {ics.mean():+.4f}  t_NW = {_nw_tstat(ics.values):+.2f}  (n={len(ics)})")
    results["jan"]["ic"] = round(float(ics.mean()), 4)
    results["jan"]["ic_t"] = round(float(_nw_tstat(ics.values)), 2)

    # ── B) DEPLOY model: freshest cutoff, save to prod dirs ──
    ds = SEEDS30[:args.deploy_seeds]
    print(f"\n########## DEPLOY MODEL (train<2026-04-15, {len(ds)} seeds) → results_*_prod ##########", flush=True)
    cdl, cdx, _ = train_leg(df, feats30, champ_rank_excl, ds, "results_cls_prod", "cls",
                            "2026-04-15", "2026-06-01")
    sdl, sdx, _ = train_leg(df_spec, VENUE, [], ds, "results_spec_prod", "spec",
                            "2026-04-15", "2026-06-01")
    mgd = stack_forward(df, df_spec, feats30, champ_rank_excl, cdl, cdx, sdl, sdx,
                        "2026-06-01", "2026-06-18")
    portd = run_sim(mgd)
    a1_thr = float(thr.dropna().iloc[-1])
    nd = sharpe(portd["net_ret"]); rd = ((1 + portd["net_ret"]).prod() - 1) * 100
    print(f"\nDEPLOY model fwd [2026-06-01→06-18]: Sharpe={nd:+.3f} Ret={rd:+.1f}% n={len(portd)} (short window)")
    cfg = {"artifact": "deploy_r191", "deploy_train_end": "2026-04-15", "deploy_val_end": "2026-06-01",
           "deploy_seeds": len(ds), "champion_feats": len(feats30), "a1_gate_thr": round(a1_thr, 4),
           "jan_validation": results["jan"],
           "deploy_cli": "--cls --leverage 4 --net-leverage 3.5 --vt --config deploy_config_r191.json"}
    with open("deploy_config_r191.json", "w") as f:
        json.dump(cfg, f, indent=2)
    for od in ("results_cls_prod", "results_spec_prod"):
        with open(os.path.join(od, "deploy_meta.json"), "w") as f:
            json.dump(cfg, f, indent=2)
    with open("results_r191_validation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nfrozen a1_gate_thr = {a1_thr:.4f}")
    print("R191 done.")


if __name__ == "__main__":
    main()
