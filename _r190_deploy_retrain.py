#!/usr/bin/env python3
"""R190 — DEPLOY retrain: champion + specialist boosters, fresh cutoff. VM (CPU).

Trains the two production legs as SINGLE models per seed (train < CUTOFF,
early-stop on [CUTOFF, VAL_END]) and saves boosters in the exact formats the
live loaders expect:
  champion  → results_cls_prod/{lgb_cls_seed_S.txt, xgb_cls_seed_S.json}
  specialist→ results_spec_prod/{lgb_spec_seed_S.txt, xgb_spec_seed_S.json}
Both use the VALIDATED research pipeline (_research_r68 train logic): LGB+XGB
binary on target (fwd_ret_12h>0), cross-sectional ranking (champion excludes
MARKET_LEVEL_FEATURES; specialist ranks all 5 venue feats), 30 seeds.

Champion feature set = the 30 we validated (CHAMPION_FEAT_31 minus the dropped
cg_taker_imb). Specialist = 5 venue features on the 2023-07+ covered slice.

Then a FORWARD-CHECK: load the saved boosters, predict the blended stack
(champ + 0.5*spec) on the untouched [VAL_END, FWD_END] window, report Sharpe
+ rank-IC. Finally emit deploy_config_r190.json with the frozen a1_gate_thr.

Run:  python3.11 _r190_deploy_retrain.py
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
import lightgbm as lgb
import xgboost as xgb

import _research_r68_continuous_wf as r68
from _research_r68_continuous_wf import (CHAMPION_FEAT_31, sharpe, cs_rank_cols,
                                         LGB_PARAMS, XGB_PARAMS, N_ROUNDS, EARLY_STOP)
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r121_realistic_costs import R114B_CFG
from src.costs import cost_prod_blended
from _r136_s6_retest import simulate_r136, A1_FROZEN, L_FROZEN, Q_FROZEN
import _r129_persistence_gate as r129

SEEDS30 = [0, 7, 13, 42, 99, 1, 8, 14, 43, 100,
           2, 9, 15, 44, 101, 3, 10, 16, 45, 102,
           4, 11, 17, 46, 103, 5, 12, 18, 47, 104]
VENUE = ["okx_binance_basis_z168", "okx_binance_basis_mom24",
         "coinbase_premium_z168", "coinbase_premium_mom24", "basis_range_z168"]
SPEC_START = pd.Timestamp("2023-07-01", tz="UTC")


def zscore(p, w):
    return (p - p.rolling(w, min_periods=w // 2).mean()) / (p.rolling(w, min_periods=w // 2).std() + 1e-12)


def _nw_tstat(x, lags=12):
    x = np.asarray(x, dtype=float); n = len(x)
    if n < 30: return np.nan
    d = x - x.mean(); var = d @ d / n
    for k in range(1, min(lags, n - 1) + 1):
        var += 2.0 * (1.0 - k / (lags + 1.0)) * (d[:-k] @ d[k:]) / n
    return x.mean() / (np.sqrt(max(var, 1e-18) / n) + 1e-18)


def train_leg(df_slice, feats, rank_exclude, seeds, out_dir, prefix,
              train_end, val_end):
    """Train one model per seed (LGB+XGB binary), save boosters. Returns the
    fitted booster lists for the forward check."""
    import gc
    os.makedirs(out_dir, exist_ok=True)
    tz = df_slice["timestamp"].dt.tz
    te = pd.Timestamp(train_end, tz=tz)
    ve = pd.Timestamp(val_end, tz=tz)
    rank_feats = [f for f in feats if f not in set(rank_exclude)]
    # MEMORY: keep ONLY needed columns (input frame has ~107 cols, we need ~35).
    # The full-frame copy + dual LGB/XGB matrices on 1.9M rows OOM-killed the VM.
    keep = list(dict.fromkeys(feats + ["timestamp", "symbol", "fwd_ret_12h"]))
    d = df_slice[[c for c in keep if c in df_slice.columns]].copy()
    d = cs_rank_cols(d, rank_feats)
    d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)
    for c in feats:
        if c in d.columns:
            d[c] = d[c].fillna(0)
    # Build train/val feature matrices ONCE as float32, drop the frame.
    trm = d[d["timestamp"] < te].dropna(subset=["target_binary"])
    vam = d[(d["timestamp"] >= te) & (d["timestamp"] < ve)].dropna(subset=["target_binary"])
    Xtr = trm[feats].to_numpy(dtype="float32"); ytr = trm["target_binary"].to_numpy()
    Xva = vam[feats].to_numpy(dtype="float32"); yva = vam["target_binary"].to_numpy()
    n_tr, n_va = len(trm), len(vam)
    del d, trm, vam; gc.collect()
    print(f"  [{prefix}] train={n_tr:,} (<{train_end})  val={n_va:,}", flush=True)
    # Shared datasets/matrices reused across seeds (LGB seed set via params).
    dtr_l = lgb.Dataset(Xtr, label=ytr, feature_name=feats, free_raw_data=False)
    dva_l = lgb.Dataset(Xva, label=yva, reference=dtr_l, free_raw_data=False)
    dtr_x = xgb.DMatrix(Xtr, label=ytr, feature_names=feats)
    dva_x = xgb.DMatrix(Xva, label=yva, feature_names=feats)
    lgbs, xgbs = [], []
    for s in seeds:
        m = lgb.train({**LGB_PARAMS, "seed": s}, dtr_l, num_boost_round=N_ROUNDS,
                      valid_sets=[dva_l],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                 lgb.log_evaluation(-1)])
        m.save_model(os.path.join(out_dir, f"lgb_{prefix}_seed_{s}.txt"))
        lgbs.append(m)
        mx = xgb.train({**XGB_PARAMS, "seed": s}, dtr_x, num_boost_round=N_ROUNDS,
                       evals=[(dva_x, "val")], early_stopping_rounds=EARLY_STOP,
                       verbose_eval=False)
        mx.save_model(os.path.join(out_dir, f"xgb_{prefix}_seed_{s}.json"))
        xgbs.append(mx)
        if s in (seeds[0], seeds[-1]):
            print(f"    seed {s}: lgb {m.best_iteration} / xgb {mx.best_iteration} iters", flush=True)
    del dtr_l, dva_l, dtr_x, dva_x, Xtr, Xva; gc.collect()
    return lgbs, xgbs, rank_feats


def predict_leg(df_slice, feats, rank_exclude, lgbs, xgbs, lo, hi):
    """Seed-averaged centered-rank prediction on [lo, hi)."""
    tz = df_slice["timestamp"].dt.tz
    d = df_slice[(df_slice["timestamp"] >= pd.Timestamp(lo, tz=tz)) &
                 (df_slice["timestamp"] < pd.Timestamp(hi, tz=tz))].copy()
    d = cs_rank_cols(d, [f for f in feats if f not in set(rank_exclude)])
    for c in feats:
        if c in d.columns:
            d[c] = d[c].fillna(0)
    X = d[feats]
    lp = np.mean([m.predict(X) for m in lgbs], axis=0)
    xp = np.mean([m.predict(xgb.DMatrix(X, feature_names=feats)) for m in xgbs], axis=0)
    d = d[["timestamp", "symbol", "fwd_ret_12h"]].copy()
    d["raw_prob"] = 0.5 * lp + 0.5 * xp
    d["pred"] = d.groupby("timestamp")["raw_prob"].rank(pct=True) - 0.5
    return d.rename(columns={"fwd_ret_12h": "fwd_ret"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", default="2026-05-15")
    ap.add_argument("--val-end", default="2026-05-29")
    ap.add_argument("--fwd-end", default="2026-06-18")
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()
    seeds = SEEDS30[:args.seeds]

    print("Loading frame + venue features...", flush=True)
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
    champ_rank_excl = [f for f in feats30 if f in MARKET_LEVEL_FEATURES]
    df_spec = df[df["timestamp"] >= SPEC_START].copy()

    print(f"\n=== CHAMPION ({len(feats30)} feats, {len(seeds)} seeds) → results_cls_prod ===", flush=True)
    c_lgb, c_xgb, _ = train_leg(df, feats30, champ_rank_excl, seeds,
                                "results_cls_prod", "cls", args.train_end, args.val_end)
    print(f"\n=== SPECIALIST (5 venue feats, {len(seeds)} seeds) → results_spec_prod ===", flush=True)
    s_lgb, s_xgb, _ = train_leg(df_spec, VENUE, [], seeds,
                                "results_spec_prod", "spec", args.train_end, args.val_end)

    # ── forward-check on the untouched tail ──
    print(f"\n=== FORWARD CHECK [{args.val_end} → {args.fwd_end}] ===", flush=True)
    champ_fwd = predict_leg(df, feats30, champ_rank_excl, c_lgb, c_xgb, args.val_end, args.fwd_end)
    spec_fwd = predict_leg(df_spec, VENUE, [], s_lgb, s_xgb, args.val_end, args.fwd_end)
    mg = champ_fwd.merge(spec_fwd[["timestamp", "symbol", "pred"]].rename(columns={"pred": "spred"}),
                         on=["timestamp", "symbol"], how="left")
    mg["spred"] = mg["spred"].fillna(0.0)
    mg["pred"] = mg["pred"] + 0.5 * mg["spred"]
    regime_aug = r129.add_persistence(regime_df, lookback=L_FROZEN)
    thr = r129.expanding_quantile_threshold(regime_aug[f"td_persist_{L_FROZEN}h"], Q_FROZEN, min_periods=720)
    a1_thr = float(thr.dropna().iloc[-1])
    gate = (regime_aug[f"td_persist_{L_FROZEN}h"] < thr)
    port = simulate_r136(mg, regime_aug, 4, 2, dict(R114B_CFG),
                         cutoff_on=0.9, cutoff_off=0.8, min_risk_off_periods=2,
                         cost_fn=cost_prod_blended, funding_per_12h=0.00012,
                         exec_delay_penalty=0.0003, a1_cfg=A1_FROZEN, gate_series=gate)
    ns = sharpe(port["net_ret"]); ret = ((1 + port["net_ret"]).prod() - 1) * 100
    ev = mg.dropna(subset=["fwd_ret"])
    ics = pd.Series({ts: spearmanr(g["pred"], g["fwd_ret"]).correlation
                     for ts, g in ev.groupby("timestamp") if g["pred"].nunique() > 2}).dropna()
    print(f"  FORWARD stack: Sharpe={ns:+.3f}  Ret={ret:+.1f}%  n={len(port)}  "
          f"IC={ics.mean():+.4f} t_NW={_nw_tstat(ics.values):+.2f}", flush=True)

    cfg = {"artifact": "deploy_r190", "train_end": args.train_end, "val_end": args.val_end,
           "seeds": len(seeds), "champion_feats": len(feats30),
           "a1_gate_thr": round(a1_thr, 4),
           "forward": {"sharpe": round(float(ns), 3), "ret_pct": round(float(ret), 1),
                       "ic": round(float(ics.mean()), 4),
                       "ic_t": round(float(_nw_tstat(ics.values)), 2), "n": int(len(port))},
           "deploy_cli": "--cls --leverage 4 --net-leverage 3.5 --vt --config deploy_config_r190.json"}
    with open("deploy_config_r190.json", "w") as f:
        json.dump(cfg, f, indent=2)
    for od in ("results_cls_prod", "results_spec_prod"):
        with open(os.path.join(od, "deploy_meta.json"), "w") as f:
            json.dump(cfg, f, indent=2)
    print(f"\n  frozen a1_gate_thr = {a1_thr:.4f}")
    print("R190 done.")


if __name__ == "__main__":
    main()
