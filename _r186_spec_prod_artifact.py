#!/usr/bin/env python3
"""R186 — PROD specialist artifact trainer (deploy phase 3, no coinglass needed). VM.

Trains the venue-specialist leg to the DEPLOY cutoff and saves per-seed
boosters in the exact format _blend_specialist_leg (run_trading.py R183)
loads: results_spec_prod/lgb_spec_seed_{s}.txt + xgb_spec_seed_{s}.json.

Protocol = research (_r161/_r166): LGB+XGB on the 5 venue features ONLY,
slice 2023-07-01+, features cross-sectionally ranked, binary target
fwd_ret_12h > 0, per-seed early stopping on the val window.
Deploy window: train < 2026-03-01, val 2026-03-01 .. 2026-04-30.
Seeds: all 30 (prob-averaging happens in prod by averaging booster outputs).

Also emits deploy_config_r186.json with the FROZEN GATED_A1 threshold
(expanding q0.20 of td_persist_720h as of the latest data) and a sanity
rank-IC of the trained artifact on the val window.
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
import lightgbm as lgb
import xgboost as xgb

import _research_r68_continuous_wf as r68
from _research_r68_continuous_wf import cs_rank_cols, LGB_PARAMS, XGB_PARAMS, N_ROUNDS, EARLY_STOP
from _r136_s6_retest import L_FROZEN, Q_FROZEN
import _r129_persistence_gate as r129

SEEDS30 = [0, 7, 13, 42, 99, 1, 8, 14, 43, 100,
           2, 9, 15, 44, 101, 3, 10, 16, 45, 102,
           4, 11, 17, 46, 103, 5, 12, 18, 47, 104]
VENUE = ["okx_binance_basis_z168", "okx_binance_basis_mom24",
         "coinbase_premium_z168", "coinbase_premium_mom24", "basis_range_z168"]
SPEC_START = pd.Timestamp("2023-07-01", tz="UTC")
TRAIN_END = pd.Timestamp("2026-03-01", tz="UTC")
VAL_END = pd.Timestamp("2026-04-30", tz="UTC")
OUT_DIR = "results_spec_prod"


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

dfs = df[df["timestamp"] >= SPEC_START].copy()
dfs = cs_rank_cols(dfs, VENUE)
dfs["target_binary"] = (dfs["fwd_ret_12h"] > 0).astype(int)
for c in VENUE:
    dfs[c] = dfs[c].fillna(0)
train_ = dfs[dfs["timestamp"] < TRAIN_END].dropna(subset=["target_binary"])
val_ = dfs[(dfs["timestamp"] >= TRAIN_END) & (dfs["timestamp"] < VAL_END)].dropna(subset=["target_binary"])
print(f"train: {len(train_):,} rows (< {TRAIN_END.date()}), val: {len(val_):,} rows")

os.makedirs(OUT_DIR, exist_ok=True)
val_ic = []
for seed in SEEDS30:
    p_lgb = {**LGB_PARAMS, "seed": seed}
    p_xgb = {**XGB_PARAMS, "seed": seed}
    dt = lgb.Dataset(train_[VENUE], label=train_["target_binary"])
    dv = lgb.Dataset(val_[VENUE], label=val_["target_binary"])
    m = lgb.train(p_lgb, dt, num_boost_round=N_ROUNDS, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                             lgb.log_evaluation(-1)])
    m.save_model(os.path.join(OUT_DIR, f"lgb_spec_seed_{seed}.txt"))
    dt_x = xgb.DMatrix(train_[VENUE], label=train_["target_binary"], feature_names=VENUE)
    dv_x = xgb.DMatrix(val_[VENUE], label=val_["target_binary"], feature_names=VENUE)
    mx = xgb.train(p_xgb, dt_x, num_boost_round=N_ROUNDS, evals=[(dv_x, "val")],
                   early_stopping_rounds=EARLY_STOP, verbose_eval=False)
    mx.save_model(os.path.join(OUT_DIR, f"xgb_spec_seed_{seed}.json"))
    print(f"  seed {seed}: lgb iters={m.best_iteration}, xgb iters={mx.best_iteration}", flush=True)

# sanity: artifact rank-IC on the val window (seed-averaged, like prod)
lgbs = [lgb.Booster(model_file=os.path.join(OUT_DIR, f"lgb_spec_seed_{s}.txt")) for s in SEEDS30]
xgbs = []
for s in SEEDS30:
    b = xgb.Booster(); b.load_model(os.path.join(OUT_DIR, f"xgb_spec_seed_{s}.json")); xgbs.append(b)
Xv = val_[VENUE]
lp = np.mean([m.predict(Xv) for m in lgbs], axis=0)
xp = np.mean([b.predict(xgb.DMatrix(Xv, feature_names=VENUE)) for b in xgbs], axis=0)
val2 = val_[["timestamp", "symbol", "fwd_ret_12h"]].copy()
val2["pred"] = 0.5 * lp + 0.5 * xp
ics = [spearmanr(g["pred"], g["fwd_ret_12h"]).correlation
       for _, g in val2.groupby("timestamp") if g["pred"].nunique() > 2]
ics = [i for i in ics if not np.isnan(i)]
print(f"\nartifact val IC (2026-03-01..04-30): {np.mean(ics):+.4f} (n={len(ics)})")

# frozen GATED_A1 threshold as of latest data
regime_aug = r129.add_persistence(regime_df, lookback=L_FROZEN)
thr_series = r129.expanding_quantile_threshold(
    regime_aug[f"td_persist_{L_FROZEN}h"], Q_FROZEN, min_periods=720)
a1_thr = float(thr_series.dropna().iloc[-1])
print(f"frozen a1_gate_thr (expanding q{Q_FROZEN} @ {thr_series.dropna().index[-1]}): {a1_thr:.4f}")

cfg = {"artifact": "spec_prod_r186", "seeds": SEEDS30, "features": VENUE,
       "train_end": str(TRAIN_END.date()), "val_end": str(VAL_END.date()),
       "val_ic": round(float(np.mean(ics)), 4),
       "a1_gate_thr": round(a1_thr, 4),
       "deploy_cli": "--cls --leverage 4 --net-leverage 3.5 --vt (+ config a1_gate_thr)"}
with open("deploy_config_r186.json", "w") as f:
    json.dump(cfg, f, indent=2)
with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
    json.dump(cfg, f, indent=2)
print("R186 done.")
