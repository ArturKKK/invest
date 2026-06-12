#!/usr/bin/env python3
"""R168 — CatBoost companion leg (model-family diversity). VM ONLY.

Hypothesis: a third MODEL FAMILY (CatBoost) on the SAME champion 30 features
adds diversity the LGB+XGB ensemble lacks (oblivious trees, ordered boosting).
Added as a 4th leg on top of the frozen stack:
    final = champ_s10 + 0.5*spec_s10 + k_cb * cb_rank,  k_cb in {0.25, 0.5}
k_cb grid is pre-registered; adoption rule pre-registered:
    adopt iff (std batch P(blend>stack) >= 0.85) AND (alt batch delta > 0)
    for the SAME k_cb.
PRE-GATE: cb standalone rank-IC t_NW12 >= 2 on W2W3 test.
Diagnostic: corr(champ.pred, cb.pred) — expect HIGH (same features); the bar
for adoption is therefore the paired test, not the IC.
Needs cache/r167_champ30_s10_w23_preds.parquet (R167 runs first in the chain).
"""
from _preflight_check import check_versions
check_versions()

import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from catboost import CatBoostClassifier

import _research_r68_continuous_wf as r68
from _research_r68_continuous_wf import CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, sharpe, cs_rank_cols
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r121_realistic_costs import R114B_CFG
from src.costs import cost_prod_blended
from _r136_s6_retest import simulate_r136, A1_FROZEN, L_FROZEN, Q_FROZEN
import _r129_persistence_gate as r129

SEEDS_STD = [0, 7, 13, 42, 99]
SEEDS_ALT = [1, 8, 14, 43, 100]
W23 = CONTINUOUS_WINDOWS[1:]
KGRID = [0.25, 0.5]


def _nw_tstat(x, lags=12):
    x = np.asarray(x, dtype=float); n = len(x)
    if n < 50: return np.nan
    d = x - x.mean(); var = d @ d / n
    for k in range(1, min(lags, n - 1) + 1):
        var += 2.0 * (1.0 - k / (lags + 1.0)) * (d[:-k] @ d[k:]) / n
    return x.mean() / (np.sqrt(max(var, 1e-18) / n) + 1e-18)


def boot_paired(a, b, n_boot=1000, block=14, seed=168):
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


def train_cb(df, feats, windows, seeds, cs_rank_exclude=None):
    """Mirror of r68.train_ensemble, single CatBoost family.

    Same slicing, same cs-ranking, same binary target, same seed-averaging and
    per-timestamp centered pct-rank output, so the leg is protocol-identical.
    """
    avail = [f for f in feats if f in df.columns]
    rank_exclude = set(cs_rank_exclude or [])
    rank_feats = [f for f in avail if f not in rank_exclude]
    tz = df["timestamp"].dt.tz
    all_cb = []
    for seed in seeds:
        for w in windows:
            te_end = pd.Timestamp(w["test_end"], tz=tz)
            te_start = pd.Timestamp(w["test_start"], tz=tz)
            tr_end = pd.Timestamp(w["train_end"], tz=tz)
            va_start = pd.Timestamp(w["val_start"], tz=tz)
            va_end = pd.Timestamp(w["val_end"], tz=tz)
            train_ = df[df["timestamp"] < tr_end].copy()
            val_ = df[(df["timestamp"] >= va_start) & (df["timestamp"] < va_end)].copy()
            test_ = df[(df["timestamp"] >= te_start) & (df["timestamp"] <= te_end)].copy()
            if len(train_) < 5000 or len(test_) < 200: continue
            if rank_feats:
                train_ = cs_rank_cols(train_, rank_feats)
                val_ = cs_rank_cols(val_, rank_feats)
                test_ = cs_rank_cols(test_, rank_feats)
            for d in [train_, val_, test_]:
                d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)
            for col in avail:
                for d in [train_, val_, test_]:
                    if d[col].isna().any(): d[col] = d[col].fillna(0)
            tr = train_[avail + ["target_binary"]].dropna()
            va = val_[avail + ["target_binary"]].dropna()
            te = test_[avail + ["target_binary", "timestamp", "symbol"]].dropna()
            fwd = test_[["timestamp", "symbol", "fwd_ret_12h"]].rename(
                columns={"fwd_ret_12h": "fwd_ret"}).dropna()
            for d in [tr, va, te]:
                d.replace([np.inf, -np.inf], np.nan, inplace=True)
            tr, va, te = tr.dropna(), va.dropna(), te.dropna()
            if len(te) == 0: continue
            m = CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6,
                                   l2_leaf_reg=3.0, random_seed=seed,
                                   early_stopping_rounds=40, verbose=False,
                                   allow_writing_files=False)
            m.fit(tr[avail], tr["target_binary"], eval_set=(va[avail], va["target_binary"]))
            rec = te[["timestamp", "symbol"]].copy()
            rec["pred_cb"] = m.predict_proba(te[avail])[:, 1]
            rec = rec.merge(fwd, on=["timestamp", "symbol"], how="inner")
            rec["window"] = w["name"]; rec["seed"] = seed
            all_cb.append(rec)
            print(f"    cb {w['name']}/s{seed}: train={len(tr):,} test={len(te):,}", flush=True)
    if not all_cb: return None
    cb_df = pd.concat(all_cb)
    avg = cb_df.groupby(["timestamp", "symbol"]).agg(
        pred_cb=("pred_cb", "mean"), fwd_ret=("fwd_ret", "first"),
        window=("window", "first")).reset_index()
    avg["pred"] = avg.groupby("timestamp")["pred_cb"].rank(pct=True) - 0.5
    return avg[["timestamp", "symbol", "pred", "fwd_ret", "window"]]


print("Loading frame...")
df, regime_df = r68.load_data()
if "timestamp" in regime_df.columns:
    regime_df = regime_df.set_index("timestamp")
regime_aug = r129.add_persistence(regime_df, lookback=L_FROZEN)
thr = r129.expanding_quantile_threshold(regime_aug[f"td_persist_{L_FROZEN}h"], Q_FROZEN, min_periods=720)
gate = (regime_aug[f"td_persist_{L_FROZEN}h"] < thr)

feats30 = [f for f in CHAMPION_FEAT_31 if f in df.columns and f != "cg_taker_imb"]
no_rank = [f for f in feats30 if f in MARKET_LEVEL_FEATURES]


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
for seeds, tag in ((SEEDS_STD, "std"), (SEEDS_ALT, "alt")):
    print(f"\n=== CatBoost leg {tag.upper()} ===", flush=True)
    cb = train_cb(df, feats30, W23, seeds, cs_rank_exclude=no_rank)
    cb.to_parquet(f"cache/r168_cb30_{tag}_preds.parquet", index=False)
    ics = [spearmanr(g["pred"], g["fwd_ret"]).correlation
           for _, g in cb.groupby("timestamp") if g["pred"].nunique() > 2]
    ics = pd.Series([i for i in ics if not np.isnan(i)])
    t_nw = _nw_tstat(ics.values)
    dg = champ.merge(cb[["timestamp", "symbol", "pred"]].rename(columns={"pred": "cbp"}),
                     on=["timestamp", "symbol"], how="inner")
    corr = dg[["pred", "cbp"]].corr().iloc[0, 1]
    print(f"  [cb {tag}] standalone IC={ics.mean():+.4f} t_NW12={t_nw:+.2f} corr(champ)={corr:+.3f}")
    results[f"pregate_{tag}"] = {"t": round(float(t_nw), 2), "corr": round(float(corr), 3)}
    if t_nw < 2:
        print(f"  [cb {tag}] PRE-GATE FAIL — skip blends")
        continue
    mg = base.merge(cb[["timestamp", "symbol", "pred"]].rename(columns={"pred": "cbp"}),
                    on=["timestamp", "symbol"], how="left")
    mg["cbp"] = mg["cbp"].fillna(0.0)
    for k in KGRID:
        bl = mg.copy()
        bl["pred"] = bl["pred"] + k * bl["cbp"]
        ns_b, p_b = run_gated(bl, f"stack + {k}*cb {tag}")
        pwin = boot_paired(p_b, p_base)
        print(f"     -> delta {ns_b - ns_base:+.3f}, P(4leg>stack) = {pwin:.3f}")
        results[f"cb_{tag}_k{k}"] = {"ns": round(float(ns_b), 3),
                                     "delta": round(float(ns_b - ns_base), 3),
                                     "p": round(float(pwin), 3)}

with open("results_r168_cb_leg.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nGATE (pre-registered): adopt iff same k has std P>=0.85 AND alt delta>0.")
print("R168 done.")
