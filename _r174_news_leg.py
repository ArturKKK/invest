#!/usr/bin/env python3
"""R174 — news/attention specialist leg (companion protocol). VM ONLY.

R173 screen (<= 2026-04-25): news_sent24_level t=+4.80, news_sent7d_level
t=+5.85, news_cnt7d_z336 t=-5.01 — all 3/3 thirds STRONG. The news axis is
orthogonal in KIND (attention flow, not price/positioning/venue).

Construction mirrors the venue specialist (R161/R166): LGB+XGB trained ONLY
on news features, W2/W3, per-timestamp centered rank added as a 4th leg:
    final = champ_s10 + 0.5*spec_s10 + k * news_rank,  k in {0.25, 0.5}
Arms (pre-registered): PRIMARY news3 (the STRONG trio);
SECONDARY news4 (+ sv_spot_taker_buy_ratio_z30d, PASS t=-3.87 3/3).
Gate: adopt iff same arm+k has std P>=0.85 AND alt delta>0.
PRE-GATE: spec standalone rank-IC t_NW12 >= 2 on W2W3 test.
Needs: data/sentiment/{crypto_news,spot_futures_volume}.parquet on the VM,
cache/r167_champ30_s10_w23_preds.parquet, cache/r166_spec_venue5_s10_preds.parquet.
"""
from _preflight_check import check_versions
check_versions()

import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _research_r68_continuous_wf as r68
from _research_r68_continuous_wf import CONTINUOUS_WINDOWS, sharpe, train_ensemble
from _research_r121_realistic_costs import R114B_CFG
from src.costs import cost_prod_blended
from _r136_s6_retest import simulate_r136, A1_FROZEN, L_FROZEN, Q_FROZEN
import _r129_persistence_gate as r129

SEEDS_STD = [0, 7, 13, 42, 99]
SEEDS_ALT = [1, 8, 14, 43, 100]
W23 = CONTINUOUS_WINDOWS[1:]
KGRID = [0.25, 0.5]
NEWS3 = ["news_sent24_level", "news_sent7d_level", "news_cnt7d_z336"]
ARMS = {"news3": NEWS3, "news4": NEWS3 + ["sv_taker_z30d"]}


def zscore(p, w):
    return (p - p.rolling(w, min_periods=w // 2).mean()) / (p.rolling(w, min_periods=w // 2).std() + 1e-12)


def _nw_tstat(x, lags=12):
    x = np.asarray(x, dtype=float); n = len(x)
    if n < 50: return np.nan
    d = x - x.mean(); var = d @ d / n
    for k in range(1, min(lags, n - 1) + 1):
        var += 2.0 * (1.0 - k / (lags + 1.0)) * (d[:-k] @ d[k:]) / n
    return x.mean() / (np.sqrt(max(var, 1e-18) / n) + 1e-18)


def boot_paired(a, b, n_boot=1000, block=14, seed=174):
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


print("Loading frame + building news features...")
df, regime_df = r68.load_data()
if "timestamp" in regime_df.columns:
    regime_df = regime_df.set_index("timestamp")
bclose = df.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="first")
bclose.columns = [c.replace("/", "") for c in bclose.columns]
grid = bclose.index

nw = pd.read_parquet("data/sentiment/crypto_news.parquet")
nw["timestamp"] = pd.to_datetime(nw["timestamp"], utc=True)
nw["bsym"] = nw["symbol"].str.replace("/", "", regex=False)


def npanel(col):
    p = nw.pivot_table(index="timestamp", columns="bsym", values=col, aggfunc="first")
    return p.shift(1).reindex(grid)  # +1h shift, exactly as screened


panels = {
    "news_sent24_level": npanel("news_sentiment_24h"),
    "news_sent7d_level": npanel("news_sentiment_7d"),
    "news_cnt7d_z336": zscore(npanel("news_count_7d"), 336),
}
del nw

sv = pd.read_parquet("data/sentiment/spot_futures_volume.parquet")
sv["date"] = pd.to_datetime(sv["date"], utc=True)
symcol = "symbol" if "symbol" in sv.columns else sv.columns[1]
sv["bsym"] = sv[symcol].astype(str).str.replace("/", "", regex=False)
if not sv["bsym"].str.endswith("USDT").any():
    sv["bsym"] = sv["bsym"] + "USDT"
tp = sv.pivot_table(index="date", columns="bsym", values="spot_taker_buy_ratio", aggfunc="first")
panels["sv_taker_z30d"] = zscore(tp.shift(1).reindex(grid, method="ffill"), 24 * 30)
del sv, tp

df["bsym"] = df["symbol"].str.replace("/", "", regex=False)
for name, p in panels.items():
    out = p.astype("float32").reset_index()
    idc = out.columns[0]
    out = out.melt(id_vars=idc, var_name="bsym", value_name=name).rename(columns={idc: "timestamp"})
    df = df.merge(out, on=["timestamp", "bsym"], how="left")
    cov = df[name].notna().mean() * 100
    print(f"  {name}: coverage {cov:.1f}%", flush=True)
panels.clear()

regime_aug = r129.add_persistence(regime_df, lookback=L_FROZEN)
thr = r129.expanding_quantile_threshold(regime_aug[f"td_persist_{L_FROZEN}h"], Q_FROZEN, min_periods=720)
gate = (regime_aug[f"td_persist_{L_FROZEN}h"] < thr)


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
for arm, fl in ARMS.items():
    for seeds, tag in ((SEEDS_STD, "std"), (SEEDS_ALT, "alt")):
        print(f"\n=== {arm} {tag.upper()} ===", flush=True)
        nspec = train_ensemble(df, fl, W23, seeds=seeds, cs_rank_exclude=[])
        nspec.to_parquet(f"cache/r174_news_{arm}_{tag}_preds.parquet", index=False)
        ics = [spearmanr(g["pred"], g["fwd_ret"]).correlation
               for _, g in nspec.groupby("timestamp") if g["pred"].nunique() > 2]
        ics = pd.Series([i for i in ics if not np.isnan(i)])
        t_nw = _nw_tstat(ics.values)
        dg = champ.merge(nspec[["timestamp", "symbol", "pred"]].rename(columns={"pred": "np_"}),
                         on=["timestamp", "symbol"], how="inner")
        corr = dg[["pred", "np_"]].corr().iloc[0, 1]
        print(f"  [{arm} {tag}] standalone IC={ics.mean():+.4f} t_NW12={t_nw:+.2f} corr(champ)={corr:+.3f}")
        results[f"pregate_{arm}_{tag}"] = {"t": round(float(t_nw), 2), "corr": round(float(corr), 3)}
        if t_nw < 2:
            print(f"  [{arm} {tag}] PRE-GATE FAIL — skip blends")
            continue
        mg = base.merge(nspec[["timestamp", "symbol", "pred"]].rename(columns={"pred": "np_"}),
                        on=["timestamp", "symbol"], how="left")
        mg["np_"] = mg["np_"].fillna(0.0)
        for k in KGRID:
            bl = mg.copy()
            bl["pred"] = bl["pred"] + k * bl["np_"]
            ns_b, p_b = run_gated(bl, f"stack + {k}*{arm} {tag}")
            pwin = boot_paired(p_b, p_base)
            print(f"     -> delta {ns_b - ns_base:+.3f}, P = {pwin:.3f}")
            results[f"{arm}_{tag}_k{k}"] = {"ns": round(float(ns_b), 3),
                                            "delta": round(float(ns_b - ns_base), 3),
                                            "p": round(float(pwin), 3)}

with open("results_r174_news_leg.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nGATE (pre-registered): adopt iff same arm+k has std P>=0.85 AND alt delta>0.")
print("R174 done.")
