#!/usr/bin/env python3
"""Adversarial verification of D6 JUN10 analysis. Independent code."""
import numpy as np, pandas as pd
from scipy import stats
import warnings; warnings.filterwarnings("ignore")

SYM_35 = [
    "BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT",
    "ADA/USDT","DOGE/USDT","AVAX/USDT","DOT/USDT","LINK/USDT",
    "MATIC/USDT","UNI/USDT","ATOM/USDT","LTC/USDT","NEAR/USDT",
    "FIL/USDT","APT/USDT","ARB/USDT","OP/USDT","AAVE/USDT",
    "INJ/USDT","FTM/USDT","ALGO/USDT","SAND/USDT","MANA/USDT",
    "AXS/USDT","THETA/USDT","RUNE/USDT","EGLD/USDT","XTZ/USDT",
    "FLOW/USDT","CHZ/USDT","CRV/USDT","LDO/USDT","SNX/USDT",
]
TIER1 = {"BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT"}
TIER3 = {"SAND/USDT","LDO/USDT","INJ/USDT","APT/USDT","ARB/USDT","GALA/USDT","FTM/USDT","MATIC/USDT"}

df = pd.read_parquet("/Users/a.s.tabakov/Developer/invest/data_vps_d6/binance_orderbook_depth_features_JUN10.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# ===== 0. Raw data sanity: timestamps, duplicates, alignment =====
print("== 0. RAW SANITY ==")
print("rows:", len(df), "symbols:", df["symbol"].nunique())
off_hour = (df["timestamp"].dt.minute != 0) | (df["timestamp"].dt.second != 0)
print("rows not exactly on the hour:", int(off_hour.sum()))
dups = df.duplicated(["symbol","timestamp"]).sum()
print("duplicate (symbol,timestamp) rows:", int(dups))
t0, t1 = df["timestamp"].min(), df["timestamp"].max()
nhours = int((t1 - t0).total_seconds() // 3600) + 1
print(f"range {t0} -> {t1}, grid slots = {nhours}, days = {(t1-t0).total_seconds()/86400:.1f}")
rc = df.groupby("symbol").size()
print("rows/symbol: min", rc.min(), "max", rc.max(), "=> coverage", round(100*rc.min()/nhours,2), "%")

# dead symbols
print("\n== dead-symbol check ==")
for s in ["FTM/USDT","MATIC/USDT","MKR/USDT"]:
    g = df[df["symbol"]==s]
    print(s, "rows", len(g), "valid mid", int(g["mid_price"].notna().sum()),
          "valid spread", int(g["spread_bps"].notna().sum()))
dead_rows = sum(len(df[df["symbol"]==s]) for s in ["FTM/USDT","MATIC/USDT","MKR/USDT"])
print("total dead rows:", dead_rows)

# ===== 1. Forward returns (my own construction) =====
df = df.sort_values(["symbol","timestamp"])
grid = pd.date_range(t0, t1, freq="1h", tz="UTC")
mid = df.pivot_table(index="timestamp", columns="symbol", values="mid_price", aggfunc="first").reindex(grid)
# Lookahead check on construction: fwd12[t] uses mid[t+12h] only.
fwd12 = mid.shift(-12) / mid - 1.0
fwd24 = mid.shift(-24) / mid - 1.0
mom12 = mid / mid.shift(12) - 1.0
b = fwd12["BTC/USDT"].dropna()
print(f"\nBTC fwd12 mean {b.mean()*1e4:.1f}bp std {b.std()*100:.2f}% n {len(b)}")

# gap-spanning check: how many fwd12 returns span an internal gap in that symbol's data?
# (with no ffill, a NaN at either endpoint kills the return; internal NaNs between
# endpoints don't matter for a 2-point return — so the only construction risk is
# return windows where the symbol had a data outage at the start point. Quantify gaps.)
gap_counts = mid.notna().sum()
print("symbols with <1500 valid mids:", gap_counts[gap_counts<1500].to_dict())

feats = df.pivot_table(index="timestamp", columns="symbol", values=None, aggfunc="first")

def piv(col):
    return df.pivot_table(index="timestamp", columns="symbol", values=col, aggfunc="first").reindex(grid)

live35 = [s for s in SYM_35 if s in mid.columns and mid[s].notna().sum() > 100]
print("live SYM_35:", len(live35))

# ===== 2. IC recomputation =====
def cs_ic_series(featp, retp, cols, min_n=15):
    f, r = featp[cols], retp[cols]
    out = {}
    for i in range(len(f)):
        x, y = f.iloc[i], r.iloc[i]
        m = x.notna() & y.notna()
        if m.sum() < min_n or x[m].nunique() < 3: continue
        out[f.index[i]] = stats.spearmanr(x[m], y[m]).statistic
    return pd.Series(out)

print("\n== 2. IC recomputation (SYM_35) ==")
for col, retp, lbl in [("depth_total_top10", fwd12, "fwd12"),
                       ("depth_total_top10", fwd24, "fwd24"),
                       ("depth_total_top10_chg24h", fwd12, "fwd12"),
                       ("spread_bps", fwd12, "fwd12")]:
    p = piv(col)
    ics = cs_ic_series(p, retp, live35)
    m, s, n = ics.mean(), ics.std(), len(ics)
    print(f"{col:26s} {lbl}: meanIC={m:+.4f} naive_t={m/(s/np.sqrt(n)):+.2f} n={n}")

# Newey-West t (HAC, Bartlett, 24 lags) — my own implementation on the IC series
def nw_tstat(series, L=24):
    s = series.dropna().values
    n = len(s); m = s.mean(); d = s - m
    g0 = np.dot(d, d) / n
    v = g0
    for l in range(1, L+1):
        g = np.dot(d[l:], d[:-l]) / n
        v += 2 * (1 - l/(L+1)) * g
    return m, m / np.sqrt(v / n)

print("\nNW t (24 lags), my implementation:")
for col in ["depth_total_top10", "spread_bps", "depth_total_top10_chg24h"]:
    ics = cs_ic_series(piv(col), fwd12, live35)
    m, t = nw_tstat(ics)
    print(f"{col:26s}: meanIC={m:+.4f} NW_t={t:+.2f}")

# ===== 3. static vs dynamic decomposition (independent approach) =====
print("\n== 3. static vs dynamic ==")
p = piv("depth_total_top10")[live35]
# static: per-symbol median depth over full sample (lookahead diagnostic), IC on non-overlap 12h
static_rank = p.median()
f12 = fwd12[live35]
ics_s, ics_d = [], []
ranks = p.rank(axis=1)
dyn = ranks - ranks.mean()  # symbol-demeaned rank (their definition)
for i in range(0, len(p), 12):
    y = f12.iloc[i]
    m = static_rank.notna() & y.notna()
    if m.sum() >= 15:
        ics_s.append(stats.spearmanr(static_rank[m], y[m]).statistic)
    x = dyn.iloc[i]
    m2 = x.notna() & y.notna()
    if m2.sum() >= 15:
        ics_d.append(stats.spearmanr(x[m2], y[m2]).statistic)
ics_s, ics_d = pd.Series(ics_s), pd.Series(ics_d)
print(f"STATIC  IC: {ics_s.mean():+.4f} (t={ics_s.mean()/(ics_s.std()/np.sqrt(len(ics_s))):+.2f}, n={len(ics_s)})")
print(f"DYNAMIC IC: {ics_d.mean():+.4f} (t={ics_d.mean()/(ics_d.std()/np.sqrt(len(ics_d))):+.2f}, n={len(ics_d)})")

# rank autocorr 24h
ranks_all = p.rank(axis=1)
acs = []
for i in range(24, len(ranks_all), 24):
    a, b2 = ranks_all.iloc[i], ranks_all.iloc[i-24]
    m = a.notna() & b2.notna()
    if m.sum() > 20: acs.append(stats.spearmanr(a[m], b2[m]).statistic)
print(f"depth rank autocorr 24h: {np.mean(acs):.3f}")
sp = piv("spread_bps")[live35].rank(axis=1)
acs2 = []
for i in range(24, len(sp), 24):
    a, b2 = sp.iloc[i], sp.iloc[i-24]
    m = a.notna() & b2.notna()
    if m.sum() > 20: acs2.append(stats.spearmanr(a[m], b2[m]).statistic)
print(f"spread rank autocorr 24h: {np.mean(acs2):.3f}")

# ===== 4. decile L/S for chg24h: offsets 0..11 =====
print("\n== 4. decile L/S depth_total_top10_chg24h, all 12 grid offsets ==")
pc = piv("depth_total_top10_chg24h")[live35]
for off in range(0, 12):
    rets = []
    for i in range(off, len(pc)-12, 12):
        x, y = pc.iloc[i], f12.iloc[i]
        m = x.notna() & y.notna()
        if m.sum() < 20: continue
        q = x[m].rank(pct=True)
        lr = y[m][q >= 0.8].mean(); sr = y[m][q <= 0.2].mean()
        rets.append(lr - sr)
    r = pd.Series(rets)
    t = r.mean()/(r.std()/np.sqrt(len(r)))
    flag = " <— their offset-0" if off==0 else (" <— their offset-6" if off==6 else "")
    print(f"offset {off:2d}: L/S={r.mean()*1e4:+6.1f}bp/12h t={t:+5.2f} n={len(r)}{flag}")

# W1/W2/W3 split at offset 0
edges = pd.date_range(t0, t1, periods=4)
rets_by_w = {1: [], 2: [], 3: []}
for i in range(0, len(pc)-12, 12):
    x, y = pc.iloc[i], f12.iloc[i]
    m = x.notna() & y.notna()
    if m.sum() < 20: continue
    q = x[m].rank(pct=True)
    v = (y[m][q >= 0.8].mean() - y[m][q <= 0.2].mean())
    ts = pc.index[i]
    for wi in range(3):
        if edges[wi] <= ts < edges[wi+1]: rets_by_w[wi+1].append(v)
for wi, rr in rets_by_w.items():
    rr = pd.Series(rr)
    print(f"  W{wi}: chg24h L/S = {rr.mean()*1e4:+.1f}bp/12h (n={len(rr)})")

# static depth top10-bot10 per window
top10 = static_rank.nlargest(10).index.tolist()
bot10 = static_rank.nsmallest(10).index.tolist()
nz = f12.iloc[::12]
for wi in range(3):
    w = nz[(nz.index >= edges[wi]) & (nz.index < edges[wi+1])]
    tl = w[top10].mean(axis=1).mean()*1e4; bl = w[bot10].mean(axis=1).mean()*1e4
    print(f"  W{wi+1} static depth top10-bot10 L/S = {tl-bl:+.1f}bp/12h")

# ===== 5. momentum control =====
print("\n== 5. momentum control ==")
pd_depth = piv("depth_total_top10")[live35]
mm = mom12[live35]
cors, resid_ics = [], []
for i in range(len(pd_depth)):
    x, mo, y = pd_depth.iloc[i], mm.iloc[i], f12.iloc[i]
    msk = x.notna() & mo.notna() & y.notna()
    if msk.sum() < 15: continue
    cors.append(stats.spearmanr(x[msk], mo[msk]).statistic)
    xr, mr = x[msk].rank(), mo[msk].rank()
    beta = np.polyfit(mr, xr, 1)[0]
    resid = xr - beta*mr
    resid_ics.append(stats.spearmanr(resid, y[msk]).statistic)
cors, resid_ics = pd.Series(cors), pd.Series(resid_ics)
print(f"depth_total vs mom12 corr: {cors.mean():+.3f}; residual IC: {resid_ics.mean():+.4f} "
      f"(naive t={resid_ics.mean()/(resid_ics.std()/np.sqrt(len(resid_ics))):+.2f})")
mic = cs_ic_series(mm, fwd12, live35)
print(f"mom12 IC benchmark: {mic.mean():+.4f}")
# chg24h vs momentum too (they claimed |corr|<=0.034 for top feats)
pcg = piv("depth_total_top10_chg24h")[live35]
c2 = []
for i in range(len(pcg)):
    x, mo = pcg.iloc[i], mm.iloc[i]
    msk = x.notna() & mo.notna()
    if msk.sum() < 15: continue
    c2.append(stats.spearmanr(x[msk], mo[msk]).statistic)
print(f"chg24h vs mom12 corr: {np.mean(c2):+.3f}")

# ===== 6. spreads =====
print("\n== 6. spreads (median full spread, bp) ==")
med = df.groupby("symbol")["spread_bps"].median()
for s in ["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT"]:
    print(f"  T1 {s:10s}: {med[s]:.3f}")
t2syms = [s for s in SYM_35 if s not in TIER1 and s not in TIER3]
t2med = med.reindex(t2syms).dropna()
print(f"T2 SYM_35 symbols with data: {len(t2med)} | median of medians = {t2med.median():.2f}bp | "
      f">=4bp: {(t2med>=4).sum()}/{len(t2med)}")
for s in ["THETA/USDT","SNX/USDT","EGLD/USDT","RUNE/USDT","DOGE/USDT","AAVE/USDT"]:
    print(f"  T2 {s:10s}: {med.get(s, np.nan):.2f}")
for s in ["INJ/USDT","LDO/USDT"]:
    print(f"  T3 {s:10s}: {med.get(s, np.nan):.2f}")
# ZIL tick-bound claim
if "ZIL/USDT" in med.index:
    zg = df[df["symbol"]=="ZIL/USDT"]["spread_bps"].dropna()
    print(f"  ZIL: med {zg.median():.1f}bp, p10 {zg.quantile(.1):.1f}, p90 {zg.quantile(.9):.1f}, "
          f"share exactly==min {100*(zg==zg.min()).mean():.0f}%")
print("Done.")
