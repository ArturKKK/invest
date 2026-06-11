#!/usr/bin/env python3
"""D6 follow-up: is depth IC a static size bet? Honest t-stats (Newey-West), decile spreads."""
import numpy as np, pandas as pd
from scipy import stats
import warnings; warnings.filterwarnings("ignore")
from _research_round7 import SYM_35

df = pd.read_parquet("data_vps_d6/binance_orderbook_depth_features_JUN10.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values(["symbol","timestamp"]).drop_duplicates(["symbol","timestamp"])
t0, t1 = df["timestamp"].min(), df["timestamp"].max()
grid = pd.date_range(t0, t1, freq="1h", tz="UTC")
mid = df.pivot(index="timestamp", columns="symbol", values="mid_price").reindex(grid)
fwd12 = mid.shift(-12)/mid - 1
live35 = [s for s in SYM_35 if s in mid.columns and mid[s].notna().sum() > 100]
print(f"live SYM_35 symbols: {len(live35)}")

def pivf(col): return df.pivot(index="timestamp", columns="symbol", values=col).reindex(grid)[live35]

# ── A. Cross-sectional rank stability of level features ──
for col in ["depth_total_top10","spread_bps","imbalance_ratio","depth_total_top10_chg24h"]:
    p = pivf(col)
    r = p.rank(axis=1)
    # rank autocorr at lags 24h, 7d, 30d
    acs = {}
    for lag, lbl in [(24,"24h"),(168,"7d"),(720,"30d")]:
        pairs = []
        for i in range(lag, len(r), 24):
            a, b = r.iloc[i], r.iloc[i-lag]
            v = pd.concat([a,b],axis=1).dropna()
            if len(v) > 20: pairs.append(stats.spearmanr(v.iloc[:,0], v.iloc[:,1]).statistic)
        acs[lbl] = np.mean(pairs)
    print(f"rank-stability {col:26s}: 24h={acs['24h']:.3f} 7d={acs['7d']:.3f} 30d={acs['30d']:.3f}")

# ── B. Newey-West adjusted t-stat for IC series of top features ──
def ic_series(col):
    p = pivf(col); f = fwd12[live35]
    out = {}
    for i in range(len(p)):
        v = pd.concat([p.iloc[i], f.iloc[i]], axis=1).dropna()
        if len(v) < 15 or v.iloc[:,0].nunique() < 3: continue
        out[p.index[i]] = stats.spearmanr(v.iloc[:,0], v.iloc[:,1]).statistic
    return pd.Series(out)

def nw_t(s, max_lag=24):
    s = s.dropna(); n = len(s); m = s.mean(); d = s - m
    var = d.var()
    for L in range(1, max_lag+1):
        w = 1 - L/(max_lag+1)
        var += 2*w*(d.autocorr(L) * d.var())
    se = np.sqrt(var/n)
    return m, m/se, n

print("\nNewey-West (24h lags) adjusted t-stats, SYM_35, fwd12:")
for col in ["depth_total_top10","spread_bps","depth_total_top10_chg24h",
            "ask_depth_top10_chg24h","imbalance_ratio","ask_depth_top10_z24"]:
    m, t, n = nw_t(ic_series(col))
    print(f"  {col:26s}: mean IC={m:+.4f}  NW t={t:+.2f}  (naive n={n})")

# ── C. Is depth IC just 'majors beat alts'? Decile L/S using STATIC depth ranking ──
med_depth_rank = pivf("depth_total_top10").median().rank(ascending=False)
top10 = med_depth_rank.nsmallest(10).index.tolist()
bot10 = med_depth_rank.nlargest(10).index.tolist()
print(f"\nStatic top-10 depth syms: {top10}")
print(f"Static bot-10 depth syms: {bot10}")
f12 = fwd12[live35]
# non-overlapping 12h periods
nz = f12.iloc[::12]
edges = pd.date_range(t0, t1, periods=4)
for wi,(a,b) in enumerate(zip(edges[:-1],edges[1:]),1):
    w = nz[(nz.index>=a)&(nz.index<b)]
    tl, bl = w[top10].mean(axis=1).mean()*1e4, w[bot10].mean(axis=1).mean()*1e4
    print(f"  W{wi} {a.date()}–{b.date()}: top10-depth mean fwd12={tl:+.1f}bp  bot10={bl:+.1f}bp  L/S={tl-bl:+.1f}bp/12h")
tl, bl = nz[top10].mean(axis=1).mean()*1e4, nz[bot10].mean(axis=1).mean()*1e4
print(f"  FULL: top10={tl:+.1f}bp bot10={bl:+.1f}bp L/S={tl-bl:+.1f}bp/12h")

# IC of the STATIC ranking (median depth over whole sample — look-ahead, diagnostic only)
static = med_depth_rank * -1  # higher = deeper
ics = []
for i in range(0, len(f12), 12):
    v = pd.concat([static, f12.iloc[i]], axis=1).dropna()
    if len(v) < 15: continue
    ics.append(stats.spearmanr(v.iloc[:,0], v.iloc[:,1]).statistic)
ics = pd.Series(ics)
print(f"STATIC depth-rank IC (non-overlap 12h): {ics.mean():+.4f} (t={ics.mean()/(ics.std()/np.sqrt(len(ics))):+.2f}, n={len(ics)})")

# dynamic-component IC: depth rank MINUS static rank (within-symbol variation only)
p = pivf("depth_total_top10").rank(axis=1)
dyn = p.sub(med_depth_rank*0 + p.median(), axis=1)  # placeholder; do per-symbol demeaned rank
dyn = p - p.mean()  # demean each symbol's rank over time -> dynamic part
ics_d = []
for i in range(0, len(f12), 12):
    v = pd.concat([dyn.iloc[i], f12.iloc[i]], axis=1).dropna()
    if len(v) < 15: continue
    ics_d.append(stats.spearmanr(v.iloc[:,0], v.iloc[:,1]).statistic)
ics_d = pd.Series(ics_d)
print(f"DYNAMIC depth-rank IC (symbol-demeaned, non-overlap): {ics_d.mean():+.4f} (t={ics_d.mean()/(ics_d.std()/np.sqrt(len(ics_d))):+.2f}, n={len(ics_d)})")

# ── D. Decile L/S economics for the best dynamic feature: depth_total_top10_chg24h ──
print("\nDecile L/S (Q5-Q1, equal-weight, non-overlapping 12h) for dynamic features:")
for col in ["depth_total_top10_chg24h","ask_depth_top10_chg24h","imbalance_ratio","ask_depth_top10_z24","spread_bps"]:
    p = pivf(col)
    rets = []
    for i in range(0, len(p)-12, 12):
        row = pd.concat([p.iloc[i], f12.iloc[i]], axis=1).dropna()
        if len(row) < 20: continue
        q = row.iloc[:,0].rank(pct=True)
        long_r = row[q >= 0.8].iloc[:,1].mean()
        short_r = row[q <= 0.2].iloc[:,1].mean()
        rets.append(long_r - short_r)
    rets = pd.Series(rets)
    ann_sharpe = rets.mean()/rets.std()*np.sqrt(730)
    print(f"  {col:26s}: mean L/S={rets.mean()*1e4:+.1f}bp/12h  t={rets.mean()/(rets.std()/np.sqrt(len(rets))):+.2f}  annSharpe~{ann_sharpe:+.2f}  n={len(rets)}")
print("Done.")
