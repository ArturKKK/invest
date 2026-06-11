#!/usr/bin/env python3
"""D6 orderbook depth dataset — 65-day alpha & execution-value audit (JUN10 snapshot)."""
import numpy as np, pandas as pd
from scipy import stats
import warnings; warnings.filterwarnings("ignore")

from _research_round7 import SYM_35

PARQ = "data_vps_d6/binance_orderbook_depth_features_JUN10.parquet"
df = pd.read_parquet(PARQ)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"])

FEATURES = ["bid_depth_top10","ask_depth_top10","imbalance_ratio","bid_depth_top5",
            "ask_depth_top5","spread_bps","bid_ask_depth_ratio","depth_total_top10",
            "bid_depth_top10_z24","ask_depth_top10_z24","imbalance_ratio_z24",
            "depth_total_top10_z24","bid_depth_top10_chg24h","ask_depth_top10_chg24h",
            "depth_total_top10_chg24h"]

# ───────────────────────── 1. DATA QUALITY ─────────────────────────
print("="*100)
print("1. DATA QUALITY")
print("="*100)
t0, t1 = df["timestamp"].min(), df["timestamp"].max()
full_grid = pd.date_range(t0, t1, freq="1h", tz="UTC")
n_expected = len(full_grid)
print(f"Range: {t0} -> {t1}  ({(t1-t0).days} days, expected hourly rows/symbol = {n_expected})")

qual = []
for sym, g in df.groupby("symbol"):
    ts = g["timestamp"]
    gaps = ts.diff().dt.total_seconds().div(3600).dropna()
    qual.append({
        "symbol": sym, "rows": len(g),
        "coverage_pct": 100*len(g)/n_expected,
        "max_gap_h": gaps.max(), "n_gaps_gt1h": int((gaps > 1.001).sum()),
        "nan_mid": int(g["mid_price"].isna().sum()),
        "nan_spread": int(g["spread_bps"].isna().sum()),
        "spread_med": g["spread_bps"].median(), "spread_p90": g["spread_bps"].quantile(0.9),
        "spread_max": g["spread_bps"].max(), "spread_neg": int((g["spread_bps"] < 0).sum()),
        "spread_zero": int((g["spread_bps"] <= 0).sum()),
    })
qual = pd.DataFrame(qual).sort_values("spread_med")
print(f"\nRows/symbol: min={qual['rows'].min()} max={qual['rows'].max()}  "
      f"coverage min={qual['coverage_pct'].min():.2f}%")
print(f"Symbols with coverage <99%: {qual[qual['coverage_pct']<99]['symbol'].tolist()}")
print(f"Max gap overall: {qual['max_gap_h'].max():.0f}h ; symbols w/ >6h gap: "
      f"{qual[qual['max_gap_h']>6][['symbol','max_gap_h','n_gaps_gt1h']].to_dict('records')}")
print(f"Negative spreads total: {qual['spread_neg'].sum()}, zero/neg: {qual['spread_zero'].sum()}")

# NaN rates per column (excluding expected z24/chg24h warmup of first 24h)
warm = df[df["timestamp"] >= t0 + pd.Timedelta(hours=25)]
nanr = warm[FEATURES + ["mid_price"]].isna().mean().sort_values(ascending=False)
print("\nNaN rates after 24h warmup (cols >0.1%):")
print(nanr[nanr > 0.001].to_string())

print("\nPer-symbol spread_bps (sorted by median) — top 10 cheapest / 10 most expensive:")
cols = ["symbol","spread_med","spread_p90","spread_max","rows","max_gap_h"]
print(qual[cols].head(10).to_string(index=False))
print("...")
print(qual[cols].tail(10).to_string(index=False))

# ───────────────────────── forward returns from mid_price ─────────────────────────
mid = df.pivot(index="timestamp", columns="symbol", values="mid_price").reindex(full_grid)
miss_pct = 100 * mid.isna().mean().mean()
print(f"\nHourly grid: {len(mid)} rows; mean missing-cell rate after reindex: {miss_pct:.2f}%")
# do NOT ffill across gaps; fwd ret only valid when both endpoints present
fwd12 = mid.shift(-12) / mid - 1.0
fwd24 = mid.shift(-24) / mid - 1.0
mom12 = mid / mid.shift(12) - 1.0   # trailing 12h momentum
print(f"fwd12 valid cells: {fwd12.notna().sum().sum()} / {fwd12.size}")

# sanity: BTC fwd ret vs known scale
b = fwd12["BTC/USDT"].dropna()
print(f"BTC fwd12: mean={b.mean()*1e4:.1f}bp std={b.std()*100:.2f}% n={len(b)}")

def melt(p, name):
    return p.stack().rename(name).reset_index().rename(columns={"level_0":"timestamp","level_1":"symbol"})

long = df.set_index(["timestamp","symbol"])[FEATURES].copy()
ret_panel = pd.concat([fwd12.stack().rename("fwd12"), fwd24.stack().rename("fwd24"),
                       mom12.stack().rename("mom12")], axis=1)
ret_panel.index.names = ["timestamp","symbol"]
panel = long.merge(ret_panel, left_index=True, right_index=True, how="left").reset_index()

# ───────────────────────── 2.+3. CROSS-SECTIONAL IC ─────────────────────────
def cs_ic(panel_sub, feat, ret_col, min_n=15):
    """Per-timestamp Spearman IC; returns series indexed by timestamp."""
    out = {}
    for ts, g in panel_sub.groupby("timestamp"):
        v = g[[feat, ret_col]].dropna()
        if len(v) < min_n: continue
        if v[feat].nunique() < 3: continue
        out[ts] = stats.spearmanr(v[feat], v[ret_col]).statistic
    return pd.Series(out)

def ic_summary(panel_sub, ret_col, label, window_bounds=None):
    rows = []
    for feat in FEATURES:
        ics = cs_ic(panel_sub, feat, ret_col)
        if len(ics) < 50:
            continue
        m, s = ics.mean(), ics.std()
        t = m / (s / np.sqrt(len(ics)))
        row = {"feature": feat, "mean_IC": m, "t_stat": t, "n_ts": len(ics)}
        if window_bounds is not None:
            for wi, (a, bnd) in enumerate(window_bounds, 1):
                wic = ics[(ics.index >= a) & (ics.index < bnd)]
                row[f"W{wi}_IC"] = wic.mean() if len(wic) > 20 else np.nan
        rows.append(row)
    res = pd.DataFrame(rows).sort_values("mean_IC", key=abs, ascending=False)
    print(f"\n--- {label} ({ret_col}) ---")
    print(res.to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    return res

# 3 sub-windows ~21.6d each
edges = pd.date_range(t0, t1, periods=4)
WB = [(edges[i], edges[i+1]) for i in range(3)]
print(f"\nSub-windows: {[(str(a.date()), str(b.date())) for a,b in WB]}")

p35 = panel[panel["symbol"].isin(SYM_35)]
in35 = sorted(set(panel["symbol"]) & set(SYM_35))
print(f"SYM_35 coverage in D6: {len(in35)}/35; missing: {sorted(set(SYM_35)-set(panel['symbol']))}")

print("\n" + "="*100)
print("2.+3. CROSS-SECTIONAL IC (Spearman, per-timestamp -> mean/t-stat) + 3-window stability")
print("="*100)
res35_12 = ic_summary(p35, "fwd12", "SYM_35", WB)
res35_24 = ic_summary(p35, "fwd24", "SYM_35", WB)
res50_12 = ic_summary(panel, "fwd12", "ALL_50", WB)
res50_24 = ic_summary(panel, "fwd24", "ALL_50", WB)

def interesting(res, wcols=("W1_IC","W2_IC","W3_IC")):
    keep = []
    for _, r in res.iterrows():
        signs = [np.sign(r[c]) for c in wcols if not np.isnan(r[c])]
        if len(signs) < 3: continue
        main = np.sign(r["mean_IC"])
        same = sum(1 for s in signs if s == main)
        if same >= 2 and abs(r["mean_IC"]) > 0.01:
            keep.append((r["feature"], r["mean_IC"], r["t_stat"], same))
    return keep

print("\nINTERESTING (same-sign >=2/3 windows AND |mean IC|>0.01), SYM_35:")
print("  fwd12:", interesting(res35_12))
print("  fwd24:", interesting(res35_24))

# ───────────────────────── 4. TIME-SERIES / MARKET-LEVEL IC ─────────────────────────
print("\n" + "="*100)
print("4. MARKET-LEVEL (TIME-SERIES) SIGNALS")
print("="*100)
spread_p = df.pivot(index="timestamp", columns="symbol", values="spread_bps").reindex(full_grid)
imb_p    = df.pivot(index="timestamp", columns="symbol", values="imbalance_ratio").reindex(full_grid)
dep_p    = df.pivot(index="timestamp", columns="symbol", values="depth_total_top10_z24").reindex(full_grid)

mkt = pd.DataFrame({
    "med_spread": spread_p[in35].median(axis=1),
    "med_imb": imb_p[in35].median(axis=1),
    "med_depth_z": dep_p[in35].median(axis=1),
})
mkt["med_spread_z24"] = (mkt["med_spread"] - mkt["med_spread"].rolling(24).mean()) / mkt["med_spread"].rolling(24).std()
mkt["xs_mean_fwd12"] = fwd12[in35].mean(axis=1)
mkt["btc_fwd12"] = fwd12["BTC/USDT"]
mkt["xs_mean_fwd24"] = fwd24[in35].mean(axis=1)

for sig in ["med_spread","med_spread_z24","med_imb","med_depth_z"]:
    for tgt in ["xs_mean_fwd12","btc_fwd12","xs_mean_fwd24"]:
        v = mkt[[sig,tgt]].dropna()
        r, p = stats.spearmanr(v[sig], v[tgt])
        # Newey-West-ish honesty: effective N for 12h-overlapping returns ~ N/12
        n_eff = len(v) / 12
        t_eff = r * np.sqrt(n_eff) / np.sqrt(max(1e-9, 1 - r**2))
        print(f"  {sig:16s} vs {tgt:14s}: rho={r:+.4f} (n={len(v)}, t_eff~{t_eff:+.2f} after overlap adj)")

# ───────────────────────── 5. ORTHOGONALITY vs MOMENTUM ─────────────────────────
print("\n" + "="*100)
print("5. ORTHOGONALITY: top D6 features vs 12h momentum (cross-sectional)")
print("="*100)
top_feats = res35_12.head(5)["feature"].tolist()
for feat in top_feats:
    cors, resid_ics = [], []
    for ts, g in p35.groupby("timestamp"):
        v = g[[feat,"mom12","fwd12"]].dropna()
        if len(v) < 15: continue
        cors.append(stats.spearmanr(v[feat], v["mom12"]).statistic)
        # residual IC: rank-orthogonalize feature against momentum
        fr = v[feat].rank(); mr = v["mom12"].rank()
        beta = np.polyfit(mr, fr, 1)[0]
        resid = fr - beta*mr
        resid_ics.append(stats.spearmanr(resid, v["fwd12"]).statistic)
    cors, resid_ics = pd.Series(cors), pd.Series(resid_ics)
    t_r = resid_ics.mean()/(resid_ics.std()/np.sqrt(len(resid_ics)))
    print(f"  {feat:28s}: corr w/ mom12 = {cors.mean():+.3f} | residual IC = "
          f"{resid_ics.mean():+.4f} (t={t_r:+.2f})")
# momentum itself as benchmark
mic = cs_ic(p35, "mom12", "fwd12")
print(f"  [benchmark] mom12 IC on fwd12: {mic.mean():+.4f} (t={mic.mean()/(mic.std()/np.sqrt(len(mic))):+.2f})")

# ───────────────────────── 6. EXECUTION / COST MODEL ─────────────────────────
print("\n" + "="*100)
print("6. EXECUTION VALUE: actual Binance spread_bps vs modeled effective costs")
print("="*100)
TIER1 = {"BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT"}
TIER3 = {"SAND/USDT","LDO/USDT","INJ/USDT","APT/USDT","ARB/USDT","GALA/USDT","FTM/USDT","MATIC/USDT"}
MODEL = {"T1": 2.4, "T2": 5.5, "T3": 10.0}           # effective bp per side
MODEL_SPREAD = {"T1": 1.0, "T2": 2.0, "T3": 5.0}     # spread component assumed in taker model

def tier(s): return "T1" if s in TIER1 else ("T3" if s in TIER3 else "T2")
q35 = qual[qual["symbol"].isin(SYM_35)].copy()
q35["tier"] = q35["symbol"].map(tier)
q35["half_spread_med"] = q35["spread_med"] / 2
q35["half_spread_p90"] = q35["spread_p90"] / 2
agg = q35.groupby("tier").agg(
    n=("symbol","count"),
    spread_med_bp=("spread_med","median"), spread_p90_bp=("spread_p90","median"),
    half_spread_med=("half_spread_med","median"))
agg["modeled_spread_bp"] = agg.index.map(MODEL_SPREAD)
agg["modeled_eff_cost"] = agg.index.map(MODEL)
print(agg.to_string(float_format=lambda x: f"{x:.2f}"))

print("\nPer-symbol (SYM_35) actual median/p90 spread, tier, modeled spread component:")
q35s = q35.sort_values(["tier","spread_med"])[["symbol","tier","spread_med","spread_p90","spread_max"]]
print(q35s.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

print("\nMis-tiered candidates (Tier2 cheaper than modeled 2bp spread, or pricier than 5bp):")
t2 = q35[q35["tier"]=="T2"]
print("  T2 with median spread <=1.2bp (Tier1-like):", t2[t2["spread_med"]<=1.2]["symbol"].tolist())
print("  T2 with median spread >=4bp  (Tier3-like):", t2[t2["spread_med"]>=4]["symbol"].tolist())
t3 = q35[q35["tier"]=="T3"]
print("  T3 with median spread <=2bp (over-penalized):", t3[t3["spread_med"]<=2]["symbol"].tolist())
t1 = q35[q35["tier"]=="T1"]
print("  T1 median spreads:", dict(zip(t1["symbol"], t1["spread_med"].round(2))))

# spread stability over time (does it spike in stress?)
ms = mkt["med_spread"].dropna()
print(f"\nMarket median spread over 65d: p10={ms.quantile(.1):.2f} med={ms.median():.2f} "
      f"p90={ms.quantile(.9):.2f} max={ms.max():.2f} bp")
print("Done.")
