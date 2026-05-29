"""R128c — D6 orderbook IC scan vs 12h fwd return computed from mid_price.

Self-contained: uses only features_ob.parquet (no main feature parquet needed).
Computes per-symbol forward 12h return from mid_price, then IC of each OB
feature. Reports both Pearson IC, Spearman RankIC, ICIR.

Usage:
    python _r128_d6_ic_scan.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

PARQUET = Path("data_vps_d6/features_ob.parquet")
HORIZONS_H = [1, 4, 12, 24]
FEATURES = [
    "imbalance_ratio", "spread_bps", "bid_ask_depth_ratio",
    "depth_total_top10", "bid_depth_top10", "ask_depth_top10",
    "bid_depth_top10_z24", "ask_depth_top10_z24",
    "imbalance_ratio_z24", "depth_total_top10_z24",
    "bid_depth_top10_chg24h", "ask_depth_top10_chg24h",
    "depth_total_top10_chg24h",
]

def main() -> None:
    df = pd.read_parquet(PARQUET)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    print(f"Loaded: {len(df):,} rows, {df['symbol'].nunique()} symbols, "
          f"{df['timestamp'].min()} → {df['timestamp'].max()}")

    # Per-symbol forward return at multiple horizons
    df = df.set_index(["symbol", "timestamp"])
    for h in HORIZONS_H:
        # data is hourly snapshots -> shift by h periods within each symbol
        df[f"fwd_ret_{h}h"] = (
            df.groupby(level="symbol")["mid_price"].shift(-h) / df["mid_price"] - 1.0
        )
    df = df.reset_index()

    rows = []
    for h in HORIZONS_H:
        target = f"fwd_ret_{h}h"
        for feat in FEATURES:
            sub = df[[feat, target, "timestamp"]].dropna()
            if len(sub) < 500:
                continue
            ic = sub[feat].corr(sub[target])
            rho, _ = spearmanr(sub[feat], sub[target])
            # daily IC for ICIR
            sub2 = sub.assign(day=sub["timestamp"].dt.floor("1D"))
            daily = sub2.groupby("day").apply(
                lambda g: g[feat].corr(g[target]) if len(g) > 30 else np.nan
            ).dropna()
            icir = daily.mean() / daily.std() if len(daily) > 3 and daily.std() > 0 else np.nan
            rows.append({
                "feature": feat, "horizon_h": h, "IC": ic, "RankIC": rho,
                "ICIR": icir, "n_obs": len(sub), "n_days": len(daily),
            })
    res = pd.DataFrame(rows).sort_values(["horizon_h", "IC"], key=lambda x: x if x.name != "IC" else x.abs(), ascending=[True, False])
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 200)
    print()
    print(res.to_string(index=False, float_format=lambda v: f"{v:+.4f}" if isinstance(v, float) else v))
    out = Path("data_vps_d6/_r128_ic_result.csv")
    res.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Pretty top hits
    print("\n=== TOP |IC| per horizon ===")
    for h in HORIZONS_H:
        top = res[res["horizon_h"] == h].assign(absIC=lambda d: d["IC"].abs()).sort_values("absIC", ascending=False).head(5)
        print(f"\nhorizon {h}h:")
        for _, r in top.iterrows():
            flag = " ★" if abs(r["IC"]) > 0.02 else ""
            print(f"  {r['feature']:<28s} IC={r['IC']:+.4f}  RankIC={r['RankIC']:+.4f}  ICIR={r['ICIR']:+.3f}  n={int(r['n_obs']):>5d}{flag}")

if __name__ == "__main__":
    main()
