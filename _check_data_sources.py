#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
DATA = Path(__file__).parent / "data" / "sentiment"
out = open("/tmp/data_check_out.txt", "w")

def p(s=""): out.write(str(s) + "\n"); out.flush()

try:
    news = pd.read_parquet(DATA / "crypto_news.parquet")
    p(f"NEWS: {news.shape} {news.columns.tolist()}")
    p(f"  range: {news['timestamp'].min()} -> {news['timestamp'].max()}")
    p(f"  symbols: {news['symbol'].nunique()}")
except Exception as e:
    p(f"News error: {e}")

p()
try:
    feat = pd.read_parquet(Path(__file__).parent / "data" / "crypto_features_1h.parquet")
    p(f"FEATURES: {feat.shape}")
    p(f"  cols: {feat.columns.tolist()[:40]}")
    p(f"  range: {feat['timestamp'].min()} -> {feat['timestamp'].max()}")
except Exception as e:
    p(f"Features error: {e}")

p()
try:
    fng = pd.read_parquet(DATA / "fear_greed.parquet")
    p(f"FNG: {fng.shape} {fng.columns.tolist()}")
    p(f"  range: {fng['timestamp'].min()} -> {fng['timestamp'].max()}")
except Exception as e:
    p(f"FNG error: {e}")

p()
try:
    macro = pd.read_parquet(DATA / "macro_daily.parquet")
    p(f"MACRO: {macro.shape} {macro.columns.tolist()}")
    p(f"  range: {macro['date'].min()} -> {macro['date'].max()}")
except Exception as e:
    p(f"Macro error: {e}")

p()
try:
    dv = pd.read_parquet(DATA / "deribit_dvol.parquet")
    p(f"DVOL: {dv.shape} {dv.columns.tolist()}")
    p(f"  range: {dv['timestamp'].min()} -> {dv['timestamp'].max()}")
except Exception as e:
    p(f"DVOL error: {e}")

p()
try:
    fm = pd.read_parquet(DATA / "binance_futures_metrics.parquet")
    p(f"FUTURES cols: {fm.columns.tolist()}")
    for c in ["top_long_pct", "global_long_pct"]:
        if c in fm.columns:
            pct = fm[c].isna().mean()*100
            p(f"  {c} NaN%: {pct:.1f}")
except Exception as e:
    p(f"Futures error: {e}")

out.close()
