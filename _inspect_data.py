"""Quick inspection of all new data sources."""
import pandas as pd

for name, path in [
    ("DVOL", "data/sentiment/deribit_dvol.parquet"),
    ("Futures Metrics", "data/sentiment/binance_futures_metrics.parquet"),
    ("Premium Index", "data/sentiment/binance_premium_index.parquet"),
    ("Binance Funding", "data/sentiment/binance_funding_rates.parquet"),
]:
    print(f"=== {name} ===")
    df = pd.read_parquet(path)
    print(f"  Rows: {len(df):,}, Cols: {len(df.columns)}")
    print(f"  Columns: {list(df.columns)}")
    if 'symbol' in df.columns:
        print(f"  Symbols: {df.symbol.nunique()}")
    if 'timestamp' in df.columns:
        print(f"  Range: {df.timestamp.min()} → {df.timestamp.max()}")
    # Show sample values for numeric cols
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if num_cols:
        print(f"  Sample stats:")
        for c in num_cols[:8]:
            print(f"    {c}: mean={df[c].mean():.6f}, std={df[c].std():.6f}, "
                  f"null%={df[c].isna().mean()*100:.1f}%")
    print()
