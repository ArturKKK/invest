import pandas as pd
df = pd.read_parquet("data/features/crypto_features_1h.parquet", columns=["timestamp","symbol"])
print("Shape:", df.shape)
print("Range:", df.timestamp.min(), "to", df.timestamp.max())
print("Symbols:", df.symbol.nunique())

oos_start = pd.Timestamp('2026-02-09', tz='UTC')
oos_end = pd.Timestamp('2026-03-07', tz='UTC')
train_end = pd.Timestamp('2025-12-01', tz='UTC')
val_start = pd.Timestamp('2025-12-09', tz='UTC')
val_end = pd.Timestamp('2026-03-07', tz='UTC')

n_oos = df[(df.timestamp >= oos_start) & (df.timestamp <= oos_end)].shape[0]
n_train = df[df.timestamp <= train_end].shape[0]
n_val = df[(df.timestamp >= val_start) & (df.timestamp <= val_end)].shape[0]
print(f"\nTrain data (up to 2025-12-01): {n_train:,} rows")
print(f"Val window (2025-12-09 to 2026-03-07): {n_val:,} rows")
print(f"Sim OOS window (2026-02-09 to 2026-03-07): {n_oos:,} rows")
print(f"\nSim period overlaps val? -> YES (Feb 9 - Mar 7 is subset of Dec 9 - Mar 7)")
