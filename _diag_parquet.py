#!/usr/bin/env python3
"""Check which parquet files are valid."""
import pyarrow.parquet as pq
from pathlib import Path

bad = []
ok = 0
for f in sorted(Path("data/raw").glob("*_1h.parquet")):
    try:
        pq.read_table(f, columns=["timestamp"])
        ok += 1
    except Exception as e:
        bad.append((f.name, f.stat().st_size, str(e)[:80]))

print(f"OK: {ok} files")
print(f"BAD: {len(bad)} files")
for name, size, err in bad:
    print(f"  {name} ({size} bytes): {err}")
