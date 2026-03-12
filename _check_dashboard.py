#!/usr/bin/env python3
"""Check dashboard trades timestamps."""
import json

with open('dashboard/data/dashboard.json') as f:
    d = json.load(f)

print("Updated:", d.get("updated"))
trades = d.get("trades", [])
print(f"\n{len(trades)} trades:")
for t in trades:
    sym = t.get("symbol", "?")
    side = t.get("side", "?")
    opened = t.get("opened", "?")
    closed = t.get("closed", "?")
    pnl = t.get("pnl", 0)
    print(f"  {sym:<15} {side:<6} opened={opened}  closed={closed}  pnl={pnl}")
