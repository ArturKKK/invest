#!/usr/bin/env python3
import json
d = json.load(open("trading_logs/trade_20260312_0530.json"))
pos = d.get("positions", [])
longs = [p for p in pos if p["side"] == "long"]
shorts = [p for p in pos if p["side"] == "short"]
la = sum(p["score"] for p in longs) / len(longs) if longs else 0
sa = sum(p["score"] for p in shorts) / len(shorts) if shorts else 0
print(f"Positions: {len(longs)}L/{len(shorts)}S")
print(f"Long avg:  {la:.4f}")
print(f"Short avg: {sa:.4f}")
print(f"L/S spread: {la - sa:.4f}")
for p in pos:
    sym = p["symbol"]
    side = p["side"]
    sc = p["score"]
    print(f"  {sym:<15} {side:<6} score={sc:+.4f}")
