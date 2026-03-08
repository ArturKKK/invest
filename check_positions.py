#!/usr/bin/env python3
"""Check OKX positions and orders."""
import os
from dotenv import load_dotenv
load_dotenv()
import ccxt

ex = ccxt.okx({
    "apiKey": os.environ["OKX_API_KEY"],
    "secret": os.environ["OKX_SECRET"],
    "password": os.environ["OKX_PASSPHRASE"],
    "enableRateLimit": True,
    "options": {"defaultType": "swap"},
})
ex.set_sandbox_mode(True)
ex.load_markets()

# Positions
positions = ex.fetch_positions()
open_pos = [p for p in positions if float(p["contracts"]) > 0]
print(f"=== Open Positions: {len(open_pos)} ===")
total_notional = 0
for p in open_pos:
    notional = float(p.get("notional", 0))
    pnl = float(p.get("unrealizedPnl", 0))
    total_notional += notional
    icon = "🟢" if pnl >= 0 else "🔴"
    sym = p["symbol"]
    side = p["side"]
    cts = p["contracts"]
    print(f"  {icon} {sym:20s} {side:5s} {cts:>8} cts  ${notional:>8.2f}  uPnL=${pnl:+.2f}")

print(f"\n  Total notional: ${total_notional:.2f}")

# Balance
bal = ex.fetch_balance()
free = bal.get("USDT", {}).get("free", 0)
total = bal.get("USDT", {}).get("total", 0)
print(f"\n=== Balance ===")
print(f"  Free:  ${free:.2f}")
print(f"  Total: ${total:.2f}")

# Open orders (unfilled limit orders etc)
orders = ex.fetch_open_orders()
print(f"\n=== Open Orders: {len(orders)} ===")
for o in orders:
    sym = o["symbol"]
    side = o["side"]
    typ = o["type"]
    amt = o["amount"]
    price = o.get("price")
    status = o.get("status")
    print(f"  {sym} {side} {typ} amt={amt} price={price} status={status}")
