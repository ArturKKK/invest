#!/usr/bin/env python3
"""Quick check: open positions + balance on OKX demo."""
import ccxt, os
from dotenv import load_dotenv
load_dotenv()

ex = ccxt.okx({
    "apiKey": os.getenv("OKX_API_KEY"),
    "secret": os.getenv("OKX_API_SECRET"),
    "password": os.getenv("OKX_PASSPHRASE"),
})
ex.set_sandbox_mode(True)

pos = ex.fetch_positions()
open_pos = [p for p in pos if float(p.get("contracts") or 0) > 0]
print(f"Open positions: {len(open_pos)}")
for p in open_pos:
    sym = p["symbol"]
    side = p["side"]
    ct = p["contracts"]
    notional = p.get("notional", "?")
    pnl = p.get("unrealizedPnl", "?")
    print(f"  {sym:25s} {side:6s} ct={ct:>6} notional={notional:>10} uPnL={pnl}")

bal = ex.fetch_balance()
free = bal["free"].get("USDT", 0)
total = bal["total"].get("USDT", 0)
print(f"\nFree USDT:  {free}")
print(f"Total USDT: {total}")
