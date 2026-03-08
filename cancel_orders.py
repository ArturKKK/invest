#!/usr/bin/env python3
"""Cancel all open orders."""
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

orders = ex.fetch_open_orders()
print(f"Found {len(orders)} open order(s)")
for o in orders:
    sym = o["symbol"]
    oid = o["id"]
    print(f"  Cancelling {sym} {oid}...")
    ex.cancel_order(oid, sym)
    print(f"  Cancelled OK")

print("Done.")
