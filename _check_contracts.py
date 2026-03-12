#!/usr/bin/env python3
"""Check OKX contract sizes to understand amount parameter."""
import ccxt, os
from dotenv import load_dotenv
load_dotenv()

ex = ccxt.okx({
    'apiKey': os.environ['OKX_API_KEY'],
    'secret': os.environ['OKX_SECRET'],
    'password': os.environ['OKX_PASSPHRASE'],
    'enableRateLimit': True,
})
ex.set_sandbox_mode(True)
ex.load_markets()

symbols = ['BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'SOL-USDT-SWAP', 'SAND-USDT-SWAP',
           'AVAX-USDT-SWAP', 'ATOM-USDT-SWAP', 'OP-USDT-SWAP', 'DOGE-USDT-SWAP',
           'AAVE-USDT-SWAP', 'IOTA-USDT-SWAP', 'ZIL-USDT-SWAP', 'FIL-USDT-SWAP']

print(f"{'Symbol':<20} {'ctVal':<10} {'Price':<12} {'225cts USD':<14} {'cts for $225':<14}")
print("-" * 70)
for sym in symbols:
    if sym not in ex.markets:
        print(f"{sym:<20} NOT FOUND")
        continue
    m = ex.markets[sym]
    ctVal = float(m.get('contractSize', 1))
    t = ex.fetch_ticker(sym)
    price = t['last']
    notional_225 = 225 * ctVal * price
    cts_for_225 = 225 / (ctVal * price)
    print(f"{sym:<20} {ctVal:<10} ${price:<11.2f} ${notional_225:<13.0f} {cts_for_225:<14.4f}")
