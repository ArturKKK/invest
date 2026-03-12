#!/usr/bin/env python3
"""Check OKX balance and margin usage."""
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

b = ex.fetch_balance()
usdt = b.get('USDT', {})
free = float(usdt.get('free', 0))
used = float(usdt.get('used', 0))
total = float(usdt.get('total', 0))
print(f"Balance: free=${free:.2f}, used=${used:.2f}, total=${total:.2f}")

positions = ex.fetch_positions()
op = [p for p in positions if float(p.get('contracts', 0)) > 0]
print(f"Open: {len(op)} positions")
tot_margin = 0
tot_notional = 0
for p in op:
    m = float(p.get('initialMargin', 0))
    n = abs(float(p.get('notional', 0)))
    upnl = float(p.get('unrealizedPnl', 0))
    lev = p.get('leverage', '?')
    tot_margin += m
    tot_notional += n
    print(f"  {p['symbol']:<20} {p['side']:<5} lev={lev}x notional=${n:.0f} margin=${m:.0f} upnl=${upnl:+.2f}")
print(f"Total: notional=${tot_notional:.0f} margin=${tot_margin:.0f}")
print(f"Margin capacity at 3x: ${free * 3:.0f} notional")
print(f"Need 20x$225 = $4500 notional = $1500 margin")
print(f"Shortfall: ${max(0, 1500 - free):.0f}")
