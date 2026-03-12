#!/usr/bin/env python3
"""Check OKX Demo balance and positions."""
import ccxt, os

# Load env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, _, v = line.partition('=')
                os.environ[k.strip()] = v.strip().strip("'\"")

exchange = ccxt.okx({
    'apiKey': os.environ.get('OKX_API_KEY', ''),
    'secret': os.environ.get('OKX_SECRET', ''),
    'password': os.environ.get('OKX_PASSPHRASE', ''),
    'enableRateLimit': True,
})
exchange.set_sandbox_mode(True)

b = exchange.fetch_balance()
free = b.get('USDT', {}).get('free', 0)
total = b.get('USDT', {}).get('total', 0)
used = b.get('USDT', {}).get('used', 0)
print(f"Balance: free=${free:.2f}, used=${used:.2f}, total=${total:.2f}")

positions = exchange.fetch_positions()
open_pos = [p for p in positions if float(p.get('contracts', 0)) > 0]
print(f"Open positions: {len(open_pos)}")
total_margin = 0
total_notional = 0
for p in open_pos:
    sym = p['symbol']
    side = p['side']
    notional = abs(float(p.get('notional', 0)))
    margin = float(p.get('initialMargin', 0))
    upnl = float(p.get('unrealizedPnl', 0))
    total_margin += margin
    total_notional += notional
    print(f"  {sym:<20} {side:<5} notional=${notional:.0f} margin=${margin:.2f} upnl=${upnl:.2f}")

print(f"\nTotal margin used: ${total_margin:.2f}")
print(f"Total notional: ${total_notional:.0f}")
