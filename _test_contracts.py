#!/usr/bin/env python3
"""Test USD to contract conversion on OKX sandbox."""
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

test_syms = [
    'BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'SOL-USDT-SWAP', 'SAND-USDT-SWAP',
    'AVAX-USDT-SWAP', 'ATOM-USDT-SWAP', 'OP-USDT-SWAP', 'DOGE-USDT-SWAP',
    'AAVE-USDT-SWAP', 'IOTA-USDT-SWAP', 'ZIL-USDT-SWAP', 'FIL-USDT-SWAP',
    'LTC-USDT-SWAP', 'ETC-USDT-SWAP', 'ADA-USDT-SWAP', 'COMP-USDT-SWAP',
    'NEAR-USDT-SWAP', 'CRV-USDT-SWAP', 'DOT-USDT-SWAP', 'ALGO-USDT-SWAP',
]

usd = 225.0
print(f"Target: ${usd} per position")
print(f"{'Symbol':<20} {'ctVal':<8} {'Price':<12} {'Contracts':<12} {'Actual USD':<12} {'Precision'}")
print("-" * 80)

for inst_id in test_syms:
    # Find unified symbol
    market = None
    for sym, m in ex.markets.items():
        if m.get('id') == inst_id:
            market = m
            break
    if not market:
        print(f"{inst_id:<20} NOT FOUND")
        continue
    
    ct_val = float(market.get('contractSize', 1))
    prec = market.get('precision', {}).get('amount', 1)
    
    try:
        ticker = ex.fetch_ticker(market['symbol'])
        price = ticker['last']
    except Exception as e:
        print(f"{inst_id:<20} TICKER ERROR: {e}")
        continue
    
    raw_contracts = usd / (ct_val * price)
    
    # Round to precision
    if isinstance(prec, (int, float)) and prec > 0:
        contracts = int(raw_contracts / prec) * prec
    else:
        contracts = int(raw_contracts)
    
    actual_usd = contracts * ct_val * price
    print(f"{inst_id:<20} {ct_val:<8} ${price:<11.4f} {contracts:<12} ${actual_usd:<11.2f} {prec}")

# Check balance
b = ex.fetch_balance()
free = float(b.get('USDT', {}).get('free', 0))
total = float(b.get('USDT', {}).get('total', 0))
print(f"\nBalance: free=${free:.2f}, total=${total:.2f}")
