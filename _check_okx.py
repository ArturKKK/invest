#!/usr/bin/env python3
"""Check which 'blocked' symbols are actually available on OKX demo."""
import ccxt

ex = ccxt.okx({
    'apiKey': '932bb5fb-c534-4c4c-95e1-6e24e6215440',
    'secret': '3E47539C27A58F64A288C3ED6CCB396E',
    'password': 'Starz7z7z7!',
    'sandbox': True,
})

markets = ex.load_markets()
swap_usdt = {k: v for k, v in markets.items() if v['type'] == 'swap' and v['quote'] == 'USDT'}
print(f"Total USDT-M swap markets on OKX demo: {len(swap_usdt)}\n")

blocked = [
    'MATIC/USDT', 'UNI/USDT', 'APT/USDT', 'FTM/USDT', 'MANA/USDT',
    'RUNE/USDT', 'EGLD/USDT', 'FLOW/USDT', 'SNX/USDT', 'ENJ/USDT',
    'BAT/USDT', 'ONE/USDT', 'ICX/USDT', 'ENS/USDT', 'GALA/USDT',
    'GRT/USDT', 'CHZ/USDT', 'MKR/USDT', 'ZIL/USDT',
]

print(f"{'Symbol':<16} {'Swap?':<8} {'Active?':<10} {'Note'}")
print("-" * 55)

can_unblock = []
for sym in sorted(blocked):
    swap = sym + ':USDT'
    if swap in markets:
        m = markets[swap]
        active = m.get('active', False)
        note = ""
        if active:
            can_unblock.append(sym)
            note = "← CAN UNBLOCK"
        print(f"{sym:<16} {'YES':<8} {'YES' if active else 'NO':<10} {note}")
    else:
        base = sym.split('/')[0]
        # Check rebranding (MATIC→POL, FTM→S, etc.)
        similar = [k for k in swap_usdt if base in k]
        note = f"similar: {similar}" if similar else "not found"
        print(f"{sym:<16} {'NO':<8} {'':<10} {note}")

print(f"\n=== Can unblock: {len(can_unblock)} ===")
for s in can_unblock:
    print(f"  ✅ {s}")

# Also try order a tiny amount to really verify
print("\n=== Quick trade test on potentially unblockable ===")
for sym in can_unblock[:3]:
    swap = sym + ':USDT'
    try:
        ex.set_leverage(3, swap)
        print(f"  {sym}: set_leverage OK")
    except Exception as e:
        err = str(e)
        if '51155' in err:
            print(f"  {sym}: ❌ 51155 compliance restricted")
        elif '51202' in err:
            print(f"  {sym}: ❌ 51202 max order exceeded")
        else:
            print(f"  {sym}: ❌ {err[:80]}")

# Show popular coins NOT in our SYMBOLS that are available
our_symbols = {
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'LINK/USDT',
    'MATIC/USDT', 'UNI/USDT', 'ATOM/USDT', 'LTC/USDT', 'FIL/USDT',
    'APT/USDT', 'ARB/USDT', 'OP/USDT', 'NEAR/USDT', 'AAVE/USDT',
    'INJ/USDT', 'FTM/USDT', 'ALGO/USDT', 'SAND/USDT', 'MANA/USDT',
    'AXS/USDT', 'THETA/USDT', 'RUNE/USDT', 'EGLD/USDT', 'XTZ/USDT',
    'FLOW/USDT', 'CHZ/USDT', 'CRV/USDT', 'LDO/USDT', 'SNX/USDT',
    'COMP/USDT', 'YFI/USDT', 'SUSHI/USDT', 'ENJ/USDT', 'BAT/USDT',
    'ZIL/USDT', 'ONE/USDT', 'IOTA/USDT', 'ICX/USDT', 'ENS/USDT',
    'IMX/USDT', 'GALA/USDT', 'MKR/USDT', 'GRT/USDT', 'ETC/USDT',
}

print("\n=== Available on OKX demo but NOT in our SYMBOLS ===")
available_new = []
for sym_swap, m in sorted(swap_usdt.items()):
    if not m.get('active'):
        continue
    sym = sym_swap.replace(':USDT', '')
    if sym not in our_symbols:
        available_new.append(sym)

# Show top ones (well-known)
well_known = ['PEPE/USDT', 'WIF/USDT', 'WLD/USDT', 'TIA/USDT', 'SEI/USDT',
              'SUI/USDT', 'ORDI/USDT', 'JTO/USDT', 'PYTH/USDT', 'STX/USDT',
              'RENDER/USDT', 'FET/USDT', 'TAO/USDT', 'PENDLE/USDT',
              'TRX/USDT', 'TON/USDT', 'SHIB/USDT', 'BCH/USDT', 'ICP/USDT',
              'HBAR/USDT', 'VET/USDT', 'WOO/USDT', 'BLUR/USDT',
              'POL/USDT', 'S/USDT']

for sym in well_known:
    status = "✅ available" if sym in available_new else "❌ not found"
    print(f"  {sym:<16} {status}")

print(f"\nTotal active USDT swaps we don't use: {len(available_new)}")
