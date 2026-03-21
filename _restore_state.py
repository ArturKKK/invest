#!/usr/bin/env python3
"""Restore trading_state.json from trade_*.json logs on VPS."""
import json, glob, os

logs = sorted(glob.glob("/home/trader/invest/trading_logs/trade_*.json"))
print(f"Found {len(logs)} trade logs")

# Rebuild equity history
equity_history = []
all_positions = []

for log_path in logs:
    try:
        d = json.load(open(log_path))
        state = d.get("state", {})
        ts = d.get("timestamp", "")
        eq = state.get("equity", 0)
        peak = state.get("peak", 5000)
        if eq > 0:
            equity_history.append({
                "timestamp": ts,
                "equity": round(eq, 2),
                "pnl": round(eq - 5000, 2),
                "dd_pct": round(eq / peak - 1, 4) if peak else 0,
            })
        # Collect positions opened at each cycle
        positions = d.get("positions", [])
        for p in positions:
            sym = p.get("symbol", "?").replace("/USDT", "")
            all_positions.append({
                "symbol": sym,
                "side": p.get("side", "?"),
                "usd": p.get("usd", 0),
                "score": round(p.get("score", 0), 4),
                "pnl": 0,
                "closed": None,
                "opened": ts,
            })
    except Exception:
        pass

print(f"Equity points recovered: {len(equity_history)}")
if equity_history:
    e0 = equity_history[0]
    eN = equity_history[-1]
    print(f"  First: {e0['timestamp'][:19]} eq={e0['equity']}")
    print(f"  Last:  {eN['timestamp'][:19]} eq={eN['equity']}")
print(f"Position records: {len(all_positions)}")

# Track open/close by detecting when a position disappears
# Build dash_trades with close detection
dash_trades = []
prev_syms = set()
sym_to_trade = {}

for log_path in sorted(logs):
    try:
        d = json.load(open(log_path))
        ts = d.get("timestamp", "")
        positions = d.get("positions", [])
        curr_syms = set()
        for p in positions:
            sym = p.get("symbol", "?").replace("/USDT", "")
            side = p.get("side", "?")
            key = (sym, side)
            curr_syms.add(key)
            if key not in sym_to_trade:
                trade = {
                    "symbol": sym,
                    "side": side,
                    "usd": p.get("usd", 0),
                    "score": round(p.get("score", 0), 4),
                    "pnl": 0,
                    "closed": None,
                    "opened": ts,
                }
                sym_to_trade[key] = trade
                dash_trades.append(trade)
        # Close trades that disappeared
        for key in list(sym_to_trade.keys()):
            if key not in curr_syms:
                sym_to_trade[key]["closed"] = ts
                del sym_to_trade[key]
    except Exception:
        pass

# Keep last 100 trades
dash_trades = dash_trades[-100:]
closed_trades = [t for t in dash_trades if t.get("closed")]
open_trades = [t for t in dash_trades if not t.get("closed")]
print(f"Dash trades: {len(dash_trades)} (open={len(open_trades)}, closed={len(closed_trades)})")

# Load current state and patch it
state_path = "/home/trader/invest/trading_logs/trading_state.json"
state = json.load(open(state_path))

# Keep current equity/peak (fresh from exchange)
# Restore historical data
state["equity_history"] = equity_history[-2000:]
state["dash_trades"] = dash_trades
state["n_cycles"] = len(logs)

# Also restore cycle_pnls from equity diffs
cycle_pnls = []
for i in range(1, len(equity_history)):
    delta = equity_history[i]["equity"] - equity_history[i-1]["equity"]
    cycle_pnls.append(round(delta, 2))
state["cycle_pnls"] = cycle_pnls[-200:]

with open(state_path, "w") as f:
    json.dump(state, f, indent=2, default=str)

print(f"\n✅ State restored!")
print(f"   Equity history: {len(state['equity_history'])} points")
print(f"   Dash trades: {len(state['dash_trades'])}")
print(f"   Cycle PnLs: {len(state['cycle_pnls'])}")
print(f"   n_cycles: {state['n_cycles']}")
