"""Reconstruct equity history from per-cycle trade logs and merge into state."""
import json, glob, os, tempfile

ROOT = "/home/trader/invest"
LOG_DIR = os.path.join(ROOT, "trading_logs")
STATE_PATH = os.path.join(LOG_DIR, "trading_state.json")

logs = sorted(glob.glob(os.path.join(LOG_DIR, "trade_*.json")))
reconstructed = []
for lf in logs:
    try:
        with open(lf) as f:
            d = json.load(f)
        ts_raw = d.get("timestamp", os.path.basename(lf).replace("trade_","").replace(".json",""))
        st = d.get("state", {})
        eq = st.get("equity", st.get("prev_equity"))
        if eq:
            reconstructed.append({"timestamp": ts_raw, "equity": round(float(eq), 2), "pnl": 0, "dd_pct": 0})
            print(f"  {ts_raw}: equity={eq:.2f}")
    except Exception as e:
        print(f"  error: {e}")

print(f"\nReconstructed {len(reconstructed)} points from trade logs")

with open(STATE_PATH) as f:
    state = json.load(f)
existing = state.get("equity_history", [])
print(f"Existing equity_history: {len(existing)} points")
if existing:
    first_ts = existing[0]["timestamp"]
    print(f"Existing range: {first_ts[:19]} to {existing[-1]['timestamp'][:19]}")
    # Only add points before existing history
    new_points = [p for p in reconstructed if p["timestamp"] < first_ts]
    print(f"Points to prepend (before existing data): {len(new_points)}")
    if new_points:
        merged = new_points + existing
        state["equity_history"] = merged
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.rename(tmp, STATE_PATH)
        print(f"SUCCESS: Merged {len(merged)} total equity points")
        print(f"New range: {merged[0]['timestamp'][:19]} to {merged[-1]['timestamp'][:19]}")
    else:
        print("No new points to prepend (all already covered)")
else:
    print("No existing equity history - setting reconstructed as full history")
    state["equity_history"] = reconstructed
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.rename(tmp, STATE_PATH)
    print(f"Set {len(reconstructed)} points as equity history")
