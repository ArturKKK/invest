#!/usr/bin/env python3
"""Check model signals, backtest results, and live feature snapshot."""
import json, glob, os

# === Cycle equity analysis ===
logs = sorted(glob.glob("trading_logs/trade_*.json"))
print("=== CYCLE EQUITY ===")
prev_eq = None
for lf in logs:
    d = json.load(open(lf))
    st = d.get("state", {})
    eq = st.get("equity", None)
    npos = len(d.get("positions", []))
    ts = d.get("timestamp", "?")
    pnl_str = ""
    if prev_eq is not None and eq is not None:
        pnl = eq - prev_eq
        pnl_str = f"  PnL={pnl:+.1f} ({pnl/prev_eq*100:+.1f}%)"
    print(f"  {os.path.basename(lf)}: eq={eq}, pos={npos}{pnl_str}")
    if eq is not None:
        prev_eq = eq

# === Backtest results from all_results files ===
print("\n=== BACKTEST RESULTS ===")
for d in ["results_v6_huber_prod", "results_v7_huber_prod", "results_catboost_prod", "results_xgboost_prod"]:
    rpath = d
    if not os.path.exists(rpath):
        continue
    for f in sorted(os.listdir(rpath)):
        if f.startswith("all_results"):
            data = json.load(open(os.path.join(rpath, f)))
            print(f"\n  {d}/{f}:")
            if isinstance(data, dict):
                for k, v in sorted(data.items()):
                    if isinstance(v, dict):
                        parts = []
                        for mk in ["win_rate", "wr", "sharpe", "sharpe_ratio", "max_dd", "max_drawdown", 
                                   "total_return", "annual_return", "n_trades", "profit_factor"]:
                            if mk in v:
                                parts.append(f"{mk}={v[mk]}")
                        print(f"    {k}: {', '.join(parts)}")
                    else:
                        print(f"    {k}: {v}")
            elif isinstance(data, list):
                for item in data[:3]:
                    print(f"    {item}")

# === Signal distribution in latest cycle ===
print("\n=== LATEST SIGNALS DISTRIBUTION ===")
if logs:
    d = json.load(open(logs[-1]))
    top5 = d.get("signals_top5", [])
    bot5 = d.get("signals_bot5", [])
    all_sigs = top5 + bot5
    if all_sigs:
        scores = [s.get("score", 0) if isinstance(s, dict) else 0 for s in all_sigs]
        print(f"  Range: [{min(scores):.3f}, {max(scores):.3f}]")
        print(f"  Mean: {sum(scores)/len(scores):.3f}")
        n_strong = sum(1 for s in scores if abs(s) > 1.5)
        print(f"  Strong signals (|z|>1.5): {n_strong}/{len(scores)}")

# === Check production_meta.json ===
print("\n=== PRODUCTION META ===")
meta_path = "results_v7_huber_prod/production_meta.json"
if os.path.exists(meta_path):
    meta = json.load(open(meta_path))
    print(f"  {meta}")

# === Feature comparison: what model expects vs what bot builds === 
print("\n=== FEATURE SETS ===")
for d in ["results_v6_huber_prod", "results_v7_huber_prod", "results_catboost_prod", "results_xgboost_prod"]:
    fn_path = os.path.join(d, "feature_names.json")
    if os.path.exists(fn_path):
        feats = json.load(open(fn_path))
        # Group features
        groups = {}
        for f in feats:
            prefix = f.split("_")[0]
            groups[prefix] = groups.get(prefix, 0) + 1
        top_groups = sorted(groups.items(), key=lambda x: -x[1])[:10]
        print(f"  {d}: {len(feats)} feats, groups: {dict(top_groups)}")
