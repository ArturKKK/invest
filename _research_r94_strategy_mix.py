#!/usr/bin/env python3
"""
R94 — Strategy Mix: combine R68 + R91 + R92 returns.

Grid weights + risk parity. Evaluate combined portfolio Sharpe/MaxDD.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EPS = 1e-10
PERIODS_PER_YEAR = 2 * 365


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def sharpe(rets) -> float:
    if len(rets) < 2:
        return 0.0
    r = np.array(rets, dtype=float)
    eq = np.cumprod(1 + r)
    pct = np.diff(eq) / eq[:-1]
    if len(pct) < 2 or np.std(pct) < EPS:
        return 0.0
    return float(np.mean(pct) / (np.std(pct) + EPS) * np.sqrt(PERIODS_PER_YEAR))


def max_dd(rets) -> float:
    eq = np.cumprod(1 + np.array(rets, dtype=float))
    running_max = np.maximum.accumulate(eq)
    dd = eq / running_max - 1
    return float(np.min(dd))


def load_equity(path: Path, name: str) -> pd.Series:
    """Load equity CSV, return net_ret Series indexed by timestamp."""
    if not path.exists():
        log(f"  ✗ {name}: NOT FOUND ({path})")
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    log(f"  ✓ {name}: {len(df)} periods, "
        f"{str(df.index.min())[:10]} → {str(df.index.max())[:10]}")
    return df["net_ret"]


def risk_parity_weights(vols: list) -> list:
    """Compute risk parity weights: w_i = (1/vol_i) / sum(1/vol_j)."""
    inv_vol = [1.0 / (v + EPS) for v in vols]
    total = sum(inv_vol)
    return [iv / total for iv in inv_vol]


def main():
    t0 = time.time()
    log("=" * 70)
    log("  R94 — STRATEGY MIX")
    log("=" * 70)

    # Load equity curves
    log("\n[0] Loading equity curves ...")

    # R68 baseline
    r68_path = RESULTS_DIR / "r86_r84_baseline_equity.csv"
    r68_rets = load_equity(r68_path, "R68")

    # R91 best
    r91_path = RESULTS_DIR / "r91_best_equity.csv"
    r91_rets = load_equity(r91_path, "R91")

    # R92 best
    r92_path = RESULTS_DIR / "r92_best_equity.csv"
    r92_rets = load_equity(r92_path, "R92")

    # Determine which strategies are available
    strategies = {}
    if r68_rets is not None:
        strategies["R68"] = r68_rets
    if r91_rets is not None:
        strategies["R91"] = r91_rets
    if r92_rets is not None:
        strategies["R92"] = r92_rets

    if "R68" not in strategies:
        log("  ✗ R68 baseline not found — cannot proceed")
        return

    n_strats = len(strategies)
    log(f"\n  Available strategies: {list(strategies.keys())}")

    # Align on common timestamps
    common_idx = strategies["R68"].index
    for name, rets in strategies.items():
        common_idx = common_idx.intersection(rets.index)

    log(f"  Common timestamps: {len(common_idx)}")

    aligned = {name: rets.reindex(common_idx).fillna(0)
               for name, rets in strategies.items()}

    # Correlation matrix
    log("\n[1] Correlation matrix ...")
    corr_df = pd.DataFrame(aligned).corr()
    log(f"\n{corr_df.round(3).to_string()}")

    # Individual metrics
    log("\n[2] Individual strategy metrics ...")
    ind_metrics = {}
    for name, rets in aligned.items():
        s = sharpe(rets)
        dd = max_dd(rets)
        vol = float(rets.std())
        ind_metrics[name] = {
            "sharpe": round(s, 4),
            "max_dd_pct": round(dd * 100, 2),
            "return_pct": round(float((1 + rets).prod() - 1) * 100, 1),
            "vol": round(vol, 6),
        }
        log(f"  {name}: Sharpe={s:.3f}  MaxDD={dd*100:.1f}%  Vol={vol:.6f}")

    # Grid search
    log("\n[3] Grid search ...")
    results = []

    if n_strats >= 3:
        # 3-strategy mix
        weight_configs = [
            {"label": "70_15_15", "R68": 0.70, "R91": 0.15, "R92": 0.15},
            {"label": "60_20_20", "R68": 0.60, "R91": 0.20, "R92": 0.20},
            {"label": "50_25_25", "R68": 0.50, "R91": 0.25, "R92": 0.25},
            {"label": "80_10_10", "R68": 0.80, "R91": 0.10, "R92": 0.10},
            {"label": "40_30_30", "R68": 0.40, "R91": 0.30, "R92": 0.30},
        ]
    elif n_strats == 2:
        other = [k for k in strategies.keys() if k != "R68"][0]
        weight_configs = [
            {"label": f"80_{other}20", "R68": 0.80, other: 0.20},
            {"label": f"70_{other}30", "R68": 0.70, other: 0.30},
            {"label": f"60_{other}40", "R68": 0.60, other: 0.40},
            {"label": f"50_{other}50", "R68": 0.50, other: 0.50},
        ]
    else:
        log("  Only R68 available — nothing to mix")
        return

    # Add risk parity
    vols = [ind_metrics[name]["vol"] for name in strategies.keys()]
    rp_weights = risk_parity_weights(vols)
    rp_config = {"label": "risk_parity"}
    for name, w in zip(strategies.keys(), rp_weights):
        rp_config[name] = round(w, 4)
    weight_configs.append(rp_config)

    best_sharpe = -999
    best_port_rets = None
    best_label = ""

    for cfg in weight_configs:
        label = cfg["label"]
        mix_rets = pd.Series(0.0, index=common_idx)
        for name in strategies.keys():
            w = cfg.get(name, 0)
            mix_rets += w * aligned[name]

        s = sharpe(mix_rets)
        dd = max_dd(mix_rets)
        ret_pct = float((1 + mix_rets).prod() - 1) * 100
        win = float((mix_rets > 0).mean())

        weights_str = ", ".join(f"{name}={cfg.get(name, 0):.2f}" for name in strategies.keys())
        m = {
            "label": label,
            "weights": {name: cfg.get(name, 0) for name in strategies.keys()},
            "net_sharpe": round(s, 4),
            "max_dd_pct": round(dd * 100, 2),
            "total_ret_pct": round(ret_pct, 1),
            "win_rate": round(win, 3),
            "n_periods": len(mix_rets),
        }
        results.append(m)

        log(f"  {label:<16} [{weights_str}]: Sharpe={s:.3f}  DD={dd*100:.1f}%  Ret={ret_pct:.1f}%")

        if s > best_sharpe:
            best_sharpe = s
            best_port_rets = mix_rets
            best_label = label

    # Compare best mix vs R68
    log("\n[4] Comparison with R68 ...")
    r68_sharpe = ind_metrics["R68"]["sharpe"]
    r68_dd = ind_metrics["R68"]["max_dd_pct"]
    best_result = [r for r in results if r["label"] == best_label][0]

    sharpe_improvement = best_result["net_sharpe"] / (r68_sharpe + EPS)
    dd_improvement = best_result["max_dd_pct"] / (r68_dd + EPS)

    log(f"  R68 alone:    Sharpe={r68_sharpe:.3f}  MaxDD={r68_dd:.1f}%")
    log(f"  Best mix:     Sharpe={best_result['net_sharpe']:.3f}  MaxDD={best_result['max_dd_pct']:.1f}%")
    log(f"  Sharpe ratio: {sharpe_improvement:.3f}x (need >1.05)")
    log(f"  DD ratio:     {dd_improvement:.3f}x (need <0.85)")

    accept = sharpe_improvement > 1.05 or dd_improvement < 0.85
    log(f"  Verdict: {'✅ ACCEPT' if accept else '✗ REJECT'}")

    # Save
    log("\n[5] Saving ...")
    summary = {
        "script": "r94_strategy_mix",
        "n_strategies": n_strats,
        "correlation_matrix": {k: {k2: round(v, 4) for k2, v in row.items()}
                               for k, row in corr_df.to_dict().items()},
        "individual_metrics": ind_metrics,
        "best_mix": best_label,
        "best_sharpe": best_sharpe,
        "sharpe_vs_r68": round(sharpe_improvement, 4),
        "dd_vs_r68": round(dd_improvement, 4),
        "accept": accept,
        "grid_results": results,
        "runtime_sec": round(time.time() - t0, 1),
    }
    (RESULTS_DIR / "r94_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))

    if best_port_rets is not None:
        eq_df = pd.DataFrame({
            "timestamp": best_port_rets.index,
            "net_ret": best_port_rets.values,
        })
        eq_df.to_csv(RESULTS_DIR / "r94_best_equity.csv", index=False)

    pd.DataFrame(results).to_csv(RESULTS_DIR / "r94_grid.csv", index=False)

    log(f"\n{'=' * 70}")
    log(f"  R94 COMPLETE — {time.time()-t0:.0f}s")
    log(f"{'=' * 70}")


if __name__ == "__main__":
    main()
