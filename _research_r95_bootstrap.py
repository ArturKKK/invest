#!/usr/bin/env python3
"""
R95 — Bootstrap Significance Test.

Block bootstrap (B=10, N=1000) to test:
1. R91-best vs cash (Sharpe > 0)
2. R92-best vs cash (Sharpe > 0)
3. R94-best-mix vs R68 alone
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


def _sharpe_numpy(r):
    """Compute Sharpe directly on numpy returns (no pct_change double-conversion)."""
    if len(r) < 2 or np.std(r) < EPS:
        return 0.0
    return float(np.mean(r) / (np.std(r) + EPS) * np.sqrt(PERIODS_PER_YEAR))


def block_bootstrap_sharpe(rets_base, rets_exp, n_boot=1000, block=10, seed=42):
    """Block bootstrap comparing two return series."""
    rng = np.random.default_rng(seed)
    n = min(len(rets_base), len(rets_exp))
    rb = np.array(rets_base[:n], dtype=float)
    re = np.array(rets_exp[:n], dtype=float)
    n_blocks = n // block

    sb_list, se_list = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n - block + 1, size=n_blocks)
        block_idx = np.concatenate([np.arange(i, i + block) for i in idx])[:n]

        sb_list.append(_sharpe_numpy(rb[block_idx]))
        se_list.append(_sharpe_numpy(re[block_idx]))

    sb, se = np.array(sb_list), np.array(se_list)
    delta = se - sb

    return {
        "p_exp_better": round(float((se > sb).mean()), 3),
        "median_delta": round(float(np.median(delta)), 4),
        "mean_delta": round(float(np.mean(delta)), 4),
        "p5_delta": round(float(np.percentile(delta, 5)), 4),
        "p95_delta": round(float(np.percentile(delta, 95)), 4),
        "base_sharpe_med": round(float(np.median(sb)), 4),
        "exp_sharpe_med": round(float(np.median(se)), 4),
    }


def bootstrap_vs_cash(rets, n_boot=1000, block=10, seed=42):
    """Bootstrap test: Sharpe > 0 (vs cash/zero returns)."""
    rng = np.random.default_rng(seed)
    r = np.array(rets, dtype=float)
    n = len(r)
    n_blocks = n // block

    s_list = []
    for _ in range(n_boot):
        idx = rng.integers(0, n - block + 1, size=n_blocks)
        block_idx = np.concatenate([np.arange(i, i + block) for i in idx])[:n]
        s_list.append(_sharpe_numpy(r[block_idx]))

    s_arr = np.array(s_list)
    return {
        "p_positive": round(float((s_arr > 0).mean()), 3),
        "sharpe_med": round(float(np.median(s_arr)), 4),
        "sharpe_p5": round(float(np.percentile(s_arr, 5)), 4),
        "sharpe_p95": round(float(np.percentile(s_arr, 95)), 4),
    }


def load_rets(path: Path, col: str = "net_ret") -> np.ndarray:
    """Load returns from equity CSV."""
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp")[col].values.astype(float)


def main():
    t0 = time.time()
    log("=" * 70)
    log("  R95 — BOOTSTRAP SIGNIFICANCE")
    log("=" * 70)

    results = {}

    # Load all equity curves
    r68_rets = load_rets(RESULTS_DIR / "r86_r84_baseline_equity.csv")
    r91_rets = load_rets(RESULTS_DIR / "r91_best_equity.csv")
    r92_rets = load_rets(RESULTS_DIR / "r92_best_equity.csv")
    r94_rets = load_rets(RESULTS_DIR / "r94_best_equity.csv")

    # Test 1: R91 vs cash
    log("\n[1] R91 (Funding Carry) vs Cash ...")
    if r91_rets is not None:
        bs1 = bootstrap_vs_cash(r91_rets)
        results["R91_vs_cash"] = bs1
        accept = bs1["p_positive"] > 0.8
        log(f"  P(Sharpe>0)={bs1['p_positive']}  MedianSh={bs1['sharpe_med']}  "
            f"[{bs1['sharpe_p5']}, {bs1['sharpe_p95']}]  "
            f"{'✅' if accept else '✗'}")
    else:
        log("  R91 equity not found — skipping")

    # Test 2: R92 vs cash
    log("\n[2] R92 (Liq Events) vs Cash ...")
    if r92_rets is not None:
        bs2 = bootstrap_vs_cash(r92_rets)
        results["R92_vs_cash"] = bs2
        accept = bs2["p_positive"] > 0.8
        log(f"  P(Sharpe>0)={bs2['p_positive']}  MedianSh={bs2['sharpe_med']}  "
            f"[{bs2['sharpe_p5']}, {bs2['sharpe_p95']}]  "
            f"{'✅' if accept else '✗'}")
    else:
        log("  R92 equity not found — skipping")

    # Test 3: R94 best mix vs R68
    log("\n[3] R94 (Best Mix) vs R68 ...")
    if r94_rets is not None and r68_rets is not None:
        bs3 = block_bootstrap_sharpe(r68_rets, r94_rets)
        results["R94_vs_R68"] = bs3
        accept = bs3["p_exp_better"] > 0.8 and bs3["median_delta"] > 0.08
        log(f"  P(mix>R68)={bs3['p_exp_better']}  medianΔSh={bs3['median_delta']:+.4f}  "
            f"base_med={bs3['base_sharpe_med']}  exp_med={bs3['exp_sharpe_med']}  "
            f"{'✅ ACCEPT' if accept else '✗ REJECT'}")
    else:
        log("  R94 or R68 equity not found — skipping")

    # Test 4: R91 vs R68 (direct)
    log("\n[4] R91 vs R68 (direct) ...")
    if r91_rets is not None and r68_rets is not None:
        bs4 = block_bootstrap_sharpe(r68_rets, r91_rets)
        results["R91_vs_R68"] = bs4
        log(f"  P(R91>R68)={bs4['p_exp_better']}  medianΔSh={bs4['median_delta']:+.4f}")
    else:
        log("  Skipping")

    # Test 5: R92 vs R68 (direct)
    log("\n[5] R92 vs R68 (direct) ...")
    if r92_rets is not None and r68_rets is not None:
        bs5 = block_bootstrap_sharpe(r68_rets, r92_rets)
        results["R92_vs_R68"] = bs5
        log(f"  P(R92>R68)={bs5['p_exp_better']}  medianΔSh={bs5['median_delta']:+.4f}")
    else:
        log("  Skipping")

    # Summary
    log(f"\n{'=' * 70}")
    log(f"  BOOTSTRAP SUMMARY")
    log(f"{'=' * 70}")
    for test_name, r in results.items():
        if "p_exp_better" in r:
            log(f"  {test_name}: P={r['p_exp_better']}  ΔSh={r['median_delta']:+.4f}")
        else:
            log(f"  {test_name}: P(>0)={r['p_positive']}  Sh_med={r['sharpe_med']}")

    # Save
    summary = {
        "script": "r95_bootstrap",
        "n_boot": 1000,
        "block_size": 10,
        "results": results,
        "runtime_sec": round(time.time() - t0, 1),
    }
    (RESULTS_DIR / "r95_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))
    log(f"\n  Saved: r95_summary.json")
    log(f"  Runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
