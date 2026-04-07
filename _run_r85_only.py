"""
Standalone R85 block bootstrap — reads existing equity CSVs, no heavy data loading.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR  = Path("results")
PERIODS_PER_YEAR = 730   # 12h bars
EPS          = 1e-10

def block_bootstrap_sharpe(rets_base, rets_exp, n_boot=1000, block=10, seed=42):
    rng      = np.random.default_rng(seed)
    n        = min(len(rets_base), len(rets_exp))
    rb, re   = rets_base[:n], rets_exp[:n]
    n_blocks = n // block

    sb_list, se_list = [], []
    for _ in range(n_boot):
        idx       = rng.integers(0, n - block + 1, size=n_blocks)
        block_idx = np.concatenate([np.arange(i, i + block) for i in idx])[:n]
        rb_s, re_s = rb[block_idx], re[block_idx]

        def _sh(r):
            if len(r) < 2 or r.std() < 1e-10:
                return 0.0
            return float(r.mean() / (r.std() + EPS) * np.sqrt(PERIODS_PER_YEAR))

        sb_list.append(_sh(rb_s))
        se_list.append(_sh(re_s))

    sb, se = np.array(sb_list), np.array(se_list)
    delta  = se - sb
    return {
        "p_exp_better":    round(float((se > sb).mean()), 3),
        "median_delta":    round(float(np.median(delta)), 4),
        "mean_delta":      round(float(np.mean(delta)), 4),
        "p5_delta":        round(float(np.percentile(delta, 5)), 4),
        "p95_delta":       round(float(np.percentile(delta, 95)), 4),
        "base_sharpe_med": round(float(np.median(sb)), 4),
        "exp_sharpe_med":  round(float(np.median(se)), 4),
    }


def portfolio_metrics(port):
    eq = (1 + port["net_ret"]).cumprod()
    sh = port["net_ret"].mean() / (port["net_ret"].std() + EPS) * np.sqrt(PERIODS_PER_YEAR)
    roll_max = eq.cummax()
    dd = (eq - roll_max) / (roll_max + EPS)
    return {"net_sharpe": round(float(sh), 4), "max_dd_pct": round(float(dd.min() * 100), 2)}


def main():
    r85_path = RESULTS_DIR / "r85_summary.json"
    if r85_path.exists():
        print("R85 already exists:", json.loads(r85_path.read_text()))
        return

    base_path = RESULTS_DIR / "r84_baseline_equity.csv"
    r81_path  = RESULTS_DIR / "r81_best_equity.csv"

    base_port = pd.read_csv(base_path, parse_dates=["timestamp"])
    rets_base = base_port["net_ret"].values.astype(float)
    bm = portfolio_metrics(base_port)
    print(f"R68 baseline: Sharpe={bm['net_sharpe']}  MaxDD={bm['max_dd_pct']}%")

    r85_results = {}

    # R85a: R81 vs R68
    r81_eq    = pd.read_csv(r81_path, parse_dates=["timestamp"])
    rets_r81  = r81_eq["net_ret"].values.astype(float)
    r81m      = portfolio_metrics(r81_eq)
    print(f"R81 best:     Sharpe={r81m['net_sharpe']}  MaxDD={r81m['max_dd_pct']}%")
    print("Bootstrap: R68 vs R81_best …")
    bsr = block_bootstrap_sharpe(rets_base, rets_r81)
    r85_results["R81_best_vs_R68"] = bsr
    accept = bsr["p_exp_better"] > 0.8 and bsr["median_delta"] > 0.08
    print(f"  P(R81>R68)={bsr['p_exp_better']}  medianΔSh={bsr['median_delta']:+.4f}  "
          f"{'✅ ACCEPT' if accept else '✗ REJECT'}")

    # R85b: R84 exp (skip — no experiments passed gate in R84)
    r84_exp_path = RESULTS_DIR / "r84_exp1_equity.csv"
    if r84_exp_path.exists():
        r84_port  = pd.read_csv(r84_exp_path, parse_dates=["timestamp"])
        rets_r84  = r84_port["net_ret"].values.astype(float)
        print("Bootstrap: R68 vs R84_best …")
        bsr2 = block_bootstrap_sharpe(rets_base, rets_r84)
        r85_results["R84_best_vs_R68"] = bsr2
        accept2 = bsr2["p_exp_better"] > 0.8 and bsr2["median_delta"] > 0.08
        print(f"  P(R84>R68)={bsr2['p_exp_better']}  medianΔSh={bsr2['median_delta']:+.4f}  "
              f"{'✅ ACCEPT' if accept2 else '✗ REJECT'}")

    out = {"script": "r85_bootstrap_standalone", "results": r85_results}
    r85_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"Saved → {r85_path}")


if __name__ == "__main__":
    main()
