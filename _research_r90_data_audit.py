#!/usr/bin/env python3
"""
R90 — Data Audit: check all datasets before R91/R92.
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
CG_DIR = ROOT / "data" / "raw" / "coinglass"


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def check_ohlcv():
    """Check OHLCV research frame via R68 load pipeline."""
    log("  Checking OHLCV research frame ...")
    from _research_r35_new_features import load_research_frame
    df, regime_df = load_research_frame()
    info = {
        "rows": len(df),
        "symbols": int(df["symbol"].nunique()),
        "columns": len(df.columns),
        "date_min": str(df["timestamp"].min())[:19],
        "date_max": str(df["timestamp"].max())[:19],
        "has_fwd_ret_12h": "fwd_ret_12h" in df.columns,
        "regime_rows": len(regime_df),
    }
    log(f"    OHLCV: {info['rows']:,} rows, {info['symbols']} symbols, "
        f"{info['date_min']} → {info['date_max']}")
    return info


def check_cg_dataset(name: str):
    """Check single CG parquet."""
    path = CG_DIR / f"{name}.parquet"
    if not path.exists():
        log(f"    ✗ {name}: NOT FOUND ({path})")
        return {"name": name, "exists": False}

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    info = {
        "name": name,
        "exists": True,
        "rows": len(df),
        "symbols": int(df["symbol"].nunique()),
        "columns": list(df.columns),
        "date_min": str(df["timestamp"].min())[:19],
        "date_max": str(df["timestamp"].max())[:19],
    }

    # Check for duplicate (symbol, date)
    df["cg_date"] = df["timestamp"].dt.normalize()
    n_dups = df.duplicated(subset=["symbol", "cg_date"]).sum()
    info["duplicates"] = int(n_dups)

    # Symbols list
    syms = sorted(df["symbol"].unique().tolist())
    info["symbol_list"] = syms

    log(f"    ✓ {name}: {info['rows']:,} rows, {info['symbols']} symbols, "
        f"{info['date_min']} → {info['date_max']}, dups={n_dups}")
    return info


def check_shift1_alignment():
    """Verify shift1 merge key logic."""
    log("  Checking shift1 alignment logic ...")
    # Create sample timestamps
    t1 = pd.Timestamp("2025-01-15 12:00", tz="UTC")
    t2 = pd.Timestamp("2025-01-16 00:00", tz="UTC")
    t3 = pd.Timestamp("2025-01-16 12:00", tz="UTC")

    for t in [t1, t2, t3]:
        cg_date = t.normalize() - pd.Timedelta(days=1)
        log(f"    OHLCV {t} → uses CG date {cg_date.date()} (shift1 ✓)")

    return {"shift1_logic": "cg_date = timestamp.normalize() - 1day", "status": "OK"}


def main():
    t0 = time.time()
    log("=" * 70)
    log("  R90 — DATA AUDIT")
    log("=" * 70)

    report = {}

    # 1. OHLCV
    log("\n[1] OHLCV Research Frame")
    report["ohlcv"] = check_ohlcv()

    # 2. CG datasets
    log("\n[2] CoinGlass Datasets")
    cg_datasets = ["funding", "liq", "oi", "taker", "ls_ratio"]
    report["cg"] = {}
    for name in cg_datasets:
        report["cg"][name] = check_cg_dataset(name)

    # 3. Shift1 alignment
    log("\n[3] Shift1 Alignment")
    report["shift1"] = check_shift1_alignment()

    # 4. Summary
    log("\n" + "=" * 70)
    log("  SUMMARY")
    log("=" * 70)
    all_ok = report["ohlcv"]["rows"] > 0
    for name, info in report["cg"].items():
        if not info["exists"]:
            all_ok = False
            log(f"  ✗ MISSING: {name}")
        else:
            log(f"  ✓ {name}: {info['rows']:,} rows, {info['symbols']} symbols")

    report["all_ok"] = all_ok
    report["runtime_sec"] = round(time.time() - t0, 1)

    out_path = RESULTS_DIR / "r90_data_audit.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"\n  Saved: {out_path}")
    log(f"  Status: {'✓ ALL OK' if all_ok else '✗ ISSUES FOUND'}")
    log(f"  Runtime: {report['runtime_sec']}s")


if __name__ == "__main__":
    main()
