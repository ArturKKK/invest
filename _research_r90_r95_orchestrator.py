#!/usr/bin/env python3
"""
R90→R95 Orchestrator: runs all experiments sequentially.
Designed to run detached on MLC VM overnight.
"""

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).parent

SCRIPTS = [
    ("R90", "_research_r90_data_audit.py"),
    ("R91", "_research_r91_funding_carry.py"),
    ("R92", "_research_r92_liq_events.py"),
    ("R94", "_research_r94_strategy_mix.py"),
    ("R95", "_research_r95_bootstrap.py"),
]


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_script(label: str, script: str, python: str) -> dict:
    """Run a single script and capture output."""
    path = ROOT / script
    if not path.exists():
        log(f"  ✗ {label}: script not found ({script})")
        return {"label": label, "status": "NOT_FOUND"}

    log(f"\n{'#' * 70}")
    log(f"  STARTING {label}: {script}")
    log(f"{'#' * 70}")
    t0 = time.time()

    try:
        result = subprocess.run(
            [python, str(path)],
            capture_output=True, text=True, timeout=7200,  # 2h max
            cwd=str(ROOT),
        )
        runtime = time.time() - t0

        # Print stdout
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                print(f"  {line}", flush=True)

        if result.returncode != 0:
            log(f"  ✗ {label} FAILED (exit code {result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-20:]:
                    print(f"  ERR: {line}", flush=True)
            return {"label": label, "status": "FAILED",
                    "exit_code": result.returncode,
                    "runtime_sec": round(runtime, 1),
                    "stderr_tail": result.stderr[-500:] if result.stderr else ""}

        log(f"  ✓ {label} DONE in {runtime:.0f}s")
        return {"label": label, "status": "OK", "runtime_sec": round(runtime, 1)}

    except subprocess.TimeoutExpired:
        runtime = time.time() - t0
        log(f"  ✗ {label} TIMEOUT after {runtime:.0f}s")
        return {"label": label, "status": "TIMEOUT", "runtime_sec": round(runtime, 1)}
    except Exception as e:
        runtime = time.time() - t0
        log(f"  ✗ {label} ERROR: {e}")
        return {"label": label, "status": "ERROR", "error": str(e),
                "runtime_sec": round(runtime, 1)}


def main():
    t_start = time.time()
    log("=" * 70)
    log("  R90-R95 ORCHESTRATOR — OVERNIGHT RUN")
    log("=" * 70)

    # Determine Python executable
    python = sys.executable
    log(f"  Python: {python}")
    log(f"  Working dir: {ROOT}")

    results = []
    for label, script in SCRIPTS:
        r = run_script(label, script, python)
        results.append(r)

        # If R90 fails, abort
        if label == "R90" and r["status"] != "OK":
            log(f"\n  ✗ R90 data audit failed — aborting pipeline")
            break

    # Final summary
    log(f"\n{'=' * 70}")
    log(f"  ORCHESTRATOR SUMMARY")
    log(f"{'=' * 70}")
    total_runtime = time.time() - t_start
    for r in results:
        s = r.get("runtime_sec", 0)
        log(f"  {r['label']:<6}: {r['status']:<10} ({s:.0f}s)")

    log(f"\n  Total runtime: {total_runtime:.0f}s ({total_runtime/60:.1f}min)")

    # Save orchestrator report
    report = {
        "script": "r90_r95_orchestrator",
        "results": results,
        "total_runtime_sec": round(total_runtime, 1),
        "all_ok": all(r["status"] == "OK" for r in results),
    }
    out_path = ROOT / "results" / "r90_r95_orchestrator.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"  Report: {out_path}")
    log("  DONE.")


if __name__ == "__main__":
    main()
