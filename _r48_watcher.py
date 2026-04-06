#!/usr/bin/env python3
"""Waits for PID 87276 to finish, then runs Phase 3+4."""
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/Users/a.s.tabakov/Developer/invest")
LOG = BASE / "results_r48_phase34.log"
WAIT_LOG = BASE / "results_r48_watcher.log"
PHASE12_PID = 87276

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(WAIT_LOG, "a") as f:
        f.write(line + "\n")

def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False

log(f"Watcher started. Monitoring PID {PHASE12_PID}...")

# Wait for Phase 1+2 to finish
while pid_alive(PHASE12_PID):
    lines = 0
    try:
        lines = sum(1 for _ in open(BASE / "results_r48_phase12.log"))
    except Exception:
        pass
    log(f"Phase1+2 (PID {PHASE12_PID}) still running... log={lines} lines")
    time.sleep(60)

log(f"Phase1+2 FINISHED. Starting Phase 3+4...")

# Small grace period
time.sleep(5)

# Run Phase 3+4
python = str(BASE / "venv" / "bin" / "python3.10")
script = str(BASE / "_research_r48_phase34.py")

with open(LOG, "w") as logf:
    logf.write(f"=== Phase 3+4 start: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    logf.flush()
    proc = subprocess.Popen(
        [python, script],
        cwd=str(BASE),
        stdout=logf,
        stderr=subprocess.STDOUT,
    )
    log(f"Phase3+4 started PID={proc.pid}")
    proc.wait()
    log(f"Phase3+4 finished with exit code {proc.returncode}")

# Append to master log
with open(BASE / "results_r48_master_full.log", "a") as mf:
    mf.write("\n=== Phase 3+4 results appended ===\n")
    try:
        with open(LOG) as lf:
            mf.write(lf.read())
    except Exception:
        pass
    mf.write(f"\n=== ALL R48 DONE: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

log("ALL DONE.")
