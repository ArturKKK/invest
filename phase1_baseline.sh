#!/bin/bash
set -e

echo "=== Phase 1: Baseline R114b Reproduction ==="
echo "Current dir: $(pwd)"
echo "Disk space:"
df -h /data /workdir 2>/dev/null || df -h /

# Create venv in /data (unlimited storage)
if [ ! -d "/data/.venv" ]; then
    echo "[1/5] Creating venv in /data/.venv..."
    python3 -m venv /data/.venv
else
    echo "[1/5] venv already exists at /data/.venv"
fi

# Activate venv
echo "[2/5] Activating venv..."
source /data/.venv/bin/activate

# Install requirements
echo "[3/5] Installing requirements (pinned versions)..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Run preflight check
echo "[4/5] Running preflight check..."
python _preflight_check.py

# Run baseline
echo "[5/5] Running R114b baseline (R68 continuous WF, expect Net Sharpe 2.831)..."
python _research_r68_continuous_wf.py

echo ""
echo "=== Phase 1 COMPLETE ==="
