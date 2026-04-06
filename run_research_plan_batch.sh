#!/usr/bin/env zsh

set -euo pipefail

cd "$(dirname "$0")"

PYTHON="/Users/a.s.tabakov/.pyenv/versions/3.10.14/bin/python"

PYTHONUNBUFFERED=1 "$PYTHON" -u _research_r35_new_features.py > results_r35.log 2>&1
PYTHONUNBUFFERED=1 "$PYTHON" -u _research_r36_regime_gating.py > results_r36.log 2>&1
PYTHONUNBUFFERED=1 "$PYTHON" -u _research_r37_cost_aware.py > results_r37.log 2>&1
PYTHONUNBUFFERED=1 "$PYTHON" -u _research_r38_targets.py > results_r38.log 2>&1