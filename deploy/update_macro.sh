#!/usr/bin/env bash
# update_macro.sh — Cron job for auto-updating macro/cross-market data (FRED)
# Add to crontab: 0 6 * * * /home/trader/invest/deploy/update_macro.sh >> /home/trader/invest/logs/macro_update.log 2>&1
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

echo "$(date -u '+%Y-%m-%d %H:%M UTC') — Starting macro data update (FRED)..."

python src/data/download_macro.py

echo "$(date -u '+%Y-%m-%d %H:%M UTC') — Macro data update complete"
