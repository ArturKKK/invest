#!/usr/bin/env bash
# update_dvol.sh — Cron job for auto-updating Deribit DVOL data
# Add to crontab: 30 */6 * * * /home/trader/invest/deploy/update_dvol.sh >> /home/trader/invest/logs/dvol_update.log 2>&1
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

echo "$(date -u '+%Y-%m-%d %H:%M UTC') — Starting DVOL update (Deribit)..."

python src/data/download_deribit_dvol.py

echo "$(date -u '+%Y-%m-%d %H:%M UTC') — DVOL update complete"
