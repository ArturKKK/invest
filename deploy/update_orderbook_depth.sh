#!/usr/bin/env bash
# update_orderbook_depth.sh — Cron job for hourly Binance orderbook depth snapshots
# Add to crontab: 35 * * * * /home/trader/invest/deploy/update_orderbook_depth.sh >> /home/trader/invest/logs/orderbook_depth.log 2>&1
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

echo "$(date -u '+%Y-%m-%d %H:%M UTC') — Starting Binance orderbook depth snapshot update..."

python src/data/download_binance_depth.py
python src/features/build_orderbook_depth_features.py

echo "$(date -u '+%Y-%m-%d %H:%M UTC') — Binance orderbook depth update complete"