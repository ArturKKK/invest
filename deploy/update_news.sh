#!/usr/bin/env bash
# update_news.sh — Cron job for auto-updating news sentiment data
# Add to crontab: 0 */6 * * * /home/trader/invest/deploy/update_news.sh >> /home/trader/invest/logs/news_update.log 2>&1
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

echo "$(date -u '+%Y-%m-%d %H:%M UTC') — Starting news update..."

# Fetch last 3 days of news (overlap for safety)
python fetch_crypto_news.py \
    --days 3 \
    --source crypto \
    --skip-political \
    --scorer vader \
    --workers 1

echo "$(date -u '+%Y-%m-%d %H:%M UTC') — News update complete"
