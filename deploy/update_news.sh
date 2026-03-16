#!/usr/bin/env bash
# update_news.sh — Cron job for auto-updating news sentiment data
# Add to crontab: 0 */6 * * * /home/trader/invest/deploy/update_news.sh >> /home/trader/invest/logs/news_update.log 2>&1
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

# Read API key from .env if available
if [[ -f .env ]]; then
    CC_KEY=$(grep -oP '^CC_API_KEY=\K.*' .env 2>/dev/null || true)
fi

echo "$(date -u '+%Y-%m-%d %H:%M UTC') — Starting news update..."

# Fetch last 3 days of news (overlap for safety)
# --feature-days 30: only rebuild features for last 30 days (saves RAM on 4GB VPS)
python fetch_crypto_news.py \
    --days 3 \
    --source crypto \
    --scorer vader \
    --workers 1 \
    --feature-days 30 \
    ${CC_KEY:+--cc-api-key "$CC_KEY"}

echo "$(date -u '+%Y-%m-%d %H:%M UTC') — News update complete"
