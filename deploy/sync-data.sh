#!/bin/bash
# sync-data.sh — Rsync data/ folder to VPS
# Usage: ./deploy/sync-data.sh            (default: root@185.42.163.63)
#        ./deploy/sync-data.sh user@host
set -e

VPS="${1:-root@185.42.163.63}"
REMOTE_DIR="/home/trader/invest"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "📦 Syncing data/ → ${VPS}:${REMOTE_DIR}/data/"
rsync -avz --progress \
    --exclude='raw_news.parquet' \
    --exclude='raw_news.parquet.checkpoint' \
    --exclude='raw_news_backup.parquet' \
    --exclude='*.tmp' \
    --exclude='overnight_*' \
    --exclude='results_*' \
    "${LOCAL_DIR}/data/" "${VPS}:${REMOTE_DIR}/data/"

echo ""
echo "✅ Data sync complete!"
echo "   Tip: raw_news.parquet excluded (updated by cron on VPS)"
