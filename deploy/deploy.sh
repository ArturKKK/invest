#!/usr/bin/env bash
# deploy.sh — Deploy gen8 champion (31f hybrid CLS) to VPS
# Usage: ./deploy/deploy.sh user@your-vps-ip
#
# Strategy:
#   - git pull for code changes (VPS has the repo)
#   - rsync ONLY data that doesn't exist on VPS (coinglass, prod models)
#   - never overwrite data/raw/ that VPS updates daily
set -euo pipefail

VPS="${1:?Usage: deploy.sh user@vps-ip}"
REMOTE_DIR="/home/trader/invest"

echo "═══════════════════════════════════════════"
echo "  Deploying gen8 champion to $VPS"
echo "═══════════════════════════════════════════"

# 1. Pull latest code via git
echo "📦 Pulling latest code..."
ssh "$VPS" bash -s <<'REMOTE'
set -euo pipefail
cd /home/trader/invest
git pull --ff-only
echo "✅ Code updated"
REMOTE

# 2. Sync production models (results_cls_prod/)
echo "📦 Syncing production models..."
rsync -avz --progress \
    results_cls_prod/ "${VPS}:${REMOTE_DIR}/results_cls_prod/"

# 3. Sync CoinGlass data (only if not present on VPS)
#    VPS may not have coinglass data yet — use --ignore-existing
echo "📦 Syncing CoinGlass data (new files only)..."
rsync -avz --progress --ignore-existing \
    data/raw/coinglass/ "${VPS}:${REMOTE_DIR}/data/raw/coinglass/"

# 4. Remote setup — install deps + update service
echo "🔧 Setting up remote environment..."
ssh "$VPS" bash -s <<'REMOTE'
set -euo pipefail
cd /home/trader/invest

# Python venv
if [ ! -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Dependencies (added xgboost for gen8)
pip install -q --upgrade pip
pip install -q pandas numpy lightgbm xgboost catboost ccxt requests python-dotenv

# Directories
mkdir -p logs trading_logs data/raw/coinglass

# Check .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env not found! Copy .env.example and fill in your keys:"
    echo "    cp .env.example .env && nano .env"
    exit 1
fi

echo "✅ Remote setup complete"
REMOTE

# 5. Install systemd service
echo "🔄 Installing systemd service..."
ssh "$VPS" bash -s <<'REMOTE'
set -euo pipefail

# Stop existing service if running
sudo systemctl stop crypto-trader 2>/dev/null || true

# Copy updated service file
sudo cp /home/trader/invest/deploy/crypto-trader.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable crypto-trader

# Setup daily CoinGlass data refresh (06:00 UTC, before any 12:00 rebal)
CRON_CMD="0 6 * * * cd /home/trader/invest && venv/bin/python src/data/download_coinglass_v4.py --only taker >> logs/coinglass_cron.log 2>&1"
(crontab -l 2>/dev/null | grep -v 'download_coinglass'; echo "$CRON_CMD") | crontab -

echo "✅ Service installed (CLS mode, leverage=1, capital=100)"
echo "✅ CoinGlass cron: daily 06:00 UTC (taker data refresh)"
echo ""
echo "Commands:"
echo "  sudo systemctl start crypto-trader    # Start"
echo "  sudo systemctl stop crypto-trader     # Stop"
echo "  sudo systemctl status crypto-trader   # Status"
echo "  journalctl -u crypto-trader -f        # Live logs"
echo "  tail -f /home/trader/invest/logs/bot.log  # Application logs"
REMOTE

echo ""
echo "═══════════════════════════════════════════"
echo "  ✅ Deployment complete!"
echo ""
echo "  Launch: ssh $VPS 'sudo systemctl start crypto-trader'"
echo "  Logs:   ssh $VPS 'journalctl -u crypto-trader -f'"
echo "═══════════════════════════════════════════"
