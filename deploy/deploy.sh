#!/usr/bin/env bash
# deploy.sh — Deploy trading bot to VPS
# Usage: ./deploy/deploy.sh user@your-vps-ip
set -euo pipefail

VPS="${1:?Usage: deploy.sh user@vps-ip}"
REMOTE_DIR="/home/trader/invest"

echo "═══════════════════════════════════════════"
echo "  Deploying to $VPS"
echo "═══════════════════════════════════════════"

# 1. Sync code (exclude data, models cached locally)
echo "📦 Syncing code..."
rsync -avz --progress \
    --exclude='venv/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='data/raw/' \
    --exclude='data/sentiment/raw_news*' \
    --exclude='.git/' \
    --exclude='logs/' \
    --exclude='.env' \
    --exclude='*.tar.gz' \
    --exclude='models_archive/' \
    --exclude='results/' \
    --exclude='results_v6/' \
    --exclude='results_v7/' \
    --exclude='results_catboost/' \
    --exclude='results_xgboost/' \
    --exclude='results_deriv/' \
    --exclude='results_catboost_prod 3/' \
    --exclude='trading_logs/' \
    ./ "${VPS}:${REMOTE_DIR}/"

# 2. Remote setup
echo "🔧 Setting up remote environment..."
ssh "$VPS" bash -s <<'REMOTE'
set -euo pipefail
cd /home/trader/invest

# Python venv
if [ ! -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Dependencies
pip install -q --upgrade pip
pip install -q pandas numpy lightgbm catboost ccxt requests python-dotenv

# Directories
mkdir -p logs data/sentiment

# Check .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env not found! Copy .env.example and fill in your keys:"
    echo "    cp .env.example .env && nano .env"
    exit 1
fi

echo "✅ Remote setup complete"
REMOTE

# 4. Install systemd service
echo "🔄 Installing systemd service..."
ssh "$VPS" bash -s <<'REMOTE'
set -euo pipefail

# Copy service file
sudo cp /home/trader/invest/deploy/crypto-trader.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable crypto-trader

echo "✅ Service installed"
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
echo "  Next steps:"
echo "  1. ssh $VPS"
echo "  2. cd $REMOTE_DIR && cp .env.example .env && nano .env"
echo "  3. sudo systemctl start crypto-trader"
echo "  4. journalctl -u crypto-trader -f"
echo "═══════════════════════════════════════════"
