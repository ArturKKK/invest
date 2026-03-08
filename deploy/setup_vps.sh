#!/usr/bin/env bash
# setup_vps.sh — Initial VPS setup (run once as root)
# Usage: ssh root@vps-ip 'bash -s' < deploy/setup_vps.sh
set -euo pipefail

echo "═══════════════════════════════════════════"
echo "  VPS Initial Setup"
echo "═══════════════════════════════════════════"

# 1. System updates
apt-get update && apt-get upgrade -y

# 2. Install Python 3.11+
apt-get install -y python3 python3-pip python3-venv git htop tmux

# 3. Create trader user
if ! id trader &>/dev/null; then
    useradd -m -s /bin/bash trader
    echo "Created user 'trader'"
fi

# 4. Setup directories
mkdir -p /home/trader/invest/logs
chown -R trader:trader /home/trader/invest

# 5. UFW firewall (allow SSH only)
apt-get install -y ufw
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
echo "y" | ufw enable

# 6. Fail2ban for SSH protection
apt-get install -y fail2ban
systemctl enable fail2ban
systemctl start fail2ban

# 7. Automatic security updates
apt-get install -y unattended-upgrades
dpkg-reconfigure -f noninteractive unattended-upgrades

# 8. Swap (2GB for 4GB VPS)
if [ ! -f /swapfile ]; then
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo "Created 2GB swap"
fi

# 9. Timezone
timedatectl set-timezone UTC

echo ""
echo "═══════════════════════════════════════════"
echo "  ✅ VPS setup complete!"
echo ""
echo "  Now from your local machine run:"
echo "    ./deploy/deploy.sh trader@$(hostname -I | awk '{print $1}')"
echo "═══════════════════════════════════════════"
