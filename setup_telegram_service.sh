#!/bin/bash
# Setup MAUDE Telegram as a systemd service

set -e

echo "Setting up MAUDE Telegram Service..."

# Create log file
sudo touch /var/log/maude-telegram.log
sudo chown mboard76:mboard76 /var/log/maude-telegram.log

# Copy service file
sudo cp maude-telegram.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start
sudo systemctl enable maude-telegram
sudo systemctl start maude-telegram

echo ""
echo "Service installed and started!"
echo ""
echo "Commands:"
echo "  sudo systemctl status maude-telegram    # Check status"
echo "  sudo systemctl restart maude-telegram   # Restart"
echo "  sudo systemctl stop maude-telegram      # Stop"
echo "  sudo journalctl -u maude-telegram -f    # View logs"
echo "  tail -f /var/log/maude-telegram.log     # View log file"
echo ""
