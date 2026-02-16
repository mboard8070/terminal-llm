#!/bin/bash
#
# MAUDE Client Setup Script
# Run this on your Mac or PC to set up the MAUDE client.
#

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║               MAUDE Client Setup                              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    echo "Install Python from https://python.org"
    exit 1
fi

echo "Python 3 found: $(python3 --version)"

# Install dependencies
echo
echo "Installing Python dependencies..."
pip3 install requests

# Create config directory
echo
echo "Creating config directory..."
mkdir -p ~/.maude/transfers

# Check Tailscale connectivity
echo
echo "Checking Tailscale connection to Spark server..."
if ping -c 1 -W 3 spark-e26c > /dev/null 2>&1; then
    echo "Tailscale connection OK"
else
    echo "Warning: Cannot reach spark-e26c via Tailscale."
    echo "Make sure Tailscale is installed and connected:"
    echo "  https://tailscale.com/download"
fi

# Create launcher script
echo
echo "Creating launcher script..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cat > ~/.maude/start_client.sh << EOF
#!/bin/bash
# MAUDE Client Launcher (via Tailscale)

# Check Tailscale connectivity
if ! ping -c 1 -W 2 spark-e26c > /dev/null 2>&1; then
    echo "Error: Cannot reach spark-e26c. Is Tailscale connected?"
    exit 1
fi

# Run client
cd "$SCRIPT_DIR"
python3 maude_client.py
EOF

chmod +x ~/.maude/start_client.sh

# Create alias
echo
echo "Adding 'maude' alias to shell config..."
SHELL_RC=""
if [ -f ~/.zshrc ]; then
    SHELL_RC=~/.zshrc
elif [ -f ~/.bashrc ]; then
    SHELL_RC=~/.bashrc
fi

if [ -n "$SHELL_RC" ]; then
    if ! grep -q "alias maude=" "$SHELL_RC" 2>/dev/null; then
        echo "" >> "$SHELL_RC"
        echo "# MAUDE Client" >> "$SHELL_RC"
        echo "alias maude='~/.maude/start_client.sh'" >> "$SHELL_RC"
        echo "Added alias to $SHELL_RC"
    else
        echo "Alias already exists in $SHELL_RC"
    fi
fi

echo
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║               Setup Complete!                                 ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo
echo "To start MAUDE client:"
echo "  1. Open a new terminal (to load the alias)"
echo "  2. Run: maude"
echo
echo "Or manually:"
echo "  ~/.maude/start_client.sh"
echo
echo "Make sure Tailscale is connected before running."
