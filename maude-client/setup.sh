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

# Check SSH key
echo
echo "Checking SSH connection to Spark server..."
if ssh -o BatchMode=yes -o ConnectTimeout=5 mboard76@spark-e26c echo "OK" 2>/dev/null; then
    echo "SSH connection OK"
else
    echo "Warning: Cannot connect to Spark server automatically."
    echo "Make sure you have SSH key authentication set up:"
    echo "  ssh-copy-id mboard76@spark-e26c"
fi

# Create launcher script
echo
echo "Creating launcher script..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cat > ~/.maude/start_client.sh << EOF
#!/bin/bash
# MAUDE Client Launcher

# Start SSH tunnel in background if not already running
if ! pgrep -f "ssh.*30000:localhost:30000.*spark-e26c" > /dev/null; then
    echo "Starting SSH tunnel..."
    ssh -L 30000:localhost:30000 mboard76@spark-e26c -N &
    sleep 2
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
echo "The client will automatically start the SSH tunnel if needed."
