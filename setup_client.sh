#!/bin/bash
# Setup MAUDE as a client on Mac/PC
# Connects to Spark via Tailscale for LLM inference

set -e

echo "MAUDE Client Setup"
echo "=================="
echo ""

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="mac"
    echo "Detected: macOS"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    OS="windows"
    echo "Detected: Windows"
else
    OS="linux"
    echo "Detected: Linux"
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "Step 1: Check Python..."
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "ERROR: Python not found. Please install Python 3.10+"
    exit 1
fi
echo "  Using: $PYTHON ($($PYTHON --version))"

echo ""
echo "Step 2: Create/update virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON -m venv venv
    echo "  Created venv"
else
    echo "  venv exists"
fi

# Activate venv
if [[ "$OS" == "windows" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo ""
echo "Step 3: Install dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "  Dependencies installed"

echo ""
echo "Step 4: Configure environment..."

# Create client .env if it doesn't exist
if [ ! -f ".env.client" ]; then
    cat > .env.client << 'EOF'
# MAUDE Client Configuration
# Points to Spark for LLM inference via Tailscale

# Spark LLM Server (via Tailscale)
LLM_SERVER_URL=http://spark-e26c:30000/v1
OLLAMA_URL=http://spark-e26c:11434/v1

# Telegram Bot Token (get from @BotFather)
TELEGRAM_BOT_TOKEN=

# Optional: API keys for cloud fallback
# ANTHROPIC_API_KEY=
# OPENAI_API_KEY=
# GOOGLE_API_KEY=
EOF
    echo "  Created .env.client - EDIT THIS FILE to add your TELEGRAM_BOT_TOKEN"
else
    echo "  .env.client exists"
fi

echo ""
echo "Step 5: Create run script..."

if [[ "$OS" == "windows" ]]; then
    # Windows batch file
    cat > run_maude_client.bat << 'EOF'
@echo off
cd /d "%~dp0"
call venv\Scripts\activate
set /p vars=<.env.client
for /f "tokens=*" %%a in (.env.client) do (
    set "%%a"
)
python maude_telegram_service.py
EOF
    echo "  Created run_maude_client.bat"
else
    # Unix shell script
    cat > run_maude_client.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
export $(grep -v '^#' .env.client | xargs)
python maude_telegram_service.py
EOF
    chmod +x run_maude_client.sh
    echo "  Created run_maude_client.sh"
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env.client and add your TELEGRAM_BOT_TOKEN"
echo "  2. Make sure Spark is running (LLM server on port 30000)"
echo "  3. Run MAUDE:"
if [[ "$OS" == "windows" ]]; then
    echo "     run_maude_client.bat"
else
    echo "     ./run_maude_client.sh"
fi
echo ""
echo "Spark address: http://spark-e26c:30000/v1"
echo "  (or use IP: http://100.107.132.16:30000/v1)"
echo ""
