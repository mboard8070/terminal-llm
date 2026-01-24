#!/bin/bash
# Setup script for PersonaPlex on NVIDIA GB10 (DGX Spark)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERSONAPLEX_DIR="$SCRIPT_DIR/personaplex"

echo "=========================================="
echo "  PersonaPlex Setup for NVIDIA GB10"
echo "=========================================="

# Load .env file if exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "Loading environment from .env..."
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Check for HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "WARNING: HF_TOKEN not set!"
    echo "You need to:"
    echo "  1. Go to https://huggingface.co/nvidia/personaplex-7b-v1"
    echo "  2. Accept the license agreement"
    echo "  3. Create a token at https://huggingface.co/settings/tokens"
    echo "  4. Run: export HF_TOKEN=your_token_here"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install system dependencies
echo ""
echo "[1/5] Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y libopus-dev libopus0 ffmpeg
elif command -v dnf &> /dev/null; then
    sudo dnf install -y opus-devel opus ffmpeg
else
    echo "Please install libopus manually for your system"
fi

# Clone PersonaPlex if not exists
echo ""
echo "[2/5] Cloning PersonaPlex repository..."
if [ -d "$PERSONAPLEX_DIR" ]; then
    echo "PersonaPlex directory exists, pulling latest..."
    cd "$PERSONAPLEX_DIR"
    git pull
else
    git clone https://github.com/NVIDIA/personaplex.git "$PERSONAPLEX_DIR"
    cd "$PERSONAPLEX_DIR"
fi

# Setup virtual environment if using project venv
echo ""
echo "[3/5] Setting up Python environment..."
if [ -d "$SCRIPT_DIR/venv" ]; then
    VENV_DIR="$SCRIPT_DIR/venv"
    source "$VENV_DIR/bin/activate"
    PIP="$VENV_DIR/bin/pip"
    PYTHON="$VENV_DIR/bin/python"
else
    PIP="pip"
    PYTHON="python3"
fi

# Install PyTorch with CUDA 13.0 for Blackwell
echo ""
echo "[4/5] Installing PyTorch with CUDA 13.0 (Blackwell support)..."
$PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install PersonaPlex (moshi)
echo ""
echo "[5/5] Installing PersonaPlex (moshi)..."
cd "$PERSONAPLEX_DIR"
$PIP install ./moshi/
$PIP install accelerate  # For CPU offload if needed

echo ""
echo "=========================================="
echo "  PersonaPlex Setup Complete!"
echo "=========================================="
echo ""
echo "To run PersonaPlex server:"
echo "  export HF_TOKEN=your_token"
echo "  SSL_DIR=\$(mktemp -d); python -m moshi.server --ssl \"\$SSL_DIR\""
echo ""
echo "Then open: https://localhost:8998"
echo ""
echo "Available voices:"
echo "  Natural: NATF0-3 (female), NATM0-3 (male)"
echo "  Variety: VARF0-4 (female), VARM0-4 (male)"
echo ""
