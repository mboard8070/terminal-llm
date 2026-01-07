#!/bin/bash
set -e

echo "=== Setting up Local Nemotron-3-Nano-30B ==="

cd /home/mboard76/nvidia-workbench/terminal-llm

# Step 1: Create venv and install huggingface_hub
echo "[1/4] Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -U "huggingface_hub[cli]"

# Step 2: Clone and build llama.cpp with CUDA
echo "[2/4] Building llama.cpp with CUDA support..."
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggml-org/llama.cpp
fi

cd llama.cpp
mkdir -p build && cd build

# Build with CUDA - sm_121 for GB10, but let's detect it
cmake .. -DGGML_CUDA=ON -DLLAMA_CURL=OFF
make -j$(nproc)

cd ../..

# Step 3: Download the model (~38GB)
echo "[3/4] Downloading Nemotron-3-Nano-30B model (~38GB)..."
mkdir -p models
./venv/bin/python -c "
from huggingface_hub import hf_hub_download
print('Downloading... this will take a while for ~38GB')
hf_hub_download(
    repo_id='unsloth/Nemotron-3-Nano-30B-A3B-GGUF',
    filename='Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf',
    local_dir='./models'
)
print('Download complete!')
"

echo "[4/4] Setup complete!"
echo ""
echo "To start the server, run:"
echo "  ./start_server.sh"
echo ""
echo "Then use the chat with:"
echo "  ./chat_local.py"
