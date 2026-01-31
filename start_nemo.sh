#!/bin/bash
cd /home/mboard76/nvidia-workbench/terminal-llm

echo "Starting Nemotron 30B server on port 30000..."
./llama.cpp/build/bin/llama-server \
    --model ./models/Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf \
    --host 0.0.0.0 \
    --port 30000 \
    --n-gpu-layers 99 \
    --ctx-size 32768 \
    --threads 8
