#!/bin/bash
cd /home/mboard76/nvidia-workbench/terminal-llm

echo "Starting Gemma 3 12B server on port 30000..."
/home/mboard76/llama.cpp/build/bin/llama-server \
    --model ./models/gemma-3-12b-it-Q6_K.gguf \
    --host 0.0.0.0 \
    --port 30000 \
    --n-gpu-layers 99 \
    --ctx-size 8192 \
    --threads 8
