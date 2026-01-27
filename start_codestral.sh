#!/bin/bash
cd /home/mboard76/nvidia-workbench/terminal-llm

echo "Starting Codestral 22B server on port 30001..."
/home/mboard76/llama.cpp/build/bin/llama-server \
    --model ./models/codestral-22b-v0.1-Q6_K.gguf \
    --host 0.0.0.0 \
    --port 30001 \
    --n-gpu-layers 99 \
    --ctx-size 16384 \
    --threads 8
