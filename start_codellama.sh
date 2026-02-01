#!/bin/bash
cd /home/mboard76/nvidia-workbench/terminal-llm

echo "Starting Code Llama 7B server on port 30000..."
./llama.cpp/build/bin/llama-server \
    --model ./models/codellama-7b-instruct.Q6_K.gguf \
    --host 0.0.0.0 \
    --port 30000 \
    --n-gpu-layers 99 \
    --ctx-size 16384 \
    --threads 8 \
    --chat-template llama2
