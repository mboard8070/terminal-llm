#!/bin/bash
cd /home/mboard76/nvidia-workbench/terminal-llm
source venv/bin/activate

# Start llama-server on internal port 30010
echo "Starting Nemotron on internal port 30010..."
./llama.cpp/build/bin/llama-server \
    --model ./models/Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf \
    --host 127.0.0.1 \
    --port 30010 \
    --n-gpu-layers 99 \
    --ctx-size 32768 \
    --threads 8 &
LLM_PID=$!

# Wait for LLM to be ready
echo "Waiting for LLM..."
for i in $(seq 1 60); do
    curl -s http://localhost:30010/v1/models > /dev/null 2>&1 && break
    sleep 1
done

# Start gateway on port 30000 (LLM + file server)
echo "Starting Gateway on port 30000 (LLM + files)..."
python3 gateway.py &
GW_PID=$!

echo ""
echo "MAUDE Gateway ready on port 30000"
echo "  /v1/*       -> LLM (Nemotron)"
echo "  /list       -> shared folder"
echo "  /download/* -> pull files"
echo "  /upload/*   -> push files"
echo ""
echo "Client connects via Tailscale to spark-e26c:30000"
echo ""

trap "kill $LLM_PID $GW_PID 2>/dev/null" EXIT
wait
