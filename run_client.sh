#!/bin/bash
# Run MAUDE terminal as a client connecting to Spark
# For use on PC/Mac machines that don't have local LLMs

cd "$(dirname "$0")"

# Point to Spark's LLM server via Tailscale
export LLM_SERVER_URL="${LLM_SERVER_URL:-http://spark-e26c:30000/v1}"
export VISION_SERVER_URL="${VISION_SERVER_URL:-http://spark-e26c:11434/v1}"

# ComfyUI on mattwell (for video generation)
export COMFYUI_HOST="${COMFYUI_HOST:-mattwell}"
export COMFYUI_PORT="${COMFYUI_PORT:-8188}"

# Enable mesh for multi-node capabilities
export MAUDE_MESH_ENABLED=true

echo "MAUDE Terminal Client"
echo "====================="
echo "LLM Server: $LLM_SERVER_URL"
echo "Vision Server: $VISION_SERVER_URL"
echo ""

# Check if we can reach Spark
if curl -s --connect-timeout 2 "$LLM_SERVER_URL/models" > /dev/null 2>&1; then
    echo "Connected to Spark LLM server"
else
    echo "WARNING: Cannot reach Spark at $LLM_SERVER_URL"
    echo "Make sure:"
    echo "  1. Tailscale is connected"
    echo "  2. LLM server is running on Spark"
    echo ""
fi

# Activate venv and run
if [ -d "venv" ]; then
    source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
fi

python3 chat_local.py "$@"
