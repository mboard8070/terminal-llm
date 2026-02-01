"""
MAUDE Client Configuration

Edit these settings for your environment.
"""

# Server connection (via SSH tunnel)
SERVER_HOST = "localhost"
SERVER_LLM_PORT = 30000  # Tunneled to Spark's Nemotron
SERVER_SSH_HOST = "mboard76@spark-e26c"

# File transfer settings
SERVER_WORK_DIR = "~/nvidia-workbench/terminal-llm"
LOCAL_TRANSFER_DIR = "~/.maude/transfers"

# Model settings (must match server)
MODEL_NAME = "nemotron"
CONTEXT_SIZE = 32768
TEMPERATURE = 0.7

# Client identity (shown in logs)
CLIENT_NAME = "maude-client"
