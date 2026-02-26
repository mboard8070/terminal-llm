"""
MAUDE Client Configuration

Edit these settings for your environment.
"""

# Server connection (via Tailscale)
SERVER_HOST = "spark-e26c"
SERVER_LLM_PORT = 30000  # Spark's Nemotron via gateway
SERVER_FILE_PORT = 30000  # Same port as LLM — gateway handles both
SERVER_SSH_HOST = "mboard76@spark-e26c"

# File transfer settings
SERVER_WORK_DIR = "~/nvidia-workbench/terminal-llm"
LOCAL_TRANSFER_DIR = "~/.maude/transfers"

# Shared folder settings
LOCAL_SHARED_DIR = "~/.maude/shared"
SERVER_SHARED_DIR = "~/nvidia-workbench/terminal-llm/shared"
SYNC_INTERVAL = 30  # seconds, same as heartbeat

# File server URL (via Tailscale, no SSH tunnel needed)
FILE_SERVER_URL = f"https://{SERVER_HOST}:{SERVER_FILE_PORT}"

# Model settings (must match server)
MODEL_NAME = "claude-opus-4-20250514"
CONTEXT_SIZE = 32768
TEMPERATURE = 0.7

# Client identity (shown in logs)
CLIENT_NAME = "maude-client"
