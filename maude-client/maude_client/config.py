"""
MAUDE Client Configuration

Edit these settings for your environment.
Override with MAUDE_SERVER_HOST / MAUDE_SERVER_PORT if needed.
"""

import os

# Server connection (via Tailscale)
SERVER_HOST = os.environ.get("MAUDE_SERVER_HOST", "desktop-aveak19")
SERVER_LLM_PORT = int(os.environ.get("MAUDE_SERVER_PORT", "30000"))
SERVER_FILE_PORT = int(os.environ.get("MAUDE_FILE_PORT", str(SERVER_LLM_PORT)))
SERVER_SSH_HOST = os.environ.get("MAUDE_SSH_HOST", "Matt@desktop-aveak19")

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
MODEL_NAME = "nemotron-super"
CONTEXT_SIZE = 32768
TEMPERATURE = 0.7

# Client identity (shown in logs)
CLIENT_NAME = "maude-client"
