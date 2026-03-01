"""
Configuration constants — can be overridden via environment variables.
"""

import os

LOCAL_URL = os.environ.get("LLM_SERVER_URL", "http://localhost:30080/v1")
MODEL = os.environ.get("MAUDE_MODEL", "nemotron")
NUM_CTX = int(os.environ.get("MAUDE_NUM_CTX", "32768"))
VISION_URL = os.environ.get("VISION_SERVER_URL", "http://localhost:11434/v1")
VISION_MODEL = os.environ.get("MAUDE_VISION_MODEL", "llava:13b")

# Shared session ID for conversation sync
SESSION_ID = os.environ.get("MAUDE_SESSION_ID", "default")
