"""MAUDE client configuration.

Values default to a local gateway so the portable client works out of the box.
Set environment variables to target a remote MAUDE server.
"""

import os
from urllib.parse import urlparse


def _with_v1(path: str) -> str:
    url = path.rstrip("/")
    return url if url.endswith("/v1") else f"{url}/v1"


def _without_v1(path: str) -> str:
    url = path.rstrip("/")
    return url[:-3].rstrip("/") if url.endswith("/v1") else url


_gateway_source = (
    os.environ.get("MAUDE_GATEWAY_URL")
    or os.environ.get("LLM_SERVER_URL")
    or os.environ.get("MAUDE_GATEWAY_BASE_URL")
)

if _gateway_source:
    GATEWAY_URL = _with_v1(_gateway_source)
    GATEWAY_BASE_URL = _without_v1(GATEWAY_URL)
else:
    _host = os.environ.get("MAUDE_GATEWAY_HOST", "127.0.0.1")
    _port = int(os.environ.get("MAUDE_GATEWAY_PORT", "8080"))
    GATEWAY_BASE_URL = f"http://{_host}:{_port}"
    GATEWAY_URL = f"{GATEWAY_BASE_URL}/v1"

_parsed_gateway = urlparse(GATEWAY_BASE_URL)

# Server connection
SERVER_HOST = _parsed_gateway.hostname or os.environ.get("MAUDE_GATEWAY_HOST", "127.0.0.1")
SERVER_LLM_PORT = _parsed_gateway.port or int(os.environ.get("MAUDE_GATEWAY_PORT", "8080"))
SERVER_FILE_PORT = SERVER_LLM_PORT
SERVER_SSH_HOST = os.environ.get("MAUDE_SERVER_SSH_HOST") or os.environ.get("SERVER_SSH_HOST") or ""

# File transfer settings
SERVER_WORK_DIR = "~/nvidia-workbench/terminal-llm"
LOCAL_TRANSFER_DIR = "~/.maude/transfers"

# Shared folder settings
LOCAL_SHARED_DIR = "~/.maude/shared"
SERVER_SHARED_DIR = "~/nvidia-workbench/terminal-llm/shared"
SYNC_INTERVAL = 30  # seconds, same as heartbeat

# File server URL
FILE_SERVER_URL = os.environ.get("MAUDE_FILE_SERVER_URL", GATEWAY_BASE_URL)

# Model settings (must match server)
MODEL_NAME = "mistral"
CONTEXT_SIZE = 32768
TEMPERATURE = 0.7

# Client identity (shown in logs)
CLIENT_NAME = "maude-client"
