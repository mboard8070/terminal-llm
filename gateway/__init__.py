"""
MAUDE Gateway — Single port proxy for multi-model LLM + file server + SSH + PWA.

Sits on port 30000. Routes:
  /v1/*          -> Multi-model routing (Mistral, Codestral, Nemotron, LLaVA)
  /models        -> Available models list
  /ws/terminal   -> SSH WebSocket proxy (PTY)
  /list          -> shared folder listing
  /download/*    -> download from shared
  /upload/*      -> upload to transfers
  /share/*       -> upload to shared
  /transfers     -> list transfers
  /health        -> health check
  /proxy?url=    -> Web proxy for browser app
  /app/*         -> PWA static files (maude-phone/dist/)
  /              -> Redirect to /app/
"""

# Re-export main entry points for backwards compatibility.
# `from gateway import GatewayHandler` and `from gateway import main` both work.
from .main import main
from .server import GatewayHandler, ThreadedHTTPServer
from .state import (
    GATEWAY_PORT,
    LLM_PORT,
    MODEL_ALIASES,
    MODEL_ROUTES,
    SHARED_DIR,
    TRANSFERS_DIR,
    VOICE_PORT,
    get_model_route,
)

__all__ = [
    "GATEWAY_PORT",
    "LLM_PORT",
    "MODEL_ALIASES",
    "MODEL_ROUTES",
    "SHARED_DIR",
    "TRANSFERS_DIR",
    "VOICE_PORT",
    "GatewayHandler",
    "ThreadedHTTPServer",
    "get_model_route",
    "main",
]
