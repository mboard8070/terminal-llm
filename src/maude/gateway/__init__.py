"""MAUDE gateway package.

The canonical gateway implementation now lives under `maude.gateway`.
The top-level `gateway` package remains as a compatibility shim.
"""

from .governance import CapabilityPolicy, RequestContext
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
    "CapabilityPolicy",
    "GatewayHandler",
    "RequestContext",
    "ThreadedHTTPServer",
    "get_model_route",
    "main",
]
