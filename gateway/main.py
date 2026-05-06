"""
Main entry point for the MAUDE Gateway.

Configures logging, starts the HTTP(S) server, and runs the gateway heartbeat.
"""

import logging
import ssl
import threading
import time
from pathlib import Path

from .server import GatewayHandler, ThreadedHTTPServer
from .state import (
    GATEWAY_PORT,
    LLM_PORT,
    MODEL_ROUTES,
    PWA_DIR,
    SHARED_DIR,
    TRANSFERS_DIR,
    VOICE_PORT,
    logger,
)


def main():
    """Start the MAUDE Gateway server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    CERT_DIR = Path(__file__).parent.parent / "certs"
    USE_SSL = (CERT_DIR / "cert.pem").exists() and (CERT_DIR / "key.pem").exists()

    logger.info("MAUDE Gateway on port %d (%s)", GATEWAY_PORT, "HTTPS" if USE_SSL else "HTTP")
    logger.info("  LLM proxy  -> localhost:%d", LLM_PORT)
    logger.info("  Voice      -> localhost:%d", VOICE_PORT)
    logger.info("  PWA dir    : %s", PWA_DIR)
    logger.info("  Shared     : %s", SHARED_DIR)
    logger.info("  Transfers  : %s", TRANSFERS_DIR)
    logger.info("  Models     : %s", ", ".join(MODEL_ROUTES.keys()))

    # Startup health check
    try:
        from health import HealthChecker

        _report = HealthChecker().check_all()
        logger.info("  Health     : %s", _report["status"])
        for _dep, _info in _report.get("dependencies", {}).items():
            if not _info.get("available"):
                logger.warning("  %s — %s", _dep, _info.get("error", "not available"))
        _degraded = _report.get("tools", {}).get("degraded", [])
        if _degraded:
            logger.warning("  Degraded tools (%d): %s", len(_degraded), ", ".join(_degraded[:10]))
    except Exception as _e:
        logger.error("  Health check failed: %s", _e)

    # Start gateway-level presence heartbeat so this node always appears
    def _gateway_heartbeat():
        import socket

        from collab import get_hub

        hub = get_hub()
        hostname = socket.gethostname().lower()
        while True:
            try:
                hub.heartbeat(f"gateway-{hostname}", "gateway", "serving requests")
            except Exception:
                pass
            time.sleep(30)

    threading.Thread(target=_gateway_heartbeat, daemon=True).start()
    logger.info("  Collab     : heartbeat started")

    server = ThreadedHTTPServer(("0.0.0.0", GATEWAY_PORT), GatewayHandler)

    if USE_SSL:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(CERT_DIR / "cert.pem"), str(CERT_DIR / "key.pem"))
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        logger.info("  SSL cert   : %s", CERT_DIR / "cert.pem")

        # Also start an HTTP server for the native Android app
        HTTP_PORT = 30080
        http_server = ThreadedHTTPServer(("0.0.0.0", HTTP_PORT), GatewayHandler)
        http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
        http_thread.start()
        logger.info("  HTTP mirror: port %d (for native app)", HTTP_PORT)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping.")
