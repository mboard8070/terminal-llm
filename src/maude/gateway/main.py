"""
Main entry point for the MAUDE Gateway.

Configures logging, starts the HTTP(S) server, and runs the gateway heartbeat.
"""

import asyncio
import json
import logging
import os
import ssl
import threading
import time
from pathlib import Path
from urllib import request as urlrequest

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


def _start_scheduler_thread():
    """Start the proactive scheduler inside the gateway process."""

    async def _scheduled_prompt_callback(prompt: str) -> str:
        model = os.environ.get("MAUDE_SCHEDULER_MODEL", "codex")
        max_tokens = int(os.environ.get("MAUDE_SCHEDULER_MAX_TOKENS", "2048"))
        timeout = int(os.environ.get("MAUDE_SCHEDULER_TIMEOUT", "3600"))
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are MAUDE's scheduler runner. Execute the scheduled task using tools when needed. "
                        "For mission prompts, call the requested mission tool and return a concise result."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }

        def _post() -> str:
            data = json.dumps(payload).encode("utf-8")
            req = urlrequest.Request(
                "http://localhost:30080/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlrequest.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if "error" in body:
                return f"Error: {body['error']}"
            return body.get("choices", [{}])[0].get("message", {}).get("content", "") or "(no response)"

        return await asyncio.to_thread(_post)

    async def _run_scheduler():
        from scheduler import get_scheduler

        scheduler = get_scheduler()
        scheduler.set_maude_callback(_scheduled_prompt_callback)
        await scheduler.start()
        enabled = len([task for task in scheduler.tasks.values() if task.enabled])
        logger.info("  Scheduler  : started with %d enabled tasks", enabled)
        while True:
            await asyncio.sleep(3600)

    def _thread_main():
        try:
            asyncio.run(_run_scheduler())
        except Exception as exc:
            logger.error("  Scheduler failed: %s", exc)

    threading.Thread(target=_thread_main, daemon=True, name="maude-scheduler").start()


def main():
    """Start the MAUDE Gateway server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    CERT_DIR = Path(__file__).resolve().parents[3] / "certs"
    USE_SSL = (CERT_DIR / "cert.pem").exists() and (CERT_DIR / "key.pem").exists()
    PHONE_HTTP_PRIMARY = os.environ.get("MAUDE_PHONE_HTTP_PRIMARY", "false").lower() in ("1", "true", "yes")

    logger.info("MAUDE Gateway on port %d (%s)", GATEWAY_PORT, "HTTP" if PHONE_HTTP_PRIMARY else ("HTTPS" if USE_SSL else "HTTP"))
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

    _start_scheduler_thread()

    server = ThreadedHTTPServer(("0.0.0.0", GATEWAY_PORT), GatewayHandler)

    if USE_SSL and not PHONE_HTTP_PRIMARY:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(CERT_DIR / "cert.pem"), str(CERT_DIR / "key.pem"))
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        logger.info("  SSL cert   : %s", CERT_DIR / "cert.pem")

        HTTP_PORT = 30080
        http_server = ThreadedHTTPServer(("0.0.0.0", HTTP_PORT), GatewayHandler)
        http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
        http_thread.start()
        logger.info("  HTTP mirror: port %d", HTTP_PORT)
    elif USE_SSL:
        HTTPS_PORT = 30080
        https_server = ThreadedHTTPServer(("0.0.0.0", HTTPS_PORT), GatewayHandler)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(CERT_DIR / "cert.pem"), str(CERT_DIR / "key.pem"))
        https_server.socket = ctx.wrap_socket(https_server.socket, server_side=True)
        https_thread = threading.Thread(target=https_server.serve_forever, daemon=True)
        https_thread.start()
        logger.info("  HTTP primary: port %d (phone app)", GATEWAY_PORT)
        logger.info("  HTTPS mirror: port %d", HTTPS_PORT)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping.")
