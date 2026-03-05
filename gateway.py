#!/usr/bin/env python3
"""
MAUDE Gateway - Single port proxy for multi-model LLM + file server + SSH + PWA.

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
  /app/*         -> PWA static files (personaplex/client/dist/)
  /              -> Redirect to /app/
"""

import os
import sys
import json
import re
import ssl
import logging
import mimetypes
import hashlib
import struct
import base64
import subprocess
import select
import threading
import pty
import signal
import fcntl
import termios
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import unquote, urlparse, parse_qs, urljoin
import http.client
import time
import uuid

logger = logging.getLogger("maude.gateway")

# Tool execution support for cloud model requests
try:
    import maude_core
    from maude_core import TOOLS as ALL_TOOLS, execute_tool, get_tools_for_message, reset_rate_limits
    TOOL_SUPPORT = True
    logger.info("Tool support enabled via maude_core")
except ImportError as e:
    logger.warning("maude_core not available, tool support disabled: %s", e)
    TOOL_SUPPORT = False

# Load environment from variables.env
ENV_FILE = Path(__file__).parent / "variables.env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

SHARED_DIR = Path.home() / "nvidia-workbench" / "terminal-llm" / "shared"
TRANSFERS_DIR = Path.home() / "nvidia-workbench" / "terminal-llm" / "transfers"
PWA_DIR = Path(__file__).parent / "maude-phone" / "dist"
LLM_PORT = 30010  # llama-server (Nemotron) runs here internally
GATEWAY_PORT = 30000  # clients connect here
PERSONAPLEX_PORT = 8998  # PersonaPlex voice server

CONVERSATIONS_DIR = Path(__file__).parent / "data" / "conversations"

SHARED_DIR.mkdir(parents=True, exist_ok=True)
TRANSFERS_DIR.mkdir(parents=True, exist_ok=True)
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Model routing configuration
# ─────────────────────────────────────────────────────────────────

MODEL_ROUTES = {
    # Mistral Large — general conversation (MAUDE default)
    "mistral-large-latest": {
        "provider": "mistral",
        "base_url": "https://api.mistral.ai",
        "api_key_env": "MISTRAL_API_KEY",
        "max_context": 128000,
    },
    # Codestral — code tasks (uses separate endpoint + key)
    "codestral-latest": {
        "provider": "mistral",
        "base_url": "https://codestral.mistral.ai",
        "api_key_env": "CODESTRAL_API_KEY",
        "max_context": 32000,
    },
    # Devstral 2 — frontier code agents (123B, 256K context)
    "devstral-2512": {
        "provider": "mistral",
        "base_url": "https://api.mistral.ai",
        "api_key_env": "MISTRAL_API_KEY",
        "max_context": 256000,
    },
    # Devstral Small — lightweight code agents
    "devstral-small-latest": {
        "provider": "mistral",
        "base_url": "https://api.mistral.ai",
        "api_key_env": "MISTRAL_API_KEY",
        "max_context": 32000,
    },
    # Devstral Medium — mid-tier code agents
    "devstral-medium-latest": {
        "provider": "mistral",
        "base_url": "https://api.mistral.ai",
        "api_key_env": "MISTRAL_API_KEY",
        "max_context": 128000,
    },
    # Nemotron — local fallback (llama-server)
    "nemotron": {
        "provider": "local",
        "base_url": f"http://localhost:{LLM_PORT}",
        "api_key_env": None,
        "max_context": 8192,
    },
    # LLaVA — vision (local)
    "llava": {
        "provider": "local",
        "base_url": f"http://localhost:{LLM_PORT}",
        "api_key_env": None,
        "max_context": 4096,
    },
    # Claude Opus — most capable, deep reasoning
    "claude-opus-4-20250514": {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_key_env": "CLAUDE_API_KEY",
        "max_context": 200000,
    },
    # Claude Sonnet — fast, capable, cost-effective
    "claude-sonnet-4-20250514": {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_key_env": "CLAUDE_API_KEY",
        "max_context": 200000,
    },
}

# Aliases for convenience
MODEL_ALIASES = {
    "mistral": "mistral-large-latest",
    "codestral": "codestral-latest",
    "devstral": "devstral-2512",
    "devstral-small": "devstral-small-latest",
    "devstral-medium": "devstral-medium-latest",
    "local": "nemotron",
    "vision": "llava",
    "claude": "claude-opus-4-20250514",
    "sonnet": "claude-sonnet-4-20250514",
}


def get_model_route(model_name):
    """Resolve model name to routing config."""
    name = MODEL_ALIASES.get(model_name, model_name)
    return name, MODEL_ROUTES.get(name)


# ─────────────────────────────────────────────────────────────────
# WebSocket helpers (RFC 6455)
# ─────────────────────────────────────────────────────────────────

WS_MAGIC = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

def ws_accept_key(key):
    """Compute Sec-WebSocket-Accept from client key."""
    return base64.b64encode(
        hashlib.sha1(key.encode() + WS_MAGIC).digest()
    ).decode()


def ws_encode_frame(data, opcode=0x01):
    """Encode a WebSocket frame (text=0x01, binary=0x02, close=0x08, ping=0x09, pong=0x0A)."""
    frame = bytearray()
    frame.append(0x80 | opcode)  # FIN + opcode
    if isinstance(data, str):
        data = data.encode("utf-8")
    length = len(data)
    if length < 126:
        frame.append(length)
    elif length < 65536:
        frame.append(126)
        frame.extend(struct.pack(">H", length))
    else:
        frame.append(127)
        frame.extend(struct.pack(">Q", length))
    frame.extend(data)
    return bytes(frame)


def ws_decode_frame(raw):
    """Decode a WebSocket frame. Returns (opcode, payload, bytes_consumed)."""
    if len(raw) < 2:
        return None, None, 0
    b0, b1 = raw[0], raw[1]
    opcode = b0 & 0x0F
    masked = b1 & 0x80
    length = b1 & 0x7F
    offset = 2
    if length == 126:
        if len(raw) < 4:
            return None, None, 0
        length = struct.unpack(">H", raw[2:4])[0]
        offset = 4
    elif length == 127:
        if len(raw) < 10:
            return None, None, 0
        length = struct.unpack(">Q", raw[2:10])[0]
        offset = 10
    if masked:
        if len(raw) < offset + 4 + length:
            return None, None, 0
        mask = raw[offset:offset + 4]
        offset += 4
        payload = bytearray(raw[offset:offset + length])
        for i in range(length):
            payload[i] ^= mask[i % 4]
        payload = bytes(payload)
    else:
        if len(raw) < offset + length:
            return None, None, 0
        payload = raw[offset:offset + length]
    return opcode, payload, offset + length


# ─────────────────────────────────────────────────────────────────
# SSH WebSocket Terminal
# ─────────────────────────────────────────────────────────────────

def handle_terminal_websocket(handler):
    """Handle a WebSocket connection for terminal access."""
    # Complete WebSocket handshake
    key = handler.headers.get("Sec-WebSocket-Key", "")
    accept = ws_accept_key(key)
    handler.send_response(101)
    handler.send_header("Upgrade", "websocket")
    handler.send_header("Connection", "Upgrade")
    handler.send_header("Sec-WebSocket-Accept", accept)
    handler.end_headers()

    sock = handler.request

    # Open a PTY with bash
    master_fd, slave_fd = pty.openpty()
    pid = os.fork()
    if pid == 0:
        # Child process
        os.close(master_fd)
        os.setsid()
        fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        os.dup2(slave_fd, 2)
        os.close(slave_fd)
        os.execvp("/bin/bash", ["/bin/bash", "--login"])
        sys.exit(0)

    os.close(slave_fd)

    # Set master_fd non-blocking
    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    ws_buffer = bytearray()
    running = True

    last_ping = [time.time()]

    def read_from_pty():
        """Read PTY output and send as WebSocket frames. Also sends keepalive pings."""
        nonlocal running
        while running:
            try:
                r, _, _ = select.select([master_fd], [], [], 0.1)
                if master_fd in r:
                    data = os.read(master_fd, 4096)
                    if not data:
                        running = False
                        break
                    frame = ws_encode_frame(data, opcode=0x02)  # binary
                    try:
                        sock.sendall(frame)
                    except Exception:
                        running = False
                        break
                # Send keepalive ping every 25 seconds
                now = time.time()
                if now - last_ping[0] > 25:
                    try:
                        sock.sendall(ws_encode_frame(b"keepalive", opcode=0x09))
                        last_ping[0] = now
                    except Exception:
                        running = False
                        break
            except OSError:
                running = False
                break

    pty_thread = threading.Thread(target=read_from_pty, daemon=True)
    pty_thread.start()

    try:
        while running:
            r, _, _ = select.select([sock], [], [], 0.1)
            if sock not in r:
                continue
            try:
                chunk = sock.recv(8192)
            except Exception:
                break
            if not chunk:
                break
            ws_buffer.extend(chunk)

            while True:
                opcode, payload, consumed = ws_decode_frame(ws_buffer)
                if opcode is None:
                    break
                ws_buffer = ws_buffer[consumed:]

                if opcode == 0x01 or opcode == 0x02:
                    # Text or binary — write to PTY
                    # Check for resize message
                    if opcode == 0x01:
                        try:
                            msg = json.loads(payload)
                            if isinstance(msg, dict) and msg.get("type") == "resize":
                                cols = msg.get("cols", 80)
                                rows = msg.get("rows", 24)
                                winsize = struct.pack("HHHH", rows, cols, 0, 0)
                                fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
                                continue
                            # Regular text input (including valid JSON that isn't a resize command)
                            os.write(master_fd, payload)
                        except (json.JSONDecodeError, ValueError):
                            os.write(master_fd, payload)
                    else:
                        os.write(master_fd, payload)
                elif opcode == 0x08:
                    # Close
                    running = False
                    break
                elif opcode == 0x09:
                    # Ping → Pong
                    sock.sendall(ws_encode_frame(payload, opcode=0x0A))
    except Exception:
        pass
    finally:
        running = False
        try:
            os.close(master_fd)
        except Exception:
            pass
        try:
            os.kill(pid, signal.SIGTERM)
            os.waitpid(pid, 0)
        except Exception:
            pass
        pty_thread.join(timeout=2)


# ─────────────────────────────────────────────────────────────────
# HTTP Terminal Transport (iOS fallback — SSE output + POST input)
# iOS WKWebView can't do WSS with self-signed certs, so we provide
# an HTTP-based terminal transport using SSE for output streaming
# and POST requests for input/resize.
# ─────────────────────────────────────────────────────────────────

_http_terminal_sessions = {}  # session_id -> {master_fd, pid, created}

# iOS WKWebView can't reliably stream fetch() responses with self-signed certs.
# Same pattern as terminal: POST to create session, GET EventSource for stream.
_chat_sessions = {}  # session_id -> {"req": dict, "created": float}


def _create_terminal_session():
    """Create a PTY session for HTTP-based terminal access. Returns session_id."""
    session_id = uuid.uuid4().hex[:12]
    master_fd, slave_fd = pty.openpty()
    pid = os.fork()
    if pid == 0:
        os.close(master_fd)
        os.setsid()
        fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        os.dup2(slave_fd, 2)
        os.close(slave_fd)
        os.execvp("/bin/bash", ["/bin/bash", "--login"])
        sys.exit(0)
    os.close(slave_fd)
    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    _http_terminal_sessions[session_id] = {
        "master_fd": master_fd,
        "pid": pid,
        "created": time.time(),
    }
    logger.info("HTTP terminal session created: %s (pid=%d)", session_id, pid)
    return session_id


def _cleanup_terminal_session(session_id):
    """Clean up a terminal session."""
    session = _http_terminal_sessions.pop(session_id, None)
    if session:
        try:
            os.close(session["master_fd"])
        except Exception:
            pass
        try:
            os.kill(session["pid"], signal.SIGTERM)
            os.waitpid(session["pid"], 0)
        except Exception:
            pass
        logger.info("HTTP terminal session cleaned up: %s", session_id)


def handle_terminal_stream(handler, session_id):
    """Stream PTY output as SSE events with base64-encoded data."""
    session = _http_terminal_sessions.get(session_id)
    if not session:
        handler.send_response(404)
        handler.end_headers()
        handler.wfile.write(b'{"error":"session not found"}')
        return

    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()

    master_fd = session["master_fd"]
    try:
        while session_id in _http_terminal_sessions:
            r, _, _ = select.select([master_fd], [], [], 0.1)
            if master_fd in r:
                try:
                    data = os.read(master_fd, 4096)
                except OSError:
                    break
                if not data:
                    break
                b64 = base64.b64encode(data).decode()
                handler.wfile.write(f"data: {b64}\n\n".encode())
                handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
    finally:
        _cleanup_terminal_session(session_id)


# ─────────────────────────────────────────────────────────────────
# PersonaPlex WebSocket Proxy
# ─────────────────────────────────────────────────────────────────

def handle_personaplex_proxy(handler):
    """Proxy WebSocket from client to PersonaPlex server on port 8998."""
    import socket as _socket

    # Complete WebSocket handshake with the client
    client_key = handler.headers.get("Sec-WebSocket-Key", "")
    accept = ws_accept_key(client_key)
    handler.send_response(101)
    handler.send_header("Upgrade", "websocket")
    handler.send_header("Connection", "Upgrade")
    handler.send_header("Sec-WebSocket-Accept", accept)
    handler.end_headers()

    client_sock = handler.request

    # Build the upstream WebSocket URL with query params
    parsed = urlparse(handler.path)
    upstream_path = parsed.path
    if parsed.query:
        upstream_path += "?" + parsed.query

    # Connect to PersonaPlex upstream (bridge runs with --ssl)
    upstream_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    upstream_sock.settimeout(10)
    try:
        upstream_sock.connect(("localhost", PERSONAPLEX_PORT))
        # Wrap with TLS — bridge uses a self-signed cert
        import ssl as _ssl
        ctx = _ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = _ssl.CERT_NONE
        upstream_sock = ctx.wrap_socket(upstream_sock, server_hostname="localhost")
    except Exception as e:
        try:
            client_sock.sendall(ws_encode_frame(f"PersonaPlex connection failed: {e}", opcode=0x08))
        except Exception:
            pass
        return

    # Send WebSocket upgrade to upstream
    import secrets
    ws_key = base64.b64encode(secrets.token_bytes(16)).decode()
    upgrade_request = (
        f"GET {upstream_path} HTTP/1.1\r\n"
        f"Host: localhost:{PERSONAPLEX_PORT}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {ws_key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"\r\n"
    )
    upstream_sock.sendall(upgrade_request.encode())

    # Read upstream handshake response
    response_data = b""
    while b"\r\n\r\n" not in response_data:
        chunk = upstream_sock.recv(4096)
        if not chunk:
            upstream_sock.close()
            return
        response_data += chunk

    if b"101" not in response_data.split(b"\r\n")[0]:
        upstream_sock.close()
        try:
            client_sock.sendall(ws_encode_frame("PersonaPlex handshake failed", opcode=0x08))
        except Exception:
            pass
        return

    # Any extra data after the HTTP headers is WebSocket data
    extra = response_data.split(b"\r\n\r\n", 1)[1] if b"\r\n\r\n" in response_data else b""

    upstream_sock.setblocking(False)
    running = True

    def upstream_to_client():
        """Forward raw bytes from upstream to client."""
        nonlocal running
        buf = bytearray(extra) if extra else bytearray()
        while running:
            try:
                r, _, _ = select.select([upstream_sock], [], [], 0.1)
                if upstream_sock in r:
                    data = upstream_sock.recv(65536)
                    if not data:
                        running = False
                        break
                    try:
                        client_sock.sendall(data)
                    except Exception:
                        running = False
                        break
            except Exception:
                running = False
                break

    def client_to_upstream():
        """Forward raw bytes from client to upstream."""
        nonlocal running
        while running:
            try:
                r, _, _ = select.select([client_sock], [], [], 0.1)
                if client_sock not in r:
                    continue
                data = client_sock.recv(65536)
                if not data:
                    running = False
                    break
                upstream_sock.sendall(data)
            except Exception:
                running = False
                break

    # Send any extra data from handshake to client
    if extra:
        try:
            client_sock.sendall(extra)
        except Exception:
            running = False

    t1 = threading.Thread(target=upstream_to_client, daemon=True)
    t2 = threading.Thread(target=client_to_upstream, daemon=True)
    t1.start()
    t2.start()

    t1.join()
    t2.join()

    try:
        upstream_sock.close()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────
# Threaded HTTP Server
# ─────────────────────────────────────────────────────────────────

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class GatewayHandler(BaseHTTPRequestHandler):
    # Use HTTP/1.1 — required for WebSocket upgrade responses (RFC 6455)
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        logger.debug("HTTP %s", format % args)

    def _add_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
        self.send_header("Access-Control-Max-Age", "86400")

    def do_OPTIONS(self):
        self.send_response(204)
        self._add_cors()
        self.end_headers()

    def do_GET(self):
        path = unquote(self.path)
        parsed = urlparse(path)
        query = parse_qs(parsed.query)

        # WebSocket upgrade for terminal
        if parsed.path == "/ws/terminal":
            upgrade = self.headers.get("Upgrade", "").lower()
            if upgrade == "websocket":
                handle_terminal_websocket(self)
                return
            self._json_response({"error": "WebSocket upgrade required"}, 400)
            return

        # WebSocket proxy for PersonaPlex voice
        if parsed.path == "/api/chat":
            upgrade = self.headers.get("Upgrade", "").lower()
            if upgrade == "websocket":
                handle_personaplex_proxy(self)
                return
            self._json_response({"error": "WebSocket upgrade required"}, 400)
            return

        # HTTP terminal SSE stream (iOS fallback)
        if parsed.path == "/api/terminal/stream":
            sid = query.get("sid", [None])[0]
            if sid:
                handle_terminal_stream(self, sid)
            else:
                self._json_response({"error": "missing sid"}, 400)
            return

        # HTTP chat SSE stream (iOS fallback — same pattern as terminal)
        if parsed.path == "/api/chat/stream":
            sid = query.get("sid", [None])[0]
            if not sid:
                self._json_response({"error": "missing sid"}, 400)
                return
            session = _chat_sessions.pop(sid, None)
            if not session:
                self._json_response({"error": "session not found"}, 404)
                return
            req = session["req"]
            req["stream"] = True
            self._eventstream_mode = True  # traces as named events for EventSource
            self._route_model_request(pre_parsed_req=req)
            return

        if parsed.path == "/list":
            req_path = query.get("path", [None])[0]
            if req_path:
                target = Path(req_path)
                if target.exists() and target.is_dir():
                    self._list_dir(target)
                else:
                    self._json_response({"error": "Directory not found"}, 404)
            else:
                self._list_dir(SHARED_DIR)
        elif parsed.path == "/transfers":
            self._list_dir(TRANSFERS_DIR)
        elif parsed.path.startswith("/download/"):
            self._send_file(SHARED_DIR / parsed.path[len("/download/"):])
        elif parsed.path.startswith("/download-transfer/"):
            self._send_file(TRANSFERS_DIR / parsed.path[len("/download-transfer/"):])
        elif parsed.path.startswith("/api/collab/"):
            self._handle_collab_get(parsed.path, query)
        elif parsed.path == "/api/conversations":
            self._get_conversations()
        elif parsed.path.startswith("/api/conversations/") and parsed.path.endswith("/messages"):
            conv_id = parsed.path.split("/")[3]
            self._get_messages(conv_id)
        elif parsed.path == "/health":
            self._serve_health()
        elif parsed.path == "/api/tools":
            self._serve_tools(query)
        elif parsed.path == "/models":
            self._serve_models()
        elif parsed.path.startswith("/proxy"):
            self._web_proxy(query)
        elif parsed.path.startswith("/v1"):
            self._proxy_to_llm()
        elif parsed.path.startswith("/app") or parsed.path == "/":
            self._serve_static(parsed.path)
        elif parsed.path == "/manifest.json":
            self._serve_static("/manifest.json")
        elif parsed.path.startswith("/assets"):
            self._serve_static(parsed.path)
        elif parsed.path in ("/maude", "/maude/voice", "/terminal", "/browser",
                              "/messages", "/files", "/settings", "/collab"):
            # SPA routes — serve index.html
            self._serve_static("/index.html")
        else:
            # Try static first, then proxy to LLM
            if self._try_serve_static(parsed.path):
                return
            self._proxy_to_llm()

    def do_POST(self):
        path = unquote(self.path)
        parsed = urlparse(path)

        # HTTP terminal endpoints (iOS fallback)
        if parsed.path == "/api/terminal/create":
            sid = _create_terminal_session()
            self._json_response({"sid": sid})
            return
        if parsed.path == "/api/terminal/input":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            sid = body.get("sid", "")
            data = body.get("data", "")
            session = _http_terminal_sessions.get(sid)
            if not session:
                self._json_response({"error": "session not found"}, 404)
                return
            try:
                os.write(session["master_fd"], data.encode() if isinstance(data, str) else data)
            except OSError:
                _cleanup_terminal_session(sid)
                self._json_response({"error": "session closed"}, 410)
                return
            self._json_response({"ok": True})
            return
        if parsed.path == "/api/terminal/resize":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            sid = body.get("sid", "")
            session = _http_terminal_sessions.get(sid)
            if not session:
                self._json_response({"error": "session not found"}, 404)
                return
            cols = body.get("cols", 80)
            rows = body.get("rows", 24)
            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(session["master_fd"], termios.TIOCSWINSZ, winsize)
            self._json_response({"ok": True})
            return

        # HTTP chat session create (iOS fallback — EventSource-based streaming)
        if parsed.path == "/api/chat/create":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b""
            try:
                req = json.loads(body)
            except (json.JSONDecodeError, ValueError):
                self._json_response({"error": "invalid JSON"}, 400)
                return
            sid = uuid.uuid4().hex[:12]
            _chat_sessions[sid] = {"req": req, "created": time.time()}
            # Cleanup stale sessions (>5 min)
            cutoff = time.time() - 300
            for k in [k for k, v in _chat_sessions.items() if v["created"] < cutoff]:
                del _chat_sessions[k]
            self._json_response({"sid": sid})
            return

        if parsed.path == "/api/tools/execute":
            self._execute_tool_api()
        elif parsed.path.startswith("/api/collab/"):
            self._handle_collab_post(parsed.path)
        elif parsed.path == "/api/conversations":
            self._save_conversations()
        elif parsed.path.startswith("/api/conversations/") and parsed.path.endswith("/messages"):
            conv_id = parsed.path.split("/")[3]
            self._save_messages(conv_id)
        elif parsed.path.startswith("/api/conversations/") and parsed.path.endswith("/delete"):
            conv_id = parsed.path.split("/")[3]
            self._delete_conversation(conv_id)
        elif parsed.path.startswith("/upload/"):
            self._receive_file(TRANSFERS_DIR / parsed.path[len("/upload/"):])
        elif parsed.path.startswith("/share/"):
            self._receive_file(SHARED_DIR / parsed.path[len("/share/"):])
        elif parsed.path == "/api/analyze-image":
            self._analyze_image()
        elif parsed.path.startswith("/v1"):
            self._route_model_request()
        else:
            self._proxy_to_llm()

    def _serve_models(self):
        """Return available models for the UI."""
        models = []
        for model_id, config in MODEL_ROUTES.items():
            available = True
            if config["api_key_env"]:
                available = bool(os.environ.get(config["api_key_env"]))
            models.append({
                "id": model_id,
                "provider": config["provider"],
                "available": available,
            })
        self._json_response({"models": models})

    def _route_model_request(self, pre_parsed_req=None):
        """Route POST /v1/chat/completions to the right provider based on model field."""
        if pre_parsed_req is not None:
            req = pre_parsed_req
        else:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b""
            try:
                req = json.loads(body)
            except (json.JSONDecodeError, ValueError):
                req = {}

        model_name = req.get("model", "mistral-large-latest")
        resolved_name, route = get_model_route(model_name)
        logger.debug("route requested=%s resolved=%s", model_name, resolved_name)

        if not route:
            self._json_response({"error": f"Unknown model: {model_name}"}, 400)
            return

        # Local models → proxy to llama-server
        if route["provider"] == "local":
            req["model"] = resolved_name
            self._proxy_to_llm(override_body=json.dumps(req).encode())
            return

        # Cloud models → forward to provider API
        api_key = os.environ.get(route["api_key_env"], "") if route["api_key_env"] else ""
        if not api_key:
            self._json_response({
                "error": f"No API key for {route['provider']} ({route['api_key_env']})"
            }, 503)
            return

        # Use tool-enabled path for cloud models if maude_core is available
        # Skip server-side tool loop if client already sent its own tools
        client_sent_tools = bool(req.get("tools"))
        if TOOL_SUPPORT and not client_sent_tools:
            if route["provider"] == "mistral":
                self._cloud_model_with_tools(req, route, resolved_name)
                return
            if route["provider"] == "anthropic":
                self._claude_tool_loop(req, route, resolved_name)
                return

        # Ensure model name in body matches
        req["model"] = resolved_name
        body = json.dumps(req).encode()

        parsed_url = urlparse(route["base_url"])
        use_ssl = parsed_url.scheme == "https"
        host = parsed_url.hostname
        port = parsed_url.port or (443 if use_ssl else 80)

        try:
            if use_ssl:
                ctx = ssl.create_default_context()
                conn = http.client.HTTPSConnection(host, port, timeout=120, context=ctx)
            else:
                conn = http.client.HTTPConnection(host, port, timeout=120)

            # Build path: provider base path + our request path
            api_path = parsed_url.path.rstrip("/") + self.path
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "Content-Length": str(len(body)),
                "Accept": "text/event-stream",
            }

            conn.request("POST", api_path, body=body, headers=headers)
            resp = conn.getresponse()

            is_streaming = (
                resp.headers.get("Transfer-Encoding") == "chunked"
                or "text/event-stream" in (resp.headers.get("Content-Type", ""))
            )

            self.send_response(resp.status)
            self._add_cors()
            for key, val in resp.getheaders():
                if key.lower() not in ("transfer-encoding", "access-control-allow-origin",
                                        "access-control-allow-methods",
                                        "access-control-allow-headers"):
                    self.send_header(key, val)
            if is_streaming:
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            if is_streaming:
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            else:
                data = resp.read()
                self.wfile.write(data)

            conn.close()
        except ConnectionRefusedError:
            self._json_response({"error": f"Provider {route['provider']} connection refused"}, 503)
        except Exception as e:
            self._json_response({"error": f"Provider proxy error: {e}"}, 502)

    def _start_sse_headers(self):
        """Send SSE response headers for streaming. Call once before any SSE writes."""
        self.send_response(200)
        self._add_cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _send_trace_sse(self, trace_type: str, data: dict):
        """Send a trace event via SSE.

        Normal mode: SSE comment (ignored by OpenAI SDK / fetch parser).
        EventSource mode: named event (visible to EventSource on iOS).
        Headers must already be sent."""
        payload = json.dumps({"type": trace_type, **data})
        if getattr(self, '_eventstream_mode', False):
            line = f"event: trace\ndata: {payload}\n\n".encode()
        else:
            line = f": trace {payload}\n\n".encode()
        try:
            self.wfile.write(b"%x\r\n%s\r\n" % (len(line), line))
            self.wfile.flush()
        except Exception:
            pass

    def _close_sse_with_error(self, error_msg: str):
        """Send an error trace and close the SSE stream cleanly."""
        try:
            self._send_trace_sse("error", {"message": error_msg})
            # Send [DONE] and close chunked encoding
            done_line = b"data: [DONE]\n\n"
            self.wfile.write(b"%x\r\n%s\r\n" % (len(done_line), done_line))
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception:
            pass

    @staticmethod
    def _compact_tool_result(name: str, result: str) -> str:
        """Truncate tool results to prevent context bloat across loop iterations.

        The full result is already sent to the client via SSE trace — this only
        affects what the LLM sees when planning its next step."""
        if not result:
            return result
        n = len(result)
        # write_file / edit_file results are already short
        if name in ("write_file", "edit_file", "change_directory", "get_working_directory"):
            return result
        # read_file: keep first 80 + last 20 lines, drop middle
        if name == "read_file" and n > 3000:
            lines = result.split("\n")
            if len(lines) > 100:
                head = "\n".join(lines[:80])
                tail = "\n".join(lines[-20:])
                return f"{head}\n\n... ({len(lines) - 100} lines omitted) ...\n\n{tail}"
            return result[:3000] + f"\n... (truncated, {n} chars total)"
        # run_command: keep first 2000 + last 800
        if name == "run_command" and n > 3000:
            return result[:2000] + f"\n\n... ({n - 2800} chars omitted) ...\n\n" + result[-800:]
        # list_directory: keep first 60 entries
        if name == "list_directory" and n > 2000:
            lines = result.split("\n")
            if len(lines) > 65:
                return "\n".join(lines[:65]) + f"\n... ({len(lines) - 65} more entries)"
            return result[:2000] + "\n... (truncated)"
        # Everything else: hard cap at 4000
        if n > 4000:
            return result[:3500] + f"\n... (truncated, {n} chars total)"
        return result

    @staticmethod
    def _estimate_tokens(messages):
        """Rough token estimate: chars / 4. Handles both string and list content (Claude blocks)."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += len(json.dumps(block))
                    elif isinstance(block, str):
                        total += len(block)
            # Also count tool_calls if present
            tc = msg.get("tool_calls")
            if tc:
                total += len(json.dumps(tc))
        return total // 4

    @staticmethod
    def _trim_messages(messages, max_tokens, format="openai"):
        """Remove oldest middle messages to fit within max_tokens (80% threshold).

        Preserves first 2 messages (system/user) and the most recent messages.
        In openai format: removes assistant+tool pairs together.
        In claude format: removes assistant+user-tool-result pairs together.
        Returns number of messages removed.
        """
        threshold = int(max_tokens * 0.8)

        # Check if we even need to trim
        est = GatewayHandler._estimate_tokens(messages)
        if est <= threshold:
            return 0

        removed = 0
        # Keep trimming from position 2 (after system/first-user) until under threshold
        while GatewayHandler._estimate_tokens(messages) > threshold and len(messages) > 4:
            idx = 2  # Start removing from after the first 2 messages

            if idx >= len(messages) - 2:
                break  # Don't remove the most recent messages

            if format == "openai":
                # Remove assistant message + any following tool messages
                if messages[idx].get("role") == "assistant":
                    messages.pop(idx)
                    removed += 1
                    # Remove consecutive tool messages that followed it
                    while idx < len(messages) - 2 and messages[idx].get("role") == "tool":
                        messages.pop(idx)
                        removed += 1
                else:
                    messages.pop(idx)
                    removed += 1
            elif format == "claude":
                # Remove assistant message + following user message (tool results)
                if messages[idx].get("role") == "assistant":
                    messages.pop(idx)
                    removed += 1
                    # Remove following user message with tool results
                    if idx < len(messages) - 2 and messages[idx].get("role") == "user":
                        content = messages[idx].get("content", "")
                        is_tool_result = isinstance(content, list) and any(
                            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
                        )
                        if is_tool_result:
                            messages.pop(idx)
                            removed += 1
                else:
                    messages.pop(idx)
                    removed += 1

        return removed

    def _send_content_chunks(self, content: str, model_name: str, chunk_id: str, created: int):
        """Send content as word-boundary SSE chunks for typewriter effect.
        Headers must already be sent."""
        words = content.split(" ") if content else [""]
        chunks = []
        current = ""
        for word in words:
            current += (" " if current else "") + word
            if len(current) > 20:
                chunks.append(current)
                current = ""
        if current:
            chunks.append(current)

        for i, chunk_text in enumerate(chunks):
            spacer = " " if i < len(chunks) - 1 else ""
            sse_data = json.dumps({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk_text + spacer},
                    "finish_reason": None,
                }]
            })
            line = f"data: {sse_data}\n\n".encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(line), line))
            self.wfile.flush()

    def _send_sse_done(self, model_name: str, chunk_id: str, created: int):
        """Send finish + [DONE] events and close chunked encoding."""
        finish_data = json.dumps({
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }]
        })
        finish_line = f"data: {finish_data}\n\n".encode()
        self.wfile.write(b"%x\r\n%s\r\n" % (len(finish_line), finish_line))

        done_line = b"data: [DONE]\n\n"
        self.wfile.write(b"%x\r\n%s\r\n" % (len(done_line), done_line))

        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _cloud_model_with_tools(self, req, route, resolved_name):
        """Handle cloud model request with server-side tool execution loop.

        Sends non-streaming requests to the cloud API in a loop, executing
        tool calls locally via maude_core, until a final text response is
        produced. The final response is streamed back to the client as SSE.
        """
        api_key = os.environ.get(route["api_key_env"], "")

        # Get the user's latest message for tool selection
        user_msg = ""
        for msg in reversed(req.get("messages", [])):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        # Use pre-scoped tools if provided, otherwise select by message content
        active_tools = req.get("tools") or get_tools_for_message(user_msg)

        # Enhance system prompt so Mistral knows it has tool access
        tool_addendum = (
            "\n\nYou have access to tools for file operations, shell commands, web browsing, and more "
            "on this DGX Spark system. When the user asks about files, directories, code, or system "
            "information, USE THE TOOLS to get real data. Do NOT guess or make up file contents, "
            "directory listings, or system information. The working directory defaults to the user's "
            "home directory. Workbench projects are in ~/nvidia-workbench/. "
            "\n\nCRITICAL RULES — FOLLOW THESE EXACTLY: "
            "1. DO the work. NEVER tell the user to run commands themselves. You have run_command — use it. "
            "2. VERIFY your work. After writing code: run the build/compile step and check for errors. "
            "After starting a server: curl it and confirm it responds. After creating files: check they exist. "
            "3. FIX errors yourself. If a build fails, read the error, fix the code, and rebuild. "
            "Do NOT just report the error and stop. "
            "4. NEVER give tutorials or step-by-step instructions for things you can do yourself. "
            "If the user asks for a URL, start the server and give them the working URL. "
            "5. NEVER guess system info. Use run_command to check IPs, ports, processes, etc. "
            "6. When serving web apps: bind to 0.0.0.0, check which ports are free first "
            "(run_command: ss -tlnp), and give the Tailscale URL (run_command: tailscale ip -4). "
            "7. Complete the ENTIRE task. Don't stop halfway and describe what remains. "
            "If building a site, it should be built, running, and accessible before you respond. "
            "\n\nIf the user has attached an image, use the view_image tool to analyze it before responding. "
            "IMPORTANT: When a user sends a photo from the phone app, the image is ALREADY uploaded "
            "and saved on this server in /home/mboard76/nvidia-workbench/terminal-llm/shared/. "
            "The image path is provided in the message. Do NOT tell the user to upload it — it is "
            "already here. If the user asks to 'upload to the server' or 'save to the server', "
            "the file is already on the server in the shared/ folder. You can move or copy it "
            "elsewhere if they want, but acknowledge it is already present. "
            "\n\nYou also have full access to Google Workspace: Gmail (read, send), Google Drive "
            "(list, search, read, upload, create docs, create folders), Sheets (read, write, create), "
            "Calendar (list, create, update events), Contacts, YouTube, and Slides. Use these tools "
            "when the user asks about their email, documents, spreadsheets, schedule, or contacts. "
            "CRITICAL: ALWAYS use the provided tool functions for Google operations. NEVER write "
            "Python scripts or code that calls Google APIs directly via run_command. The tools "
            "already handle authentication and API calls. For example, to upload a file to Drive, "
            "use drive_upload — do NOT write a Python script that imports googleapiclient. "
            "To repeat the same operation on multiple files, call the same tool multiple times. "
            "\n\nYou can SEARCH THE WEB using the web_search tool (DuckDuckGo) and READ WEB PAGES "
            "using web_browse. When the user asks about current events, recent news, documentation, "
            "prices, reviews, or anything that requires up-to-date information, USE web_search. "
            "Do NOT say you cannot search the web — you CAN and SHOULD when relevant."
            "\n\nIMAGE DISPLAY: When you use generate_image, share_file, or web_image_search and want "
            "to display images inline, include the markdown ![description](url) in your response. "
            "For generated/shared images use ![description](/download/filename.png). "
            "For web images use the full URL from the search results."
        )
        messages = list(req.get("messages", []))
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] = msg["content"] + tool_addendum
                break

        # Connection details for the cloud API
        parsed_url = urlparse(route["base_url"])
        use_ssl = parsed_url.scheme == "https"
        host = parsed_url.hostname
        port = parsed_url.port or (443 if use_ssl else 80)
        api_path = parsed_url.path.rstrip("/") + "/v1/chat/completions"

        reset_rate_limits()
        max_iterations = 40
        recent_tool_calls = []
        pending_images = []  # Collect image URLs from tool results for auto-injection
        is_streaming = req.get("stream", False)
        sse_started = False

        # Start SSE headers immediately for streaming so client gets trace events
        if is_streaming:
            self._start_sse_headers()
            sse_started = True

        for iteration in range(max_iterations):
            # On last iteration, drop tools to force a summary response
            is_final = iteration == max_iterations - 1
            if is_final:
                messages.append({"role": "user", "content": "(System: You've used many tool calls. Wrap up now — summarize what you've done and what remains.)"})

            # Context trimming — keep messages within model's context window
            max_ctx = route.get("max_context", 128000)
            tool_schema_overhead = len(json.dumps(active_tools)) // 4 if active_tools else 0
            effective_max = max_ctx - tool_schema_overhead
            trimmed = self._trim_messages(messages, effective_max, format="openai")
            if trimmed:
                logger.info("Trimmed %d messages to fit %s context (%d tokens)", trimmed, resolved_name, effective_max)
                if sse_started:
                    self._send_trace_sse("context_trim", {"removed": trimmed, "max_tokens": effective_max})

            # Build non-streaming request for tool loop
            loop_req = {
                "model": resolved_name,
                "messages": messages,
                "stream": False,
                "max_tokens": req.get("max_tokens", 4096),
                "temperature": req.get("temperature", 0.7),
            }
            if not is_final:
                loop_req["tools"] = active_tools
                loop_req["tool_choice"] = "auto"

            body = json.dumps(loop_req).encode()
            llm_start = time.time()

            try:
                # Fresh connection per iteration — avoids stale sockets after long tool calls
                if use_ssl:
                    ctx = ssl.create_default_context()
                    conn = http.client.HTTPSConnection(host, port, timeout=300, context=ctx)
                else:
                    conn = http.client.HTTPConnection(host, port, timeout=300)

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "Content-Length": str(len(body)),
                }

                # Run LLM call in thread with keepalive pings
                llm_result_box = [None, None, None]  # [status, body, exception]
                def _llm_call():
                    try:
                        conn.request("POST", api_path, body=body, headers=headers)
                        resp = conn.getresponse()
                        llm_result_box[0] = resp.status
                        llm_result_box[1] = resp.read()
                        conn.close()
                    except Exception as exc:
                        llm_result_box[2] = exc
                t = threading.Thread(target=_llm_call)
                t.start()
                while t.is_alive():
                    t.join(timeout=15)
                    if t.is_alive() and sse_started:
                        elapsed_so_far = time.time() - llm_start
                        self._send_trace_sse("keepalive", {
                            "name": "llm_call",
                            "elapsed": round(elapsed_so_far, 1),
                        })
                t.join()

                if llm_result_box[2] is not None:
                    raise llm_result_box[2]

                resp_status = llm_result_box[0]
                resp_body = llm_result_box[1]

                if resp_status != 200:
                    try:
                        err = json.loads(resp_body)
                    except Exception:
                        err = {"error": resp_body.decode(errors="replace")}
                    if sse_started:
                        self._close_sse_with_error(f"LLM error: {resp_status}")
                    else:
                        self._json_response(err, resp_status)
                    return

                result = json.loads(resp_body)
                choice = result.get("choices", [{}])[0]
                message = choice.get("message", {})
                finish_reason = choice.get("finish_reason", "")

            except ConnectionRefusedError:
                err_msg = f"Provider {route['provider']} connection refused"
                if sse_started:
                    self._close_sse_with_error(err_msg)
                else:
                    self._json_response({"error": err_msg}, 503)
                return
            except Exception as e:
                err_msg = f"Tool loop error: {e}"
                if sse_started:
                    self._close_sse_with_error(err_msg)
                else:
                    self._json_response({"error": err_msg}, 502)
                return

            # Emit LLM call trace
            llm_elapsed = time.time() - llm_start
            usage = result.get("usage", {})
            prompt_tok = usage.get("prompt_tokens", 0)
            compl_tok = usage.get("completion_tokens", 0)
            if sse_started:
                self._send_trace_sse("llm_call", {
                    "prompt_tokens": prompt_tok,
                    "completion_tokens": compl_tok,
                    "elapsed": round(llm_elapsed, 2),
                })
            logger.info("LLM: %d+%d tokens in %.1fs", prompt_tok, compl_tok, llm_elapsed)

            # Check for tool calls
            tool_calls = message.get("tool_calls")
            if tool_calls and finish_reason == "tool_calls":
                # Add assistant message with tool_calls to conversation
                messages.append(message)

                # Execute each tool call
                for tc in tool_calls:
                    func_name = tc["function"]["name"]
                    try:
                        func_args = json.loads(tc["function"]["arguments"])
                    except (json.JSONDecodeError, ValueError):
                        func_args = {}

                    # Emit tool_call trace
                    args_preview = json.dumps(func_args, ensure_ascii=False)
                    if len(args_preview) > 80:
                        args_preview = args_preview[:80] + "..."
                    if sse_started:
                        self._send_trace_sse("tool_call", {
                            "name": func_name,
                            "args": args_preview,
                        })

                    # Duplicate detection
                    call_sig = (func_name, json.dumps(func_args, sort_keys=True))
                    if call_sig in recent_tool_calls:
                        tool_result = "(Already called with same arguments. Respond with the best answer you have.)"
                    else:
                        recent_tool_calls.append(call_sig)
                        logger.info("tool %s(%s)", func_name, func_args)
                        tool_start = time.time()

                        # Run tool in thread so we can send keepalive pings
                        tool_result_box = [None]
                        def _run_tool():
                            tool_result_box[0] = execute_tool(func_name, func_args)
                        t = threading.Thread(target=_run_tool)
                        t.start()
                        while t.is_alive():
                            t.join(timeout=15)
                            if t.is_alive() and sse_started:
                                elapsed_so_far = time.time() - tool_start
                                self._send_trace_sse("keepalive", {
                                    "name": func_name,
                                    "elapsed": round(elapsed_so_far, 1),
                                })
                        tool_result = tool_result_box[0]
                        tool_elapsed = time.time() - tool_start

                        # Emit tool_result trace
                        preview = (tool_result or "")[:80].replace("\n", " ").strip()
                        if len(tool_result or "") > 80:
                            preview += "..."
                        if sse_started:
                            self._send_trace_sse("tool_result", {
                                "name": func_name,
                                "preview": preview,
                                "elapsed": round(tool_elapsed, 2),
                            })

                    # Add compacted tool result to messages (full result already sent via SSE trace)
                    messages.append({
                        "role": "tool",
                        "name": func_name,
                        "content": self._compact_tool_result(func_name, tool_result or ""),
                        "tool_call_id": tc["id"],
                    })

                    # Collect image URLs from tool results for auto-injection
                    for _m in re.finditer(r'!\[([^\]]*)\]\(([^)]+)\)', tool_result or ""):
                        pending_images.append((_m.group(1), _m.group(2)))

                continue  # Loop back to get next response from model

            # No tool calls — final text response
            final_content = message.get("content", "")

            # Auto-inject images from tool results that the LLM didn't include
            if pending_images:
                for alt, url in pending_images:
                    if url not in final_content:
                        final_content += f"\n\n![{alt}]({url})"

            if not is_streaming:
                self._json_response({
                    "id": f"chatcmpl-tool-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": resolved_name,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": final_content},
                        "finish_reason": "stop",
                    }],
                    "usage": result.get("usage", {}),
                })
            elif sse_started:
                # SSE headers already sent — just send content + done
                chunk_id = f"chatcmpl-tool-{int(time.time())}"
                created = int(time.time())
                self._send_content_chunks(final_content, resolved_name, chunk_id, created)
                self._send_sse_done(resolved_name, chunk_id, created)
            else:
                self._send_as_sse(final_content, resolved_name)
            conn.close()
            return

        # Max iterations reached (shouldn't normally happen — last iteration drops tools)
        fallback = "I've completed as many steps as I can in one go. Ask me to continue if there's more to do."
        if not is_streaming:
            self._json_response({
                "id": f"chatcmpl-tool-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": resolved_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": fallback},
                    "finish_reason": "stop",
                }],
            })
        elif sse_started:
            chunk_id = f"chatcmpl-tool-{int(time.time())}"
            created = int(time.time())
            self._send_content_chunks(fallback, resolved_name, chunk_id, created)
            self._send_sse_done(resolved_name, chunk_id, created)
        else:
            self._send_as_sse(fallback, resolved_name)
        conn.close()

    def _send_as_sse(self, content, model_name):
        """Send a text response to the client as SSE, matching Mistral's streaming format."""
        self.send_response(200)
        self._add_cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        chunk_id = f"chatcmpl-tool-{int(time.time())}"
        created = int(time.time())

        # Split content into small chunks at word boundaries for streaming feel
        words = content.split(" ") if content else [""]
        chunks = []
        current = ""
        for word in words:
            current += (" " if current else "") + word
            if len(current) > 20:
                chunks.append(current)
                current = ""
        if current:
            chunks.append(current)

        for i, chunk_text in enumerate(chunks):
            spacer = " " if i < len(chunks) - 1 else ""
            sse_data = json.dumps({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk_text + spacer},
                    "finish_reason": None,
                }]
            })
            line = f"data: {sse_data}\n\n".encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(line), line))
            self.wfile.flush()

        # Send finish event
        finish_data = json.dumps({
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }]
        })
        finish_line = f"data: {finish_data}\n\n".encode()
        self.wfile.write(b"%x\r\n%s\r\n" % (len(finish_line), finish_line))

        done_line = b"data: [DONE]\n\n"
        self.wfile.write(b"%x\r\n%s\r\n" % (len(done_line), done_line))

        # End chunked encoding
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _claude_tool_loop(self, req, route, resolved_name):
        """Handle Claude (Anthropic) model request with server-side tool execution loop.

        Claude's API differs from OpenAI/Mistral:
        - Auth via x-api-key header (not Bearer token)
        - Endpoint: /v1/messages (not /v1/chat/completions)
        - System prompt is a top-level field (not a message)
        - Tool schema uses { name, description, input_schema } (not function wrapper)
        - Tool results go in user messages as { type: "tool_result", tool_use_id }
        - Stop reasons: "tool_use" and "end_turn" (not "tool_calls" and "stop")
        """
        api_key = os.environ.get(route["api_key_env"], "")

        # Get the user's latest message for tool selection
        user_msg = ""
        for msg in reversed(req.get("messages", [])):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        # Use pre-scoped tools if provided, otherwise select by message content
        active_tools_openai = req.get("tools") or get_tools_for_message(user_msg)

        # Convert OpenAI tool format to Claude format
        claude_tools = []
        for tool in active_tools_openai:
            func = tool.get("function", {})
            claude_tools.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            })

        # Mark last tool with cache_control so tools + system get cached
        if claude_tools:
            claude_tools[-1]["cache_control"] = {"type": "ephemeral"}

        # Extract system prompt from messages and convert to Claude format
        system_text = ""
        claude_messages = []
        for msg in req.get("messages", []):
            if msg.get("role") == "system":
                system_text = msg.get("content", "")
            else:
                claude_messages.append({
                    "role": msg.get("role"),
                    "content": msg.get("content", ""),
                })

        # Enhance system prompt with tool context (same addendum as Mistral loop)
        tool_addendum = (
            "\n\nYou have access to tools for file operations, shell commands, web browsing, and more "
            "on this DGX Spark system. When the user asks about files, directories, code, or system "
            "information, USE THE TOOLS to get real data. Do NOT guess or make up file contents, "
            "directory listings, or system information. The working directory defaults to the user's "
            "home directory. Workbench projects are in ~/nvidia-workbench/. "
            "\n\nCRITICAL RULES — FOLLOW THESE EXACTLY: "
            "1. DO the work. NEVER tell the user to run commands themselves. You have run_command — use it. "
            "2. VERIFY your work. After writing code: run the build/compile step and check for errors. "
            "After starting a server: curl it and confirm it responds. After creating files: check they exist. "
            "3. FIX errors yourself. If a build fails, read the error, fix the code, and rebuild. "
            "Do NOT just report the error and stop. "
            "4. NEVER give tutorials or step-by-step instructions for things you can do yourself. "
            "If the user asks for a URL, start the server and give them the working URL. "
            "5. NEVER guess system info. Use run_command to check IPs, ports, processes, etc. "
            "6. When serving web apps: bind to 0.0.0.0, check which ports are free first "
            "(run_command: ss -tlnp), and give the Tailscale URL (run_command: tailscale ip -4). "
            "7. Complete the ENTIRE task. Don't stop halfway and describe what remains. "
            "If building a site, it should be built, running, and accessible before you respond. "
            "\n\nIf the user has attached an image, use the view_image tool to analyze it before responding. "
            "IMPORTANT: When a user sends a photo from the phone app, the image is ALREADY uploaded "
            "and saved on this server in /home/mboard76/nvidia-workbench/terminal-llm/shared/. "
            "The image path is provided in the message. Do NOT tell the user to upload it — it is "
            "already here. If the user asks to 'upload to the server' or 'save to the server', "
            "the file is already on the server in the shared/ folder. You can move or copy it "
            "elsewhere if they want, but acknowledge it is already present. "
            "\n\nYou also have full access to Google Workspace: Gmail (read, send), Google Drive "
            "(list, search, read, upload, create docs, create folders), Sheets (read, write, create), "
            "Calendar (list, create, update events), Contacts, YouTube, and Slides. Use these tools "
            "when the user asks about their email, documents, spreadsheets, schedule, or contacts. "
            "CRITICAL: ALWAYS use the provided tool functions for Google operations. NEVER write "
            "Python scripts or code that calls Google APIs directly via run_command. The tools "
            "already handle authentication and API calls. For example, to upload a file to Drive, "
            "use drive_upload — do NOT write a Python script that imports googleapiclient. "
            "To repeat the same operation on multiple files, call the same tool multiple times. "
            "\n\nYou can SEARCH THE WEB using the web_search tool (DuckDuckGo) and READ WEB PAGES "
            "using web_browse. When the user asks about current events, recent news, documentation, "
            "prices, reviews, or anything that requires up-to-date information, USE web_search. "
            "Do NOT say you cannot search the web — you CAN and SHOULD when relevant."
            "\n\nIMAGE DISPLAY: When you use generate_image, share_file, or web_image_search and want "
            "to display images inline, include the markdown ![description](url) in your response. "
            "For generated/shared images use ![description](/download/filename.png). "
            "For web images use the full URL from the search results."
        )
        system_text += tool_addendum

        # Use block format for system prompt with cache_control
        # This caches the full system prompt + tool addendum across loop iterations
        # and across requests within the 5-minute TTL window
        system_blocks = [{
            "type": "text",
            "text": system_text,
            "cache_control": {"type": "ephemeral"},
        }]

        # Connection details for Claude API
        parsed_url = urlparse(route["base_url"])
        use_ssl = parsed_url.scheme == "https"
        host = parsed_url.hostname
        port = parsed_url.port or (443 if use_ssl else 80)
        api_path = "/v1/messages"

        reset_rate_limits()
        max_iterations = 40
        recent_tool_calls = []
        pending_images = []  # Collect image URLs from tool results for auto-injection
        is_streaming = req.get("stream", False)
        sse_started = False

        # Start SSE headers immediately for streaming so client gets trace events
        if is_streaming:
            self._start_sse_headers()
            sse_started = True

        for iteration in range(max_iterations):
            # On last iteration, drop tools to force a summary response
            is_final = iteration == max_iterations - 1
            if is_final:
                claude_messages.append({"role": "user", "content": "(System: You've used many tool calls. Wrap up now — summarize what you've done and what remains.)"})

            # Context trimming — keep messages within model's context window
            max_ctx = route.get("max_context", 200000)
            system_overhead = len(json.dumps(system_blocks)) // 4
            tool_schema_overhead = len(json.dumps(claude_tools)) // 4 if claude_tools else 0
            effective_max = max_ctx - system_overhead - tool_schema_overhead
            trimmed = self._trim_messages(claude_messages, effective_max, format="claude")
            if trimmed:
                logger.info("Trimmed %d Claude messages to fit %s context (%d tokens)", trimmed, resolved_name, effective_max)
                if sse_started:
                    self._send_trace_sse("context_trim", {"removed": trimmed, "max_tokens": effective_max})

            # Build non-streaming request for tool loop
            loop_req = {
                "model": resolved_name,
                "max_tokens": req.get("max_tokens", 4096),
                "system": system_blocks,
                "messages": claude_messages,
            }
            if not is_final:
                loop_req["tools"] = claude_tools

            body = json.dumps(loop_req).encode()
            llm_start = time.time()

            try:
                # Fresh connection per iteration — avoids stale sockets after long tool calls
                if use_ssl:
                    ctx = ssl.create_default_context()
                    conn = http.client.HTTPSConnection(host, port, timeout=300, context=ctx)
                else:
                    conn = http.client.HTTPConnection(host, port, timeout=300)

                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Length": str(len(body)),
                }

                # Run LLM call in thread with keepalive pings
                llm_result_box = [None, None, None]  # [status, body, exception]
                def _llm_call():
                    try:
                        conn.request("POST", api_path, body=body, headers=headers)
                        resp = conn.getresponse()
                        llm_result_box[0] = resp.status
                        llm_result_box[1] = resp.read()
                        conn.close()
                    except Exception as exc:
                        llm_result_box[2] = exc
                t = threading.Thread(target=_llm_call)
                t.start()
                while t.is_alive():
                    t.join(timeout=15)
                    if t.is_alive() and sse_started:
                        elapsed_so_far = time.time() - llm_start
                        self._send_trace_sse("keepalive", {
                            "name": "llm_call",
                            "elapsed": round(elapsed_so_far, 1),
                        })
                t.join()

                if llm_result_box[2] is not None:
                    raise llm_result_box[2]

                resp_status = llm_result_box[0]
                resp_body = llm_result_box[1]

                if resp_status != 200:
                    try:
                        err = json.loads(resp_body)
                    except Exception:
                        err = {"error": resp_body.decode(errors="replace")}
                    if not sse_started:
                        self._json_response(err, resp_status)
                    else:
                        err_msg = err.get("error", {}).get("message", str(err)) if isinstance(err.get("error"), dict) else str(err)
                        self._close_sse_with_error(f"Claude error: {err_msg}")
                    return

                result = json.loads(resp_body)
                stop_reason = result.get("stop_reason", "")

            except ConnectionRefusedError:
                err_msg = f"Provider {route['provider']} connection refused"
                if sse_started:
                    self._close_sse_with_error(err_msg)
                else:
                    self._json_response({"error": err_msg}, 503)
                return
            except Exception as e:
                err_msg = f"Claude tool loop error: {e}"
                if sse_started:
                    self._close_sse_with_error(err_msg)
                else:
                    self._json_response({"error": err_msg}, 502)
                return

            # Emit LLM call trace (include cache stats)
            llm_elapsed = time.time() - llm_start
            usage = result.get("usage", {})
            prompt_tok = usage.get("input_tokens", 0)
            compl_tok = usage.get("output_tokens", 0)
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_create = usage.get("cache_creation_input_tokens", 0)
            trace_data = {
                "prompt_tokens": prompt_tok,
                "completion_tokens": compl_tok,
                "elapsed": round(llm_elapsed, 2),
            }
            if cache_read:
                trace_data["cache_read_tokens"] = cache_read
            if cache_create:
                trace_data["cache_create_tokens"] = cache_create
            if sse_started:
                self._send_trace_sse("llm_call", trace_data)
            cache_info = ""
            if cache_read:
                cache_info += f" cache_read={cache_read}"
            if cache_create:
                cache_info += f" cache_write={cache_create}"
            logger.info("Claude LLM: %d+%d tokens in %.1fs%s", prompt_tok, compl_tok, llm_elapsed, cache_info)

            # Parse content blocks from Claude response
            content_blocks = result.get("content", [])
            text_parts = []
            tool_use_blocks = []
            for block in content_blocks:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_use_blocks.append(block)

            # Check for tool use
            if tool_use_blocks and stop_reason == "tool_use":
                # Add assistant message (full content blocks) to conversation
                claude_messages.append({
                    "role": "assistant",
                    "content": content_blocks,
                })

                # Execute each tool and collect results
                tool_results = []
                for tu in tool_use_blocks:
                    func_name = tu.get("name", "")
                    func_args = tu.get("input", {})
                    tool_use_id = tu.get("id", "")

                    # Emit tool_call trace
                    args_preview = json.dumps(func_args, ensure_ascii=False)
                    if len(args_preview) > 80:
                        args_preview = args_preview[:80] + "..."
                    if sse_started:
                        self._send_trace_sse("tool_call", {
                            "name": func_name,
                            "args": args_preview,
                        })

                    # Duplicate detection
                    call_sig = (func_name, json.dumps(func_args, sort_keys=True))
                    if call_sig in recent_tool_calls:
                        tool_result = "(Already called with same arguments. Respond with the best answer you have.)"
                    else:
                        recent_tool_calls.append(call_sig)
                        logger.info("claude tool %s(%s)", func_name, func_args)
                        tool_start = time.time()

                        # Run tool in thread so we can send keepalive pings
                        tool_result_box = [None]
                        def _run_tool(fn=func_name, fa=func_args):
                            tool_result_box[0] = execute_tool(fn, fa)
                        t = threading.Thread(target=_run_tool)
                        t.start()
                        while t.is_alive():
                            t.join(timeout=15)
                            if t.is_alive() and sse_started:
                                elapsed_so_far = time.time() - tool_start
                                self._send_trace_sse("keepalive", {
                                    "name": func_name,
                                    "elapsed": round(elapsed_so_far, 1),
                                })
                        tool_result = tool_result_box[0]
                        tool_elapsed = time.time() - tool_start

                        # Emit tool_result trace
                        preview = (tool_result or "")[:80].replace("\n", " ").strip()
                        if len(tool_result or "") > 80:
                            preview += "..."
                        if sse_started:
                            self._send_trace_sse("tool_result", {
                                "name": func_name,
                                "preview": preview,
                                "elapsed": round(tool_elapsed, 2),
                            })

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": self._compact_tool_result(func_name, tool_result or ""),
                    })

                    # Collect image URLs from tool results for auto-injection
                    for _m in re.finditer(r'!\[([^\]]*)\]\(([^)]+)\)', tool_result or ""):
                        pending_images.append((_m.group(1), _m.group(2)))

                # Add tool results as a user message (Claude API format)
                claude_messages.append({
                    "role": "user",
                    "content": tool_results,
                })

                continue  # Loop back to get next response from model

            # No tool calls — final text response
            final_content = "\n\n".join(text_parts) if text_parts else ""

            # Auto-inject images from tool results that the LLM didn't include
            if pending_images:
                for alt, url in pending_images:
                    if url not in final_content:
                        final_content += f"\n\n![{alt}]({url})"

            if not is_streaming:
                self._json_response({
                    "id": f"chatcmpl-claude-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": resolved_name,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": final_content},
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tok,
                        "completion_tokens": compl_tok,
                        "total_tokens": prompt_tok + compl_tok,
                    },
                })
            elif sse_started:
                chunk_id = f"chatcmpl-claude-{int(time.time())}"
                created = int(time.time())
                self._send_content_chunks(final_content, resolved_name, chunk_id, created)
                self._send_sse_done(resolved_name, chunk_id, created)
            else:
                self._send_as_sse(final_content, resolved_name)
            conn.close()
            return

        # Max iterations reached (shouldn't normally happen — last iteration drops tools)
        fallback = "I've completed as many steps as I can in one go. Ask me to continue if there's more to do."
        if not is_streaming:
            self._json_response({
                "id": f"chatcmpl-claude-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": resolved_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": fallback},
                    "finish_reason": "stop",
                }],
            })
        elif sse_started:
            chunk_id = f"chatcmpl-claude-{int(time.time())}"
            created = int(time.time())
            self._send_content_chunks(fallback, resolved_name, chunk_id, created)
            self._send_sse_done(resolved_name, chunk_id, created)
        else:
            self._send_as_sse(fallback, resolved_name)
        conn.close()

    def _proxy_to_llm(self, override_body=None):
        """Forward request to local llama-server."""
        try:
            conn = http.client.HTTPConnection("localhost", LLM_PORT, timeout=300)

            if override_body is not None:
                body = override_body
            else:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length) if content_length > 0 else None

            headers = {}
            for key, val in self.headers.items():
                if key.lower() not in ("host", "transfer-encoding", "content-length"):
                    headers[key] = val
            if body is not None:
                headers["Content-Length"] = str(len(body))

            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()

            is_streaming = (
                resp.headers.get("Transfer-Encoding") == "chunked"
                or "text/event-stream" in (resp.headers.get("Content-Type", ""))
            )

            self.send_response(resp.status)
            self._add_cors()
            for key, val in resp.getheaders():
                if key.lower() not in ("transfer-encoding",):
                    self.send_header(key, val)
            if is_streaming:
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            if is_streaming:
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            else:
                data = resp.read()
                self.wfile.write(data)

            conn.close()
        except ConnectionRefusedError:
            self._json_response({"error": "LLM server not ready yet"}, 503)
        except Exception as e:
            self._json_response({"error": f"Proxy error: {e}"}, 502)

    def _web_proxy(self, query):
        """Proxy web pages for the browser app."""
        url = query.get("url", [None])[0]
        if not url:
            self._json_response({"error": "Missing url parameter"}, 400)
            return

        try:
            parsed = urlparse(url)
            use_ssl = parsed.scheme == "https"
            host = parsed.hostname
            port = parsed.port or (443 if use_ssl else 80)
            request_path = parsed.path or "/"
            if parsed.query:
                request_path += "?" + parsed.query

            if use_ssl:
                ctx = ssl.create_default_context()
                conn = http.client.HTTPSConnection(host, port, timeout=15, context=ctx)
            else:
                conn = http.client.HTTPConnection(host, port, timeout=15)

            conn.request("GET", request_path, headers={
                "User-Agent": "Mozilla/5.0 (Linux; Android 14) MAUDE/1.0",
                "Accept": "text/html,application/xhtml+xml,*/*",
                "Host": host,
            })
            resp = conn.getresponse()
            data = resp.read()
            content_type = resp.headers.get("Content-Type", "text/html")

            # Handle redirects
            if resp.status in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location", "")
                if location and not location.startswith("http"):
                    location = urljoin(url, location)
                self._json_response({"redirect": location, "status": resp.status})
                conn.close()
                return

            self.send_response(200)
            self._add_cors()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            conn.close()
        except Exception as e:
            self._json_response({"error": f"Proxy error: {e}"}, 502)

    def _serve_static(self, path):
        """Serve PWA static files from dist/."""
        if path == "/" or path == "/app" or path == "/app/":
            path = "/index.html"
        elif path.startswith("/app/"):
            path = path[4:]  # Strip /app prefix

        # Map to file
        filepath = PWA_DIR / path.lstrip("/")

        # SPA fallback: if file not found and no extension, serve index.html
        if not filepath.exists() or filepath.is_dir():
            if "." not in filepath.name:
                filepath = PWA_DIR / "index.html"

        if not filepath.exists():
            self._json_response({"error": "Not found"}, 404)
            return

        try:
            data = filepath.read_bytes()
            content_type = mimetypes.guess_type(filepath.name)[0] or "application/octet-stream"
            self.send_response(200)
            self._add_cors()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            if filepath.name == "index.html" or content_type == "application/javascript":
                self.send_header("Cache-Control", "no-store, must-revalidate")
            elif content_type.startswith("text/"):
                self.send_header("Cache-Control", "no-cache")
            else:
                self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _try_serve_static(self, path):
        """Try to serve as static file, return True if served."""
        filepath = PWA_DIR / path.lstrip("/")
        if filepath.exists() and filepath.is_file():
            self._serve_static(path)
            return True
        return False

    def _list_dir(self, directory):
        entries = []
        try:
            for entry in sorted(directory.iterdir()):
                stat = entry.stat()
                entries.append({
                    "name": entry.name,
                    "size": stat.st_size,
                    "is_dir": entry.is_dir(),
                    "modified": stat.st_mtime,
                })
        except Exception:
            pass
        self._json_response({"path": str(directory), "files": entries})

    def _send_file(self, filepath):
        if not filepath.exists():
            self._json_response({"error": f"File not found: {filepath.name}"}, 404)
            return
        try:
            data = filepath.read_bytes()
            content_type = mimetypes.guess_type(filepath.name)[0] or "application/octet-stream"
            self.send_response(200)
            self._add_cors()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            # Inline display for images/videos, attachment for everything else
            if content_type.startswith(("image/", "video/", "audio/")):
                self.send_header("Content-Disposition", f'inline; filename="{filepath.name}"')
            else:
                self.send_header("Content-Disposition", f'attachment; filename="{filepath.name}"')
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _receive_file(self, filepath):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(data)
            self._json_response({"status": "ok", "filename": filepath.name, "size": len(data)})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _analyze_image(self):
        """Analyze an image in shared/ using LLaVA via maude_core.execute_tool."""
        if not TOOL_SUPPORT:
            self._json_response({"error": "Tool support not available (maude_core not loaded)"}, 503)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length > 0 else b""
            req = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            self._json_response({"error": "Invalid JSON body"}, 400)
            return

        filename = req.get("filename", "")
        question = req.get("question", "Describe this image in detail.")

        if not filename:
            self._json_response({"error": "Missing 'filename' field"}, 400)
            return

        filepath = SHARED_DIR / filename
        if not filepath.exists():
            self._json_response({"error": f"File not found: {filename}"}, 404)
            return

        try:
            reset_rate_limits()
            result = execute_tool("view_image", {"path": str(filepath), "question": question})
            self._json_response({"analysis": result, "filename": filename})
        except Exception as e:
            self._json_response({"error": f"Image analysis failed: {e}"}, 500)

    # ── Conversation sync API ─────────────────────────────────────

    def _get_conversations(self):
        """Return conversation index."""
        index_file = CONVERSATIONS_DIR / "index.json"
        if index_file.exists():
            self._json_response(json.loads(index_file.read_text()))
        else:
            self._json_response([])

    def _save_conversations(self):
        """Save conversation index."""
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        data = json.loads(body)
        (CONVERSATIONS_DIR / "index.json").write_text(json.dumps(data))
        self._json_response({"ok": True})

    def _get_messages(self, conv_id):
        """Return messages for a conversation."""
        # Sanitize ID to prevent path traversal
        safe_id = conv_id.replace("/", "").replace("..", "")
        msg_file = CONVERSATIONS_DIR / f"{safe_id}.json"
        if msg_file.exists():
            self._json_response(json.loads(msg_file.read_text()))
        else:
            self._json_response([])

    def _save_messages(self, conv_id):
        """Save messages for a conversation."""
        safe_id = conv_id.replace("/", "").replace("..", "")
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        data = json.loads(body)
        (CONVERSATIONS_DIR / f"{safe_id}.json").write_text(json.dumps(data))
        self._json_response({"ok": True})

    def _delete_conversation(self, conv_id):
        """Delete a conversation's messages file."""
        safe_id = conv_id.replace("/", "").replace("..", "")
        msg_file = CONVERSATIONS_DIR / f"{safe_id}.json"
        if msg_file.exists():
            msg_file.unlink()
        self._json_response({"ok": True})

    # ── Collaboration API ────────────────────────────────────────────

    def _read_post_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        return json.loads(body) if body else {}

    def _handle_collab_get(self, path: str, query: dict):
        """Route GET /api/collab/* requests."""
        from collab import get_hub
        hub = get_hub()

        if path == "/api/collab/presence":
            self._json_response(hub.presence.get_all())
        elif path == "/api/collab/activity":
            since = float(query.get("since", [0])[0])
            limit = int(query.get("limit", [50])[0])
            self._json_response(hub.activity.get_recent(since, limit))
        elif path == "/api/collab/projects":
            self._json_response(hub.list_projects())
        elif path.startswith("/api/collab/projects/"):
            project_id = path.split("/")[4]
            proj = hub.get_project(project_id)
            if proj:
                self._json_response(proj)
            else:
                self._json_response({"error": "Project not found"}, 404)
        elif path == "/api/collab/tasks":
            self._json_response(hub.tasks.list_all())
        elif path == "/api/collab/tasks/poll":
            client_id = query.get("client_id", [""])[0]
            if not client_id:
                self._json_response({"error": "client_id required"}, 400)
            else:
                # Resolve platform-targeted tasks first
                hub.tasks.resolve_platform_targets(hub.presence.get_all())
                tasks = hub.tasks.get_queued_for_client(client_id)
                self._json_response(tasks)
        elif path.startswith("/api/collab/tasks/"):
            task_id = path.split("/")[4]
            task = hub.tasks.get(task_id)
            if task:
                self._json_response(task)
            else:
                self._json_response({"error": "Task not found"}, 404)
        elif path == "/api/collab/gossip":
            self._json_response(hub.get_gossip_bundle())
        elif path == "/api/collab/status":
            self._json_response(hub.get_status())
        else:
            self._json_response({"error": "Not found"}, 404)

    def _handle_collab_post(self, path: str):
        """Route POST /api/collab/* requests."""
        from collab import get_hub
        hub = get_hub()
        data = self._read_post_body()

        if path == "/api/collab/presence":
            hub.heartbeat(
                client_id=data.get("client_id", ""),
                client_type=data.get("client_type", "unknown"),
                activity=data.get("activity", ""),
                conversation_id=data.get("conversation_id", ""),
                project_id=data.get("project_id", ""),
                hostname=data.get("hostname", ""),
                platform=data.get("platform", ""),
            )
            self._json_response({"ok": True})
        elif path == "/api/collab/activity":
            hub.emit(
                event_type=data.get("type", "custom"),
                summary=data.get("summary", ""),
                data=data.get("data"),
                client_id=data.get("client_id", ""),
                conversation_id=data.get("conversation_id", ""),
                project_id=data.get("project_id", ""),
            )
            self._json_response({"ok": True})
        elif path == "/api/collab/projects":
            proj = hub.create_project(
                name=data.get("name", "Untitled"),
                description=data.get("description", ""),
                tags=data.get("tags", []),
            )
            self._json_response(proj, 201)
        elif path.startswith("/api/collab/projects/") and path.endswith("/delete"):
            project_id = path.split("/")[4]
            hub.delete_project(project_id)
            self._json_response({"ok": True})
        elif path.startswith("/api/collab/projects/"):
            project_id = path.split("/")[4]
            proj = hub.update_project(project_id, **data)
            if proj:
                self._json_response(proj)
            else:
                self._json_response({"error": "Project not found"}, 404)
        elif path == "/api/collab/tasks":
            task = hub.dispatch_task(
                prompt=data.get("prompt", ""),
                target=data.get("target", ""),
                capability=data.get("capability", "LLM"),
                project_id=data.get("project_id", ""),
                target_client_id=data.get("target_client_id", ""),
                target_platform=data.get("target_platform", ""),
            )
            self._json_response(task, 201)
        elif path.endswith("/claim") and path.startswith("/api/collab/tasks/"):
            # POST /api/collab/tasks/{id}/claim
            task_id = path.split("/")[4]
            task = hub.tasks.get(task_id)
            if not task:
                self._json_response({"error": "Task not found"}, 404)
            elif task.get("status") != "queued":
                self._json_response({"error": "Task already claimed", "status": task.get("status")}, 409)
            else:
                hub.tasks.update_status(task_id, "running")
                self._json_response({"ok": True, "task_id": task_id})
        elif path.endswith("/result") and path.startswith("/api/collab/tasks/"):
            # POST /api/collab/tasks/{id}/result
            task_id = path.split("/")[4]
            task = hub.tasks.get(task_id)
            if not task:
                self._json_response({"error": "Task not found"}, 404)
            else:
                status = data.get("status", "completed")
                result = data.get("result", "")
                hub.tasks.update_status(task_id, status, result)
                self._json_response({"ok": True, "task_id": task_id})
        elif path == "/api/collab/tasks/execute":
            result = hub.execute_task(data)
            self._json_response(result)
        else:
            self._json_response({"error": "Not found"}, 404)

    # ── Health & Tool Catalog API ──────────────────────────────────

    def _serve_health(self):
        """Enhanced /health — structured report with deps, services, tools."""
        try:
            from health import HealthChecker
            report = HealthChecker().check_all()
        except ImportError:
            report = {
                "status": "ok",
                "llm_port": LLM_PORT,
                "personaplex_port": PERSONAPLEX_PORT,
                "gateway_port": GATEWAY_PORT,
            }
        self._json_response(report)

    def _serve_tools(self, query: dict):
        """GET /api/tools — full catalog or filtered by ?message=."""
        try:
            from tool_catalog import get_catalog, get_filtered_tools
        except ImportError:
            self._json_response({"error": "tool_catalog not available"}, 503)
            return

        message = query.get("message", [None])[0]
        if message:
            tools = get_filtered_tools(message)
            self._json_response({"tools": tools, "message": message})
        else:
            catalog = get_catalog()
            self._json_response(catalog)

    def _execute_tool_api(self):
        """POST /api/tools/execute — execute a server-side tool."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            self._json_response({"error": "Invalid JSON"}, 400)
            return

        name = data.get("name", "")
        arguments = data.get("arguments", {})
        if not name:
            self._json_response({"error": "Missing 'name' field"}, 400)
            return

        try:
            from tool_catalog import execute_server_tool
            result = execute_server_tool(name, arguments)
        except ImportError:
            self._json_response({"error": "tool_catalog not available"}, 503)
            return

        if result.get("error"):
            code = 400 if "local tool" in result["error"] else 500
            self._json_response(result, code)
        else:
            self._json_response(result)

    def _json_response(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self._add_cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    CERT_DIR = Path(__file__).parent / "certs"
    USE_SSL = (CERT_DIR / "cert.pem").exists() and (CERT_DIR / "key.pem").exists()

    logger.info("MAUDE Gateway on port %d (%s)", GATEWAY_PORT, "HTTPS" if USE_SSL else "HTTP")
    logger.info("  LLM proxy  -> localhost:%d", LLM_PORT)
    logger.info("  PersonaPlex-> localhost:%d", PERSONAPLEX_PORT)
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
        import threading
        HTTP_PORT = 30080
        http_server = ThreadedHTTPServer(("0.0.0.0", HTTP_PORT), GatewayHandler)
        http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
        http_thread.start()
        logger.info("  HTTP mirror: port %d (for native app)", HTTP_PORT)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping.")
