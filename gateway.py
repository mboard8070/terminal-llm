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
import ssl
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

# Tool execution support for cloud model requests
try:
    import maude_core
    from maude_core import TOOLS as ALL_TOOLS, execute_tool, get_tools_for_message, reset_rate_limits
    TOOL_SUPPORT = True
    print("Tool support enabled via maude_core")
except ImportError as e:
    print(f"Warning: maude_core not available, tool support disabled: {e}")
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

SHARED_DIR.mkdir(parents=True, exist_ok=True)
TRANSFERS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Model routing configuration
# ─────────────────────────────────────────────────────────────────

MODEL_ROUTES = {
    # Mistral Large — general conversation (MAUDE default)
    "mistral-large-latest": {
        "provider": "mistral",
        "base_url": "https://api.mistral.ai",
        "api_key_env": "MISTRAL_API_KEY",
    },
    # Codestral — code tasks (uses separate endpoint + key)
    "codestral-latest": {
        "provider": "mistral",
        "base_url": "https://codestral.mistral.ai",
        "api_key_env": "CODESTRAL_API_KEY",
    },
    # Nemotron — local fallback (llama-server)
    "nemotron": {
        "provider": "local",
        "base_url": f"http://localhost:{LLM_PORT}",
        "api_key_env": None,
    },
    # LLaVA — vision (local)
    "llava": {
        "provider": "local",
        "base_url": f"http://localhost:{LLM_PORT}",
        "api_key_env": None,
    },
}

# Aliases for convenience
MODEL_ALIASES = {
    "mistral": "mistral-large-latest",
    "codestral": "codestral-latest",
    "local": "nemotron",
    "vision": "llava",
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

    # Connect to PersonaPlex upstream
    upstream_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    upstream_sock.settimeout(10)
    try:
        upstream_sock.connect(("localhost", PERSONAPLEX_PORT))
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
        pass

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
        elif parsed.path == "/health":
            self._json_response({
                "status": "ok",
                "llm_port": LLM_PORT,
                "personaplex_port": PERSONAPLEX_PORT,
                "gateway_port": GATEWAY_PORT,
            })
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
                              "/messages", "/files", "/settings"):
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

        if parsed.path.startswith("/upload/"):
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

    def _route_model_request(self):
        """Route POST /v1/chat/completions to the right provider based on model field."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        # Parse the model from the request
        try:
            req = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            req = {}

        model_name = req.get("model", "mistral-large-latest")
        resolved_name, route = get_model_route(model_name)

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

        # Use tool-enabled path for Mistral if maude_core is available
        if TOOL_SUPPORT and route["provider"] == "mistral":
            self._cloud_model_with_tools(req, route, resolved_name)
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

        # Select relevant tools based on message content
        active_tools = get_tools_for_message(user_msg)

        # Enhance system prompt so Mistral knows it has tool access
        tool_addendum = (
            "\n\nYou have access to tools for file operations, shell commands, web browsing, and more "
            "on this DGX Spark system. When the user asks about files, directories, code, or system "
            "information, USE THE TOOLS to get real data. Do NOT guess or make up file contents, "
            "directory listings, or system information. The working directory defaults to the user's "
            "home directory. Workbench projects are in ~/nvidia-workbench/. "
            "If the user has attached an image, use the view_image tool to analyze it before responding. "
            "IMPORTANT: When a user sends a photo from the phone app, the image is ALREADY uploaded "
            "and saved on this server in /home/mboard76/nvidia-workbench/terminal-llm/shared/. "
            "The image path is provided in the message. Do NOT tell the user to upload it — it is "
            "already here. If the user asks to 'upload to the server' or 'save to the server', "
            "the file is already on the server in the shared/ folder. You can move or copy it "
            "elsewhere if they want, but acknowledge it is already present."
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
        max_iterations = 15
        recent_tool_calls = []

        for iteration in range(max_iterations):
            # Build non-streaming request for tool loop
            loop_req = {
                "model": resolved_name,
                "messages": messages,
                "tools": active_tools,
                "tool_choice": "auto",
                "stream": False,
                "max_tokens": req.get("max_tokens", 4096),
                "temperature": req.get("temperature", 0.7),
            }

            body = json.dumps(loop_req).encode()

            try:
                if use_ssl:
                    ctx = ssl.create_default_context()
                    conn = http.client.HTTPSConnection(host, port, timeout=120, context=ctx)
                else:
                    conn = http.client.HTTPConnection(host, port, timeout=120)

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "Content-Length": str(len(body)),
                }

                conn.request("POST", api_path, body=body, headers=headers)
                resp = conn.getresponse()
                resp_body = resp.read()
                conn.close()

                if resp.status != 200:
                    try:
                        err = json.loads(resp_body)
                    except Exception:
                        err = {"error": resp_body.decode(errors="replace")}
                    self._json_response(err, resp.status)
                    return

                result = json.loads(resp_body)
                choice = result.get("choices", [{}])[0]
                message = choice.get("message", {})
                finish_reason = choice.get("finish_reason", "")

            except ConnectionRefusedError:
                self._json_response({"error": f"Provider {route['provider']} connection refused"}, 503)
                return
            except Exception as e:
                self._json_response({"error": f"Tool loop error: {e}"}, 502)
                return

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

                    # Duplicate detection
                    call_sig = (func_name, json.dumps(func_args, sort_keys=True))
                    if call_sig in recent_tool_calls:
                        tool_result = "(Already called with same arguments. Respond with the best answer you have.)"
                    else:
                        recent_tool_calls.append(call_sig)
                        print(f"  [tool] {func_name}({func_args})")
                        tool_result = execute_tool(func_name, func_args)

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "name": func_name,
                        "content": tool_result or "",
                        "tool_call_id": tc["id"],
                    })

                continue  # Loop back to get next response from model

            # No tool calls — final text response
            final_content = message.get("content", "")
            self._send_as_sse(final_content, resolved_name)
            return

        # Max iterations reached
        self._send_as_sse(
            "I've reached the maximum number of tool calls. Here's what I found so far.",
            resolved_name,
        )

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
                if key.lower() not in ("host", "transfer-encoding"):
                    headers[key] = val

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
            if content_type.startswith("text/") or content_type == "application/javascript":
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

    def _json_response(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self._add_cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    CERT_DIR = Path(__file__).parent / "certs"
    USE_SSL = (CERT_DIR / "cert.pem").exists() and (CERT_DIR / "key.pem").exists()

    print(f"MAUDE Gateway on port {GATEWAY_PORT} ({'HTTPS' if USE_SSL else 'HTTP'})")
    print(f"  LLM proxy  -> localhost:{LLM_PORT}")
    print(f"  PersonaPlex-> localhost:{PERSONAPLEX_PORT}")
    print(f"  PWA dir    : {PWA_DIR}")
    print(f"  Shared     : {SHARED_DIR}")
    print(f"  Transfers  : {TRANSFERS_DIR}")
    print(f"  Models     : {', '.join(MODEL_ROUTES.keys())}")

    server = ThreadedHTTPServer(("0.0.0.0", GATEWAY_PORT), GatewayHandler)

    if USE_SSL:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(CERT_DIR / "cert.pem"), str(CERT_DIR / "key.pem"))
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        print(f"  SSL cert   : {CERT_DIR / 'cert.pem'}")

        # Also start an HTTP server for the native Android app
        import threading
        HTTP_PORT = 30080
        http_server = ThreadedHTTPServer(("0.0.0.0", HTTP_PORT), GatewayHandler)
        http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
        http_thread.start()
        print(f"  HTTP mirror: port {HTTP_PORT} (for native app)")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
