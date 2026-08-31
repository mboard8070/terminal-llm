"""
HTTP server class for the MAUDE Gateway.

GatewayHandler is the BaseHTTPRequestHandler subclass that dispatches
all HTTP requests. It inherits tool-loop methods from CloudMixin and
route-handler methods from RoutesMixin.
"""

import html
import http.client
import json
import os
import ssl
import struct
import sys
import threading
import time
import uuid

try:
    import fcntl
    import termios
except ImportError:  # Windows — PTY resize is unavailable
    fcntl = None
    termios = None
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, unquote, urlparse

from .cloud import CloudMixin
from .routes import RoutesMixin
from .state import (
    SHARED_DIR,
    TOOL_SUPPORT,
    TRANSFERS_DIR,
    chat_sessions,
    device_location,
    get_model_route,
    http_terminal_sessions,
    logger,
)
from .websocket import (
    cleanup_terminal_session,
    create_terminal_session,
    handle_terminal_stream,
    handle_terminal_websocket,
    handle_voice_proxy,
)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class GatewayHandler(CloudMixin, RoutesMixin, BaseHTTPRequestHandler):
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

        if parsed.path in ("/reset-cache", "/reset"):
            logger.info(
                "phone reset page from=%s ua=%s", self.client_address[0], self.headers.get("User-Agent", "")[:80]
            )
            self._serve_reset_cache_page()
            return

        if parsed.path in ("/plain-chat", "/plain", "/plainchat", "/chat-test", "/maude/plain-chat"):
            self._serve_plain_chat()
            return

        if parsed.path == "/api/ping":
            logger.info("phone ping from=%s ua=%s", self.client_address[0], self.headers.get("User-Agent", "")[:80])
            self._json_response({"ok": True, "ts": time.time(), "client": self.client_address[0]})
            return

        if parsed.path in ("/diag", "/diagnostic", "/live-check"):
            logger.info(
                "phone diag page from=%s ua=%s", self.client_address[0], self.headers.get("User-Agent", "")[:80]
            )
            self._serve_gateway_diag()
            return

        # WebSocket upgrade for terminal
        if parsed.path == "/ws/terminal":
            upgrade = self.headers.get("Upgrade", "").lower()
            if upgrade == "websocket":
                handle_terminal_websocket(self)
                return
            self._json_response({"error": "WebSocket upgrade required"}, 400)
            return

        # WebSocket proxy for voice server
        if parsed.path == "/api/chat":
            upgrade = self.headers.get("Upgrade", "").lower()
            if upgrade == "websocket":
                handle_voice_proxy(self)
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
            offset = int(query.get("offset", ["0"])[0] or "0")
            logger.info("phone chat stream sid=%s offset=%s from=%s", sid, offset, self.client_address[0])
            if not sid:
                self._json_response({"error": "missing sid"}, 400)
                return
            session = chat_sessions.get(sid)
            if not session:
                self._json_response({"error": "session not found"}, 404)
                return
            self._stream_chat_session(session, offset)
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
            self._send_file(SHARED_DIR / parsed.path[len("/download/") :])
        elif parsed.path.startswith("/download-transfer/"):
            self._send_file(TRANSFERS_DIR / parsed.path[len("/download-transfer/") :])
        elif parsed.path.startswith("/api/collab/"):
            self._handle_collab_get(parsed.path, query)
        elif parsed.path == "/api/conversations":
            self._get_conversations()
        elif parsed.path.startswith("/api/conversations/") and parsed.path.endswith("/messages"):
            conv_id = parsed.path.split("/")[3]
            self._get_messages(conv_id)
        elif parsed.path == "/api/location":
            if device_location and time.time() - device_location.get("ts", 0) < 3600:
                self._json_response(device_location)
            else:
                self._json_response({"error": "no recent location"}, 404)
            return
        elif parsed.path.startswith("/api/command-center/"):
            logger.info(
                "phone command-center path=%s from=%s ua=%s",
                parsed.path,
                self.client_address[0],
                self.headers.get("User-Agent", "")[:80],
            )
            self._handle_command_center(parsed.path, query)
        elif parsed.path == "/health":
            logger.info("phone health from=%s ua=%s", self.client_address[0], self.headers.get("User-Agent", "")[:80])
            self._serve_health()
        elif parsed.path == "/vnc":
            self._redirect_vnc()
        elif parsed.path == "/api/tools":
            self._serve_tools(query)
        elif parsed.path == "/api/tasks":
            self._serve_tasks()
        elif parsed.path == "/models":
            self._serve_models()
        elif parsed.path.startswith("/proxy"):
            self._web_proxy(query)
        elif parsed.path == "/v1/models":
            self._serve_v1_models()
        elif parsed.path.startswith("/v1"):
            self._proxy_to_llm()
        elif parsed.path.startswith("/app") or parsed.path == "/":
            logger.info(
                "phone app load path=%s from=%s ua=%s",
                parsed.path,
                self.client_address[0],
                self.headers.get("User-Agent", "")[:80],
            )
            self._serve_static(parsed.path)
        elif parsed.path == "/manifest.json":
            self._serve_static("/manifest.json")
        elif parsed.path.startswith("/assets"):
            self._serve_static(parsed.path)
        elif parsed.path in (
            "/maude",
            "/maude/voice",
            "/terminal",
            "/browser",
            "/messages",
            "/files",
            "/settings",
            "/collab",
            "/command-center",
        ):
            logger.info(
                "phone app load path=%s from=%s ua=%s",
                parsed.path,
                self.client_address[0],
                self.headers.get("User-Agent", "")[:80],
            )
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

        if parsed.path in ("/plain-chat", "/plain", "/plainchat", "/chat-test", "/maude/plain-chat"):
            self._handle_plain_chat_post()
            return

        # HTTP terminal endpoints (iOS fallback)
        if parsed.path == "/api/terminal/create":
            sid = create_terminal_session()
            self._json_response({"sid": sid})
            return
        if parsed.path == "/api/terminal/input":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            sid = body.get("sid", "")
            data = body.get("data", "")
            session = http_terminal_sessions.get(sid)
            if not session:
                self._json_response({"error": "session not found"}, 404)
                return
            try:
                os.write(session["master_fd"], data.encode() if isinstance(data, str) else data)
            except OSError:
                cleanup_terminal_session(sid)
                self._json_response({"error": "session closed"}, 410)
                return
            self._json_response({"ok": True})
            return
        if parsed.path == "/api/terminal/resize":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            sid = body.get("sid", "")
            session = http_terminal_sessions.get(sid)
            if not session:
                self._json_response({"error": "session not found"}, 404)
                return
            cols = body.get("cols", 80)
            rows = body.get("rows", 24)
            if fcntl is None or termios is None:
                self._json_response({"error": "PTY resize not supported on this OS"}, 501)
                return
            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(session["master_fd"], termios.TIOCSWINSZ, winsize)
            self._json_response({"ok": True})
            return

        # HTTP chat session create (iOS fallback — EventSource-based streaming)
        if parsed.path == "/api/chat/create":
            content_length = int(self.headers.get("Content-Length", 0))
            logger.info(
                "phone chat create bytes=%s from=%s ua=%s",
                content_length,
                self.client_address[0],
                self.headers.get("User-Agent", "")[:80],
            )
            body = self.rfile.read(content_length) if content_length > 0 else b""
            try:
                req = json.loads(body)
            except (json.JSONDecodeError, ValueError):
                self._json_response({"error": "invalid JSON"}, 400)
                return
            sid = uuid.uuid4().hex[:12]
            req["stream"] = True
            session = {
                "sid": sid,
                "req": req,
                "created": time.time(),
                "events": [],
                "done": False,
                "error": None,
                "cond": threading.Condition(),
            }
            chat_sessions[sid] = session
            threading.Thread(target=self._run_chat_session, args=(sid,), daemon=True).start()
            # Cleanup stale sessions (>5 min)
            cutoff = time.time() - 1800
            for k in [k for k, v in chat_sessions.items() if v["created"] < cutoff and v.get("done")]:
                del chat_sessions[k]
            self._json_response({"sid": sid})
            return

        if parsed.path == "/api/image/stylize":
            self._handle_image_stylize()
        elif parsed.path == "/api/image/generate":
            self._handle_image_generate()
        elif parsed.path == "/api/tools/execute":
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
            self._receive_file(TRANSFERS_DIR / parsed.path[len("/upload/") :])
        elif parsed.path.startswith("/share/"):
            self._receive_file(SHARED_DIR / parsed.path[len("/share/") :])
        elif parsed.path.startswith("/delete/"):
            self._delete_shared(parsed.path[len("/delete/") :])
        elif parsed.path.startswith("/delete-transfer/"):
            self._delete_transfer(parsed.path[len("/delete-transfer/") :])
        elif parsed.path == "/api/analyze-image":
            self._analyze_image()
        elif parsed.path.startswith("/v1"):
            self._route_model_request()
        else:
            self._proxy_to_llm()

    def _serve_gateway_diag(self):
        page = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MAUDE Gateway Diagnostic</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#0d1117;color:#e6edf3;margin:0;padding:20px}}code,pre{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;display:block;white-space:pre-wrap;word-break:break-word}}a{{display:block;margin-top:12px;padding:12px;border-radius:8px;background:#58a6ff;color:#07111f;text-align:center;text-decoration:none;font-weight:700;box-sizing:border-box}}small{{color:#8b949e}}</style>
</head><body>
<h1>MAUDE Gateway Live</h1>
<p>This page was generated by the gateway process, not the React app.</p>
<code>server_time: {time.strftime("%Y-%m-%d %H:%M:%S %Z")}</code>
<code>client_ip: {html.escape(self.client_address[0])}</code>
<code>path: {html.escape(self.path)}</code>
<a href="/api/ping?from=diag">Open /api/ping</a>
<a href="/?fresh=diag-{int(time.time())}">Open MAUDE app fresh</a>
<pre id="result">Testing fetch('/api/ping')...</pre>
<script>
fetch('/api/ping?from=diag-js', {{ cache: 'no-store' }})
  .then(r => r.text().then(t => document.getElementById('result').textContent = 'fetch status: ' + r.status + '\n' + t))
  .catch(e => document.getElementById('result').textContent = 'fetch failed: ' + e.name + ': ' + e.message);
</script>
<small>If this page does not load, the phone is not reaching port 30000.</small>
</body></html>"""
        body = page.encode()
        self.send_response(200)
        self._add_cors()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def _serve_plain_chat(self, answer: str = ""):
        logger.info("plain chat page from=%s ua=%s", self.client_address[0], self.headers.get("User-Agent", "")[:80])
        escaped = html.escape(answer)
        response_block = ""
        if escaped:
            response_block = "<h2>Response</h2><pre>" + escaped + "</pre>"
        page = (
            """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MAUDE Plain Chat</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background:#0d1117; color:#e6edf3; margin:0; padding:16px; }
textarea { width:100%; min-height:120px; box-sizing:border-box; font:16px -apple-system; background:#161b22; color:#e6edf3; border:1px solid #30363d; border-radius:8px; padding:10px; }
button { margin-top:10px; width:100%; padding:14px; font:600 16px -apple-system; border:0; border-radius:8px; background:#58a6ff; color:#07111f; }
pre { white-space:pre-wrap; background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px; }
small { color:#8b949e; }
</style></head><body>
<h1>MAUDE Plain Chat</h1>
<small>No React. No service worker. No streaming fetch.</small>
<form method="post" action="/plain-chat">
<textarea name="message" placeholder="Message MAUDE"></textarea>
<button type="submit">Send</button>
</form>
"""
            + response_block
            + """
</body></html>"""
        )
        body = page.encode()
        self.send_response(200)
        self._add_cors()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def _handle_plain_chat_post(self):
        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length).decode(errors="replace") if content_length else ""
        params = parse_qs(raw)
        message = (params.get("message") or [""])[0].strip()
        logger.info(
            "plain chat post bytes=%s from=%s ua=%s",
            content_length,
            self.client_address[0],
            self.headers.get("User-Agent", "")[:80],
        )
        if not message:
            self._serve_plain_chat("No message entered.")
            return
        req = {
            "model": os.environ.get("MAUDE_MODEL", "nemotron-super"),
            "messages": [
                {"role": "system", "content": "You are MAUDE. Be concise and helpful."},
                {"role": "user", "content": message},
            ],
            "stream": False,
            "max_tokens": 1024,
            "temperature": 0.3,
        }
        conn = None
        try:
            body = json.dumps(req).encode()
            conn = http.client.HTTPConnection("localhost", 30080, timeout=180)
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            )
            resp = conn.getresponse()
            data = json.loads(resp.read().decode(errors="replace"))
            answer = data.get("choices", [{}])[0].get("message", {}).get("content") or data.get("error") or str(data)
        except Exception as exc:
            answer = f"Error: {exc}"
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
        self._serve_plain_chat(answer)

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

        model_name = req.get("model", os.environ.get("MAUDE_MODEL", "nemotron-super"))
        model_absent = "model" not in req
        resolved_name, route = get_model_route(model_name)
        logger.info(
            "route requested=%s resolved=%s%s",
            model_name,
            resolved_name,
            " (client sent no model field -> default)" if model_absent else "",
        )

        # Cache phone location and use as fallback for non-phone clients
        if "location" in req:
            loc = req["location"]
            if isinstance(loc, dict) and loc.get("lat") is not None:
                device_location.clear()
                device_location.update({**loc, "ts": time.time()})
        elif device_location and time.time() - device_location.get("ts", 0) < 3600:
            # Use cached phone location for CLI/client requests (fresh within 1 hour)
            req["location"] = dict(device_location)

        if not route:
            self._json_response({"error": f"Unknown model: {model_name}"}, 400)
            return

        req["_route_trace"] = {
            "requested_model": model_name,
            "resolved_model": resolved_name,
            "provider": route["provider"],
            "max_context": route.get("max_context", 0),
        }

        if route["provider"] == "codex-cli":
            self._codex_cli_response(req, resolved_name)
            return

        if route["provider"] == "grok-cli":
            self._grok_cli_response(req, resolved_name)
            return

        # Local models — tool-capable ones go through the same tool loop as cloud
        if route["provider"] == "local":
            req["model"] = resolved_name
            client_sent_tools = bool(req.get("tools"))
            plain_api = bool(req.get("response_format"))
            # Vision model and JSON mode -> raw proxy (no tool loop)
            if resolved_name in ("llava",) or plain_api or client_sent_tools:
                self._proxy_to_llm(override_body=json.dumps(req).encode())
                return
            # Tool-capable local models -> same tool loop as cloud
            if TOOL_SUPPORT:
                req["_route_trace"]["tool_mode"] = "server"
                self._cloud_model_with_tools(req, route, resolved_name)
                return
            # Fallback: raw proxy
            self._proxy_to_llm(override_body=json.dumps(req).encode())
            return

        # Cloud models -> forward to provider API
        api_key = os.environ.get(route["api_key_env"], "") if route["api_key_env"] else ""
        if not api_key:
            self._json_response({"error": f"No API key for {route['provider']} ({route['api_key_env']})"}, 503)
            return

        # Use tool-enabled path for cloud models if maude_core is available
        # Skip server-side tool loop if client already sent its own tools
        # Also skip if response_format is set (JSON mode -- plain API call)
        client_sent_tools = bool(req.get("tools"))
        plain_api = bool(req.get("response_format"))
        if TOOL_SUPPORT and not client_sent_tools and not plain_api:
            if route["provider"] in ("mistral", "openrouter"):
                req["_route_trace"]["tool_mode"] = "server"
                self._cloud_model_with_tools(req, route, resolved_name)
                return
            if route["provider"] == "anthropic":
                req["_route_trace"]["tool_mode"] = "server"
                self._claude_tool_loop(req, route, resolved_name)
                return

        # Inject location into system prompt before stripping from body
        req["model"] = resolved_name
        location = req.pop("location", None)
        if location and isinstance(location, dict):
            lat = location.get("lat")
            lng = location.get("lng")
            if lat is not None and lng is not None:
                loc_ctx = (
                    f"\nDEVICE LOCATION: The user's phone is at latitude {lat:.6f}, "
                    f"longitude {lng:.6f} (accuracy: {location.get('accuracy', 'unknown')}m). "
                    "Use this for location-aware responses."
                )
                for msg in req.get("messages", []):
                    if msg.get("role") == "system":
                        msg["content"] = msg["content"] + loc_ctx
                        break
        body = json.dumps(req).encode()

        parsed_url = urlparse(route["base_url"])
        use_ssl = parsed_url.scheme == "https"
        host = parsed_url.hostname
        port = parsed_url.port or (443 if use_ssl else 80)

        try:
            if use_ssl:
                ctx = ssl.create_default_context()
                # Disable older versions of SSL/TLS that are known to have issues
                ctx.options |= ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3 | ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
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

            is_streaming = resp.headers.get("Transfer-Encoding") == "chunked" or "text/event-stream" in (
                resp.headers.get("Content-Type", "")
            )

            self.send_response(resp.status)
            self._add_cors()
            for key, val in resp.getheaders():
                if key.lower() not in (
                    "transfer-encoding",
                    "access-control-allow-origin",
                    "access-control-allow-methods",
                    "access-control-allow-headers",
                ):
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

    def _run_chat_session(self, sid: str):
        """Run an iOS/mobile chat request independently of any client connection."""
        session = chat_sessions.get(sid)
        if not session:
            return
        body = json.dumps(session["req"]).encode()
        try:
            conn = http.client.HTTPConnection("localhost", 30080, timeout=900)
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            )
            resp = conn.getresponse()
            if resp.status >= 400:
                err = resp.read().decode(errors="replace")
                self._append_chat_event(
                    session, f"event: trace\ndata: {json.dumps({'type': 'error', 'message': err})}\n\n"
                )
                self._append_chat_event(session, "data: [DONE]\n\n")
                return

            block_lines = []
            while True:
                raw = resp.readline()
                if not raw:
                    break
                line = raw.decode(errors="replace")
                if line.strip() == "":
                    if block_lines:
                        event = self._normalize_chat_event_block("".join(block_lines) + "\n")
                        if event:
                            self._append_chat_event(session, event)
                            if "data: [DONE]" in event:
                                break
                        block_lines = []
                    continue
                block_lines.append(line)
            if block_lines:
                event = self._normalize_chat_event_block("".join(block_lines) + "\n")
                if event:
                    self._append_chat_event(session, event)
        except Exception as e:
            self._append_chat_event(
                session,
                f"event: trace\ndata: {json.dumps({'type': 'error', 'message': f'Background chat failed: {e}'})}\n\n",
            )
            self._append_chat_event(session, "data: [DONE]\n\n")
        finally:
            try:
                conn.close()
            except Exception:
                pass
            with session["cond"]:
                session["done"] = True
                session["cond"].notify_all()

    @staticmethod
    def _normalize_chat_event_block(block: str) -> str:
        """Convert gateway SSE comments into EventSource-visible trace events."""
        stripped = block.strip()
        if not stripped:
            return ""
        if stripped.startswith(": trace "):
            payload = stripped[len(": trace ") :]
            return f"event: trace\ndata: {payload}\n\n"
        return block if block.endswith("\n\n") else block.rstrip("\n") + "\n\n"

    @staticmethod
    def _append_chat_event(session: dict, event: str):
        with session["cond"]:
            session["events"].append(event)
            session["cond"].notify_all()

    def _stream_chat_session(self, session: dict, offset: int):
        """Attach/re-attach an EventSource client to a background chat session."""
        self.send_response(200)
        self._add_cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()

        idx = max(0, offset)
        last_ping = time.time()
        while True:
            with session["cond"]:
                while idx >= len(session["events"]) and not session["done"]:
                    session["cond"].wait(timeout=10)
                    if time.time() - last_ping >= 10:
                        break
                events = session["events"][idx:]
                done = session["done"] and idx >= len(session["events"])

            if events:
                for event in events:
                    try:
                        self.wfile.write(f"id: {idx}\n".encode())
                        self.wfile.write(event.encode())
                        self.wfile.flush()
                    except Exception:
                        return
                    idx += 1
                    if "data: [DONE]" in event:
                        self.close_connection = True
                        return
                last_ping = time.time()
                continue

            if done:
                return

            try:
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
                last_ping = time.time()
            except Exception:
                return

    def _handle_image_stylize(self):
        """POST /api/image/stylize — img2img via Replicate Flux 2 Klein."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        try:
            req = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            self._json_response({"error": "Invalid JSON"}, 400)
            return

        prompt = req.get("prompt", "")
        image = req.get("image", "")
        if not prompt or not image:
            self._json_response({"error": "Both 'prompt' and 'image' (base64) are required"}, 400)
            return

        try:
            from .replicate import stylize_image

            result = stylize_image(
                prompt=prompt,
                image_base64=image,
                strength=req.get("strength", 0.55),
                model=req.get("model", "black-forest-labs/flux-2-klein"),
                width=req.get("width", 1024),
                height=req.get("height", 1024),
            )
            self._json_response(result)
        except Exception as e:
            logger.error("Image stylize failed: %s", e)
            self._json_response({"error": str(e)}, 500)

    def _handle_image_generate(self):
        """POST /api/image/generate — text-to-image via Replicate."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        try:
            req = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            self._json_response({"error": "Invalid JSON"}, 400)
            return

        prompt = req.get("prompt", "")
        if not prompt:
            self._json_response({"error": "'prompt' is required"}, 400)
            return

        try:
            from .replicate import generate_image

            result = generate_image(
                prompt=prompt,
                model=req.get("model", "black-forest-labs/flux-2-klein"),
                width=req.get("width", 1024),
                height=req.get("height", 1024),
            )
            self._json_response(result)
        except Exception as e:
            logger.error("Image generate failed: %s", e)
            self._json_response({"error": str(e)}, 500)

    def _json_response(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self._add_cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        self.wfile.write(data)
