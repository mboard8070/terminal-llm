#!/usr/bin/env python3
"""
MAUDE Gateway - Single port proxy for LLM + file server.

Sits on port 30000. Routes:
  /v1/*          -> llama-server (port 30010)
  /list          -> shared folder listing
  /download/*    -> download from shared
  /upload/*      -> upload to transfers
  /share/*       -> upload to shared
  /transfers     -> list transfers
  /health        -> health check
"""

import os
import json
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import unquote, urlparse
import http.client
import threading

SHARED_DIR = Path.home() / "nvidia-workbench" / "terminal-llm" / "shared"
TRANSFERS_DIR = Path.home() / "nvidia-workbench" / "terminal-llm" / "transfers"
LLM_PORT = 30010  # llama-server runs here internally
GATEWAY_PORT = 30000  # clients connect here

SHARED_DIR.mkdir(parents=True, exist_ok=True)
TRANSFERS_DIR.mkdir(parents=True, exist_ok=True)


class GatewayHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        path = unquote(self.path)

        if path == "/list":
            self._list_dir(SHARED_DIR)
        elif path == "/transfers":
            self._list_dir(TRANSFERS_DIR)
        elif path.startswith("/download/"):
            self._send_file(SHARED_DIR / path[len("/download/"):])
        elif path.startswith("/download-transfer/"):
            self._send_file(TRANSFERS_DIR / path[len("/download-transfer/"):])
        elif path == "/health":
            self._json_response({"status": "ok", "llm_port": LLM_PORT})
        elif path.startswith("/v1"):
            self._proxy_to_llm()
        else:
            self._proxy_to_llm()

    def do_POST(self):
        path = unquote(self.path)

        if path.startswith("/upload/"):
            self._receive_file(TRANSFERS_DIR / path[len("/upload/"):])
        elif path.startswith("/share/"):
            self._receive_file(SHARED_DIR / path[len("/share/"):])
        elif path.startswith("/v1"):
            self._proxy_to_llm()
        else:
            self._proxy_to_llm()

    def do_OPTIONS(self):
        self._proxy_to_llm()

    def _proxy_to_llm(self):
        """Forward request to llama-server."""
        try:
            conn = http.client.HTTPConnection("localhost", LLM_PORT, timeout=300)

            # Read request body if present
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else None

            # Forward headers
            headers = {}
            for key, val in self.headers.items():
                if key.lower() not in ("host", "transfer-encoding"):
                    headers[key] = val

            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()

            # Check if streaming response
            is_streaming = resp.headers.get("Transfer-Encoding") == "chunked" or \
                          "text/event-stream" in (resp.headers.get("Content-Type", ""))

            self.send_response(resp.status)
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
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
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

    def _json_response(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    print(f"MAUDE Gateway on port {GATEWAY_PORT}")
    print(f"  LLM proxy -> localhost:{LLM_PORT}")
    print(f"  Shared:    {SHARED_DIR}")
    print(f"  Transfers: {TRANSFERS_DIR}")
    server = HTTPServer(("0.0.0.0", GATEWAY_PORT), GatewayHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
