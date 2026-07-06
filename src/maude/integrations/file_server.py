#!/usr/bin/env python3
"""
MAUDE File Server - Simple HTTP server for the shared and transfers folders.

Runs on port 30002. The server is the single source of truth for both folders;
clients pull on demand and push when they want to share. There is no
background sync — every operation is an explicit HTTP call.

Routes:
  GET  /list                     -> list shared folder
  GET  /transfers                -> list transfers folder
  GET  /download/<file>          -> download from shared
  GET  /download-transfer/<file> -> download from transfers
  GET  /health                   -> health check
  POST /share/<file>             -> upload into shared
  POST /upload/<file>            -> upload into transfers
  POST /delete/<file>            -> delete from shared
  POST /delete-transfer/<file>   -> delete from transfers
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from maude.config import runtime_paths
from urllib.parse import unquote

_PATHS = runtime_paths()
SHARED_DIR = _PATHS.shared_dir
TRANSFERS_DIR = _PATHS.transfers_dir
PORT = 30002

SHARED_DIR.mkdir(parents=True, exist_ok=True)
TRANSFERS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_name(name: str) -> bool:
    return bool(name) and "/" not in name and "\\" not in name and not name.startswith(".")


class FileHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Quiet logging

    def do_GET(self):
        path = unquote(self.path)

        if path == "/list":
            self._list_dir(SHARED_DIR)
        elif path == "/transfers":
            self._list_dir(TRANSFERS_DIR)
        elif path.startswith("/download/"):
            self._send_file(SHARED_DIR / path[len("/download/") :])
        elif path.startswith("/download-transfer/"):
            self._send_file(TRANSFERS_DIR / path[len("/download-transfer/") :])
        elif path == "/health":
            self._json_response({"status": "ok"})
        else:
            self._error(404, "Not found")

    def do_POST(self):
        path = unquote(self.path)

        if path.startswith("/upload/"):
            self._receive_file(TRANSFERS_DIR / path[len("/upload/") :])
        elif path.startswith("/share/"):
            self._receive_file(SHARED_DIR / path[len("/share/") :])
        elif path.startswith("/delete/"):
            self._delete_in(SHARED_DIR, path[len("/delete/") :])
        elif path.startswith("/delete-transfer/"):
            self._delete_in(TRANSFERS_DIR, path[len("/delete-transfer/") :])
        else:
            self._error(404, "Not found")

    def _delete_in(self, base: Path, filename: str):
        if not _safe_name(filename):
            self._error(400, "Invalid filename")
            return
        target = base / filename
        if not target.exists():
            self._json_response({"status": "ok", "filename": filename, "existed": False})
            return
        try:
            target.unlink()
            self._json_response({"status": "ok", "filename": filename, "existed": True})
        except Exception as e:
            self._error(500, str(e))

    def _list_dir(self, directory: Path):
        entries = []
        for entry in sorted(directory.iterdir()):
            if entry.name.startswith("."):
                continue
            stat = entry.stat()
            entries.append(
                {
                    "name": entry.name,
                    "size": stat.st_size,
                    "is_dir": entry.is_dir(),
                    "modified": stat.st_mtime,
                }
            )
        self._json_response({"path": str(directory), "files": entries})

    def _send_file(self, filepath: Path):
        if not filepath.exists():
            self._error(404, f"File not found: {filepath.name}")
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
            self._error(500, str(e))

    def _receive_file(self, filepath: Path):
        if not _safe_name(filepath.name):
            self._error(400, "Invalid filename")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(data)
            self._json_response({"status": "ok", "filename": filepath.name, "size": len(data)})
        except Exception as e:
            self._error(500, str(e))

    def _json_response(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code, msg):
        self._json_response({"error": msg}, code)


if __name__ == "__main__":
    print(f"MAUDE File Server on port {PORT}")
    print(f"  Shared:    {SHARED_DIR}")
    print(f"  Transfers: {TRANSFERS_DIR}")

    server = HTTPServer(("0.0.0.0", PORT), FileHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
