#!/usr/bin/env python3
"""
MAUDE File Server - Simple HTTP server for shared folder access.

Runs on port 30002. Clients tunnel to it the same way they tunnel to the LLM.
Supports: list files, download files, upload files.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote

SHARED_DIR = Path.home() / "nvidia-workbench" / "terminal-llm" / "shared"
TRANSFERS_DIR = Path.home() / "nvidia-workbench" / "terminal-llm" / "transfers"
PORT = 30002

SHARED_DIR.mkdir(parents=True, exist_ok=True)
TRANSFERS_DIR.mkdir(parents=True, exist_ok=True)


class FileHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Quiet logging

    def do_GET(self):
        path = unquote(self.path)

        # GET /list — list shared folder
        if path == "/list":
            self._list_dir(SHARED_DIR)

        # GET /transfers — list transfers folder
        elif path == "/transfers":
            self._list_dir(TRANSFERS_DIR)

        # GET /download/<filename> — download from shared
        elif path.startswith("/download/"):
            filename = path[len("/download/") :]
            self._send_file(SHARED_DIR / filename)

        # GET /download-transfer/<filename> — download from transfers
        elif path.startswith("/download-transfer/"):
            filename = path[len("/download-transfer/") :]
            self._send_file(TRANSFERS_DIR / filename)

        # GET /health — health check
        elif path == "/health":
            self._json_response({"status": "ok"})

        else:
            self._error(404, "Not found. Use /list, /download/<file>, or POST /upload")

    def do_POST(self):
        path = unquote(self.path)

        # POST /upload — upload to transfers (client -> server)
        if path.startswith("/upload/"):
            filename = path[len("/upload/") :]
            self._receive_file(TRANSFERS_DIR / filename)

        # POST /share — upload to shared folder
        elif path.startswith("/share/"):
            filename = path[len("/share/") :]
            self._receive_file(SHARED_DIR / filename)

        else:
            self._error(404, "Use POST /upload/<filename> or /share/<filename>")

    def _list_dir(self, directory):
        entries = []
        for entry in sorted(directory.iterdir()):
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

    def _send_file(self, filepath):
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

    def _receive_file(self, filepath):
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


DELETIONS_FILE = SHARED_DIR / ".maude_deletions"


def _scan_shared():
    """Return set of non-hidden filenames in shared dir."""
    return {
        f.name for f in SHARED_DIR.iterdir()
        if f.is_file() and not f.name.startswith(".")
    }


def _shared_watcher(interval=5):
    """Background thread: detect files deleted from shared/ and record them."""
    known = _scan_shared()
    while True:
        time.sleep(interval)
        try:
            current = _scan_shared()
            deleted = known - current
            if deleted:
                with open(DELETIONS_FILE, "a") as f:
                    for name in sorted(deleted):
                        f.write(name + "\n")
                print(f"  [watcher] Recorded deletions: {sorted(deleted)}")
            known = current
        except Exception as e:
            print(f"  [watcher] Error: {e}")


if __name__ == "__main__":
    print(f"MAUDE File Server on port {PORT}")
    print(f"  Shared:    {SHARED_DIR}")
    print(f"  Transfers: {TRANSFERS_DIR}")

    # Start shared folder watcher (detects deletions for sync propagation)
    threading.Thread(target=_shared_watcher, daemon=True).start()
    print("  Watcher:   active (scanning every 5s)")

    server = HTTPServer(("0.0.0.0", PORT), FileHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
