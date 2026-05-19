"""
Route handler methods for the MAUDE Gateway.

Contains RoutesMixin — methods for handling non-cloud API endpoints
(files, collaboration, conversations, health, tools, static serving, etc.).
Mixed into GatewayHandler.
"""

import http.client
import json
import mimetypes
import os
import ssl
import subprocess
import time
from urllib.parse import urljoin, urlparse

from .state import (
    CONVERSATIONS_DIR,
    GATEWAY_PORT,
    LLM_PORT,
    MODEL_ALIASES,
    MODEL_ROUTES,
    PWA_DIR,
    SHARED_DIR,
    TOOL_SUPPORT,
    VOICE_PORT,
    execute_tool,
    reset_rate_limits,
)


class RoutesMixin:
    """Mixin providing route handler methods for GatewayHandler."""

    def _serve_models(self):
        """Return available models for the UI."""
        models = []
        for model_id, config in MODEL_ROUTES.items():
            available = True
            if config["api_key_env"]:
                available = bool(os.environ.get(config["api_key_env"]))
            models.append(
                {
                    "id": model_id,
                    "provider": config["provider"],
                    "available": available,
                }
            )
        route_ids = {model["id"] for model in models}
        for alias, target in MODEL_ALIASES.items():
            if alias in route_ids:
                continue
            config = MODEL_ROUTES.get(target)
            if not config:
                continue
            available = True
            if config["api_key_env"]:
                available = bool(os.environ.get(config["api_key_env"]))
            models.append(
                {
                    "id": alias,
                    "provider": config["provider"],
                    "available": available,
                }
            )
        self._json_response({"models": models})

    def _serve_v1_models(self):
        """Return models in OpenAI-compatible /v1/models format."""
        data = []
        for model_id in MODEL_ROUTES:
            data.append({"id": model_id, "object": "model", "owned_by": "maude"})
        for alias in MODEL_ALIASES:
            data.append({"id": alias, "object": "model", "owned_by": "maude"})
        self._json_response({"object": "list", "data": data})

    def _serve_tasks(self):
        """Return observable long-running MAUDE-related background tasks."""
        tasks = []
        try:
            proc = subprocess.run(
                ["ps", "-eo", "pid,ppid,etime,stat,cmd"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in proc.stdout.splitlines()[1:]:
                parts = line.strip().split(None, 4)
                if len(parts) < 5:
                    continue
                pid, ppid, elapsed, stat, cmd = parts
                cmd_lower = cmd.lower()
                if "rg -i" in cmd_lower or "ps -eo" in cmd_lower:
                    continue
                label = None
                kind = None
                if "codex exec" in cmd_lower:
                    kind = "codex"
                    label = "Codex is still working"
                elif "hyperframes" in cmd_lower and ("render" in cmd_lower or "npm run" in cmd_lower):
                    kind = "render"
                    label = "HyperFrames is rendering"
                elif "ffmpeg" in cmd_lower:
                    kind = "render"
                    label = "FFmpeg is rendering video"
                elif "youtube" in cmd_lower and "upload" in cmd_lower:
                    kind = "upload"
                    label = "YouTube upload is running"
                elif "python app/main.py" in cmd_lower and "comfy" not in cmd_lower:
                    kind = "comfyui"
                    label = "ComfyUI is running"
                if label:
                    tasks.append(
                        {
                            "pid": int(pid),
                            "ppid": int(ppid),
                            "elapsed": elapsed,
                            "stat": stat,
                            "kind": kind,
                            "label": label,
                            "cmd": cmd[:240],
                        }
                    )
        except Exception as e:
            self._json_response({"error": str(e)}, 500)
            return

        recent_outputs = []
        try:
            cutoff = time.time() - 3600
            for path in SHARED_DIR.iterdir():
                if path.is_file() and path.suffix.lower() in {".mp4", ".png", ".jpg", ".jpeg"}:
                    st = path.stat()
                    if st.st_mtime >= cutoff:
                        recent_outputs.append(
                            {
                                "filename": path.name,
                                "size": st.st_size,
                                "modified": st.st_mtime,
                                "url": f"/download/{path.name}",
                            }
                        )
            recent_outputs.sort(key=lambda item: item["modified"], reverse=True)
            recent_outputs = recent_outputs[:10]
        except Exception:
            recent_outputs = []

        self._json_response({"tasks": tasks, "recent_outputs": recent_outputs})

    def _proxy_to_llm(self, override_body=None):
        """Forward request to local llama-server."""
        try:
            conn = http.client.HTTPConnection("localhost", LLM_PORT, timeout=300)

            if override_body is not None:
                body = override_body
                method = "POST"
                path = "/v1/chat/completions"
            else:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length) if content_length > 0 else None
                method = self.command
                path = self.path

            headers = {}
            for key, val in self.headers.items():
                if key.lower() not in ("host", "transfer-encoding", "content-length"):
                    headers[key] = val
            if body is not None:
                headers["Content-Length"] = str(len(body))
                headers["Content-Type"] = "application/json"

            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()

            is_streaming = resp.headers.get("Transfer-Encoding") == "chunked" or "text/event-stream" in (
                resp.headers.get("Content-Type", "")
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

            conn.request(
                "GET",
                request_path,
                headers={
                    "User-Agent": "Mozilla/5.0 (Linux; Android 14) MAUDE/1.0",
                    "Accept": "text/html,application/xhtml+xml,*/*",
                    "Host": host,
                },
            )
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

    def _serve_reset_cache_page(self):
        """Serve a standalone page that clears MAUDE PWA cache/state."""
        html = b'''<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Reset MAUDE</title>
  <style>
    body { background:#0d1117; color:#e6edf3; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; padding:32px; }
    code { background:#161b22; padding:2px 5px; border-radius:4px; }
    .ok { color:#7ee787; }
    .warn { color:#ffa657; }
  </style>
</head>
<body>
  <h1>Resetting MAUDE</h1>
  <p id="status">Clearing service workers, caches, and local app state...</p>
  <script>
    async function reset() {
      const lines = [];
      try {
        if ('serviceWorker' in navigator) {
          const regs = await navigator.serviceWorker.getRegistrations();
          await Promise.all(regs.map((r) => r.unregister()));
          lines.push('Unregistered ' + regs.length + ' service worker(s).');
        }
      } catch (e) { lines.push('Service worker reset failed: ' + e); }
      try {
        if ('caches' in window) {
          const names = await caches.keys();
          await Promise.all(names.map((n) => caches.delete(n)));
          lines.push('Deleted ' + names.length + ' cache(s).');
        }
      } catch (e) { lines.push('Cache reset failed: ' + e); }
      try {
        localStorage.clear();
        sessionStorage.clear();
        lines.push('Cleared local storage.');
      } catch (e) { lines.push('Storage reset failed: ' + e); }
      document.getElementById('status').innerHTML = '<span class="ok">Reset complete.</span><br>' + lines.map((l) => '<code>' + l + '</code>').join('<br>') + '<br><br>Redirecting to MAUDE...';
      setTimeout(() => { window.location.replace('/?fresh=' + Date.now()); }, 1000);
    }
    reset();
  </script>
  <noscript><p class="warn">JavaScript is required to clear the Safari cache.</p></noscript>
</body>
</html>
'''
        self.send_response(200)
        self._add_cors()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        self.wfile.write(html)

    def _serve_static(self, path):
        """Serve PWA static files from dist/."""
        if path == "/" or path == "/app" or path == "/app/":
            path = "/index.html"
        elif path.startswith("/app/"):
            path = path[4:]  # Strip /app prefix

        # Map to file
        filepath = PWA_DIR / path.lstrip("/")

        # SPA fallback: if file not found and no extension, serve index.html
        if (not filepath.exists() or filepath.is_dir()) and "." not in filepath.name:
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

    def _delete_shared(self, filename: str):
        """Delete a file from the shared folder."""
        self._delete_in(SHARED_DIR, filename)

    def _delete_transfer(self, filename: str):
        """Delete a file from the transfers folder."""
        from .state import TRANSFERS_DIR

        self._delete_in(TRANSFERS_DIR, filename)

    def _delete_in(self, base, filename: str):
        if not filename or "/" in filename or "\\" in filename or filename.startswith("."):
            self._json_response({"error": "Invalid filename"}, 400)
            return
        target = base / filename
        if not target.exists():
            self._json_response({"status": "ok", "filename": filename, "existed": False})
            return
        try:
            target.unlink()
            self._json_response({"status": "ok", "filename": filename, "existed": True})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _analyze_image(self):
        """Analyze an image in shared/ using the active vision model via maude_core.execute_tool."""
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

    # -- Conversation sync API ----------------------------------------

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

    # -- Collaboration API --------------------------------------------

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

    # -- Command Center API -------------------------------------------

    def _handle_command_center(self, path: str, query: dict):
        """Route /api/command-center/* to the command center tools."""
        try:
            from maude_core.tools_command_center import (
                _dispatch_activity_feed,
                _dispatch_gpu_processes,
                _dispatch_memory_browse,
                _dispatch_node_status,
                _dispatch_scheduler_status,
                _dispatch_session_list,
                _dispatch_system_stats,
            )
        except ImportError:
            self._json_response({"error": "command center module not available"}, 503)
            return

        endpoint = path.replace("/api/command-center/", "")

        if endpoint == "system":
            result = _dispatch_system_stats({})
        elif endpoint == "gpu-processes":
            result = _dispatch_gpu_processes({})
        elif endpoint == "memory":
            args = {}
            if "category" in query:
                args["category"] = query["category"][0]
            if "query" in query:
                args["query"] = query["query"][0]
            if "limit" in query:
                args["limit"] = int(query["limit"][0])
            result = _dispatch_memory_browse(args)
        elif endpoint == "sessions":
            args = {}
            if "limit" in query:
                args["limit"] = int(query["limit"][0])
            result = _dispatch_session_list(args)
        elif endpoint == "activity":
            args = {}
            if "limit" in query:
                args["limit"] = int(query["limit"][0])
            result = _dispatch_activity_feed(args)
        elif endpoint == "scheduler":
            result = _dispatch_scheduler_status({})
        elif endpoint == "nodes":
            result = _dispatch_node_status({})
        else:
            self._json_response({"error": f"Unknown endpoint: {endpoint}"}, 404)
            return

        # Tools return JSON strings, parse and re-serve
        try:
            self._json_response(json.loads(result))
        except (json.JSONDecodeError, TypeError):
            self._json_response({"result": result})

    # -- Health & Tool Catalog API ------------------------------------

    def _redirect_vnc(self):
        """GET /vnc -- redirect to the noVNC web viewer on port 6080.

        This endpoint exists so browser_login can return a gateway-relative URL
        (e.g. http://<gateway>:30080/vnc) instead of a raw noVNC URL that the
        LLM might mangle.
        """
        try:
            # Build redirect using the Host header the client used to reach us
            host = self.headers.get("Host", "localhost:30080")
            # Strip port from host to get the hostname/IP the client used
            hostname = host.split(":")[0]
            target = f"http://{hostname}:6080/vnc.html?autoconnect=true"
            self.send_response(302)
            self.send_header("Location", target)
            self.end_headers()
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _serve_health(self):
        """Enhanced /health -- structured report with deps, services, tools."""
        try:
            from health import HealthChecker

            report = HealthChecker().check_all()
        except ImportError:
            report = {
                "status": "ok",
                "llm_port": LLM_PORT,
                "voice_server_port": VOICE_PORT,
                "gateway_port": GATEWAY_PORT,
            }
        self._json_response(report)

    def _serve_tools(self, query: dict):
        """GET /api/tools -- full catalog or filtered by ?message=."""
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
        """POST /api/tools/execute -- execute a server-side tool."""
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
