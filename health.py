from maude_bootstrap import ensure_local_maude

ensure_local_maude()

"""
MAUDE Health Check System — centralized dependency and service verification.

Provides structured health reports for the gateway /health endpoint,
tool pre-flight checks, and startup diagnostics.
"""

import importlib
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

# ── Dependency → version mapping ──────────────────────────────────

_OPTIONAL_DEPS = {
    "ddgs": "ddgs",
    "beautifulsoup4": "bs4",
    "playwright": "playwright",
    "anthropic": "anthropic",
    "google-api-python-client": "googleapiclient",
    "google-auth-oauthlib": "google_auth_oauthlib",
    "croniter": "croniter",
    "faster-whisper": "faster_whisper",
    "python-telegram-bot": "telegram",
    "google-generativeai": "google.generativeai",
    "openai": "openai",
    "requests": "requests",
    "rich": "rich",
}

# ── Tool → required dependency mapping ────────────────────────────

_TOOL_DEPS = {
    "web_search": ["ddgs"],
    "web_browse": ["beautifulsoup4", "requests"],
    "web_view": ["playwright"],
    "view_image": [],  # needs LLaVA reachable, checked via service
    "generate_image": [],  # needs ComfyUI reachable, checked via service
    "gmail_list": ["google-api-python-client", "google-auth-oauthlib"],
    "gmail_read": ["google-api-python-client", "google-auth-oauthlib"],
    "gmail_send": ["google-api-python-client", "google-auth-oauthlib"],
    "drive_list": ["google-api-python-client"],
    "drive_search": ["google-api-python-client"],
    "drive_read": ["google-api-python-client"],
    "drive_upload": ["google-api-python-client"],
    "drive_create_doc": ["google-api-python-client"],
    "drive_create_folder": ["google-api-python-client"],
    "drive_create_sheet": ["google-api-python-client"],
    "drive_update_doc": ["google-api-python-client"],
    "drive_delete": ["google-api-python-client"],
    "sheets_read": ["google-api-python-client"],
    "sheets_write": ["google-api-python-client"],
    "sheets_append": ["google-api-python-client"],
    "sheets_create": ["google-api-python-client"],
    "sheets_list_sheets": ["google-api-python-client"],
    "sheets_clear": ["google-api-python-client"],
    "calendar_list_events": ["google-api-python-client"],
    "calendar_create_event": ["google-api-python-client"],
    "calendar_update_event": ["google-api-python-client"],
    "calendar_delete_event": ["google-api-python-client"],
    "calendar_search_events": ["google-api-python-client"],
    "calendar_list_calendars": ["google-api-python-client"],
    "slides_get_presentation": ["google-api-python-client"],
    "slides_get_slide": ["google-api-python-client"],
    "slides_create_presentation": ["google-api-python-client"],
    "slides_add_slide": ["google-api-python-client"],
    "slides_add_text": ["google-api-python-client"],
    "contacts_list": ["google-api-python-client"],
    "contacts_get": ["google-api-python-client"],
    "contacts_create": ["google-api-python-client"],
    "contacts_update": ["google-api-python-client"],
    "contacts_delete": ["google-api-python-client"],
    "contacts_search": ["google-api-python-client"],
    "youtube_search": ["google-api-python-client"],
    "youtube_get_video": ["google-api-python-client"],
    "youtube_get_channel": ["google-api-python-client"],
    "youtube_list_playlists": ["google-api-python-client"],
    "youtube_get_playlist_items": ["google-api-python-client"],
    "youtube_create_playlist": ["google-api-python-client"],
    "youtube_add_to_playlist": ["google-api-python-client"],
    "youtube_get_comments": ["google-api-python-client"],
    "youtube_post_comment": ["google-api-python-client"],
    "youtube_my_channel": ["google-api-python-client"],
    "substack_create_draft": [],
    "substack_list_drafts": [],
    "substack_list_posts": [],
    "substack_get_post": [],
    "substack_update_draft": [],
    "substack_delete_draft": [],
    "substack_get_stats": [],
    "schedule_task": ["croniter"],
    "send_to_claude": ["anthropic"],
    "hyperframes_doctor": [],
    "hyperframes_browser_ensure": [],
    "hyperframes_init": [],
    "hyperframes_lint": [],
    "hyperframes_render": [],
}

# Tools with no external deps (always ready)
_ALWAYS_READY = {
    "read_file",
    "write_file",
    "edit_file",
    "list_directory",
    "search_file",
    "search_directory",
    "run_command",
    "change_directory",
    "get_working_directory",
    "ask_frontier",
    "list_shared",
    "share_file",
    "list_transfers",
    "get_transfer",
    "mesh_status",
    "dispatch_task",
    "create_project",
    "list_projects",
    "add_to_project",
    "list_tasks",
}

_start_time = time.time()


class HealthChecker:
    """Centralized health checking for MAUDE services and tools."""

    def check_all(self) -> dict:
        """Full health report: services, deps, tools."""
        services = self.check_services()
        deps = self.check_dependencies()
        tools = self.check_tools(deps)

        # Determine overall status
        svc_down = any(s["status"] != "ok" for s in services.values())
        has_degraded = bool(tools.get("degraded"))

        if svc_down:
            status = "error"
        elif has_degraded:
            status = "degraded"
        else:
            status = "ok"

        return {
            "status": status,
            "services": services,
            "dependencies": deps,
            "tools": tools,
            "uptime": round(time.time() - _start_time, 1),
        }

    def check_services(self) -> dict:
        """Ping key services by TCP connect."""
        result = {}
        for name, port in [("llama_server", 30010), ("voice_server", 8998)]:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                sock.connect(("127.0.0.1", port))
                sock.close()
                result[name] = {"status": "ok", "port": port}
            except (TimeoutError, ConnectionRefusedError, OSError):
                result[name] = {"status": "down", "port": port}
        return result

    def check_dependencies(self) -> dict:
        """Try-import each optional package, report version + availability."""
        result = {}
        for pkg_name, import_name in _OPTIONAL_DEPS.items():
            try:
                mod = importlib.import_module(import_name)
                version = getattr(mod, "__version__", getattr(mod, "VERSION", "unknown"))
                result[pkg_name] = {"available": True, "version": str(version)}
            except ImportError as e:
                result[pkg_name] = {"available": False, "error": str(e)}
        return result

    def check_tools(self, deps: dict = None) -> dict:
        """Map each tool to its deps, report which are functional."""
        if deps is None:
            deps = self.check_dependencies()

        try:
            from maude_core import TOOLS

            total = len(TOOLS)
            tool_names = {t["function"]["name"] for t in TOOLS}
        except ImportError:
            return {"total": 0, "functional": 0, "degraded": [], "reasons": {}}

        degraded = []
        reasons = {}
        functional = 0

        for name in tool_names:
            ready, reason = self._check_single(name, deps)
            if ready:
                functional += 1
            else:
                degraded.append(name)
                reasons[name] = reason

        return {
            "total": total,
            "functional": functional,
            "degraded": degraded,
            "reasons": reasons,
        }

    def check_tool_ready(self, name: str) -> tuple:
        """Pre-flight check for a single tool. Returns (ready: bool, reason: str)."""
        deps = self.check_dependencies()
        return self._check_single(name, deps)

    def _check_single(self, name: str, deps: dict) -> tuple:
        """Check a single tool against dep status dict and runtime capability."""
        if name in _ALWAYS_READY:
            return (True, "")

        capability = self._check_capability(name, deps)
        if capability is not None:
            return capability

        required = _TOOL_DEPS.get(name)
        if required is None:
            return (True, "")

        if not required:
            return (True, "")

        missing = []
        for dep in required:
            info = deps.get(dep, {})
            if not info.get("available"):
                missing.append(dep)

        if missing:
            return (False, f"missing: {', '.join(missing)}")
        return (True, "")

    def _check_capability(self, name: str, deps: dict) -> tuple | None:
        if name.startswith("browser_"):
            info = deps.get("playwright", {})
            if not info.get("available"):
                return (False, "playwright not installed; run `pip install playwright` and `playwright install`")
            if name in {"browser_login", "browser_check_session"} and not self._tcp_ready("127.0.0.1", 6080):
                return (False, "noVNC is not reachable on 127.0.0.1:6080; start/configure the browser login service")
            return (True, "")

        if name.startswith("workflow_"):
            info = deps.get("playwright", {})
            if info.get("available"):
                return (True, "")
            return (False, "playwright not installed; browser workflows are unavailable")

        if name == "generate_image":
            url = f"http://{os.environ.get('COMFYUI_HOST', '127.0.0.1')}:{os.environ.get('COMFYUI_PORT', '8188')}"
            parsed = urlparse(url)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 8188
            if self._tcp_ready(host, port):
                return (True, "")
            if shutil.which("systemctl"):
                return (False, f"ComfyUI is not reachable at {url}; start it with `systemctl --user start maude-comfyui.service`")
            return (False, f"ComfyUI is not reachable at {url}; set COMFYUI_HOST/COMFYUI_PORT or start ComfyUI")

        if name == "generate_image_flux2":
            if os.environ.get("MAUDE_ENABLE_FLUX2", "").lower() not in {"1", "true", "yes", "on"}:
                return (False, "Flux 2 is disabled; set MAUDE_ENABLE_FLUX2=true to enable cloud Flux 2")
            if not os.environ.get("REPLICATE_API_TOKEN"):
                return (False, "REPLICATE_API_TOKEN is not set")
            return (True, "")

        if name in {"view_image", "web_view"}:
            if os.environ.get("OPEN_ROUTER_API_KEY") or os.environ.get("CLAUDE_API_KEY") or os.environ.get("MISTRAL_API_KEY"):
                return (True, "")
            vision_url = os.environ.get("VISION_SERVER_URL", "http://127.0.0.1:11434/v1")
            parsed = urlparse(vision_url)
            if self._tcp_ready(parsed.hostname or "127.0.0.1", parsed.port or 11434):
                return (True, "")
            return (False, "no vision provider available; set OPEN_ROUTER_API_KEY/CLAUDE_API_KEY/MISTRAL_API_KEY or start the local vision server")

        if name.startswith("github_"):
            if not shutil.which("gh"):
                return (False, "gh CLI not found; install from https://cli.github.com/")
            ok, detail = self._command_ok(["gh", "auth", "status"] )
            if not ok:
                return (False, f"gh CLI is not authenticated; run `gh auth login` ({detail})")
            return (True, "")

        if self._is_google_tool(name):
            missing = []
            for dep in ("google-api-python-client", "google-auth-oauthlib"):
                if not deps.get(dep, {}).get("available"):
                    missing.append(dep)
            if missing:
                return (False, f"missing: {', '.join(missing)}")
            if not self._google_credentials_present():
                return (False, "Google credentials not found; configure OAuth credentials under ~/.config/maude or run the Google setup flow")
            return (True, "")

        if name == "schedule_task":
            if not deps.get("croniter", {}).get("available"):
                return (False, "croniter not installed; install scheduler dependencies")
            return (True, "")

        if name.startswith("hyperframes_"):
            if not shutil.which("npx"):
                return (False, "npx not found; install Node.js 22+ and npm")
            return (True, "")

        if name.startswith("skill_"):
            try:
                importlib.import_module("skills")
                return (True, "")
            except ImportError:
                return (False, "skills framework not available")

        return None

    def _tcp_ready(self, host: str, port: int, timeout: float = 0.5) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def _command_ok(self, command: list[str]) -> tuple[bool, str]:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
        except FileNotFoundError:
            return False, "command not found"
        except Exception as exc:
            return False, str(exc)
        output = (result.stdout or result.stderr or "").splitlines()
        detail = output[0] if output else f"exit {result.returncode}"
        return result.returncode == 0, detail

    def _google_credentials_present(self) -> bool:
        config_dir = Path(os.environ.get("MAUDE_CONFIG_DIR", Path.home() / ".config" / "maude")).expanduser()
        candidates = [
            os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
            os.environ.get("MAUDE_GOOGLE_CREDENTIALS"),
            str(config_dir / "credentials.json"),
            str(config_dir / "google_credentials.json"),
            str(config_dir / "google_token.json"),
            str(config_dir / "token.json"),
        ]
        return any(value and Path(value).expanduser().exists() for value in candidates)

    def _is_google_tool(self, name: str) -> bool:
        prefixes = (
            "gmail_",
            "drive_",
            "sheets_",
            "calendar_",
            "slides_",
            "contacts_",
            "youtube_",
        )
        return name.startswith(prefixes)


# Module-level singleton for convenience
_checker = HealthChecker()
check_tool_ready = _checker.check_tool_ready
