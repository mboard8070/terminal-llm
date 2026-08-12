"""
MAUDE Client Tool Router — fetches tool catalog from gateway,
routes execution between local and server.

Replaces duplicated tool definitions in client_tools.py with a
single source of truth from the gateway's /api/tools endpoint.
"""

import os
import re
import time
import json
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from maude_client.config import FILE_SERVER_URL
from maude_client import client_tools


# ── Client-only tool definitions ──────────────────────────────────
# These tools only exist on the client (file transfers, server interaction).
# They are NOT in the server's tool catalog, so we define them here
# and merge them with the server catalog at fetch time.

CLIENT_ONLY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "upload_to_server",
            "description": "Upload/push a local file to the Spark server. Use when user says 'upload', 'push', or 'send to server'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "local_path": {"type": "string", "description": "Local file path"},
                    "remote_path": {"type": "string", "description": "Destination path on server (relative to work dir)"}
                },
                "required": ["local_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "download_from_server",
            "description": "Download/pull a file from the Spark server. Use when user says 'pull', 'download', 'grab', or 'fetch from server'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "remote_path": {"type": "string", "description": "File path on server"},
                    "local_path": {"type": "string", "description": "Local destination (default: transfers folder)"}
                },
                "required": ["remote_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_server_files",
            "description": "List files in a directory on the Spark server. Use to find files before pulling/downloading.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path on server (default: work dir)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_server_command",
            "description": "Run a command on the Spark server via SSH.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute on server"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_to_server_maude",
            "description": "Send a message to the MAUDE instance running on the server.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message to send to server MAUDE"}
                },
                "required": ["message"]
            }
        }
    },
]

# Names of tools that must execute locally on the client
_LOCAL_TOOL_NAMES = {
    "read_file", "write_file", "edit_file", "list_directory",
    "search_files", "run_command",
    # File transfer tools — implemented locally
    "upload_to_server", "download_from_server", "list_server_files",
    "run_server_command", "send_to_server_maude",
    # Shared folder tools — implemented locally
    "list_shared", "sync_shared", "pull_shared", "clean_shared",
    "list_transfers", "clean_transfers",
}

# Minimal local-only catalog for offline mode
_OFFLINE_TOOLS = [t for t in CLIENT_ONLY_TOOLS] + [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a local file. Returns content with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "start_line": {"type": "integer", "description": "Starting line (1-indexed, optional)"},
                    "end_line": {"type": "integer", "description": "Ending line (optional)"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a local file. Creates directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace lines in a local file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "start_line": {"type": "integer", "description": "First line to replace (1-indexed)"},
                    "end_line": {"type": "integer", "description": "Last line to replace"},
                    "new_content": {"type": "string", "description": "Replacement content"}
                },
                "required": ["path", "start_line", "end_line", "new_content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List contents of a local directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: current directory)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for text in local files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text or regex to search for"},
                    "path": {"type": "string", "description": "Directory to search (default: current)"},
                    "file_pattern": {"type": "string", "description": "File glob pattern (e.g., '*.py')"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command locally.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_shared",
            "description": "List files in the server's shared folder.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sync_shared",
            "description": "Pull all files from the server's shared folder into the local cache (~/.maude/shared/).",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pull_shared",
            "description": "Download a specific file from the server's shared folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Name of the file in the server's shared folder"},
                    "local_path": {"type": "string", "description": "Local destination path (defaults to ~/.maude/shared/)"}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clean_shared",
            "description": "Delete files from the server's shared folder. Also clears the local cache copy if present.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Specific file to remove (omit to clean all files)"},
                    "confirm": {"type": "boolean", "description": "Must be true to proceed with deletion"}
                },
                "required": ["confirm"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_transfers",
            "description": "List files in the server's transfers folder (uploads from clients).",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clean_transfers",
            "description": "Delete files from the server's transfers folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Specific file to remove (omit to clean all files)"},
                    "confirm": {"type": "boolean", "description": "Must be true to proceed with deletion"}
                },
                "required": ["confirm"]
            }
        }
    },
]

# Client-only tool name set
_CLIENT_ONLY_NAMES = {t["function"]["name"] for t in CLIENT_ONLY_TOOLS}

# Fast dispatch patterns (client-local + shared intents).
# Prefer specific paths/commands first; multi-step messages are skipped in fast_dispatch().
_PATH_TOKEN_RE = re.compile(
    r"(?:~|/|\.{1,2}/)[^\s?\"']+|[A-Za-z0-9_.@+-]+\.[A-Za-z0-9]{1,8}\b"
)
_MULTI_STEP_RE = re.compile(
    r"\b(?:and then|then |after that|also |as well as|followed by)\b"
    r"|\b(?:fix|debug|implement|refactor|deploy|investigate)\b",
    re.I,
)
_SHELL_STATUS = [
    (re.compile(r"^(?:run\s+|please\s+)?git\s+status\s*$", re.I), "git status"),
    (re.compile(r"^(?:run\s+|please\s+)?git\s+status\s+--short\s*$", re.I), "git status --short"),
    (re.compile(r"^(?:run\s+|please\s+)?docker\s+ps\s*$", re.I), "docker ps"),
    (re.compile(r"^(?:run\s+|please\s+)?docker\s+ps\s+-a\s*$", re.I), "docker ps -a"),
    (re.compile(r"^(?:run\s+|please\s+)?pytest(?:\s+[\w./-]+)?\s*$", re.I), None),
    (re.compile(r"^(?:run\s+|please\s+)?npm\s+test\s*$", re.I), "npm test"),
    (re.compile(r"^(?:run\s+)?(?:the\s+)?tests?\s*$", re.I), "pytest"),
]


def _client_extract_path(msg: str) -> str | None:
    m = _PATH_TOKEN_RE.search(msg)
    if not m:
        return None
    return m.group(0).strip().rstrip(".,;:!?")


def _client_shell_cmd(msg: str) -> str | None:
    cleaned = msg.strip().rstrip("?.!")
    for pattern, fixed in _SHELL_STATUS:
        m = pattern.match(cleaned)
        if not m:
            continue
        if fixed is not None:
            return fixed
        return re.sub(r"^(?:run\s+|please\s+)", "", cleaned, flags=re.I).strip()
    return None


_FAST_PATTERNS = [
    # Shared folder
    (re.compile(r'\b(?:pull|grab|fetch)\s+(.+)', re.I),
     "pull_shared", lambda m, msg: {"filename": m.group(1).strip().strip('"').strip("'")}),
    (re.compile(r'\b(?:list|show|what.?s in)\b.*\b(?:shared)\b', re.I),
     "list_shared", lambda m, msg: {}),
    # Server files
    (re.compile(r'\b(?:list|show)\b.*\bserver\b.*\b(?:files|director)\b', re.I),
     "list_server_files", lambda m, msg: {"path": ""}),
    # Local filesystem — list with path
    (re.compile(
        r'\b(?:list|show|ls)\b\s+(?:the\s+)?(?:files?|contents?|dir(?:ectory)?|folder)\s+'
        r'(?:in|at|of|from)\s+([^\s?\"\']+)', re.I),
     "list_directory", lambda m, msg: {"path": m.group(1).strip().rstrip(".,;:?")}),
    (re.compile(r'^(?:ls|dir)\s+([~/.\w][^\s]*)\s*$', re.I),
     "list_directory", lambda m, msg: {"path": m.group(1).strip()}),
    (re.compile(r'^(?:ls|dir)\s*$', re.I),
     "list_directory", lambda m, msg: {"path": "."}),
    (re.compile(r'\b(?:list|show|what.?s in)\b.*\b(?:director|folder|files)\b', re.I),
     "list_directory",
     lambda m, msg: {"path": _client_extract_path(msg) or "."}),
    # Local filesystem — read
    (re.compile(r'\b(?:cat|type)\s+([~/.\w][^\s?\"\']+)', re.I),
     "read_file", lambda m, msg: {"path": m.group(1).strip().rstrip(".,;:?")}),
    (re.compile(
        r'\b(?:read|show|open|print|display)\b\s+(?:the\s+)?(?:file\s+|contents?\s+of\s+)?'
        r'([~/.\w][^\s?\"\']+\.[A-Za-z0-9]{1,8})\b', re.I),
     "read_file", lambda m, msg: {"path": m.group(1).strip().rstrip(".,;:?")}),
    # Shell status (whitelist only)
    (re.compile(r'^(?:run\s+|please\s+)?git\s+status(?:\s+--short)?\s*$', re.I),
     "run_command",
     lambda m, msg: {"command": "git status --short" if "--short" in msg.lower() else "git status"}),
    (re.compile(r'^(?:run\s+|please\s+)?docker\s+ps(?:\s+-a)?\s*$', re.I),
     "run_command",
     lambda m, msg: {"command": "docker ps -a" if "-a" in msg.lower() else "docker ps"}),
    (re.compile(r'^(?:run\s+|please\s+)?(?:pytest(?:\s+[\w./-]+)?|npm\s+test)\s*$', re.I),
     "run_command",
     lambda m, msg: {
         "command": re.sub(r"^(?:run\s+|please\s+)", "", msg.strip(), flags=re.I).rstrip("?.!")
     }),
    # Memory (remote tools — executed via gateway)
    (re.compile(r'\b(?:list|show)\b.*\bmemories\b', re.I),
     "list_memories", lambda m, msg: {"limit": 20}),
    (re.compile(r'\bwhat do you (?:know|remember) about\s+(.+)', re.I),
     "recall_memory", lambda m, msg: {"query": m.group(1).strip().rstrip("?.")}),
    (re.compile(r'\b(?:check|search|recall)\s+(?:your\s+)?memory\s+(?:for|about)\s+(.+)', re.I),
     "recall_memory", lambda m, msg: {"query": m.group(1).strip().rstrip("?.")}),
    # Image gen (remote)
    (re.compile(
        r'\b(?:generate|create|draw|make)\b\s+(?:me\s+)?(?:an?\s+)?'
        r'(?:image|picture|illustration|photo)\b\s*(?:of\s+|:\s*)(.+)', re.I),
     "generate_image", lambda m, msg: {"prompt": m.group(1).strip().rstrip("?.")}),
    # URL summarize (remote)
    (re.compile(
        r'\b(?:summarize|summary|tldr|tl;dr|browse|fetch)\b.*?(https?://\S+)', re.I),
     "web_browse", lambda m, msg: {"url": m.group(1).rstrip(").,;]\"'")}),
    # Skips that need more info
    (re.compile(r'\b(?:upload|send)\b.*\b(?:to server|to spark)\b', re.I),
     None, None),
    (re.compile(r'\b(?:download|get)\b.*\b(?:from server|from spark)\b', re.I),
     None, None),
]

CACHE_TTL = 300  # 5 minutes


class ToolRouter:
    """Routes tool execution between local client and remote gateway.

    Fetches the tool catalog from the gateway's /api/tools endpoint,
    caches it, and routes execute() calls to either local implementations
    or the gateway's /api/tools/execute endpoint.
    """

    def __init__(self, gateway_url: str = None):
        self._gateway_url = (gateway_url or FILE_SERVER_URL).rstrip("/")
        self._catalog = None
        self._catalog_ts = 0
        self._all_tools = list(_OFFLINE_TOOLS)  # start with offline set
        self._groups = {}
        self._core_tools = set()
        self._execution_targets = {}
        self._session_groups = set()
        self._domains = []
        self._active_domains: set[str] = set()  # sticky for this client process
        self._online = False

    def fetch_catalog(self) -> dict:
        """GET /api/tools from gateway, cache for 5 min. Falls back to offline catalog."""
        now = time.time()
        if self._catalog and (now - self._catalog_ts) < CACHE_TTL:
            return self._catalog

        try:
            r = requests.get(
                f"{self._gateway_url}/api/tools",
                timeout=5, verify=False
            )
            r.raise_for_status()
            catalog = r.json()

            # Merge server tools with client-only tools
            server_tools = catalog.get("tools", [])
            server_names = {t["function"]["name"] for t in server_tools}

            # Add client-only tools that aren't on the server
            merged_tools = list(server_tools)
            for ct in CLIENT_ONLY_TOOLS:
                if ct["function"]["name"] not in server_names:
                    merged_tools.append(ct)

            self._all_tools = merged_tools
            self._groups = catalog.get("groups", {})
            self._core_tools = set(catalog.get("core_tools", []))
            self._execution_targets = catalog.get("execution_targets", {})
            self._session_groups = set(catalog.get("session_groups", []))
            self._domains = catalog.get("domains", [])

            # Mark client-only tools as local
            for name in _CLIENT_ONLY_NAMES:
                self._execution_targets[name] = "local"
            # Also mark local file tools
            for name in _LOCAL_TOOL_NAMES:
                self._execution_targets[name] = "local"

            self._catalog = catalog
            self._catalog_ts = now
            self._online = True
            return catalog

        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            # Offline — use local-only tools
            self._all_tools = list(_OFFLINE_TOOLS)
            self._online = False
            self._core_tools = {t["function"]["name"] for t in _OFFLINE_TOOLS}
            self._session_groups = set()
            self._domains = []
            return {"tools": self._all_tools, "groups": {}, "core_tools": list(self._core_tools), "execution_targets": {}}

    def activate_domain(self, domain: str) -> list:
        """Sticky-activate a catalog domain for subsequent turns on this client."""
        domain = (domain or "").strip().lower().replace("-", "_")
        if not domain:
            return []
        if domain.startswith("domain_"):
            domain = domain[len("domain_"):]
        activated = []
        if domain in self._groups:
            self._active_domains.add(domain)
            activated.append(domain)
            act = self._groups[domain].get("activates")
            if act:
                self._active_domains.add(act)
                activated.append(act)
        else:
            for d in self._domains:
                if d.get("name") == domain:
                    self._active_domains.add(domain)
                    activated.append(domain)
                    break
        return activated

    def get_tools_for_message(self, message: str, messages: list = None) -> list:
        """Filter tools using cached catalog groups + keywords + session sticky domains."""
        # Ensure catalog is loaded
        if not self._catalog:
            self.fetch_catalog()

        if not self._groups:
            # No groups available (offline or no catalog) — return all tools
            return list(self._all_tools)

        msg_lower = (message or "").lower()
        active_names = set(self._core_tools)

        # Add client-only tools that should always be available
        active_names.update(_LOCAL_TOOL_NAMES)

        matched_groups = set()
        for gname, group in self._groups.items():
            for kw in group.get("keywords", []):
                if kw in msg_lower:
                    matched_groups.add(gname)
                    act = group.get("activates")
                    if act:
                        matched_groups.add(act)
                    break

        # Sticky: session-tier groups stay active after first match
        for g in matched_groups:
            group = self._groups.get(g, {})
            tier = group.get("tier", "session")
            if tier in ("session", "rare") or g in self._session_groups:
                self._active_domains.add(g)
                act = group.get("activates")
                if act:
                    self._active_domains.add(act)

        all_groups = matched_groups | set(self._active_domains)
        for g in all_groups:
            group = self._groups.get(g)
            if group:
                active_names.update(group.get("tools", []))

        # History-based reactivation from prior tool_calls
        if messages:
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        fn = tc.get("function") or {}
                        n = fn.get("name") if isinstance(fn, dict) else None
                        if n:
                            active_names.add(n)

        result = [t for t in self._all_tools if t["function"]["name"] in active_names]

        # Optional per-domain stub tools (off by default; server list_tool_domains
        # already carries names + one-liners for discovery).
        lazy_stubs = os.environ.get("MAUDE_LAZY_TOOL_STUBS", "0").strip().lower() in (
            "1", "true", "yes", "on",
        )
        if lazy_stubs:
            for d in self._domains:
                dname = d.get("name", "")
                dtools = set(d.get("tools") or [])
                if dname in all_groups or (dtools & active_names):
                    continue
                tools_preview = ", ".join(list(dtools)[:6])
                if len(dtools) > 6:
                    tools_preview += ", …"
                result.append({
                    "type": "function",
                    "function": {
                        "name": f"domain_{dname}",
                        "description": (
                            f"[lazy domain:{dname}] {d.get('description', dname)}. "
                            f"Tools: {tools_preview}. "
                            f"Call activate_tool_domain(domain=\"{dname}\") to load full schemas."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "activate": {
                                    "type": "boolean",
                                    "description": "Set true to activate this domain",
                                }
                            },
                            "required": [],
                        },
                    },
                })

        return result

    def execute(self, name: str, arguments: dict) -> str:
        """Route: local tools → client_tools, server tools → gateway API."""
        if name.startswith("domain_"):
            activated = self.activate_domain(name)
            if activated:
                return (
                    f"Activated domain(s): {', '.join(activated)}. "
                    "Full schemas available on the next step."
                )
            return f"Unknown domain stub '{name}'"
        if name == "activate_tool_domain":
            domain = (arguments or {}).get("domain", "")
            activated = self.activate_domain(domain)
            if not activated:
                # Still try remote (server session state)
                return self._execute_remote(name, arguments or {})
            # Also activate on server
            try:
                self._execute_remote(name, arguments or {})
            except Exception:
                pass
            return f"Activated domain(s): {', '.join(activated)}. Full schemas available next step."
        if name in _LOCAL_TOOL_NAMES:
            return self._execute_local(name, arguments)
        else:
            return self._execute_remote(name, arguments)

    def _execute_local(self, name: str, arguments: dict) -> str:
        """Dispatch to client_tools local implementations."""
        return client_tools.execute_tool(name, arguments)

    def _execute_remote(self, name: str, arguments: dict) -> str:
        """POST gateway /api/tools/execute, return result string."""
        try:
            r = requests.post(
                f"{self._gateway_url}/api/tools/execute",
                json={"name": name, "arguments": arguments},
                timeout=120,
                verify=False,
            )
            data = r.json()
            if data.get("error"):
                return f"Error: {data['error']}"
            return data.get("result", f"Error: No result from {name}")
        except requests.ConnectionError:
            return f"Error: Can't reach gateway for tool '{name}'. Is the server running?"
        except Exception as e:
            return f"Error executing {name}: {e}"

    # Messages mentioning other devices should skip fast dispatch
    _CROSS_MACHINE_RE = re.compile(
        r'\b(?:windows|mac|pc|laptop|macbook|mattwell|other machine|other device|on the)\b', re.I
    )

    def fast_dispatch(self, message: str):
        """Try to match message to a direct tool call.

        Returns (tool_name, args, result) or None.
        """
        msg = (message or "").strip()
        if not msg:
            return None
        # Skip fast dispatch if message targets another device
        if self._CROSS_MACHINE_RE.search(msg):
            return None
        # Don't hijack multi-step requests
        if len(msg) > 240 or _MULTI_STEP_RE.search(msg):
            return None

        # Whitelisted shell status (before generic patterns)
        shell_cmd = _client_shell_cmd(msg)
        if shell_cmd:
            try:
                result = self.execute("run_command", {"command": shell_cmd})
                if result and not result.startswith("Error:"):
                    return "run_command", {"command": shell_cmd}, result
            except Exception:
                pass

        for pattern, tool_name, arg_builder in _FAST_PATTERNS:
            if tool_name is None:
                continue
            match = pattern.search(msg)
            if match:
                try:
                    args = arg_builder(match, msg)
                    # Skip empty required args
                    if tool_name in ("read_file",) and not args.get("path"):
                        continue
                    if tool_name in ("recall_memory", "generate_image", "web_browse") and not any(
                        args.get(k) for k in ("query", "prompt", "url")
                    ):
                        continue
                    result = self.execute(tool_name, args)
                    if result and not result.startswith("Error:"):
                        return tool_name, args, result
                except Exception:
                    continue
        return None

    @property
    def is_online(self) -> bool:
        return self._online

    def health_warnings(self) -> list:
        """Return list of warning strings for startup display."""
        warnings = []
        if not self._online:
            warnings.append("Gateway unreachable — running with local tools only")
        if self._catalog:
            tools_info = self._catalog.get("tools", [])
            if len(tools_info) < 10:
                warnings.append(f"Only {len(tools_info)} tools available (expected 70+)")
        return warnings
