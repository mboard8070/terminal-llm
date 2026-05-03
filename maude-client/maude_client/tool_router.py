"""
MAUDE Client Tool Router — fetches tool catalog from gateway,
routes execution between local and server.

Replaces duplicated tool definitions in client_tools.py with a
single source of truth from the gateway's /api/tools endpoint.
"""

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

# Fast dispatch patterns (moved from client_tools.py)
_FAST_PATTERNS = [
    (re.compile(r'\b(?:pull|grab|fetch)\s+(.+)', re.I),
     "pull_shared", lambda m, msg: {"filename": m.group(1).strip().strip('"').strip("'")}),
    (re.compile(r'\b(?:list|show|what.?s in)\b.*\b(?:shared)\b', re.I),
     "list_shared", lambda m, msg: {}),
    (re.compile(r'\b(?:list|show|what.?s in)\b.*\b(?:director|folder|files)\b', re.I),
     "list_directory", lambda m, msg: {"path": "."}),
    (re.compile(r'\b(?:list|show)\b.*\bserver\b.*\b(?:files|director)\b', re.I),
     "list_server_files", lambda m, msg: {"path": ""}),
    (re.compile(r'\b(?:upload|send)\b.*\b(?:to server|to spark)\b', re.I),
     None, None),  # Skip — needs path from user
    (re.compile(r'\b(?:download|get)\b.*\b(?:from server|from spark)\b', re.I),
     None, None),  # Skip — needs path from user
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
            return {"tools": self._all_tools, "groups": {}, "core_tools": list(self._core_tools), "execution_targets": {}}

    def get_tools_for_message(self, message: str) -> list:
        """Filter tools using cached catalog groups + keywords."""
        # Ensure catalog is loaded
        if not self._catalog:
            self.fetch_catalog()

        if not self._groups:
            # No groups available (offline or no catalog) — return all tools
            return list(self._all_tools)

        msg_lower = message.lower()
        active_names = set(self._core_tools)

        # Add client-only tools that should always be available
        active_names.update(_LOCAL_TOOL_NAMES)

        for group in self._groups.values():
            for kw in group.get("keywords", []):
                if kw in msg_lower:
                    active_names.update(group.get("tools", []))
                    break

        return [t for t in self._all_tools if t["function"]["name"] in active_names]

    def execute(self, name: str, arguments: dict) -> str:
        """Route: local tools → client_tools, server tools → gateway API."""
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
        msg = message.strip()
        # Skip fast dispatch if message targets another device
        if self._CROSS_MACHINE_RE.search(msg):
            return None
        for pattern, tool_name, arg_builder in _FAST_PATTERNS:
            if tool_name is None:
                continue
            match = pattern.search(msg)
            if match:
                try:
                    args = arg_builder(match, msg)
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
