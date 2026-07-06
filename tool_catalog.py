from maude_bootstrap import ensure_local_maude

ensure_local_maude()

"""
MAUDE Tool Catalog — single source of truth for all tool definitions.

Wraps maude_core.py's TOOLS list with execution metadata so clients
can fetch the full catalog from the gateway instead of maintaining
duplicate definitions.
"""

from health import check_tool_ready
from maude_core import (
    _CORE_TOOL_NAMES,
    _TOOL_GROUPS,
    TOOLS,
    get_tools_for_message,
)

# ── Execution targets: where each tool should run ─────────────────

_LOCAL_TOOLS = {
    "read_file",
    "write_file",
    "edit_file",
    "list_directory",
    "search_file",
    "search_directory",
    "run_command",
    "get_working_directory",
    "change_directory",
}

# Client-only tools that only exist on the Mac client (not in server TOOLS)
_CLIENT_ONLY_TOOLS = {
    "upload_to_server",
    "download_from_server",
    "list_server_files",
    "run_server_command",
    "send_to_server_maude",
    "search_files",  # client variant of search_file/search_directory
}

# Build execution target map
_EXECUTION_TARGETS = {}
for _t in TOOLS:
    _name = _t["function"]["name"]
    _EXECUTION_TARGETS[_name] = "local" if _name in _LOCAL_TOOLS else "server"


def get_catalog() -> dict:
    """Full tool catalog with metadata.

    Returns:
        dict with keys:
          - tools: list of OpenAI function-call tool definitions
          - groups: keyword routing groups {name: {keywords, tools}}
          - core_tools: list of always-included tool names
          - execution_targets: {tool_name: "local"|"server"}
    """
    return {
        "tools": TOOLS,
        "groups": {
            name: {
                "keywords": list(group["keywords"]),
                "tools": list(group["tools"]),
            }
            for name, group in _TOOL_GROUPS.items()
        },
        "core_tools": list(_CORE_TOOL_NAMES),
        "execution_targets": dict(_EXECUTION_TARGETS),
    }


def get_filtered_tools(message: str) -> list:
    """Keyword-filtered tools for a specific message.

    Wraps maude_core.get_tools_for_message() so clients can use
    the same filtering logic via the API.
    """
    return get_tools_for_message(message)


def execute_server_tool(name: str, arguments: dict) -> dict:
    """Execute a server-side tool and return the result.

    Rejects local-only tools with an error.

    Returns:
        dict with keys: result (str), elapsed (float), error (str|None)
    """
    import time

    from maude_core import execute_tool

    # Reject tools that should run on the client
    target = _EXECUTION_TARGETS.get(name, "server")
    if target == "local" or name in _CLIENT_ONLY_TOOLS:
        return {
            "result": None,
            "elapsed": 0,
            "error": f"Tool '{name}' is a local tool and must be executed on the client.",
        }

    # Pre-flight readiness check
    ready, reason = check_tool_ready(name)
    if not ready:
        return {
            "result": None,
            "elapsed": 0,
            "error": f"Tool '{name}' unavailable: {reason}",
        }

    start = time.time()
    try:
        result = execute_tool(name, arguments)
        elapsed = round(time.time() - start, 2)
        return {"result": result, "elapsed": elapsed, "error": None}
    except Exception as e:
        elapsed = round(time.time() - start, 2)
        return {"result": None, "elapsed": elapsed, "error": str(e)}
