"""Domain-owned tool schemas — Claude Code fleet orchestration."""

TOOL_NAMES = {
    "claude_register_worker",
    "ask_claude_code",
    "claude_check_reply",
    "claude_broadcast",
    "claude_list_workers",
    "claude_fleet_status",
    "claude_reset_worker",
    "claude_remove_worker",
}

_TARGET_PROPS = {
    "target": {
        "type": "string",
        "description": "Device hostname (or client_id/platform) to run Claude on. Call mesh_status/claude_fleet_status to see options.",
    },
    "target_client_id": {
        "type": "string",
        "description": "Exact client_id of the target device (from mesh_status). Preferred over target for precision.",
    },
    "target_platform": {
        "type": "string",
        "description": "Target any online device of this platform: 'windows', 'macos', or 'linux'.",
    },
}

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "claude_register_worker",
            "description": "Define a named Claude Code worker: a resumable Claude session pinned to a machine and project directory. Register one per parallel Claude instance you want in the fleet, then talk to it with ask_claude_code. Re-registering resets the session unless keep_session=true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short unique name for this worker, e.g. 'lighting', 'verse', 'ui'.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Absolute path to the project directory on the target machine where Claude should run (e.g. the UEFN/game project folder).",
                    },
                    **_TARGET_PROPS,
                    "permission_mode": {
                        "type": "string",
                        "enum": ["acceptEdits", "bypassPermissions", "plan", "default"],
                        "description": "How Claude handles tool permissions when running headless. 'acceptEdits' auto-accepts edits (default); 'bypassPermissions' runs fully autonomously (use with care); 'plan' only plans.",
                    },
                    "model": {"type": "string", "description": "Optional Claude model override for this worker."},
                    "keep_session": {
                        "type": "boolean",
                        "description": "If true, keep the existing conversation thread when re-registering instead of starting fresh.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_claude_code",
            "description": "Send a prompt to a registered Claude Code worker and wait for its reply. Continues the worker's conversation thread (resumable), so you can hold a multi-turn dialogue and have it work on the project. Claude runs detached on the target machine (no time limit); if the wait times out, Claude KEEPS WORKING — collect the reply later with claude_check_reply. Returns Claude's response text plus cost/duration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "worker": {"type": "string", "description": "Name of the registered worker to talk to."},
                    "prompt": {"type": "string", "description": "The instruction or question for Claude Code."},
                    "timeout": {
                        "type": "integer",
                        "description": "Seconds to wait for a reply before returning 'still working' (default 300, max 3600). Claude keeps running past this; use claude_check_reply to collect.",
                    },
                },
                "required": ["worker", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "claude_check_reply",
            "description": "Collect the reply from a worker whose ask_claude_code timed out or is still running in the background. Returns the reply if Claude finished, or how long it has been working. Safe to call repeatedly.",
            "parameters": {
                "type": "object",
                "properties": {"worker": {"type": "string", "description": "Name of the worker to check."}},
                "required": ["worker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "claude_broadcast",
            "description": "Send the same prompt to several Claude Code workers at once (the fleet). Dispatches them concurrently and gathers all replies. Use to fan out a task across parallel Claude instances working the same game.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The instruction to send to every targeted worker."},
                    "workers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of workers to target. Omit to broadcast to ALL registered workers.",
                    },
                    "timeout": {"type": "integer", "description": "Seconds to wait per worker (default 300)."},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "claude_list_workers",
            "description": "List all registered Claude Code workers and their target machine, project dir, permission mode, and whether their session has started.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "claude_fleet_status",
            "description": "Show which mesh clients are online and, for each Claude worker, whether it is OFFLINE, WORKING (has a running task), or idle/ready. Use to see how many instances are open and free to take work.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "claude_reset_worker",
            "description": "Start a fresh Claude conversation for a worker (new session id). The next ask_claude_code begins a clean thread.",
            "parameters": {
                "type": "object",
                "properties": {"worker": {"type": "string", "description": "Name of the worker to reset."}},
                "required": ["worker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "claude_remove_worker",
            "description": "Delete a registered Claude worker from the fleet.",
            "parameters": {
                "type": "object",
                "properties": {"worker": {"type": "string", "description": "Name of the worker to remove."}},
                "required": ["worker"],
            },
        },
    },
]


def schemas() -> list[dict]:
    return list(SCHEMAS)
