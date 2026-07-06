"""Domain-owned tool schemas."""

TOOL_NAMES = {
    "change_directory",
    "edit_file",
    "get_transfer",
    "get_working_directory",
    "list_directory",
    "list_shared",
    "list_transfers",
    "read_file",
    "remove_shared",
    "run_command",
    "search_directory",
    "search_file",
    "share_file",
    "write_file",
}

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file with line numbers. Use start_line/end_line for large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "start_line": {"type": "integer", "description": "First line to read (1-indexed, optional)"},
                    "end_line": {"type": "integer", "description": "Last line to read (1-indexed, optional)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "Content to write to the file"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories on the DGX Spark SERVER (Linux). Cannot access the user's "
            "Mac/iPhone. Shows file sizes and types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path on the server to list (defaults to working directory)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_working_directory",
            "description": "Get the current working directory path.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "change_directory",
            "description": "Change the current working directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path to change to"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command. Use for: pip, python, git, rm, mv, cp, etc. Do NOT use rm to "
            "remove shared folder files — use the remove_shared tool instead so deletions sync to "
            "clients.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The shell command to execute"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_file",
            "description": "Search for text/pattern in a single file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "pattern": {"type": "string", "description": "Text to search for"},
                },
                "required": ["path", "pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_directory",
            "description": "Search for text across all files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory to search"},
                    "pattern": {"type": "string", "description": "Text to search for"},
                },
                "required": ["directory", "pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit specific lines in a file. Read the file first to see line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "start_line": {"type": "integer", "description": "First line to replace (1-indexed)"},
                    "end_line": {"type": "integer", "description": "Last line to replace (1-indexed)"},
                    "new_content": {"type": "string", "description": "New content to insert"},
                },
                "required": ["path", "start_line", "end_line", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_shared",
            "description": "List files in the shared folder. Files placed here are synced to connected clients "
            "automatically. To remove files, use the remove_shared tool (NOT rm — rm won't propagate "
            "the deletion to clients).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "share_file",
            "description": "Copy a file into the shared folder so the client can pull/download it. Use this when "
            "the user says 'send this to the client' or 'share this file'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to share"},
                    "filename": {
                        "type": "string",
                        "description": "Optional name for the file in shared folder (defaults to original name)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_shared",
            "description": "Remove a file from the shared folder. ALWAYS use this instead of rm for shared folder "
            "files — it records the deletion so client sync removes the file locally too, preventing "
            "it from being re-synced back.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Name of the file to remove from the shared folder"}
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_transfers",
            "description": "List files uploaded by the client (in the transfers folder). Use when user asks 'what "
            "did the client send' or 'check uploads'.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_transfer",
            "description": "Copy a file from the transfers folder (client uploads) to the working directory. Use "
            "when user says 'pull that file' or 'grab the upload'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Name of the file in transfers folder"},
                    "destination": {
                        "type": "string",
                        "description": "Where to copy it (defaults to working directory)",
                    },
                },
                "required": ["filename"],
            },
        },
    },
]


def schemas() -> list[dict]:
    return list(SCHEMAS)
