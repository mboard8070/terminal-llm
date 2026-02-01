"""
MAUDE Client Tools - Local file operations and server communication.
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Optional
from config import SERVER_SSH_HOST, SERVER_WORK_DIR, LOCAL_TRANSFER_DIR

# Ensure transfer directory exists
TRANSFER_DIR = Path(LOCAL_TRANSFER_DIR).expanduser()
TRANSFER_DIR.mkdir(parents=True, exist_ok=True)

# Tool definitions for the LLM
TOOLS = [
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
            "name": "upload_to_server",
            "description": "Upload a local file to the Spark server.",
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
            "description": "Download a file from the Spark server.",
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
            "description": "List files in a directory on the Spark server.",
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
    }
]


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool and return the result."""
    try:
        if name == "read_file":
            return read_file(
                arguments["path"],
                arguments.get("start_line"),
                arguments.get("end_line")
            )
        elif name == "write_file":
            return write_file(arguments["path"], arguments["content"])
        elif name == "edit_file":
            return edit_file(
                arguments["path"],
                arguments["start_line"],
                arguments["end_line"],
                arguments["new_content"]
            )
        elif name == "list_directory":
            return list_directory(arguments.get("path", "."))
        elif name == "search_files":
            return search_files(
                arguments["pattern"],
                arguments.get("path", "."),
                arguments.get("file_pattern")
            )
        elif name == "run_command":
            return run_command(arguments["command"])
        elif name == "upload_to_server":
            return upload_to_server(
                arguments["local_path"],
                arguments.get("remote_path")
            )
        elif name == "download_from_server":
            return download_from_server(
                arguments["remote_path"],
                arguments.get("local_path")
            )
        elif name == "list_server_files":
            return list_server_files(arguments.get("path"))
        elif name == "run_server_command":
            return run_server_command(arguments["command"])
        elif name == "send_to_server_maude":
            return send_to_server_maude(arguments["message"])
        else:
            return f"Unknown tool: {name}"
    except Exception as e:
        return f"Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Local File Operations
# ─────────────────────────────────────────────────────────────────────────────

def read_file(path: str, start_line: int = None, end_line: int = None) -> str:
    """Read a local file with optional line range."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"Error: File not found: {path}"

    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        # Apply line range
        if start_line is not None:
            start_idx = max(0, start_line - 1)
            end_idx = end_line if end_line else len(lines)
            lines = lines[start_idx:end_idx]
            line_offset = start_idx
        else:
            line_offset = 0

        # Format with line numbers
        numbered = []
        for i, line in enumerate(lines):
            line_num = i + line_offset + 1
            numbered.append(f"{line_num:4d} | {line.rstrip()}")

        return "\n".join(numbered)
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a local file."""
    path = os.path.expanduser(path)
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def edit_file(path: str, start_line: int, end_line: int, new_content: str) -> str:
    """Replace lines in a file."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"Error: File not found: {path}"

    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Replace lines
        new_lines = new_content.split('\n')
        if not new_content.endswith('\n'):
            new_lines = [line + '\n' for line in new_lines]
        else:
            new_lines = [line + '\n' for line in new_lines[:-1]] + ['']

        lines[start_line-1:end_line] = new_lines

        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        return f"Successfully edited lines {start_line}-{end_line} in {path}"
    except Exception as e:
        return f"Error editing file: {e}"


def list_directory(path: str = ".") -> str:
    """List directory contents."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"Error: Directory not found: {path}"

    try:
        entries = []
        for entry in sorted(os.listdir(path)):
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                entries.append(f"[DIR]  {entry}/")
            else:
                size = os.path.getsize(full_path)
                entries.append(f"[FILE] {entry} ({size} bytes)")

        return f"Contents of {path}:\n" + "\n".join(entries)
    except Exception as e:
        return f"Error listing directory: {e}"


def search_files(pattern: str, path: str = ".", file_pattern: str = None) -> str:
    """Search for text in files."""
    path = os.path.expanduser(path)

    cmd = ["grep", "-rn", pattern, path]
    if file_pattern:
        cmd = ["grep", "-rn", "--include", file_pattern, pattern, path]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        if not output:
            return f"No matches found for '{pattern}'"

        # Limit output
        lines = output.split('\n')
        if len(lines) > 50:
            output = '\n'.join(lines[:50]) + f"\n... and {len(lines)-50} more matches"

        return output
    except subprocess.TimeoutExpired:
        return "Error: Search timed out"
    except Exception as e:
        return f"Error searching: {e}"


def run_command(command: str) -> str:
    """Run a local shell command."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=os.getcwd()
        )

        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"

        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds"
    except Exception as e:
        return f"Error running command: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Server Communication
# ─────────────────────────────────────────────────────────────────────────────

def upload_to_server(local_path: str, remote_path: str = None) -> str:
    """Upload a file to the Spark server."""
    local_path = os.path.expanduser(local_path)
    if not os.path.exists(local_path):
        return f"Error: Local file not found: {local_path}"

    filename = os.path.basename(local_path)
    if remote_path:
        dest = f"{SERVER_SSH_HOST}:{SERVER_WORK_DIR}/{remote_path}"
    else:
        dest = f"{SERVER_SSH_HOST}:{SERVER_WORK_DIR}/transfers/{filename}"

    try:
        # Ensure remote transfers directory exists
        subprocess.run(
            ["ssh", SERVER_SSH_HOST, f"mkdir -p {SERVER_WORK_DIR}/transfers"],
            capture_output=True,
            timeout=10
        )

        result = subprocess.run(
            ["scp", local_path, dest],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            return f"Uploaded {local_path} to server: {dest}"
        else:
            return f"Error uploading: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Upload timed out"
    except Exception as e:
        return f"Error: {e}"


def download_from_server(remote_path: str, local_path: str = None) -> str:
    """Download a file from the Spark server."""
    filename = os.path.basename(remote_path)

    if local_path:
        local_path = os.path.expanduser(local_path)
    else:
        local_path = str(TRANSFER_DIR / filename)

    # Handle relative paths on server
    if not remote_path.startswith('/') and not remote_path.startswith('~'):
        remote_path = f"{SERVER_WORK_DIR}/{remote_path}"

    source = f"{SERVER_SSH_HOST}:{remote_path}"

    try:
        result = subprocess.run(
            ["scp", source, local_path],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            return f"Downloaded to {local_path}"
        else:
            return f"Error downloading: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Download timed out"
    except Exception as e:
        return f"Error: {e}"


def list_server_files(path: str = None) -> str:
    """List files on the server."""
    if path:
        if not path.startswith('/') and not path.startswith('~'):
            path = f"{SERVER_WORK_DIR}/{path}"
    else:
        path = SERVER_WORK_DIR

    try:
        result = subprocess.run(
            ["ssh", SERVER_SSH_HOST, f"ls -la {path}"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            return f"Server files at {path}:\n{result.stdout}"
        else:
            return f"Error: {result.stderr}"
    except Exception as e:
        return f"Error: {e}"


def run_server_command(command: str) -> str:
    """Run a command on the server via SSH."""
    try:
        result = subprocess.run(
            ["ssh", SERVER_SSH_HOST, f"cd {SERVER_WORK_DIR} && {command}"],
            capture_output=True,
            text=True,
            timeout=60
        )

        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"

        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out"
    except Exception as e:
        return f"Error: {e}"


def send_to_server_maude(message: str) -> str:
    """Send a message to the server MAUDE instance via tmux."""
    try:
        # Sanitize message - keep only safe characters
        safe_msg = ''.join(c for c in message if c.isalnum() or c in ' .,!?-_:@')

        if not safe_msg.strip():
            return "Error: Message was empty after sanitization"

        # Send to MAUDE tmux session on server
        result = subprocess.run(
            ["ssh", SERVER_SSH_HOST, f'tmux send-keys -t maude "{safe_msg}" Enter'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            return f"Message sent to server MAUDE: {safe_msg[:100]}..."
        else:
            return f"Error sending to server MAUDE: {result.stderr}"
    except Exception as e:
        return f"Error: {e}"
