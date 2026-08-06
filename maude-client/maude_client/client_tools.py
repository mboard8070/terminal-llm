"""
MAUDE Client Tools - Local file operations and server communication.

Tool definitions are now fetched from the gateway via tool_router.py.
This module only contains tool *implementations* that run locally on the client.
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Optional
from maude_client.config import SERVER_SSH_HOST, SERVER_WORK_DIR, LOCAL_TRANSFER_DIR, LOCAL_SHARED_DIR, SERVER_SHARED_DIR, FILE_SERVER_URL
from maude_client.process_utils import run_process, shell_command
import requests as _requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Ensure transfer directory exists
TRANSFER_DIR = Path(LOCAL_TRANSFER_DIR).expanduser()
TRANSFER_DIR.mkdir(parents=True, exist_ok=True)

# Ensure shared directory exists
SHARED_DIR = Path(LOCAL_SHARED_DIR).expanduser()
SHARED_DIR.mkdir(parents=True, exist_ok=True)


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a local tool and return the result.

    Only dispatches tools that run on the client (file ops, server comm,
    shared folder). Server-side tools (web, collab, google, etc.) are
    handled by the ToolRouter via the gateway API.
    """
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
        elif name == "list_shared":
            return list_shared()
        elif name == "sync_shared":
            return sync_shared()
        elif name == "pull_shared":
            return pull_shared(
                arguments.get("filename", ""),
                arguments.get("local_path")
            )
        elif name == "clean_shared":
            return clean_shared(
                arguments.get("filename"),
                arguments.get("confirm", False)
            )
        elif name == "list_transfers":
            return list_transfers()
        elif name == "clean_transfers":
            return clean_transfers(
                arguments.get("filename"),
                arguments.get("confirm", False)
            )
        else:
            return f"Unknown local tool: {name}"
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

        total_lines = len(lines)

        # Apply line range
        if start_line is not None:
            start_idx = max(0, start_line - 1)
            end_idx = end_line if end_line else len(lines)
            lines = lines[start_idx:end_idx]
            line_offset = start_idx
        else:
            line_offset = 0

        # Cap unbounded full-file reads so the client doesn't stall dumping huge files.
        max_lines = 200
        truncated = False
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            truncated = True

        # Format with line numbers
        numbered = []
        for i, line in enumerate(lines):
            line_num = i + line_offset + 1
            numbered.append(f"{line_num:4d} | {line.rstrip()}")

        output = "\n".join(numbered)
        if truncated:
            shown_end = line_offset + max_lines
            output += (
                f"\n... truncated after {max_lines} lines "
                f"(file has {total_lines} lines; pass start_line/end_line for a slice)"
            )
            if start_line is None:
                output += f"\nHint: read_file(path, start_line=1, end_line={max_lines})"
            else:
                output += f"\nShown lines {line_offset + 1}-{shown_end}"
        return output
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
        names = sorted(os.listdir(path))
        max_entries = 100
        entries = []
        for entry in names[:max_entries]:
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                entries.append(f"[DIR]  {entry}/")
            else:
                size = os.path.getsize(full_path)
                entries.append(f"[FILE] {entry} ({size} bytes)")

        header = f"Contents of {path} ({len(names)} items)"
        body = "\n".join(entries)
        if len(names) > max_entries:
            body += f"\n... and {len(names) - max_entries} more"
        return f"{header}:\n{body}"
    except Exception as e:
        return f"Error listing directory: {e}"


def search_files(pattern: str, path: str = ".", file_pattern: str = None) -> str:
    """Search for text in files."""
    path = os.path.expanduser(path)

    # Skip heavy/irrelevant trees so local searches don't hang the client UI.
    exclude_dirs = [
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        "dist", "build", ".tox", ".mypy_cache", ".ruff_cache",
        "Library", "Movies", "Music", "Pictures",
    ]
    cmd = ["grep", "-rn", "--binary-files=without-match"]
    for d in exclude_dirs:
        cmd.extend(["--exclude-dir", d])
    if file_pattern:
        cmd.extend(["--include", file_pattern])
    cmd.extend([pattern, path])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        output = result.stdout.strip()
        if not output:
            return f"No matches found for '{pattern}'"

        # Keep model context useful but bounded.
        lines = output.split('\n')
        max_matches = 30
        if len(lines) > max_matches:
            output = '\n'.join(lines[:max_matches]) + f"\n... and {len(lines)-max_matches} more matches"

        return output
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 15s (narrow path/pattern and retry)"
    except Exception as e:
        return f"Error searching: {e}"


def run_command(command: str) -> str:
    """Run a local shell command with process-tree cleanup on timeout."""
    try:
        result = run_process(shell_command(command), timeout=60, cwd=os.getcwd())

        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        if result.timed_out:
            output += "\n[timeout: command process tree terminated after 60 seconds]"
        elif result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"

        text = output.strip() or "(no output)"
        # Bound huge command dumps so the chat loop stays responsive.
        lines = text.splitlines()
        max_lines = 80
        if len(lines) > max_lines:
            text = "\n".join(lines[:max_lines]) + f"\n... and {len(lines) - max_lines} more lines"
        if len(text) > 8000:
            text = text[:8000] + "\n... (truncated)"
        return text
    except Exception as e:
        return f"Error running command: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Server Communication
# ─────────────────────────────────────────────────────────────────────────────

def upload_to_server(local_path: str, remote_path: str = None) -> str:
    """Upload a file to the Spark server via HTTP file server."""
    local_path = os.path.expanduser(local_path)
    if not os.path.exists(local_path):
        return f"Error: Local file not found: {local_path}"

    filename = os.path.basename(local_path)

    try:
        with open(local_path, "rb") as f:
            data = f.read()
        r = _requests.post(f"{FILE_SERVER_URL}/upload/{filename}", data=data, timeout=120, verify=False)
        if r.status_code == 200:
            return f"Uploaded '{filename}' to server transfers folder ({len(data):,} bytes)"
        else:
            return f"Error uploading: {r.json().get('error', r.text)}"
    except _requests.ConnectionError:
        return "Error: Can't reach file server. Make sure Tailscale is connected and the server is running."
    except Exception as e:
        return f"Error: {e}"


def download_from_server(remote_path: str, local_path: str = None) -> str:
    """Download a file from the server's shared folder via HTTP."""
    filename = os.path.basename(remote_path)
    return pull_shared(filename, local_path)


def list_server_files(path: str = None) -> str:
    """List files on the server's shared folder."""
    return list_shared()


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


# ─────────────────────────────────────────────────────────────────────────────
# Shared Folder Operations
# ─────────────────────────────────────────────────────────────────────────────

def list_shared() -> str:
    """List files in the server's shared folder via HTTP file server."""
    try:
        r = _requests.get(f"{FILE_SERVER_URL}/list", timeout=5, verify=False)
        data = r.json()
        if "error" in data:
            return f"Error: {data['error']}"
        files = data.get("files", [])
        if not files:
            return "Shared folder on server is empty."
        entries = []
        for f in files:
            if f["is_dir"]:
                entries.append(f"  [DIR]  {f['name']}/")
            else:
                size = f["size"]
                entries.append(f"  [FILE] {f['name']} ({size:,} bytes)")
        return "Server shared folder:\n" + "\n".join(entries)
    except _requests.ConnectionError:
        return "Error: Can't reach file server. Make sure Tailscale is connected and the server is running."
    except Exception as e:
        return f"Error: {e}"


def pull_shared(filename: str, local_path: str = None) -> str:
    """Pull a file from the server's shared folder via HTTP."""
    dest_dir = Path(LOCAL_SHARED_DIR).expanduser()
    dest_dir.mkdir(parents=True, exist_ok=True)

    if local_path:
        dest = Path(os.path.expanduser(local_path))
    else:
        dest = dest_dir / filename

    try:
        r = _requests.get(f"{FILE_SERVER_URL}/download/{filename}", timeout=120, stream=True, verify=False)
        if r.status_code == 404:
            return f"Error: '{filename}' not found on server. Use list_shared to see available files."
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        size = dest.stat().st_size
        return f"Pulled '{filename}' to {dest} ({size:,} bytes)"
    except _requests.ConnectionError:
        return "Error: Can't reach file server. Make sure Tailscale is connected and the server is running."
    except Exception as e:
        return f"Error: {e}"


def sync_shared() -> str:
    """Pull all files from server shared folder."""
    try:
        r = _requests.get(f"{FILE_SERVER_URL}/list", timeout=5, verify=False)
        data = r.json()
        files = data.get("files", [])
        if not files:
            return "Nothing to sync — server shared folder is empty."
        pulled = []
        for f in files:
            if not f["is_dir"]:
                result = pull_shared(f["name"])
                pulled.append(result)
        return "Sync complete:\n" + "\n".join(pulled)
    except _requests.ConnectionError:
        return "Error: Can't reach file server. Make sure Tailscale is connected and the server is running."
    except Exception as e:
        return f"Error: {e}"


def clean_shared(filename: str = None, confirm: bool = False) -> str:
    """Delete files from the server's shared folder and clear the local cache.

    The server is the source of truth — the file is deleted there, and the
    local copy in ~/.maude/shared/ (a download cache) is removed so it
    doesn't linger.
    """
    return _clean_remote(
        endpoint_prefix="/delete",
        list_endpoint="/list",
        local_dir=Path(LOCAL_SHARED_DIR).expanduser(),
        filename=filename,
        confirm=confirm,
        label="shared",
    )


def list_transfers() -> str:
    """List files in the server's transfers folder."""
    try:
        r = _requests.get(f"{FILE_SERVER_URL}/transfers", timeout=5, verify=False)
        data = r.json()
        if "error" in data:
            return f"Error: {data['error']}"
        files = data.get("files", [])
        if not files:
            return "Server transfers folder is empty."
        entries = []
        for f in files:
            if f["is_dir"]:
                entries.append(f"  [DIR]  {f['name']}/")
            else:
                entries.append(f"  [FILE] {f['name']} ({f['size']:,} bytes)")
        return "Server transfers folder:\n" + "\n".join(entries)
    except _requests.ConnectionError:
        return "Error: Can't reach file server. Make sure Tailscale is connected and the server is running."
    except Exception as e:
        return f"Error: {e}"


def clean_transfers(filename: str = None, confirm: bool = False) -> str:
    """Delete files from the server's transfers folder."""
    return _clean_remote(
        endpoint_prefix="/delete-transfer",
        list_endpoint="/transfers",
        local_dir=None,
        filename=filename,
        confirm=confirm,
        label="transfers",
    )


def _clean_remote(
    *,
    endpoint_prefix: str,
    list_endpoint: str,
    local_dir: Optional[Path],
    filename: Optional[str],
    confirm: bool,
    label: str,
) -> str:
    """Shared implementation for clean_shared / clean_transfers."""
    if not confirm:
        return "Error: Set confirm=true to proceed with deletion."

    if filename:
        targets = [filename]
    else:
        try:
            r = _requests.get(f"{FILE_SERVER_URL}{list_endpoint}", timeout=10, verify=False)
            if r.status_code != 200:
                return f"Could not list server {label}: HTTP {r.status_code}"
            targets = sorted(
                e["name"]
                for e in r.json().get("files", [])
                if not e.get("is_dir") and not e["name"].startswith(".")
            )
        except _requests.ConnectionError:
            return "Error: Can't reach file server."
        except Exception as e:
            return f"Error listing server {label}: {e}"

    if not targets:
        return f"Nothing to delete — server {label} folder is empty."

    server_ok = 0
    failures = []
    local_ok = 0

    for name in targets:
        try:
            r = _requests.post(
                f"{FILE_SERVER_URL}{endpoint_prefix}/{name}", timeout=15, verify=False
            )
            if r.status_code == 200:
                server_ok += 1
            else:
                failures.append(f"{name}: HTTP {r.status_code} {r.text[:120]}")
        except _requests.ConnectionError:
            failures.append(f"{name}: server unreachable")
            continue
        except Exception as e:
            failures.append(f"{name}: {e}")
            continue

        if local_dir is not None:
            local_file = local_dir / name
            if local_file.exists():
                try:
                    local_file.unlink()
                    local_ok += 1
                except Exception as e:
                    failures.append(f"{name} (local cache): {e}")

    parts = [f"Deleted {server_ok}/{len(targets)} from server {label}"]
    if local_dir is not None:
        parts.append(f"cleared {local_ok} local cache file(s)")
    summary = ", ".join(parts)
    if failures:
        return summary + "\nFailures:\n  " + "\n  ".join(failures)
    return summary
