"""
File and shell tool implementations — 9 tools.
"""

import os
import subprocess
from pathlib import Path

from maude_core.log import log
from maude_core.paths import resolve_path
from tool_registry import register_tool


def tool_read_file(path: str, start_line: int = None, end_line: int = None) -> str:
    """Read file contents with line numbers."""
    try:
        file_path = resolve_path(path)
        if not file_path.exists():
            return f"Error: File not found: {file_path}"
        if not file_path.is_file():
            return f"Error: Not a file: {file_path}"
        if file_path.stat().st_size > 1_000_000:
            return f"Error: File too large (>1MB): {file_path}"

        lines = file_path.read_text(errors="replace").splitlines()
        total_lines = len(lines)

        start_idx = (start_line - 1) if start_line else 0
        end_idx = end_line if end_line else min(total_lines, start_idx + 200)
        start_idx = max(0, start_idx)
        end_idx = min(total_lines, end_idx)

        selected = lines[start_idx:end_idx]
        numbered = [f"{start_idx + i + 1:4d} | {line}" for i, line in enumerate(selected)]

        log(f"Read lines {start_idx + 1}-{end_idx} of {total_lines} from {file_path}")
        return f"File: {file_path} ({total_lines} lines)\n\n" + "\n".join(numbered)
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


def tool_write_file(path: str, content: str) -> str:
    """Write content to file."""
    try:
        file_path = resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        log(f"Wrote {len(content)} chars to {file_path}")
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def tool_search_file(path: str, pattern: str) -> str:
    """Search for pattern in file."""
    try:
        file_path = resolve_path(path)
        if not file_path.exists():
            return f"Error: File not found: {file_path}"

        lines = file_path.read_text(errors="replace").splitlines()
        matches = []
        for i, line in enumerate(lines):
            if pattern.lower() in line.lower():
                matches.append(f"{i + 1:4d} | {line}")

        if not matches:
            return f"No matches for '{pattern}' in {file_path}"

        log(f"Found {len(matches)} matches for '{pattern}'")
        return f"Matches in {file_path}:\n\n" + "\n".join(matches[:50])
    except Exception as e:
        return f"Error searching file: {e}"


def tool_search_directory(directory: str, pattern: str) -> str:
    """Search for pattern across all files in directory."""
    EXCLUDE_DIRS = {".git", "node_modules", "__pycache__", "venv", ".venv", "build", "dist", ".cache"}
    CODE_EXTENSIONS = {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".sh",
        ".yaml",
        ".yml",
        ".json",
        ".md",
        ".txt",
        ".html",
        ".css",
    }

    try:
        dir_path = resolve_path(directory)
        if not dir_path.exists():
            return f"Error: Directory not found: {dir_path}"
        if not dir_path.is_dir():
            return f"Error: Not a directory: {dir_path}"

        pattern_lower = pattern.lower()
        matches = []
        files_searched = 0

        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for name in files:
                if not any(name.endswith(ext) for ext in CODE_EXTENSIONS):
                    continue

                filepath = Path(root) / name
                files_searched += 1

                try:
                    content = filepath.read_text(errors="replace")
                    for lineno, line in enumerate(content.splitlines(), start=1):
                        if pattern_lower in line.lower():
                            rel_path = filepath.relative_to(dir_path)
                            matches.append(f"{rel_path}:{lineno}: {line.strip()[:100]}")
                            if len(matches) >= 50:
                                break
                except:
                    continue

                if len(matches) >= 50:
                    break
            if len(matches) >= 50:
                break

        if not matches:
            return f"No matches for '{pattern}' in {dir_path} ({files_searched} files searched)"

        log(f"Found {len(matches)} matches in {files_searched} files")
        result = f"Matches for '{pattern}' in {dir_path}:\n\n" + "\n".join(matches)
        if len(matches) == 50:
            result += "\n\n(Limited to 50 results)"
        return result
    except Exception as e:
        return f"Error searching directory: {e}"


def tool_edit_file(path: str, start_line: int, end_line: int, new_content: str) -> str:
    """Edit specific lines in a file."""
    try:
        file_path = resolve_path(path)
        if not file_path.exists():
            return f"Error: File not found: {file_path}"

        lines = file_path.read_text(errors="replace").splitlines()
        total = len(lines)

        if start_line < 1 or end_line < start_line:
            return f"Error: Invalid line range {start_line}-{end_line}"
        if start_line > total:
            return f"Error: start_line {start_line} > file length ({total})"

        start_idx = start_line - 1
        end_idx = min(end_line, total)

        lines[start_idx:end_idx] = new_content.splitlines()
        file_path.write_text("\n".join(lines) + "\n")

        log(f"Edited lines {start_line}-{end_line} in {file_path}")
        return f"Edited lines {start_line}-{end_line} in {file_path}"
    except Exception as e:
        return f"Error editing file: {e}"


def tool_list_directory(path: str = None) -> str:
    """List directory contents."""
    from maude_core.paths import working_dir

    try:
        dir_path = resolve_path(path) if path else working_dir

        if not dir_path.exists():
            return f"Error: Directory not found: {dir_path}"
        if not dir_path.is_dir():
            return f"Error: Not a directory: {dir_path}"

        entries = []
        for item in sorted(dir_path.iterdir()):
            try:
                if item.is_dir():
                    entries.append(f"[DIR]  {item.name}/")
                else:
                    size = item.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size // 1024}KB"
                    else:
                        size_str = f"{size // (1024 * 1024)}MB"
                    entries.append(f"[FILE] {item.name} ({size_str})")
            except PermissionError:
                entries.append(f"[????] {item.name} (permission denied)")

        result = f"Directory: {dir_path}\n\n" + "\n".join(entries) if entries else f"Directory: {dir_path}\n\n(empty)"
        log(f"Listed {len(entries)} items in {dir_path}")
        return result
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error listing directory: {e}"


def tool_get_working_directory() -> str:
    """Get current working directory."""
    from maude_core.paths import working_dir

    return str(working_dir)


def tool_change_directory(path: str) -> str:
    """Change working directory."""
    from maude_core import paths

    try:
        new_path = resolve_path(path)
        if not new_path.exists():
            return f"Error: Directory not found: {new_path}"
        if not new_path.is_dir():
            return f"Error: Not a directory: {new_path}"
        paths.working_dir = new_path
        log(f"Changed directory to {paths.working_dir}")
        return f"Changed working directory to: {paths.working_dir}"
    except Exception as e:
        return f"Error changing directory: {e}"


def tool_run_command(command: str) -> str:
    """Execute a shell command."""
    from maude_core.paths import working_dir

    try:
        log(f"$ {command}")

        # Detect server/daemon commands that block forever — run them in background
        _SERVER_PATTERNS = [
            "npm run start",
            "npm start",
            "npx next start",
            "npx serve",
            "python -m http.server",
            "python3 -m http.server",
            "node server",
            "nohup ",
        ]
        cmd_stripped = command.strip().rstrip("&").strip()
        is_server_cmd = any(p in cmd_stripped for p in _SERVER_PATTERNS)

        if is_server_cmd:
            # Run as background process, wait briefly for startup, then return
            log("  (backgrounding server command)")
            proc = subprocess.Popen(
                cmd_stripped,
                shell=True,
                cwd=str(working_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            # Wait briefly for the process to start (or fail immediately)
            import time

            time.sleep(3)
            if proc.poll() is not None:
                # Process already exited — probably an error
                stdout = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
                stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                return f"Server failed to start:\n{stdout}\n{stderr}".strip()
            return f"Server started in background (PID {proc.pid}). Use curl to verify it's running."

        result = subprocess.run(command, shell=True, cwd=str(working_dir), capture_output=True, text=True, timeout=300)
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"
        if not output.strip():
            output = "(command completed successfully)"
        if len(output) > 10000:
            output = output[:10000] + "\n... (output truncated)"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 5 minutes"
    except Exception as e:
        return f"Error executing command: {e}"


# ── Registry wrappers ──────────────────────────────────────────


@register_tool("read_file")
def _dispatch_read_file(args):
    return tool_read_file(args.get("path", ""), args.get("start_line"), args.get("end_line"))


@register_tool("write_file")
def _dispatch_write_file(args):
    return tool_write_file(args.get("path", ""), args.get("content", ""))


@register_tool("search_file")
def _dispatch_search_file(args):
    return tool_search_file(args.get("path", ""), args.get("pattern", ""))


@register_tool("search_directory")
def _dispatch_search_directory(args):
    return tool_search_directory(args.get("directory", ""), args.get("pattern", ""))


@register_tool("edit_file")
def _dispatch_edit_file(args):
    return tool_edit_file(
        args.get("path", ""), args.get("start_line", 1), args.get("end_line", 1), args.get("new_content", "")
    )


@register_tool("list_directory")
def _dispatch_list_directory(args):
    return tool_list_directory(args.get("path"))


@register_tool("get_working_directory")
def _dispatch_get_working_directory(args):
    return tool_get_working_directory()


@register_tool("change_directory")
def _dispatch_change_directory(args):
    return tool_change_directory(args.get("path", ""))


@register_tool("run_command")
def _dispatch_run_command(args):
    return tool_run_command(args.get("command", ""))
