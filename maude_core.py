"""
MAUDE Core - Shared tools and functionality.

This module provides the core tool implementations used by both
the TUI (chat_local.py) and Telegram (run_telegram.py) interfaces.
"""

import os
import subprocess
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any
from openai import OpenAI

# Configuration - can be overridden via environment variables
LOCAL_URL = os.environ.get("LLM_SERVER_URL", "http://localhost:30000/v1")
MODEL = os.environ.get("MAUDE_MODEL", "nemotron")
NUM_CTX = int(os.environ.get("MAUDE_NUM_CTX", "32768"))
VISION_URL = os.environ.get("VISION_SERVER_URL", "http://localhost:11434/v1")
VISION_MODEL = os.environ.get("MAUDE_VISION_MODEL", "llava:7b")

# Shared session ID for conversation sync
SESSION_ID = os.environ.get("MAUDE_SESSION_ID", "default")

# Working directory for file operations
working_dir = Path.home()

# Cache for web_view results
_web_view_cache = {}

# Optional logging callback - set by the interface (TUI or Telegram)
# Signature: log_callback(message: str) -> None
_log_callback: Optional[Callable[[str], None]] = None

# Memory instance (lazy loaded)
_memory = None


def get_memory():
    """Get or create the shared memory instance."""
    global _memory
    if _memory is None:
        try:
            from memory import MaudeMemory
            _memory = MaudeMemory()
        except Exception as e:
            log(f"Memory init failed: {e}")
            return None
    return _memory


def save_message(role: str, content: str, channel: str = "cli"):
    """Save a message to shared conversation history."""
    mem = get_memory()
    if mem and content:
        try:
            mem.save_message(SESSION_ID, role, content, channel)
        except Exception as e:
            log(f"Failed to save message: {e}")


def get_conversation_history(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent conversation history from all channels."""
    mem = get_memory()
    if mem:
        try:
            return mem.get_conversation(SESSION_ID, limit)
        except Exception as e:
            log(f"Failed to get history: {e}")
    return []


def build_messages_with_history(system_prompt: str, user_message: str, history_limit: int = 10) -> List[Dict[str, str]]:
    """Build messages list with system prompt, recent history, and current message."""
    messages = [{"role": "system", "content": system_prompt}]

    # Add recent history
    history = get_conversation_history(history_limit)
    for msg in history:
        # Only include user and assistant messages (skip system, tool)
        if msg["role"] in ("user", "assistant") and msg["content"]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current message
    messages.append({"role": "user", "content": user_message})

    return messages


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight Chat Sync (file-based, non-blocking)
# ─────────────────────────────────────────────────────────────────────────────

CHAT_LOG_PATH = Path.home() / ".config" / "maude" / "chat_sync.jsonl"


def append_chat_log(channel: str, role: str, content: str):
    """Append a message to the shared chat log. Fast, non-blocking."""
    import json
    from datetime import datetime
    try:
        CHAT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now().isoformat(),
            "channel": channel,
            "role": role,
            "content": content
        }
        with open(CHAT_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # Non-critical, don't break anything


def read_chat_log_since(last_position: int = 0) -> tuple:
    """Read new entries from chat log since position. Returns (entries, new_position)."""
    import json
    entries = []
    new_position = last_position
    try:
        if not CHAT_LOG_PATH.exists():
            return [], 0
        with open(CHAT_LOG_PATH, "r") as f:
            f.seek(last_position)
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except:
                        pass
            new_position = f.tell()
    except Exception:
        pass
    return entries, new_position


def set_log_callback(callback: Callable[[str], None]):
    """Set the logging callback for tool status messages."""
    global _log_callback
    _log_callback = callback


def log(message: str):
    """Log a status message if callback is set."""
    if _log_callback:
        _log_callback(message)


def set_working_directory(path: Path):
    """Set the working directory for file operations."""
    global working_dir
    working_dir = path


def get_working_directory() -> Path:
    """Get the current working directory."""
    return working_dir


def resolve_path(path_str: str) -> Path:
    """Resolve a path relative to working directory."""
    global working_dir
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = working_dir / path
    return path.resolve()


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definitions
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [
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
                    "end_line": {"type": "integer", "description": "Last line to read (1-indexed, optional)"}
                },
                "required": ["path"]
            }
        }
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
                    "content": {"type": "string", "description": "Content to write to the file"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories in a path. Shows file sizes and types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to list (defaults to working directory)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_working_directory",
            "description": "Get the current working directory path.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "change_directory",
            "description": "Change the current working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to change to"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command. Use for: pip, python, git, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_browse",
            "description": "Fetch and read content from a web URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "description": "Number of results (default 5, max 10)"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_view",
            "description": "Screenshot a webpage and analyze it visually using LLaVA.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to screenshot"},
                    "question": {"type": "string", "description": "Optional question about the page"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "view_image",
            "description": "Analyze a local image file using LLaVA vision model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the image file"},
                    "question": {"type": "string", "description": "Optional question about the image"}
                },
                "required": ["path"]
            }
        }
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
                    "pattern": {"type": "string", "description": "Text to search for"}
                },
                "required": ["path", "pattern"]
            }
        }
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
                    "pattern": {"type": "string", "description": "Text to search for"}
                },
                "required": ["directory", "pattern"]
            }
        }
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
                    "new_content": {"type": "string", "description": "New content to insert"}
                },
                "required": ["path", "start_line", "end_line", "new_content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_frontier",
            "description": "Escalate to a frontier AI model (Claude, GPT, Gemini) for complex questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question requiring expert analysis"},
                    "context": {"type": "string", "description": "Relevant context"},
                    "provider": {"type": "string", "description": "Optional: claude, openai, gemini, grok, mistral"}
                },
                "required": ["question"]
            }
        }
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# Tool Implementations
# ─────────────────────────────────────────────────────────────────────────────

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

        lines = file_path.read_text(errors='replace').splitlines()
        total_lines = len(lines)

        start_idx = (start_line - 1) if start_line else 0
        end_idx = end_line if end_line else min(total_lines, start_idx + 200)
        start_idx = max(0, start_idx)
        end_idx = min(total_lines, end_idx)

        selected = lines[start_idx:end_idx]
        numbered = [f"{start_idx + i + 1:4d} | {line}" for i, line in enumerate(selected)]

        log(f"Read lines {start_idx+1}-{end_idx} of {total_lines} from {file_path}")
        return f"File: {file_path} ({total_lines} lines)\n\n" + '\n'.join(numbered)
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

        lines = file_path.read_text(errors='replace').splitlines()
        matches = []
        for i, line in enumerate(lines):
            if pattern.lower() in line.lower():
                matches.append(f"{i+1:4d} | {line}")

        if not matches:
            return f"No matches for '{pattern}' in {file_path}"

        log(f"Found {len(matches)} matches for '{pattern}'")
        return f"Matches in {file_path}:\n\n" + '\n'.join(matches[:50])
    except Exception as e:
        return f"Error searching file: {e}"


def tool_search_directory(directory: str, pattern: str) -> str:
    """Search for pattern across all files in directory."""
    EXCLUDE_DIRS = {'.git', 'node_modules', '__pycache__', 'venv', '.venv', 'build', 'dist', '.cache'}
    CODE_EXTENSIONS = {'.py', '.js', '.ts', '.jsx', '.tsx', '.sh', '.yaml', '.yml', '.json', '.md', '.txt', '.html', '.css'}

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
                    content = filepath.read_text(errors='replace')
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
        result = f"Matches for '{pattern}' in {dir_path}:\n\n" + '\n'.join(matches)
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

        lines = file_path.read_text(errors='replace').splitlines()
        total = len(lines)

        if start_line < 1 or end_line < start_line:
            return f"Error: Invalid line range {start_line}-{end_line}"
        if start_line > total:
            return f"Error: start_line {start_line} > file length ({total})"

        start_idx = start_line - 1
        end_idx = min(end_line, total)

        lines[start_idx:end_idx] = new_content.splitlines()
        file_path.write_text('\n'.join(lines) + '\n')

        log(f"Edited lines {start_line}-{end_line} in {file_path}")
        return f"Edited lines {start_line}-{end_line} in {file_path}"
    except Exception as e:
        return f"Error editing file: {e}"


def tool_list_directory(path: str = None) -> str:
    """List directory contents."""
    global working_dir
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
    global working_dir
    return str(working_dir)


def tool_change_directory(path: str) -> str:
    """Change working directory."""
    global working_dir
    try:
        new_path = resolve_path(path)
        if not new_path.exists():
            return f"Error: Directory not found: {new_path}"
        if not new_path.is_dir():
            return f"Error: Not a directory: {new_path}"
        working_dir = new_path
        log(f"Changed directory to {working_dir}")
        return f"Changed working directory to: {working_dir}"
    except Exception as e:
        return f"Error changing directory: {e}"


def tool_run_command(command: str) -> str:
    """Execute a shell command."""
    global working_dir
    try:
        log(f"$ {command}")
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            timeout=300
        )
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


def tool_web_browse(url: str) -> str:
    """Fetch and parse web page content."""
    import requests
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return "Error: beautifulsoup4 not installed"

    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        log(f"Fetching {url}")

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'header', 'noscript', 'iframe']):
            tag.decompose()

        main_content = soup.find('main') or soup.find('article') or soup.find('div', {'class': ['content', 'post', 'article', 'main']})
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)

        if len(text) > 15000:
            text = text[:15000] + "\n\n... (content truncated)"

        log(f"Retrieved {len(text)} chars from {url}")
        return f"Content from {url}:\n\n{text}"

    except requests.exceptions.Timeout:
        return f"Error: Request timed out for {url}"
    except requests.exceptions.RequestException as e:
        return f"Error fetching {url}: {e}"
    except Exception as e:
        return f"Error parsing {url}: {e}"


def tool_web_search(query: str, num_results: int = 5) -> str:
    """Search the web using DuckDuckGo."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "Error: ddgs not installed"

    try:
        num_results = min(max(1, num_results), 10)
        log(f"Searching: {query}")

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))

        if not results:
            return f"No results found for: {query}"

        output = f"Search results for: {query}\n\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. {r.get('title', 'No title')}\n"
            output += f"   URL: {r.get('href', 'No URL')}\n"
            output += f"   {r.get('body', 'No description')}\n\n"

        log(f"Found {len(results)} results")
        return output

    except Exception as e:
        return f"Error searching: {e}"


def tool_web_view(url: str, question: str = None) -> str:
    """Screenshot a webpage and analyze it with LLaVA."""
    global _web_view_cache
    import base64

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "Error: playwright not installed"

    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        if url in _web_view_cache:
            log(f"Using cached screenshot of {url}")
            return _web_view_cache[url] + "\n\n[NOTE: Using cached analysis.]"

        log(f"Capturing screenshot of {url}")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1024, 'height': 768})
            page.goto(url, wait_until='networkidle', timeout=30000)
            page.wait_for_timeout(1000)
            screenshot_bytes = page.screenshot(full_page=False)
            browser.close()

        base64_image = base64.b64encode(screenshot_bytes).decode('utf-8')
        log(f"Screenshot captured ({len(screenshot_bytes) // 1024}KB)")

        if question:
            prompt = f"This is a screenshot of {url}. {question}"
        else:
            prompt = f"This is a screenshot of {url}. Describe what you see."

        log("Analyzing with LLaVA...")

        vision_client = OpenAI(base_url=VISION_URL, api_key="not-needed")
        response = vision_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }],
            max_tokens=1024,
            temperature=0.2
        )

        analysis = response.choices[0].message.content
        log("Vision analysis complete")

        result = f"Visual analysis of {url}:\n\n{analysis}"
        _web_view_cache[url] = result
        return result

    except Exception as e:
        error_msg = str(e)
        if "playwright" in error_msg.lower():
            return f"Error: Playwright issue. Run: playwright install chromium\n{e}"
        elif "connect" in error_msg.lower():
            return f"Error: Cannot connect to vision model at {VISION_URL}\n{e}"
        return f"Error viewing webpage: {e}"


def tool_view_image(path: str, question: str = None) -> str:
    """Analyze a local image file with LLaVA."""
    import base64

    try:
        file_path = resolve_path(path)
        if not file_path.exists():
            return f"Error: Image not found: {file_path}"
        if not file_path.is_file():
            return f"Error: Not a file: {file_path}"

        ext = file_path.suffix.lower()
        if ext not in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
            return f"Error: Unsupported format: {ext}"

        if file_path.stat().st_size > 20_000_000:
            return f"Error: Image too large (>20MB)"

        mime_types = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                      '.gif': 'image/gif', '.webp': 'image/webp'}
        mime_type = mime_types.get(ext, 'image/png')

        log(f"Reading image {file_path}")

        with open(file_path, 'rb') as f:
            image_bytes = f.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        log(f"Image loaded ({len(image_bytes) // 1024}KB)")

        if question:
            prompt = f"This is an image from {file_path.name}. {question}"
        else:
            prompt = f"This is an image from {file_path.name}. Describe what you see."

        log("Analyzing with LLaVA...")

        vision_client = OpenAI(base_url=VISION_URL, api_key="not-needed")
        response = vision_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                ]
            }],
            max_tokens=1024,
            temperature=0.2
        )

        analysis = response.choices[0].message.content
        log("Vision analysis complete")

        return f"Analysis of {file_path.name}:\n\n{analysis}"

    except Exception as e:
        if "connect" in str(e).lower():
            return f"Error: Cannot connect to vision model at {VISION_URL}"
        return f"Error analyzing image: {e}"


def tool_ask_frontier(question: str, context: str = None, provider: str = None) -> str:
    """Escalate a question to a frontier cloud model."""
    try:
        from frontier import ask_frontier, list_available_providers

        available = list_available_providers()
        if not available:
            return "Error: No frontier providers configured. Set API keys in .env"

        provider_name = provider if provider in available else None
        log("Escalating to frontier model...")

        response = ask_frontier(
            query=question,
            context=context,
            provider_name=provider_name,
            system_prompt="You are an expert assistant. Be thorough but concise."
        )

        log(f"{response.provider}: {response.input_tokens}+{response.output_tokens} tokens, ${response.cost_usd:.4f}")

        return f"[Expert response from {response.provider}]\n\n{response.content}"

    except Exception as e:
        return f"Error calling frontier model: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool Execution
# ─────────────────────────────────────────────────────────────────────────────

# Rate limiting counters (reset per turn by caller)
vision_call_count = 0
web_call_count = 0


def reset_rate_limits():
    """Reset per-turn rate limits. Call at start of each user turn."""
    global vision_call_count, web_call_count
    vision_call_count = 0
    web_call_count = 0


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool and return the result."""
    global vision_call_count, web_call_count

    # Rate limiting
    if name in ("web_view", "view_image"):
        if vision_call_count > 0:
            return "(Vision analysis already completed - see previous result.)"
        vision_call_count += 1

    if name in ("web_browse", "web_search"):
        if web_call_count >= 4:
            return "(Web research limit reached. Use gathered information now.)"
        web_call_count += 1

    # Dispatch to tool
    if name == "read_file":
        return tool_read_file(arguments.get("path", ""), arguments.get("start_line"), arguments.get("end_line"))
    elif name == "write_file":
        return tool_write_file(arguments.get("path", ""), arguments.get("content", ""))
    elif name == "search_file":
        return tool_search_file(arguments.get("path", ""), arguments.get("pattern", ""))
    elif name == "search_directory":
        return tool_search_directory(arguments.get("directory", ""), arguments.get("pattern", ""))
    elif name == "edit_file":
        return tool_edit_file(arguments.get("path", ""), arguments.get("start_line", 1), arguments.get("end_line", 1), arguments.get("new_content", ""))
    elif name == "list_directory":
        return tool_list_directory(arguments.get("path"))
    elif name == "get_working_directory":
        return tool_get_working_directory()
    elif name == "change_directory":
        return tool_change_directory(arguments.get("path", ""))
    elif name == "run_command":
        return tool_run_command(arguments.get("command", ""))
    elif name == "web_browse":
        return tool_web_browse(arguments.get("url", ""))
    elif name == "web_search":
        return tool_web_search(arguments.get("query", ""), arguments.get("num_results", 5))
    elif name == "web_view":
        return tool_web_view(arguments.get("url", ""), arguments.get("question"))
    elif name == "view_image":
        return tool_view_image(arguments.get("path", ""), arguments.get("question"))
    elif name == "ask_frontier":
        return tool_ask_frontier(arguments.get("question", ""), arguments.get("context"), arguments.get("provider"))
    else:
        return f"Error: Unknown tool: {name}"
