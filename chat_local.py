#!/usr/bin/env python3
"""
MAUDE - Terminal LLM Chat powered by Nemotron-3-Nano-30B
"""

import sys
import os
from dotenv import load_dotenv
load_dotenv()

import time
import json
import threading
from pathlib import Path
from openai import OpenAI

# Textual TUI framework
from textual.app import App, ComposeResult
from textual.widgets import Input, RichLog, Static, Footer
from textual.containers import Container, Vertical
from textual.binding import Binding
from textual import work
from rich.text import Text
from rich.panel import Panel
import pyfiglet

# Minimal imports
from keys import KeyManager

# Global reference to app for output
_app = None

class TUIConsole:
    """Console that writes to Textual RichLog when app is running, otherwise stdout."""
    def __init__(self):
        self.width = 80

    def print(self, *args, end="\n", **kwargs):
        text = " ".join(str(a) for a in args)
        if _app and hasattr(_app, 'output_log'):
            try:
                _app.call_from_thread(_app.write_output, text)
            except:
                print(text, end=end)
        else:
            print(text, end=end)

    def input(self, prompt=""):
        return input(prompt)

    def clear(self):
        if not _app:
            os.system('clear' if os.name != 'nt' else 'cls')

console = TUIConsole()

# Working directory for file operations
working_dir = Path.home()

# Main model - Nemotron via llama.cpp for coding
# Override with LLM_SERVER_URL env var for remote servers (e.g., via Tailscale)
LOCAL_URL = os.environ.get("LLM_SERVER_URL", "http://localhost:30000/v1")
MODEL = "nemotron"
# Context window size - increase for longer conversations
NUM_CTX = int(os.environ.get("MAUDE_NUM_CTX", "32768"))

# Vision model - LLaVA via Ollama (only used for web_view/view_image)
# Override with VISION_SERVER_URL env var for remote servers
VISION_URL = os.environ.get("VISION_SERVER_URL", "http://localhost:11434/v1")
VISION_MODEL = "llava:7b"

# Toggle for showing reasoning
show_thinking = False

# Cache for web_view results (URL -> analysis) to avoid repeated screenshots
_web_view_cache = {}

# Track expensive calls per turn (reset in chat loop)
_vision_call_count = 0
_web_call_count = 0  # web_browse + web_search

# Color palette for animation - fire gradient
COLORS = ["red", "bright_red", "orange1", "orange3", "yellow", "bright_yellow"]

# File system tools definition
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file with line numbers. Use start_line/end_line for large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read (1-indexed, optional)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read (1-indexed, optional)"
                    }
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
                    "path": {
                        "type": "string",
                        "description": "Path to the file (relative to working directory or absolute)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
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
                    "path": {
                        "type": "string",
                        "description": "Path to list (relative to working directory or absolute). Defaults to current working directory."
                    }
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
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
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
                    "path": {
                        "type": "string",
                        "description": "Path to change to (relative or absolute)"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command. Use for: pip install, python scripts, git, venv setup, build tools, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_browse",
            "description": "Fetch and read content from a web URL. Returns the page text/markdown. Use this to look up documentation, articles, API references, or any web content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL to fetch (must start with http:// or https://)"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo. Returns a list of results with titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 10)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_view",
            "description": "Take a screenshot of a webpage and analyze it visually using LLaVA vision model. Use this when you need to see the actual layout, images, or visual content of a page that text scraping would miss.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL to screenshot (must start with http:// or https://)"
                    },
                    "question": {
                        "type": "string",
                        "description": "Optional specific question about what to look for on the page (e.g., 'What products are shown?' or 'Describe the navigation menu')"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "view_image",
            "description": "Analyze a local image file (screenshot, photo, diagram, etc.) using LLaVA vision model. Supports PNG, JPG, JPEG, GIF, WEBP formats.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the image file (relative to working directory or absolute)"
                    },
                    "question": {
                        "type": "string",
                        "description": "Optional specific question about the image (e.g., 'What error is shown?' or 'Describe the UI layout')"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_file",
            "description": "Search for text/pattern in a single file. Returns matching lines with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to search"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Text or pattern to search for (case-insensitive)"
                    }
                },
                "required": ["path", "pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_directory",
            "description": "Search for text/pattern across all files in a directory. Use this to find code when you don't know which file it's in. Returns file:line matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory to search in"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Text or pattern to search for (case-insensitive)"
                    }
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
                    "path": {
                        "type": "string",
                        "description": "Path to the file"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to replace (1-indexed)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to replace (1-indexed)"
                    },
                    "new_content": {
                        "type": "string",
                        "description": "New content to insert"
                    }
                },
                "required": ["path", "start_line", "end_line", "new_content"]
            }
        }
    }
]


def resolve_path(path_str: str) -> Path:
    """Resolve a path relative to working directory."""
    global working_dir
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = working_dir / path
    return path.resolve()


def tool_read_file(path: str, start_line: int = None, end_line: int = None) -> str:
    """Read file contents with line numbers."""
    try:
        file_path = resolve_path(path)
        if not file_path.exists():
            return f"Error: File not found: {file_path}"
        if not file_path.is_file():
            return f"Error: Not a file: {file_path}"
        if file_path.stat().st_size > 1_000_000:  # 1MB limit
            return f"Error: File too large (>1MB): {file_path}"

        lines = file_path.read_text(errors='replace').splitlines()
        total_lines = len(lines)

        # Handle line range
        start_idx = (start_line - 1) if start_line else 0
        end_idx = end_line if end_line else min(total_lines, start_idx + 200)
        start_idx = max(0, start_idx)
        end_idx = min(total_lines, end_idx)

        selected = lines[start_idx:end_idx]
        numbered = [f"{start_idx + i + 1:4d} | {line}" for i, line in enumerate(selected)]

        console.print(f"[dim cyan]  -> Read lines {start_idx+1}-{end_idx} of {total_lines} from {file_path}[/dim cyan]")
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
        console.print(f"[dim cyan]  -> Wrote {len(content)} chars to {file_path}[/dim cyan]")
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def tool_search_file(path: str, pattern: str) -> str:
    """Search for pattern in file, return matching lines with numbers."""
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

        console.print(f"[dim cyan]  -> Found {len(matches)} matches for '{pattern}'[/dim cyan]")
        return f"Matches in {file_path}:\n\n" + '\n'.join(matches[:50])
    except Exception as e:
        return f"Error searching file: {e}"


def tool_search_directory(directory: str, pattern: str) -> str:
    """Search for pattern across all files in directory."""
    import re

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
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for name in files:
                # Only search code files
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

        console.print(f"[dim cyan]  -> Found {len(matches)} matches in {files_searched} files[/dim cyan]")
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

        console.print(f"[dim cyan]  -> Edited lines {start_line}-{end_line} in {file_path}[/dim cyan]")
        return f"Edited lines {start_line}-{end_line} in {file_path}"
    except Exception as e:
        return f"Error editing file: {e}"


def tool_list_directory(path: str = None) -> str:
    """List directory contents."""
    global working_dir
    try:
        if path:
            dir_path = resolve_path(path)
        else:
            dir_path = working_dir

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
        console.print(f"[dim cyan]  -> Listed {len(entries)} items in {dir_path}[/dim cyan]")
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
        console.print(f"[dim cyan]  -> Changed directory to {working_dir}[/dim cyan]")
        return f"Changed working directory to: {working_dir}"
    except Exception as e:
        return f"Error changing directory: {e}"


def tool_run_command(command: str) -> str:
    """Execute a shell command."""
    global working_dir
    import subprocess
    try:
        console.print(f"[dim cyan]  $ {command}[/dim cyan]")
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
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
        # Truncate very long output
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
        return "Error: beautifulsoup4 not installed. Run: pip install beautifulsoup4"

    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        console.print(f"[dim cyan]  -> Fetching {url}[/dim cyan]")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script, style, nav, footer, aside elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'header', 'noscript', 'iframe']):
            tag.decompose()

        # Try to find main content
        main_content = soup.find('main') or soup.find('article') or soup.find('div', {'class': ['content', 'post', 'article', 'main']})
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)

        # Truncate if too long
        if len(text) > 15000:
            text = text[:15000] + "\n\n... (content truncated)"

        console.print(f"[dim cyan]  -> Retrieved {len(text)} chars from {url}[/dim cyan]")
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
        return "Error: ddgs not installed. Run: pip install ddgs"

    try:
        num_results = min(max(1, num_results), 10)
        console.print(f"[dim cyan]  -> Searching: {query}[/dim cyan]")

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))

        if not results:
            return f"No results found for: {query}"

        output = f"Search results for: {query}\n\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. {r.get('title', 'No title')}\n"
            output += f"   URL: {r.get('href', 'No URL')}\n"
            output += f"   {r.get('body', 'No description')}\n\n"

        console.print(f"[dim cyan]  -> Found {len(results)} results[/dim cyan]")
        return output

    except Exception as e:
        return f"Error searching: {e}"


def tool_web_view(url: str, question: str = None) -> str:
    """Screenshot a webpage and analyze it with LLaVA vision model."""
    global _web_view_cache
    import base64
    import tempfile

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "Error: playwright not installed. Run: pip install playwright && playwright install chromium"

    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # Check cache - return previous analysis if we already screenshotted this URL
        if url in _web_view_cache:
            console.print(f"[dim cyan]  -> Using cached screenshot of {url}[/dim cyan]")
            return _web_view_cache[url] + "\n\n[NOTE: You already have this visual analysis. Use the information above to answer the user's question now. Do not call web_view again.]"

        console.print(f"[dim cyan]  -> Capturing screenshot of {url}[/dim cyan]")

        # Take screenshot with Playwright (smaller viewport for faster processing)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1024, 'height': 768})
            page.goto(url, wait_until='networkidle', timeout=30000)

            # Wait a moment for any lazy-loaded content
            page.wait_for_timeout(1000)

            # Take screenshot
            screenshot_bytes = page.screenshot(full_page=False)  # Viewport only for reasonable size
            browser.close()

        # Convert to base64
        base64_image = base64.b64encode(screenshot_bytes).decode('utf-8')
        console.print(f"[dim cyan]  -> Screenshot captured ({len(screenshot_bytes) // 1024}KB)[/dim cyan]")

        # Prepare prompt for LLaVA
        if question:
            prompt = f"This is a screenshot of the webpage {url}. {question}"
        else:
            prompt = f"This is a screenshot of the webpage {url}. Describe what you see on this page, including the main content, layout, any images, navigation, and key information displayed."

        # Send to LLaVA for visual analysis
        console.print(f"[dim cyan]  -> Analyzing with LLaVA...[/dim cyan]")

        vision_client = OpenAI(
            base_url=VISION_URL,
            api_key="not-needed"
        )

        response = vision_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1024,
            temperature=0.2
        )

        analysis = response.choices[0].message.content
        console.print(f"[dim cyan]  -> Vision analysis complete[/dim cyan]")

        result = f"Visual analysis of {url}:\n\n{analysis}"
        _web_view_cache[url] = result  # Cache for future calls
        return result

    except Exception as e:
        error_msg = str(e)
        if "playwright" in error_msg.lower() or "browser" in error_msg.lower():
            return f"Error: Playwright browser issue. Try running: playwright install chromium\n{e}"
        elif "connect" in error_msg.lower() or "refused" in error_msg.lower():
            return f"Error: Cannot connect to vision model at {VISION_URL}. Is Ollama running?\n{e}"
        else:
            return f"Error viewing webpage: {e}"


def tool_view_image(path: str, question: str = None) -> str:
    """Analyze a local image file with LLaVA vision model."""
    import base64

    try:
        file_path = resolve_path(path)
        if not file_path.exists():
            return f"Error: Image not found: {file_path}"
        if not file_path.is_file():
            return f"Error: Not a file: {file_path}"

        # Check file extension
        ext = file_path.suffix.lower()
        if ext not in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
            return f"Error: Unsupported image format: {ext}. Use PNG, JPG, JPEG, GIF, or WEBP."

        # Check file size (limit to 20MB)
        if file_path.stat().st_size > 20_000_000:
            return f"Error: Image too large (>20MB): {file_path}"

        # Determine MIME type
        mime_types = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                      '.gif': 'image/gif', '.webp': 'image/webp'}
        mime_type = mime_types.get(ext, 'image/png')

        console.print(f"[dim cyan]  -> Reading image {file_path}[/dim cyan]")

        # Read and encode image
        with open(file_path, 'rb') as f:
            image_bytes = f.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        console.print(f"[dim cyan]  -> Image loaded ({len(image_bytes) // 1024}KB)[/dim cyan]")

        # Prepare prompt
        if question:
            prompt = f"This is an image from {file_path.name}. {question}"
        else:
            prompt = f"This is an image from {file_path.name}. Describe what you see in detail, including any text, UI elements, diagrams, or notable features."

        # Send to LLaVA
        console.print(f"[dim cyan]  -> Analyzing with LLaVA...[/dim cyan]")

        vision_client = OpenAI(
            base_url=VISION_URL,
            api_key="not-needed"
        )

        response = vision_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1024,
            temperature=0.2
        )

        analysis = response.choices[0].message.content
        console.print(f"[dim cyan]  -> Vision analysis complete[/dim cyan]")

        return f"Analysis of {file_path.name}:\n\n{analysis}"

    except Exception as e:
        error_msg = str(e)
        if "connect" in error_msg.lower() or "refused" in error_msg.lower():
            return f"Error: Cannot connect to vision model at {VISION_URL}. Is Ollama running?\n{e}"
        return f"Error analyzing image: {e}"


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool and return the result."""
    global _vision_call_count, _web_call_count

    # Limit vision tools to 1 call per turn
    if name in ("web_view", "view_image"):
        if _vision_call_count > 0:
            return "(Vision analysis already completed - see previous result above. Please use that information to respond to the user.)"
        _vision_call_count += 1

    # Limit web tools to 4 calls per turn (search + browse combined)
    if name in ("web_browse", "web_search"):
        if _web_call_count >= 4:
            return "(Web research limit reached. Please use the information already gathered to respond to the user now.)"
        _web_call_count += 1

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
    else:
        return f"Error: Unknown tool: {name}"


def create_client():
    """Create local API client."""
    return OpenAI(
        base_url=LOCAL_URL,
        api_key="not-needed"
    )


def fire_text(text: str, offset: int = 0) -> Text:
    """Create fire-colored text (red -> orange -> yellow gradient)."""
    result = Text()
    for i, char in enumerate(text):
        if char.strip():
            color = COLORS[(i + offset) % len(COLORS)]
            result.append(char, style=f"bold {color}")
        else:
            result.append(char)
    return result


def get_banner_with_mech():
    """Generate banner text."""
    banner = pyfiglet.figlet_format("MAUDE", font="banner3")
    banner_lines = banner.rstrip('\n').split('\n')
    return banner_lines


def animate_banner():
    """Display animated MAUDE banner."""
    banner_lines = get_banner_with_mech()

    # Animate the banner with fire colors
    with Live(console=console, refresh_per_second=12, transient=True) as live:
        for frame in range(25):
            content = Text()

            # Add animated MAUDE banner
            for line in banner_lines:
                content.append_text(fire_text(line, frame))
                content.append('\n')

            live.update(Align.center(content))
            time.sleep(0.06)

    # Final static banner with fire gradient
    final_content = Text()
    for line in banner_lines:
        final_content.append_text(fire_text(line, 24))
        final_content.append('\n')

    console.print(Align.center(final_content))
    console.print()


def print_separator():
    """Print a styled separator."""
    console.print("─" * console.width, style="dim cyan")


def chat(client, messages: list):
    """Send chat request to local Nemotron with tool support."""
    global show_thinking, _vision_call_count, _web_call_count

    max_tool_iterations = 15
    tool_iteration = 0
    recent_tool_calls = []
    _vision_call_count = 0
    _web_call_count = 0

    while True:
        tool_iteration += 1
        if tool_iteration > max_tool_iterations:
            console.print("[dim yellow](max tool iterations reached)[/dim yellow]")
            break
        try:
            start_time = time.time()

            # Non-streaming request for TUI compatibility
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=4096,
                tools=TOOLS,
                tool_choice="auto",
                extra_body={"num_ctx": NUM_CTX}
            )

            elapsed_time = time.time() - start_time
            msg = response.choices[0].message
            full_content = msg.content or ""

            # Show response
            if full_content:
                console.print(f"[bold magenta]MAUDE:[/bold magenta] {full_content}")
                token_count = response.usage.completion_tokens if response.usage else 0
                console.print(f"[dim]{token_count} tokens in {elapsed_time:.1f}s[/dim]")

            # Collect tool calls
            tool_calls_data = {}
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_data[tc.id] = {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }

            # Handle tool calls
            if tool_calls_data:
                # Add assistant message with tool calls to history
                messages.append({
                    "role": "assistant",
                    "content": full_content or "",
                    "tool_calls": [
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": tc_data["name"],
                                "arguments": tc_data["arguments"]
                            }
                        }
                        for tc_id, tc_data in tool_calls_data.items()
                    ]
                })

                # Execute each tool call
                for tc_id, tc_data in tool_calls_data.items():
                    func_name = tc_data["name"]
                    try:
                        func_args = json.loads(tc_data["arguments"])
                    except json.JSONDecodeError:
                        func_args = {}

                    # Check for duplicate tool calls (same tool + same args)
                    call_signature = (func_name, json.dumps(func_args, sort_keys=True))
                    if call_signature in recent_tool_calls:
                        console.print(f"[dim yellow]  [{func_name}] (skipped duplicate)[/dim yellow]")
                        result = "(Already called with same arguments - see previous result)"
                    else:
                        recent_tool_calls.append(call_signature)
                        console.print(f"[dim yellow]  [{func_name}][/dim yellow]")
                        result = execute_tool(func_name, func_args)

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result
                    })

                # Continue loop to get next response
                continue

            # No tool calls - we're done
            return full_content

        except Exception as e:
            error_msg = str(e)
            # If tools aren't supported, retry without them
            if "tool" in error_msg.lower() or "function" in error_msg.lower():
                console.print("[dim yellow]Tools not supported, falling back to basic chat...[/dim yellow]")
                try:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=[m for m in messages if "tool_calls" not in m and m.get("role") != "tool"],
                        temperature=0.2,
                        max_tokens=1024,
                        extra_body={"num_ctx": NUM_CTX}
                    )
                    msg = response.choices[0].message
                    if msg.content:
                        console.print(f"[bold magenta]MAUDE:[/bold magenta] {msg.content}")
                    return msg.content
                except Exception as e2:
                    console.print(f"[red]Error: {e2}[/red]")
                    return None
            else:
                console.print(f"[red]Error: {e}[/red]")
                console.print("[dim]Is the server running? Start with: ./start_server.sh[/dim]")
                return None


def handle_command(cmd: str) -> str:
    """Handle slash commands. Returns response or None if not a command."""
    if not cmd.startswith("/"):
        return None

    parts = cmd[1:].strip().split(maxsplit=10)
    if not parts:
        return None

    command = parts[0].lower()

    if command == "help":
        return """MAUDE Commands:

/help    - Show this help

Tools available:
- search_directory, search_file, read_file, write_file, edit_file
- list_directory, change_directory, get_working_directory
- run_command (shell)
- web_browse, web_search, web_view
- view_image

Say "quit" to exit."""
    else:
        return f"Unknown command: /{command}\nType /help for available commands."


SYSTEM_PROMPT = """You are MAUDE, a coding assistant powered by Nemotron.

STYLE: Be brief. Action over explanation.

TOOLS:
- search_directory(directory, pattern): Search across all files in a project - use when you don't know which file
- search_file(path, pattern): Search within a single file
- read_file(path, start_line, end_line): Read file with line numbers
- write_file(path, content): Create/overwrite file
- edit_file(path, start_line, end_line, new_content): Edit specific lines
- list_directory, change_directory, get_working_directory
- run_command: Shell commands (git, pip, python, etc.)
- web_browse, web_search: Web access
- web_view, view_image: Visual analysis (LLaVA)

FILE EDITING WORKFLOW:
1. search_directory to find which file contains the code
2. read_file with line range to see context
3. edit_file to replace lines

Use tools proactively. Confirm before destructive operations."""


class MaudeApp(App):
    """MAUDE TUI Application with fixed input at bottom."""

    CSS = """
    #output {
        height: 1fr;
        border: solid cyan;
        padding: 0 1;
        scrollbar-gutter: stable;
    }
    #input-container {
        height: auto;
        max-height: 3;
        padding: 0 1;
    }
    Input {
        border: solid green;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.client = None

    def compose(self) -> ComposeResult:
        yield RichLog(id="output", wrap=True, highlight=True, markup=True)
        with Container(id="input-container"):
            yield Input(placeholder="Type message... (quit to exit)", id="user-input")

    def on_mount(self):
        global _app
        _app = self

        # Load API keys
        km = KeyManager()
        km.load_all_keys()

        # Create client
        try:
            self.client = create_client()
        except Exception as e:
            self.write_output(f"[red]Failed to connect: {e}[/red]")
            return

        self.output_log = self.query_one("#output", RichLog)
        self.input_widget = self.query_one("#user-input", Input)
        self.banner_lines = get_banner_with_mech()
        self.banner_frame = 0

        # Start banner animation
        self.animate_banner()

    def animate_banner(self):
        """Animate the fire-colored banner."""
        if self.banner_frame < 25:
            # Clear and redraw banner with new frame
            self.output_log.clear()
            for line in self.banner_lines:
                self.output_log.write(fire_text(line, self.banner_frame))
            self.banner_frame += 1
            self.set_timer(0.06, self.animate_banner)
        else:
            # Animation done, show final banner + info
            self.output_log.clear()
            for line in self.banner_lines:
                self.output_log.write(fire_text(line, 24))
            self.write_output("")
            self.write_output("[dim]Nemotron-30B | Files | Shell | Web | Vision[/dim]")
            self.write_output("[dim]Type /help for commands, 'quit' to exit[/dim]\n")
            self.input_widget.focus()

    def write_output(self, text):
        """Write to the output log."""
        if hasattr(self, 'output_log'):
            self.output_log.write(text)

    async def on_input_submitted(self, event: Input.Submitted):
        """Handle user input."""
        user_input = event.value.strip()
        self.input_widget.clear()

        if not user_input:
            return

        # Show user message
        self.write_output(f"\n[bold green]YOU >[/bold green] {user_input}")

        # Exit commands
        if user_input.lower() in ("quit", "exit", "bye", "goodbye"):
            self.write_output("\n[dim magenta]MAUDE signing off.[/dim magenta]")
            self.exit()
            return

        # Slash commands
        if user_input.startswith("/"):
            result = handle_command(user_input)
            if result:
                self.write_output(f"\n{result}")
            return

        # Process with LLM in background
        self.process_message(user_input)

    @work(thread=True)
    def process_message(self, user_input: str):
        """Process message with LLM in background thread."""
        self.messages.append({"role": "user", "content": user_input})

        # Show thinking indicator
        self.call_from_thread(self.write_output, "\n[cyan]thinking...[/cyan]")

        response = chat(self.client, self.messages)

        if response:
            self.messages.append({"role": "assistant", "content": response})


def main():
    app = MaudeApp()
    app.run()


if __name__ == "__main__":
    main()
