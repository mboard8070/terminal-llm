"""Cross-machine clipboard skill - copy/paste text across devices."""

import base64

from skills import skill
from skills.utils import _is_dispatch_error


def _dispatch_shell(command: str, target: str) -> str:
    """Dispatch a shell command to a remote machine via collab.

    Only passes `target` — lets collab resolve hostname/client_id/platform.
    """
    from collab_tools import execute_collab_tool

    args = {
        "prompt": command,
        "capability": "SHELL",
    }
    if target:
        args["target"] = target
    return execute_collab_tool("dispatch_task", args)


def _extract_output(result: str) -> str:
    """Extract actual command output from dispatch_task result.

    Results come back as 'Task completed on <dest>:\\n<output>'
    """
    if not result:
        return ""
    for prefix in ("Task completed on ", "Task failed on "):
        if result.startswith(prefix):
            newline = result.find("\n")
            if newline != -1:
                return result[newline + 1 :].strip()
    return result.strip()


def _platform_hint(target: str) -> str:
    """Guess target platform from target name for command construction."""
    t = target.lower()
    if t in ("windows", "win"):
        return "windows"
    if t in ("macos", "mac", "darwin"):
        return "macos"
    return ""


def _build_copy_command(b64_text: str, platform: str) -> str:
    """Build a platform-aware clipboard copy command."""
    if platform == "windows":
        return f'[System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String("{b64_text}")) | Set-Clipboard'
    # macOS / Linux universal bash
    # Try pbcopy (macOS), then xclip, xsel, wl-copy (Linux)
    return (
        f'TEXT=$(echo "{b64_text}" | base64 -d) && '
        f'{{ echo "$TEXT" | pbcopy 2>/dev/null && echo "copied"; }} || '
        f'{{ echo "$TEXT" | xclip -selection clipboard 2>/dev/null && echo "copied"; }} || '
        f'{{ echo "$TEXT" | xsel --clipboard --input 2>/dev/null && echo "copied"; }} || '
        f'{{ echo "$TEXT" | wl-copy 2>/dev/null && echo "copied"; }} || '
        f'echo "ERROR: no clipboard utility found (install xclip, xsel, or wl-clipboard)"'
    )


def _build_paste_command(platform: str) -> str:
    """Build a platform-aware clipboard paste command."""
    if platform == "windows":
        return "Get-Clipboard"
    # macOS / Linux fallback chain
    return (
        "{ pbpaste 2>/dev/null; } || "
        "{ xclip -selection clipboard -o 2>/dev/null; } || "
        "{ xsel --clipboard --output 2>/dev/null; } || "
        "{ wl-paste 2>/dev/null; } || "
        'echo "ERROR: no clipboard utility found (install xclip, xsel, or wl-clipboard)"'
    )


@skill(
    name="copy_clipboard",
    description="Copy text to a device's clipboard. Sends text to the clipboard of the target machine (Mac, Windows, or Linux).",
    version="1.0.0",
    author="MAUDE",
    triggers=["copy", "clipboard", "copy to clipboard"],
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to copy to the clipboard"},
            "target": {
                "type": "string",
                "description": "Target device — hostname, client_id, or platform (e.g. 'windows', 'macos', 'macbook'). Leave empty for default.",
                "default": "",
            },
        },
        "required": ["text"],
    },
)
def copy_clipboard(text: str, target: str = "") -> str:
    """Copy text to a remote device's clipboard."""
    if not text:
        return "Error: no text provided to copy"

    if len(text) > 100_000:
        return f"Error: text too large ({len(text)} chars). Maximum is 100,000 characters."

    # Base64 encode to avoid shell escaping issues
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")

    platform = _platform_hint(target)
    command = _build_copy_command(b64, platform)
    result = _dispatch_shell(command, target)

    if _is_dispatch_error(result):
        return f"Clipboard copy failed:\n{result}"

    preview = text[:80] + "..." if len(text) > 80 else text
    return f'Copied to clipboard on {target or "target device"}:\n"{preview}"'


@skill(
    name="paste_clipboard",
    description="Read/paste the clipboard contents from a device. Returns whatever text is currently on the target machine's clipboard.",
    version="1.0.0",
    author="MAUDE",
    triggers=["paste", "read clipboard", "clipboard contents", "get clipboard"],
    parameters={
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Target device — hostname, client_id, or platform (e.g. 'windows', 'macos', 'macbook'). Leave empty for default.",
                "default": "",
            }
        },
    },
)
def paste_clipboard(target: str = "") -> str:
    """Read clipboard contents from a remote device."""
    platform = _platform_hint(target)
    command = _build_paste_command(platform)
    result = _dispatch_shell(command, target)

    if _is_dispatch_error(result):
        return f"Clipboard paste failed:\n{result}"

    output = _extract_output(result)
    if not output:
        return f"Clipboard on {target or 'target device'} is empty."

    return f"Clipboard contents from {target or 'target device'}:\n{output}"
