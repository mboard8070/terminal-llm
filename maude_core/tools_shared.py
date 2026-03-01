"""
Shared folder and file transfer tool implementations.
"""

from pathlib import Path

from tool_registry import register_tool
from .paths import resolve_path, get_working_directory

SHARED_DIR = Path.home() / "nvidia-workbench" / "terminal-llm" / "shared"
TRANSFERS_DIR = Path.home() / "nvidia-workbench" / "terminal-llm" / "transfers"


def tool_list_shared() -> str:
    """List files in the shared folder."""
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for entry in sorted(SHARED_DIR.iterdir()):
        if entry.is_dir():
            entries.append(f"[DIR]  {entry.name}/")
        else:
            size = entry.stat().st_size
            entries.append(f"[FILE] {entry.name} ({size:,} bytes)")
    if not entries:
        return f"Shared folder is empty: {SHARED_DIR}"
    return f"Shared folder ({SHARED_DIR}):\n" + "\n".join(entries)


def tool_share_file(path: str, filename: str = None) -> str:
    """Copy a file into the shared folder for the client to access."""
    import shutil
    src = resolve_path(path)
    if not src.exists():
        return f"Error: File not found: {src}"
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    dest_name = filename or src.name
    dest = SHARED_DIR / dest_name
    try:
        shutil.copy2(str(src), str(dest))
        return f"Shared '{src.name}' as '{dest_name}' \u2014 client can now pull it from the shared folder."
    except Exception as e:
        return f"Error sharing file: {e}"


def tool_list_transfers() -> str:
    """List files uploaded by the client."""
    TRANSFERS_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for entry in sorted(TRANSFERS_DIR.iterdir()):
        if entry.is_dir():
            entries.append(f"[DIR]  {entry.name}/")
        else:
            size = entry.stat().st_size
            entries.append(f"[FILE] {entry.name} ({size:,} bytes)")
    if not entries:
        return f"Transfers folder is empty (no client uploads): {TRANSFERS_DIR}"
    return f"Client uploads ({TRANSFERS_DIR}):\n" + "\n".join(entries)


def tool_get_transfer(filename: str, destination: str = None) -> str:
    """Copy a file from the transfers folder to the working directory."""
    import shutil
    src = TRANSFERS_DIR / filename
    if not src.exists():
        return f"Error: '{filename}' not found in transfers. Use list_transfers to see available files."
    dest_dir = resolve_path(destination) if destination else get_working_directory()
    if dest_dir.is_dir():
        dest = dest_dir / filename
    else:
        dest = dest_dir
    try:
        shutil.copy2(str(src), str(dest))
        return f"Copied '{filename}' to {dest}"
    except Exception as e:
        return f"Error: {e}"


# ── Registry wrappers ──────────────────────────────────────────

@register_tool("list_shared")
def _dispatch_list_shared(args):
    return tool_list_shared()

@register_tool("share_file")
def _dispatch_share_file(args):
    return tool_share_file(args.get("path", ""), args.get("filename"))

@register_tool("list_transfers")
def _dispatch_list_transfers(args):
    return tool_list_transfers()

@register_tool("get_transfer")
def _dispatch_get_transfer(args):
    return tool_get_transfer(args.get("filename", ""), args.get("destination"))
