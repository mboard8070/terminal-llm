"""
Working directory management and path resolution.
"""

from pathlib import Path

# Working directory for file operations
working_dir = Path.home()


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
