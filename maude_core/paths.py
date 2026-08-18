"""
Working directory management and path resolution.
"""

from pathlib import Path

# Working directory for file operations
working_dir = Path.home()

# Spark used /home/mboard76; rewrite that prefix on any host.
_SPARK_HOME = "/home/mboard76"


def set_working_directory(path: Path):
    """Set the working directory for file operations."""
    global working_dir
    working_dir = path


def get_working_directory() -> Path:
    """Get the current working directory."""
    return working_dir


def shared_dir() -> Path:
    """Server shared-upload directory (created if missing)."""
    path = Path.home() / "nvidia-workbench" / "terminal-llm" / "shared"
    path.mkdir(parents=True, exist_ok=True)
    return path


def remap_spark_home(path_str: str) -> str:
    """Map Spark Linux home paths onto this machine's home directory."""
    if not path_str:
        return path_str
    normalized = path_str.replace("\\", "/")
    if normalized == _SPARK_HOME or normalized.startswith(_SPARK_HOME + "/"):
        rest = normalized[len(_SPARK_HOME) :].lstrip("/")
        mapped = Path.home() / rest if rest else Path.home()
        return str(mapped)
    return path_str


def resolve_path(path_str: str) -> Path:
    """Resolve a path relative to working directory."""
    global working_dir
    path_str = remap_spark_home(path_str)
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = working_dir / path
    return path.resolve()
