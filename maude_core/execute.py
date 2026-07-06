from maude_bootstrap import ensure_local_maude

ensure_local_maude()

"""Compatibility shim for orchestration tool execution."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from maude.orchestration.tool_execution import execute_tool

__all__ = ["execute_tool"]
