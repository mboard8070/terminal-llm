from maude_bootstrap import ensure_local_maude

ensure_local_maude()

"""Compatibility shim for migrated MAUDE tool handlers."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from maude.tools.handlers import ai as _canonical

globals().update({name: value for name, value in vars(_canonical).items() if not name.startswith("__")})

__all__ = [name for name in globals() if not name.startswith("__")]
