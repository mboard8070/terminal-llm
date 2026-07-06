from maude_bootstrap import ensure_local_maude

ensure_local_maude()

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from maude.orchestration import scheduler as _canonical

globals().update({name: value for name, value in vars(_canonical).items() if not name.startswith("__")})

__all__ = [name for name in globals() if not name.startswith("__")]


def get_scheduler():
    _canonical._scheduler = globals().get("_scheduler", getattr(_canonical, "_scheduler", None))
    scheduler = _canonical.get_scheduler()
    globals()["_scheduler"] = getattr(_canonical, "_scheduler", scheduler)
    return scheduler
