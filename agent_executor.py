from maude_bootstrap import ensure_local_maude

ensure_local_maude()

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from maude.orchestration.agents import *  # noqa: F403
