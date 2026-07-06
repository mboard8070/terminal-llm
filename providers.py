from maude_bootstrap import ensure_local_maude

ensure_local_maude()

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from maude.providers.config import *  # noqa: F403
