"""Runtime import bootstrap for legacy MAUDE entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_local_maude() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    src_text = str(src)
    sys.path[:] = [entry for entry in sys.path if entry != src_text]
    sys.path.insert(0, src_text)

    loaded = sys.modules.get("maude")
    loaded_file = Path(getattr(loaded, "__file__", "")).resolve() if loaded else None
    if loaded_file and src not in loaded_file.parents:
        for name in list(sys.modules):
            if name == "maude" or name.startswith("maude."):
                del sys.modules[name]
