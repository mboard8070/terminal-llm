from maude_bootstrap import ensure_local_maude

ensure_local_maude()

"""Compatibility alias for the migrated MAUDE xAI OAuth integration."""

import sys

from maude.integrations import xai_oauth as _canonical

sys.modules[__name__] = _canonical
