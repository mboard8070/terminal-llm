from maude_bootstrap import ensure_local_maude

ensure_local_maude()

"""Compatibility alias for the migrated MAUDE social posting integration."""

import sys

from maude.integrations import social_posting as _canonical

sys.modules[__name__] = _canonical
