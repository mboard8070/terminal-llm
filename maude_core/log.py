"""
Logging — callback-based log() for TUI/Telegram + Python logging bridge.
"""

import logging
from typing import Callable, Optional

_module_logger = logging.getLogger("maude.core")

_log_callback: Optional[Callable[[str], None]] = None


def set_log_callback(callback: Callable[[str], None]):
    """Set the logging callback for tool status messages."""
    global _log_callback
    _log_callback = callback


def log(message: str):
    """Log a status message via callback and Python logging."""
    _module_logger.debug(message)
    if _log_callback:
        _log_callback(message)
