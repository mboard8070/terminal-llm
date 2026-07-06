"""Integration adapter contracts."""

from .base import IntegrationAdapter, IntegrationError, TestDoubleAdapter

__all__ = [
    "TestDoubleAdapter",
    "xai_oauth",
    "vnc",
    "voice",
    "video",
    "substack",
    "hyperframes",
    "github",
    "file_server",
    "browser_workflows",
    "IntegrationAdapter",
    "IntegrationError",
    "browser",
    "google",
    "social_posting",
]
from . import browser, google, social_posting
from . import browser_workflows
from . import file_server
from . import github
from . import hyperframes
from . import substack
from . import video
from . import voice
from . import vnc
from . import xai_oauth
