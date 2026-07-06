"""Domain-owned tool schema catalogs."""

from . import browser, filesystem, github, google, media, memory, missions, platform, social, substack, system, web

DOMAIN_MODULES = (
    browser,
    filesystem,
    github,
    google,
    media,
    memory,
    missions,
    platform,
    social,
    substack,
    system,
    web,
)


def all_schemas() -> list[dict]:
    schemas: list[dict] = []
    for module in DOMAIN_MODULES:
        schemas.extend(module.schemas())
    return schemas


__all__ = [
    "DOMAIN_MODULES",
    "all_schemas",
    "browser",
    "filesystem",
    "github",
    "google",
    "media",
    "memory",
    "missions",
    "platform",
    "social",
    "substack",
    "system",
    "web",
]
