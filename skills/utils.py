"""Shared utilities for built-in skills (dispatch helpers, etc.)."""


def _is_dispatch_error(result: str) -> bool:
    """Check if a dispatch result indicates failure."""
    if not result:
        return True
    r = result.strip()
    if r.startswith("ERROR"):
        return True
    if "not found or offline" in r:
        return True
    if "no result yet" in r:
        return True
    return bool(r.startswith("Task failed"))
