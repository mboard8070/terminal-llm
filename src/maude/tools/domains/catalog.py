"""Domain catalog helpers for migrated tool schemas."""

from __future__ import annotations

from collections.abc import Iterable


def legacy_tool_schemas(names: Iterable[str]) -> list[dict]:
    """Return legacy tool schemas for a domain-owned set of tool names."""

    from maude_core.tool_defs import TOOLS

    wanted = set(names)
    return [tool for tool in TOOLS if tool.get("function", {}).get("name") in wanted]
