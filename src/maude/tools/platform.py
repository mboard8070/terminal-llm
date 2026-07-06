"""Permissioned tool execution facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .policy import evaluate_tool_policy, validate_result


@dataclass(frozen=True)
class ToolRequest:
    """Normalized tool request."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    approvals: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ToolResult:
    """Normalized tool result."""

    name: str
    output: str
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolPlatform:
    """Adapter over the existing registry-based tool executor."""

    def execute(self, request: ToolRequest) -> ToolResult:
        from maude_core.execute import execute_tool

        decision = evaluate_tool_policy(request.name, request.arguments, request.approvals)
        metadata = dict(decision.audit)
        if request.run_id:
            metadata["run_id"] = request.run_id
        if not decision.allowed:
            metadata["blocked"] = True
            return ToolResult(name=request.name, output=f"Tool blocked: {decision.reason}", metadata=metadata)

        output = execute_tool(request.name, request.arguments)
        if result_error := validate_result(output):
            metadata["result_error"] = result_error
            return ToolResult(name=request.name, output=f"Tool failed validation: {result_error}", metadata=metadata)
        return ToolResult(
            name=request.name,
            output=output,
            cached="[cached result]" in output,
            metadata=metadata,
        )
