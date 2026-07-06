"""Tool risk classification, validation, and approval policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ToolRisk(StrEnum):
    READ = "read"
    LOCAL_WRITE = "local_write"
    EXTERNAL_WRITE = "external_write"
    DELETE = "delete"
    PUBLISH = "publish"
    SPEND = "spend"


@dataclass(frozen=True)
class ToolPolicyDecision:
    tool_name: str
    risk: ToolRisk
    allowed: bool
    reason: str = ""
    audit: dict[str, Any] = field(default_factory=dict)


PUBLISH_TOOLS = {"youtube_upload", "youtube_post_comment", "substack_publish", "x_post", "social_post"}
DELETE_TOOLS = {"drive_delete", "calendar_delete_event", "contacts_delete", "remove_shared", "remove_transfer"}
SPEND_PREFIXES = ("purchase_", "buy_", "order_")
LOCAL_WRITE_TOOLS = {"write_file", "edit_file", "share_file", "get_transfer", "mission_update", "mission_log"}
EXTERNAL_WRITE_PREFIXES = (
    "gmail_send",
    "drive_upload",
    "drive_create",
    "drive_update",
    "sheets_write",
    "sheets_append",
    "calendar_create",
    "calendar_update",
    "contacts_create",
    "contacts_update",
    "youtube_create",
    "youtube_add",
)


def classify_tool(name: str) -> ToolRisk:
    if name in PUBLISH_TOOLS or name.endswith("_publish"):
        return ToolRisk.PUBLISH
    if name in DELETE_TOOLS or name.endswith("_delete") or name.startswith("delete_"):
        return ToolRisk.DELETE
    if any(name.startswith(prefix) for prefix in SPEND_PREFIXES):
        return ToolRisk.SPEND
    if name in LOCAL_WRITE_TOOLS:
        return ToolRisk.LOCAL_WRITE
    if any(name.startswith(prefix) for prefix in EXTERNAL_WRITE_PREFIXES):
        return ToolRisk.EXTERNAL_WRITE
    return ToolRisk.READ


def validate_arguments(name: str, arguments: dict[str, Any]) -> str | None:
    if not isinstance(arguments, dict):
        return "tool arguments must be an object"
    if name in {"read_file", "write_file", "edit_file"} and not arguments.get("path"):
        return f"{name} requires a path argument"
    if name in DELETE_TOOLS and not (arguments.get("filename") or arguments.get("id") or arguments.get("file_id")):
        return f"{name} requires an explicit target identifier"
    return None


def validate_result(result: Any) -> str | None:
    if result is None:
        return "tool returned no result"
    return None


def evaluate_tool_policy(name: str, arguments: dict[str, Any], approvals: set[str] | None = None) -> ToolPolicyDecision:
    risk = classify_tool(name)
    approvals = approvals or set()
    audit = {"tool": name, "risk": risk.value}

    if error := validate_arguments(name, arguments):
        return ToolPolicyDecision(name, risk, False, error, audit)

    approval_key = risk.value
    if (
        risk in {ToolRisk.EXTERNAL_WRITE, ToolRisk.DELETE, ToolRisk.PUBLISH, ToolRisk.SPEND}
        and approval_key not in approvals
    ):
        return ToolPolicyDecision(name, risk, False, f"{risk.value} approval required", audit)

    return ToolPolicyDecision(name, risk, True, audit=audit)
