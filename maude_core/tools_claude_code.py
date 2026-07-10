"""
Claude Code fleet tools — lazy-import registrations.

Lets MAUDE dispatch prompts to named, resumable Claude Code sessions running on
mesh clients and see which instances are online / idle / working.
"""

from tool_registry import register_tool

_TOOL_NAMES = (
    "claude_register_worker",
    "ask_claude_code",
    "claude_check_reply",
    "claude_broadcast",
    "claude_list_workers",
    "claude_fleet_status",
    "claude_reset_worker",
    "claude_remove_worker",
)


def _make(name):
    @register_tool(name)
    def _dispatch(args, _name=name):
        from claude_code_tools import execute_claude_tool

        return execute_claude_tool(_name, args)

    return _dispatch


for _n in _TOOL_NAMES:
    _make(_n)
