"""
Fast Tool Dispatch — bypass LLM for obvious tool calls.
"""

import re as _re


# Pattern -> (tool_name, argument_builder)
# argument_builder receives the match object and original message, returns dict
_FAST_PATTERNS = [
    # Google Drive
    (_re.compile(r'\b(?:list|show|what.?s (?:on|in))\b.*\b(?:drive|google drive)\b', _re.I),
     "drive_list", lambda m, msg: {"query": "", "max_results": 20}),
    (_re.compile(r'\b(?:search|find|look for)\b.*\b(?:drive|google drive)\b', _re.I),
     "drive_search", lambda m, msg: {"query": _re.sub(r'.*?(?:search|find|look for)\s+', '', msg, flags=_re.I).strip().rstrip('?.')}),
    (_re.compile(r'\b(?:search|find|look for)\b\s+(?:["\'`](.+?)["\'`]|(\S+.+?))\s+(?:on|in|from)\s+(?:drive|google drive)', _re.I),
     "drive_search", lambda m, msg: {"query": m.group(1) or m.group(2)}),

    # Gmail
    (_re.compile(r'\b(?:check|list|show|read|get|any new)\b.*\b(?:emails?|gmail|inbox|mail)\b', _re.I),
     "gmail_list", lambda m, msg: {"query": "", "max_results": 10}),
    (_re.compile(r'\b(?:search|find)\b.*\b(?:emails?|gmail|mail)\b.*(?:from|about|subject)\s+(.+)', _re.I),
     "gmail_list", lambda m, msg: {"query": m.group(1).strip().rstrip('?.'), "max_results": 10}),

    # Calendar
    (_re.compile(r'\b(?:what.?s on|check|show|list|any)\b.*\b(?:calendar|schedule|agenda)\b', _re.I),
     "calendar_list_events", lambda m, msg: {"max_results": 10}),
    (_re.compile(r'\b(?:upcoming|next|today.?s?|any)\b.*\b(?:event|meeting|appointment|calendar)\b', _re.I),
     "calendar_list_events", lambda m, msg: {"max_results": 10}),

    # Sheets
    (_re.compile(r'\b(?:read|show|open|get)\b.*\b(?:spreadsheet|sheet)\b', _re.I),
     "sheets_list_sheets", lambda m, msg: {"spreadsheet_id": ""}),

    # Contacts (specific search before generic list)
    (_re.compile(r'\b(?:find|search|look up)\b.*\bcontact.*?(?:for|named?)\s+(.+)', _re.I),
     "contacts_search", lambda m, msg: {"query": m.group(1).strip().rstrip('?.')}),
    (_re.compile(r'\b(?:list|show)\b.*\b(?:contact|contacts|address book)\b', _re.I),
     "contacts_list", lambda m, msg: {"max_results": 20}),

    # YouTube
    (_re.compile(r'\b(?:search|find|look for)\b.*\b(?:youtube|on youtube)\b', _re.I),
     "youtube_search", lambda m, msg: {"query": _re.sub(r'.*?(?:search|find|look for)\s+', '', msg, flags=_re.I).replace('on youtube', '').replace('youtube', '').strip().rstrip('?.'), "max_results": 5}),
    (_re.compile(r'\bmy (?:youtube )?channel\b', _re.I),
     "youtube_my_channel", lambda m, msg: {}),

    # Substack
    (_re.compile(r'\b(?:list|show|check)\b.*\b(?:substack|newsletter)\b.*\b(?:draft|drafts)\b', _re.I),
     "substack_list_drafts", lambda m, msg: {"limit": 10}),
    (_re.compile(r'\b(?:list|show)\b.*\b(?:substack|newsletter)\b.*\b(?:post|posts|articles?)\b', _re.I),
     "substack_list_posts", lambda m, msg: {"limit": 10}),
    (_re.compile(r'\bsubstack\b.*\bstat', _re.I),
     "substack_get_stats", lambda m, msg: {}),

    # Web search (common pattern)
    # NOTE: "google" alone must NOT match when followed by a service name
    # (doc, drive, sheet, calendar, etc.) — those should go to Google tools via LLM
    (_re.compile(r'\b(?:search|google|look up|what is|what are|who is|when is|where is)\b(?!.*\b(?:doc|drive|sheet|calendar|slide|contact|gmail|emails?|mail|inbox)\b)', _re.I),
     "web_search", lambda m, msg: {"query": _re.sub(r'^(?:search\s+(?:for\s+)?|google\s+|look\s+up\s+)', '', msg, flags=_re.I).strip().rstrip('?.'), "num_results": 5}),
]


def fast_dispatch(message: str):
    """
    Try to match the user's message to a direct tool call.

    Returns:
        (tool_name, arguments, result) if matched and executed
        None if no fast path matched
    """
    from .execute import execute_tool

    msg = message.strip()

    for pattern, tool_name, arg_builder in _FAST_PATTERNS:
        match = pattern.search(msg)
        if match:
            try:
                args = arg_builder(match, msg)
                # Skip if args seem empty/broken
                if tool_name in ("drive_search", "web_search", "youtube_search") and not args.get("query"):
                    continue
                result = execute_tool(tool_name, args)
                if result and not result.startswith("Error:"):
                    return tool_name, args, result
            except Exception:
                continue

    return None
