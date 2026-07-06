"""Domain-owned tool schemas."""

TOOL_NAMES = {
    "ask_frontier",
    "schedule_task",
    "send_to_claude",
    "xai_oauth_doctor",
    "xai_oauth_finish",
    "xai_oauth_start",
    "xai_oauth_status",
    "xai_oauth_test",
    "xai_x_search",
}

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "xai_oauth_start",
            "description": "Start xAI Grok OAuth login for SuperGrok or X Premium+ using the same PKCE loopback "
            "flow as Hermes/OpenClaw. Returns the authorization URL.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "xai_oauth_finish",
            "description": "Finish a pending xAI Grok OAuth login after the callback page was reached and save "
            "tokens for MAUDE. If loopback capture failed, pass the final redirected callback_url.",
            "parameters": {
                "type": "object",
                "properties": {
                    "callback_url": {
                        "type": "string",
                        "description": "Optional full final redirect URL from the xAI OAuth callback page.",
                    },
                    "code": {
                        "type": "string",
                        "description": "Optional authorization code. Prefer callback_url "
                        "because it includes OAuth state.",
                    },
                    "state": {"type": "string", "description": "Optional OAuth state when passing code manually."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "xai_oauth_doctor",
            "description": "Run non-secret diagnostics for MAUDE's xAI Grok OAuth setup: pending login, callback "
            "capture, discovery, stored credentials, refresh token, and optional API test.",
            "parameters": {
                "type": "object",
                "properties": {
                    "refresh": {"type": "boolean", "description": "If true, force a token refresh check."},
                    "test_api": {"type": "boolean", "description": "If true, send a small xAI API request."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "xai_oauth_status",
            "description": "Check MAUDE's xAI Grok OAuth login status without exposing tokens.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "xai_oauth_test",
            "description": "Send a small request to xAI using MAUDE's stored Grok OAuth bearer token.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Optional prompt to send"},
                    "model": {"type": "string", "description": "Optional xAI model, defaults to grok-4.3"},
                    "max_output_tokens": {"type": "integer", "description": "Optional output token cap"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "xai_x_search",
            "description": "Search X posts via xAI's server-side x_search Responses API tool using the stored Grok "
            "OAuth token.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for X"},
                    "model": {"type": "string", "description": "Optional xAI model, defaults to grok-4.3"},
                    "timeout_seconds": {"type": "number", "description": "Optional request timeout"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_frontier",
            "description": "Escalate to a frontier AI model (Claude, GPT, Gemini) for complex questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question requiring expert analysis"},
                    "context": {"type": "string", "description": "Relevant context"},
                    "provider": {
                        "type": "string",
                        "description": "Optional: claude, openai, gemini, grok-oauth, grok, mistral",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_to_claude",
            "description": "Send a message to Claude Code running in tmux. Use this to delegate complex coding "
            "tasks, get expert analysis, or have Claude work on files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The message/task to send to Claude Code"},
                    "session": {"type": "string", "description": "tmux session name (default: 'claude')"},
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_task",
            "description": "Schedule automated tasks. Convert natural language schedules to cron expressions:\n"
            '- "every morning" or "daily at 8am" → 0 8 * * *\n'
            '- "every hour" → 0 * * * *\n'
            '- "weekdays at 9am" → 0 9 * * 1-5\n'
            '- "every evening at 6pm" → 0 18 * * *\n'
            '- "weekly on Monday" → 0 9 * * 1\n'
            '- "every 30 minutes" → */30 * * * *\n'
            "\n"
            "Shortcuts: @hourly, @daily, @morning, @evening, @weekly, @workdays\n"
            "\n"
            "Actions: add (create task), list (show all), remove (delete), enable, disable, run "
            "(execute now)",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action: add, list, remove, enable, disable, run",
                        "enum": ["add", "list", "remove", "enable", "disable", "run"],
                    },
                    "name": {"type": "string", "description": "Descriptive name for the task (for add action)"},
                    "cron": {
                        "type": "string",
                        "description": "Cron expression or shortcut like @daily, @morning (for add action)",
                    },
                    "prompt": {"type": "string", "description": "What MAUDE should do when triggered (for add action)"},
                    "task_id": {"type": "string", "description": "Task ID (for remove/enable/disable/run actions)"},
                },
                "required": ["action"],
            },
        },
    },
]


def schemas() -> list[dict]:
    return list(SCHEMAS)
