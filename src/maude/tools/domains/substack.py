"""Domain-owned tool schemas."""

TOOL_NAMES = {
    "substack_create_draft",
    "substack_delete_draft",
    "substack_get_post",
    "substack_get_stats",
    "substack_list_drafts",
    "substack_list_posts",
    "substack_update_draft",
}

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "substack_create_draft",
            "description": "Create a draft post on Substack newsletter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Post title"},
                    "body": {
                        "type": "string",
                        "description": "Post body text (plain text, double newlines for paragraphs)",
                    },
                    "subtitle": {"type": "string", "description": "Post subtitle"},
                    "audience": {
                        "type": "string",
                        "description": "Audience: 'everyone' (free) or 'only_paid'. Default: 'everyone'",
                    },
                },
                "required": ["title", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "substack_list_drafts",
            "description": "List draft posts on Substack.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "Maximum drafts to return (default 10)"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "substack_list_posts",
            "description": "List published Substack posts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Maximum posts to return (default 10)"},
                    "offset": {"type": "integer", "description": "Offset for pagination (default 0)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "substack_get_post",
            "description": "Get a specific Substack post or draft by ID.",
            "parameters": {
                "type": "object",
                "properties": {"post_id": {"type": "string", "description": "The post or draft ID"}},
                "required": ["post_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "substack_update_draft",
            "description": "Update an existing Substack draft.",
            "parameters": {
                "type": "object",
                "properties": {
                    "draft_id": {"type": "string", "description": "The draft ID to update"},
                    "title": {"type": "string", "description": "New title"},
                    "body": {"type": "string", "description": "New body text"},
                    "subtitle": {"type": "string", "description": "New subtitle"},
                },
                "required": ["draft_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "substack_delete_draft",
            "description": "Delete a Substack draft.",
            "parameters": {
                "type": "object",
                "properties": {"draft_id": {"type": "string", "description": "The draft ID to delete"}},
                "required": ["draft_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "substack_get_stats",
            "description": "Get Substack publication statistics (subscribers, posts, etc.).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def schemas() -> list[dict]:
    return list(SCHEMAS)
