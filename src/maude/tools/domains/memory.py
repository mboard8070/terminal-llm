"""Domain-owned tool schemas."""

TOOL_NAMES = {"forget_memory", "list_memories", "memory_browse", "memory_ledger_status", "recall_memory", "save_memory"}

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save typed context to MAUDE's memory ledger. Use semantic for durable facts, episodic "
            "for events/decisions, procedural for workflows, working for short-lived task context, "
            "preference for standing user preferences, identity for stable profile/project identity, "
            "person for people, project for project state, mission for mission state, and artifact "
            "for important files/outputs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Short unique identifier for the memory (e.g. "
                        "'favorite_language', 'project_deadline', "
                        "'wife_name')",
                    },
                    "value": {"type": "string", "description": "The information to remember"},
                    "category": {
                        "type": "string",
                        "description": "Memory type",
                        "enum": [
                            "semantic",
                            "episodic",
                            "procedural",
                            "working",
                            "preference",
                            "identity",
                            "person",
                            "project",
                            "mission",
                            "artifact",
                        ],
                    },
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "Search MAUDE's memory ledger and compatibility memory for relevant context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for in memory"},
                    "category": {
                        "type": "string",
                        "description": "Optional memory type filter",
                        "enum": [
                            "semantic",
                            "episodic",
                            "procedural",
                            "working",
                            "preference",
                            "identity",
                            "person",
                            "project",
                            "mission",
                            "artifact",
                            "conversation",
                        ],
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_memories",
            "description": "List stored memories, optionally filtered by category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Filter by memory type",
                        "enum": [
                            "semantic",
                            "episodic",
                            "procedural",
                            "working",
                            "preference",
                            "identity",
                            "person",
                            "project",
                            "mission",
                            "artifact",
                            "conversation",
                        ],
                    },
                    "limit": {"type": "integer", "description": "Maximum results (default 20)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_memory",
            "description": "Remove a specific memory by its key.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string", "description": "The memory key to remove"}},
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_ledger_status",
            "description": "Show MAUDE's memory ledger path, counts, files, and backend role split.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_browse",
            "description": "Browse MAUDE's persistent memory database. Can filter by category or search by query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Filter by memory category (e.g. 'fact', 'preference')",
                    },
                    "query": {"type": "string", "description": "Search term to find in memory keys or values"},
                    "limit": {"type": "integer", "description": "Max results to return (default 20, max 100)"},
                },
                "required": [],
            },
        },
    },
]


def schemas() -> list[dict]:
    return list(SCHEMAS)
