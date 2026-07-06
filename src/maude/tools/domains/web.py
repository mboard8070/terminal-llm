"""Domain-owned tool schemas."""

TOOL_NAMES = {"web_browse", "view_image", "web_view", "web_search", "web_image_search"}

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_browse",
            "description": "Fetch and read content from a web URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "URL to fetch"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo. Use this for weather, news, prices, current events, or "
            "any factual query that needs up-to-date information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "description": "Number of results (default 5, max 10)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_view",
            "description": "Screenshot a webpage and analyze it visually using the active model's vision.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to screenshot"},
                    "question": {"type": "string", "description": "Optional question about the page"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_image",
            "description": "Analyze a local image file using the active model's vision.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the image file"},
                    "question": {"type": "string", "description": "Optional question about the image"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_image_search",
            "description": "Search the web for images. Returns image URLs with markdown display syntax. Use when "
            "the user wants to find pictures, photos, or images of something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Image search query"},
                    "num_results": {"type": "integer", "description": "Number of results (default 5, max 10)"},
                },
                "required": ["query"],
            },
        },
    },
]


def schemas() -> list[dict]:
    return list(SCHEMAS)
