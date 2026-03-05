"""
Dynamic tool selection — keyword-based filtering to keep token usage manageable.
"""

from .tool_defs import TOOLS

# Core tools always sent (file ops, web, shell, AI delegation)
_CORE_TOOL_NAMES = {
    "read_file", "write_file", "list_directory",
    "change_directory", "run_command", "web_browse", "web_search",
    "search_file", "search_directory", "edit_file",
    "ask_frontier",
    "save_memory", "recall_memory",
}

# Tool groups activated by keyword detection
_TOOL_GROUPS = {
    "image_gen": {
        "keywords": ["generate image", "create image", "make image", "draw", "flux",
                      "generate a picture", "make a picture", "image of", "picture of",
                      "illustration", "render", "lora", "stillion"],
        "tools": {"generate_image", "share_file"},
    },
    "shared": {
        "keywords": ["shared", "transfer", "client", "pull", "push", "send to client",
                      "grab", "fetch", "upload", "download", "sync"],
        "tools": {"list_shared", "share_file", "list_transfers", "get_transfer"},
    },
    "gmail": {
        "keywords": ["gmail", "email", "inbox", "mail"],
        "tools": {"gmail_list", "gmail_read", "gmail_send"},
    },
    "drive": {
        "keywords": ["drive", "google doc", "google drive", "my documents",
                      "my files on", "cloud files", "gdrive", "folder"],
        "tools": {"drive_list", "drive_search", "drive_read", "drive_upload",
                  "drive_create_doc", "drive_create_folder", "drive_create_sheet",
                  "drive_update_doc", "drive_delete"},
    },
    "sheets": {
        "keywords": ["sheet", "spreadsheet", "csv", "table", "cells", "rows", "columns"],
        "tools": {"sheets_read", "sheets_write", "sheets_append", "sheets_create",
                  "sheets_list_sheets", "sheets_clear"},
    },
    "calendar": {
        "keywords": ["calendar", "event", "meeting", "schedule", "appointment", "reminder"],
        "tools": {"calendar_list_events", "calendar_create_event", "calendar_update_event",
                  "calendar_delete_event", "calendar_search_events", "calendar_list_calendars"},
    },
    "slides": {
        "keywords": ["slide", "presentation", "deck", "powerpoint", "ppt"],
        "tools": {"slides_get_presentation", "slides_get_slide", "slides_create_presentation",
                  "slides_add_slide", "slides_add_text"},
    },
    "contacts": {
        "keywords": ["contact", "phone number", "address book", "people"],
        "tools": {"contacts_list", "contacts_get", "contacts_create", "contacts_update",
                  "contacts_delete", "contacts_search"},
    },
    "youtube": {
        "keywords": ["youtube", "playlist", "channel", "subscribe"],
        "tools": {"youtube_search", "youtube_get_video", "youtube_get_channel",
                  "youtube_list_playlists", "youtube_create_playlist",
                  "youtube_get_comments", "youtube_post_comment", "youtube_my_channel"},
    },
    "substack": {
        "keywords": ["substack", "newsletter", "draft", "publish", "blog post", "article"],
        "tools": {"substack_create_draft", "substack_list_drafts", "substack_list_posts",
                  "substack_get_post", "substack_update_draft", "substack_delete_draft",
                  "substack_get_stats"},
    },
    "browser": {
        "keywords": ["browser", "click", "fill form", "navigate to", "open page", "screenshot", "webpage"],
        "tools": {"browser_open", "browser_click", "browser_type", "browser_navigate",
                  "browser_screenshot", "browser_extract", "browser_fill_form",
                  "browser_select", "browser_close"},
    },
    "social": {
        "keywords": ["tweet", "post to", "social media", "twitter", "linkedin", "bluesky", "share on"],
        "tools": {"skill_post_social", "skill_social_status"},
    },
    "web_image": {
        "keywords": ["find image", "find picture", "find photo", "show me a picture",
                      "show me a photo", "show me an image", "photo of", "image of",
                      "picture of", "images of", "photos of", "pictures of",
                      "search for image", "search for photo", "web image",
                      "find me a picture", "find me an image", "find me a photo"],
        "tools": {"web_image_search"},
    },
    "vision": {
        "keywords": ["image", "photo", "picture", "camera", "see", "look at",
                      "what is this", "analyze", "describe", "view_image",
                      "attached", "screenshot", "snap"],
        "tools": {"view_image"},
    },
    "memory": {
        "keywords": ["remember", "recall", "forget", "memory", "memories",
                      "you know", "do you remember", "what do you know",
                      "i told you", "i mentioned", "last time",
                      "my preference", "my favorite", "i like", "i prefer",
                      "don't forget", "keep in mind", "note that"],
        "tools": {"save_memory", "recall_memory", "list_memories", "forget_memory"},
    },
    "collab": {
        "keywords": ["who's online", "whos online", "mesh status", "devices",
                      "dispatch", "send to spark", "send to mac", "run on",
                      "project", "collaboration", "activity", "task",
                      "what are they doing", "online",
                      "mac", "windows", "pc", "laptop", "other machine",
                      "other device", "remote", "cross-machine", "mattwell",
                      "macbook", "on the"],
        "tools": {"mesh_status", "dispatch_task", "create_project",
                  "list_projects", "add_to_project", "list_tasks"},
    },
    "github": {
        "keywords": ["pull request", "pr", "merge", "github", "repo"],
        "tools": {"github_list_prs", "github_view_pr", "github_merge_pr"},
    },
    "agents": {
        "keywords": ["research", "analyze", "investigate", "compare", "look into",
                      "deep dive", "comprehensive", "parallel", "use the .* agent",
                      "run agent", "dispatch agent"],
        "tools": {"run_agent", "run_agents"},
    },
    "google": {
        "keywords": ["google"],
        "tools": {"gmail_list", "gmail_read", "gmail_send",
                  "drive_list", "drive_search", "drive_read", "drive_upload",
                  "drive_create_doc", "drive_create_folder", "drive_create_sheet",
                  "drive_update_doc", "drive_delete",
                  "sheets_read", "sheets_write", "sheets_create",
                  "calendar_list_events", "calendar_create_event",
                  "contacts_list", "contacts_search"},
    },
}

# Build lookup: tool name -> tool definition
_TOOL_BY_NAME = {t["function"]["name"]: t for t in TOOLS}


def get_tools_for_message(message: str) -> list:
    """Return a filtered subset of TOOLS relevant to the user's message.

    Always includes core tools. Adds specialized tool groups only when
    keywords in the message suggest they're needed.
    """
    msg_lower = message.lower()

    # Start with core tools
    active_names = set(_CORE_TOOL_NAMES)

    # Add tool groups whose keywords match
    for group in _TOOL_GROUPS.values():
        for kw in group["keywords"]:
            if kw in msg_lower:
                active_names.update(group["tools"])
                break

    # Build filtered list preserving order
    return [t for t in TOOLS if t["function"]["name"] in active_names]
