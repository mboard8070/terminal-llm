"""
Tool definitions — the TOOLS list of JSON schemas.

Also adds dynamic tools (browser, skills) if available.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file with line numbers. Use start_line/end_line for large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "start_line": {"type": "integer", "description": "First line to read (1-indexed, optional)"},
                    "end_line": {"type": "integer", "description": "Last line to read (1-indexed, optional)"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "Content to write to the file"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories in a path. Shows file sizes and types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to list (defaults to working directory)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_working_directory",
            "description": "Get the current working directory path.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "change_directory",
            "description": "Change the current working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to change to"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command. Use for: pip, python, git, rm, mv, cp, etc. Also use to remove files from the shared/ or transfers/ folder when asked to clean up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_browse",
            "description": "Fetch and read content from a web URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "description": "Number of results (default 5, max 10)"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_view",
            "description": "Screenshot a webpage and analyze it visually using LLaVA.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to screenshot"},
                    "question": {"type": "string", "description": "Optional question about the page"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "view_image",
            "description": "Analyze a local image file using LLaVA vision model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the image file"},
                    "question": {"type": "string", "description": "Optional question about the image"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image using Flux AI via ComfyUI. Saves the image to the shared folder and returns a download URL for display. Use markdown ![description](/download/filename.png) to show it in chat.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Text description of the image to generate"},
                    "width": {"type": "integer", "description": "Image width (default 1024)"},
                    "height": {"type": "integer", "description": "Image height (default 1024)"},
                    "seed": {"type": "integer", "description": "Seed for reproducibility (-1 for random)"},
                    "steps": {"type": "integer", "description": "Sampling steps (default 28)"},
                    "lora": {"type": "string", "description": "Optional LoRA name: stillion-style, marker-mech-style"},
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_file",
            "description": "Search for text/pattern in a single file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "pattern": {"type": "string", "description": "Text to search for"}
                },
                "required": ["path", "pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_directory",
            "description": "Search for text across all files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory to search"},
                    "pattern": {"type": "string", "description": "Text to search for"}
                },
                "required": ["directory", "pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit specific lines in a file. Read the file first to see line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "start_line": {"type": "integer", "description": "First line to replace (1-indexed)"},
                    "end_line": {"type": "integer", "description": "Last line to replace (1-indexed)"},
                    "new_content": {"type": "string", "description": "New content to insert"}
                },
                "required": ["path", "start_line", "end_line", "new_content"]
            }
        }
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
                    "provider": {"type": "string", "description": "Optional: claude, openai, gemini, grok, mistral"}
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_to_claude",
            "description": "Send a message to Claude Code running in tmux. Use this to delegate complex coding tasks, get expert analysis, or have Claude work on files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The message/task to send to Claude Code"},
                    "session": {"type": "string", "description": "tmux session name (default: 'claude')"}
                },
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_task",
            "description": """Schedule automated tasks. Convert natural language schedules to cron expressions:
- "every morning" or "daily at 8am" \u2192 0 8 * * *
- "every hour" \u2192 0 * * * *
- "weekdays at 9am" \u2192 0 9 * * 1-5
- "every evening at 6pm" \u2192 0 18 * * *
- "weekly on Monday" \u2192 0 9 * * 1
- "every 30 minutes" \u2192 */30 * * * *

Shortcuts: @hourly, @daily, @morning, @evening, @weekly, @workdays

Actions: add (create task), list (show all), remove (delete), enable, disable, run (execute now)""",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "Action: add, list, remove, enable, disable, run", "enum": ["add", "list", "remove", "enable", "disable", "run"]},
                    "name": {"type": "string", "description": "Descriptive name for the task (for add action)"},
                    "cron": {"type": "string", "description": "Cron expression or shortcut like @daily, @morning (for add action)"},
                    "prompt": {"type": "string", "description": "What MAUDE should do when triggered (for add action)"},
                    "task_id": {"type": "string", "description": "Task ID (for remove/enable/disable/run actions)"}
                },
                "required": ["action"]
            }
        }
    },
    # Google Tools
    {"type": "function", "function": {"name": "gmail_list", "description": "List recent emails from Gmail. Use query for searching (same syntax as Gmail search).", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query (e.g., 'from:someone@example.com', 'subject:invoice', 'is:unread')"}, "max_results": {"type": "integer", "description": "Maximum emails to return (default 10)"}}, "required": []}}},
    {"type": "function", "function": {"name": "gmail_read", "description": "Read a specific email by its message ID.", "parameters": {"type": "object", "properties": {"message_id": {"type": "string", "description": "The Gmail message ID"}}, "required": ["message_id"]}}},
    {"type": "function", "function": {"name": "gmail_send", "description": "Send an email via Gmail.", "parameters": {"type": "object", "properties": {"to": {"type": "string", "description": "Recipient email address"}, "subject": {"type": "string", "description": "Email subject"}, "body": {"type": "string", "description": "Email body text"}, "cc": {"type": "string", "description": "CC recipients (optional)"}}, "required": ["to", "subject", "body"]}}},
    {"type": "function", "function": {"name": "drive_list", "description": "List files in Google Drive. Use query for filtering.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Drive query (e.g., \"name contains 'report'\")"}, "max_results": {"type": "integer", "description": "Maximum files to return (default 20)"}}, "required": []}}},
    {"type": "function", "function": {"name": "drive_search", "description": "Search Google Drive for files by name or content.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search term"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "drive_read", "description": "Read the contents of a file from Google Drive (text files, Google Docs, etc.).", "parameters": {"type": "object", "properties": {"file_id": {"type": "string", "description": "The Google Drive file ID"}}, "required": ["file_id"]}}},
    {"type": "function", "function": {"name": "drive_upload", "description": "Upload a local file to Google Drive.", "parameters": {"type": "object", "properties": {"local_path": {"type": "string", "description": "Path to the local file to upload"}, "folder_id": {"type": "string", "description": "Optional Drive folder ID to upload into"}}, "required": ["local_path"]}}},
    {"type": "function", "function": {"name": "drive_create_folder", "description": "Create a new folder in Google Drive.", "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "Name for the new folder"}, "parent_id": {"type": "string", "description": "Optional parent folder ID to create inside"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "drive_create_doc", "description": "Create a new Google Doc in Google Drive.", "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "Name for the new document"}, "folder_id": {"type": "string", "description": "Optional folder ID to create inside"}, "content": {"type": "string", "description": "Optional initial content for the document"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "drive_create_sheet", "description": "Create a new Google Sheet in Google Drive.", "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "Name for the new spreadsheet"}, "folder_id": {"type": "string", "description": "Optional folder ID to create inside"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "drive_update_doc", "description": "Write or append content to an existing Google Doc.", "parameters": {"type": "object", "properties": {"doc_id": {"type": "string", "description": "The Google Doc ID"}, "content": {"type": "string", "description": "The content to write to the document"}, "append": {"type": "boolean", "description": "If true, append to existing content. If false (default), replace all content."}}, "required": ["doc_id", "content"]}}},
    {"type": "function", "function": {"name": "drive_delete", "description": "Delete a file or folder from Google Drive.", "parameters": {"type": "object", "properties": {"file_id": {"type": "string", "description": "The Google Drive file or folder ID to delete"}}, "required": ["file_id"]}}},
    # Google Sheets tools
    {"type": "function", "function": {"name": "sheets_read", "description": "Read data from a Google Sheets spreadsheet.", "parameters": {"type": "object", "properties": {"spreadsheet_id": {"type": "string", "description": "The Google Sheets spreadsheet ID"}, "range": {"type": "string", "description": "Cell range to read (e.g., 'Sheet1!A1:D10'). Default: 'Sheet1'"}}, "required": ["spreadsheet_id"]}}},
    {"type": "function", "function": {"name": "sheets_write", "description": "Write data to a Google Sheets spreadsheet (overwrites existing data in range).", "parameters": {"type": "object", "properties": {"spreadsheet_id": {"type": "string", "description": "The Google Sheets spreadsheet ID"}, "range": {"type": "string", "description": "Cell range to write to (e.g., 'Sheet1!A1')"}, "values": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}, "description": "2D array of values (rows of columns)"}}, "required": ["spreadsheet_id", "range", "values"]}}},
    {"type": "function", "function": {"name": "sheets_append", "description": "Append rows to a Google Sheets spreadsheet.", "parameters": {"type": "object", "properties": {"spreadsheet_id": {"type": "string", "description": "The Google Sheets spreadsheet ID"}, "range": {"type": "string", "description": "Range to append after (e.g., 'Sheet1!A1')"}, "values": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}, "description": "2D array of rows to append"}}, "required": ["spreadsheet_id", "range", "values"]}}},
    {"type": "function", "function": {"name": "sheets_create", "description": "Create a new Google Sheets spreadsheet.", "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "Name for the new spreadsheet"}, "folder_id": {"type": "string", "description": "Optional Drive folder ID to create inside"}}, "required": ["title"]}}},
    {"type": "function", "function": {"name": "sheets_list_sheets", "description": "List all sheet tabs in a Google Sheets spreadsheet.", "parameters": {"type": "object", "properties": {"spreadsheet_id": {"type": "string", "description": "The Google Sheets spreadsheet ID"}}, "required": ["spreadsheet_id"]}}},
    {"type": "function", "function": {"name": "sheets_clear", "description": "Clear a range of cells in a Google Sheets spreadsheet.", "parameters": {"type": "object", "properties": {"spreadsheet_id": {"type": "string", "description": "The Google Sheets spreadsheet ID"}, "range": {"type": "string", "description": "Cell range to clear (e.g., 'Sheet1!A1:D10')"}}, "required": ["spreadsheet_id", "range"]}}},
    # Google Calendar tools
    {"type": "function", "function": {"name": "calendar_list_events", "description": "List upcoming Google Calendar events.", "parameters": {"type": "object", "properties": {"max_results": {"type": "integer", "description": "Maximum events to return (default 10)"}, "time_min": {"type": "string", "description": "Start time filter (ISO format, e.g., '2025-01-15T00:00:00Z'). Default: now"}, "time_max": {"type": "string", "description": "End time filter (ISO format)"}, "calendar_id": {"type": "string", "description": "Calendar ID (default: 'primary')"}}, "required": []}}},
    {"type": "function", "function": {"name": "calendar_create_event", "description": "Create a new Google Calendar event.", "parameters": {"type": "object", "properties": {"summary": {"type": "string", "description": "Event title"}, "start": {"type": "string", "description": "Start time (ISO format, e.g., '2025-01-15T10:00:00-05:00')"}, "end": {"type": "string", "description": "End time (ISO format)"}, "description": {"type": "string", "description": "Event description"}, "location": {"type": "string", "description": "Event location"}, "calendar_id": {"type": "string", "description": "Calendar ID (default: 'primary')"}}, "required": ["summary", "start", "end"]}}},
    {"type": "function", "function": {"name": "calendar_update_event", "description": "Update an existing Google Calendar event.", "parameters": {"type": "object", "properties": {"event_id": {"type": "string", "description": "The event ID to update"}, "summary": {"type": "string", "description": "New event title"}, "start": {"type": "string", "description": "New start time (ISO format)"}, "end": {"type": "string", "description": "New end time (ISO format)"}, "description": {"type": "string", "description": "New description"}, "location": {"type": "string", "description": "New location"}, "calendar_id": {"type": "string", "description": "Calendar ID (default: 'primary')"}}, "required": ["event_id"]}}},
    {"type": "function", "function": {"name": "calendar_delete_event", "description": "Delete a Google Calendar event.", "parameters": {"type": "object", "properties": {"event_id": {"type": "string", "description": "The event ID to delete"}, "calendar_id": {"type": "string", "description": "Calendar ID (default: 'primary')"}}, "required": ["event_id"]}}},
    {"type": "function", "function": {"name": "calendar_search_events", "description": "Search Google Calendar events by text.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search text"}, "max_results": {"type": "integer", "description": "Maximum events to return (default 10)"}, "calendar_id": {"type": "string", "description": "Calendar ID (default: 'primary')"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "calendar_list_calendars", "description": "List all available Google Calendars.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    # Google Slides tools
    {"type": "function", "function": {"name": "slides_get_presentation", "description": "Get Google Slides presentation metadata and slide list.", "parameters": {"type": "object", "properties": {"presentation_id": {"type": "string", "description": "The Google Slides presentation ID"}}, "required": ["presentation_id"]}}},
    {"type": "function", "function": {"name": "slides_get_slide", "description": "Get text content from a specific slide in a Google Slides presentation.", "parameters": {"type": "object", "properties": {"presentation_id": {"type": "string", "description": "The Google Slides presentation ID"}, "slide_index": {"type": "integer", "description": "Slide index (0-based). Default: 0"}}, "required": ["presentation_id"]}}},
    {"type": "function", "function": {"name": "slides_create_presentation", "description": "Create a new Google Slides presentation.", "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "Title for the new presentation"}, "folder_id": {"type": "string", "description": "Optional Drive folder ID to create inside"}}, "required": ["title"]}}},
    {"type": "function", "function": {"name": "slides_add_slide", "description": "Add a new slide to a Google Slides presentation.", "parameters": {"type": "object", "properties": {"presentation_id": {"type": "string", "description": "The Google Slides presentation ID"}, "layout": {"type": "string", "description": "Slide layout: BLANK, TITLE, TITLE_AND_BODY, TITLE_AND_TWO_COLUMNS, TITLE_ONLY, SECTION_HEADER. Default: BLANK"}}, "required": ["presentation_id"]}}},
    {"type": "function", "function": {"name": "slides_add_text", "description": "Add a text box to a slide in a Google Slides presentation.", "parameters": {"type": "object", "properties": {"presentation_id": {"type": "string", "description": "The Google Slides presentation ID"}, "slide_id": {"type": "string", "description": "The slide object ID to add text to"}, "text": {"type": "string", "description": "The text content to add"}, "x": {"type": "number", "description": "X position in points (default 100)"}, "y": {"type": "number", "description": "Y position in points (default 100)"}, "width": {"type": "number", "description": "Text box width in points (default 400)"}, "height": {"type": "number", "description": "Text box height in points (default 300)"}}, "required": ["presentation_id", "slide_id", "text"]}}},
    # Google Contacts tools
    {"type": "function", "function": {"name": "contacts_list", "description": "List Google Contacts. Optionally search by name or email.", "parameters": {"type": "object", "properties": {"max_results": {"type": "integer", "description": "Maximum contacts to return (default 20)"}, "query": {"type": "string", "description": "Search query to filter contacts"}}, "required": []}}},
    {"type": "function", "function": {"name": "contacts_get", "description": "Get detailed info for a single Google Contact.", "parameters": {"type": "object", "properties": {"resource_name": {"type": "string", "description": "Contact resource name (e.g., 'people/c1234567890')"}}, "required": ["resource_name"]}}},
    {"type": "function", "function": {"name": "contacts_create", "description": "Create a new Google Contact.", "parameters": {"type": "object", "properties": {"given_name": {"type": "string", "description": "First name"}, "family_name": {"type": "string", "description": "Last name"}, "email": {"type": "string", "description": "Email address"}, "phone": {"type": "string", "description": "Phone number"}, "organization": {"type": "string", "description": "Company/organization name"}}, "required": ["given_name"]}}},
    {"type": "function", "function": {"name": "contacts_update", "description": "Update an existing Google Contact.", "parameters": {"type": "object", "properties": {"resource_name": {"type": "string", "description": "Contact resource name (e.g., 'people/c1234567890')"}, "given_name": {"type": "string", "description": "New first name"}, "family_name": {"type": "string", "description": "New last name"}, "email": {"type": "string", "description": "New email address"}, "phone": {"type": "string", "description": "New phone number"}}, "required": ["resource_name"]}}},
    {"type": "function", "function": {"name": "contacts_delete", "description": "Delete a Google Contact.", "parameters": {"type": "object", "properties": {"resource_name": {"type": "string", "description": "Contact resource name (e.g., 'people/c1234567890')"}}, "required": ["resource_name"]}}},
    {"type": "function", "function": {"name": "contacts_search", "description": "Search Google Contacts by name or email.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}}},
    # YouTube tools
    {"type": "function", "function": {"name": "youtube_search", "description": "Search YouTube for videos, channels, or playlists.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}, "max_results": {"type": "integer", "description": "Maximum results (default 5)"}, "video_type": {"type": "string", "description": "Type: 'video', 'channel', or 'playlist'. Default: 'video'"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "youtube_get_video", "description": "Get detailed info about a YouTube video (title, stats, description, duration).", "parameters": {"type": "object", "properties": {"video_id": {"type": "string", "description": "The YouTube video ID"}}, "required": ["video_id"]}}},
    {"type": "function", "function": {"name": "youtube_get_channel", "description": "Get YouTube channel info and stats.", "parameters": {"type": "object", "properties": {"channel_id": {"type": "string", "description": "The YouTube channel ID"}}, "required": ["channel_id"]}}},
    {"type": "function", "function": {"name": "youtube_list_playlists", "description": "List YouTube playlists. If no channel_id, lists your own playlists.", "parameters": {"type": "object", "properties": {"channel_id": {"type": "string", "description": "Channel ID (omit for your own playlists)"}, "max_results": {"type": "integer", "description": "Maximum results (default 10)"}}, "required": []}}},
    {"type": "function", "function": {"name": "youtube_get_playlist_items", "description": "List videos in a YouTube playlist.", "parameters": {"type": "object", "properties": {"playlist_id": {"type": "string", "description": "The playlist ID"}, "max_results": {"type": "integer", "description": "Maximum results (default 20)"}}, "required": ["playlist_id"]}}},
    {"type": "function", "function": {"name": "youtube_create_playlist", "description": "Create a new YouTube playlist.", "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "Playlist title"}, "description": {"type": "string", "description": "Playlist description"}, "privacy": {"type": "string", "description": "Privacy: 'public', 'private', or 'unlisted'. Default: 'private'"}}, "required": ["title"]}}},
    {"type": "function", "function": {"name": "youtube_add_to_playlist", "description": "Add a video to a YouTube playlist.", "parameters": {"type": "object", "properties": {"playlist_id": {"type": "string", "description": "The playlist ID"}, "video_id": {"type": "string", "description": "The video ID to add"}}, "required": ["playlist_id", "video_id"]}}},
    {"type": "function", "function": {"name": "youtube_get_comments", "description": "Get comments on a YouTube video.", "parameters": {"type": "object", "properties": {"video_id": {"type": "string", "description": "The video ID"}, "max_results": {"type": "integer", "description": "Maximum comments (default 10)"}}, "required": ["video_id"]}}},
    {"type": "function", "function": {"name": "youtube_post_comment", "description": "Post a comment on a YouTube video.", "parameters": {"type": "object", "properties": {"video_id": {"type": "string", "description": "The video ID to comment on"}, "text": {"type": "string", "description": "Comment text"}}, "required": ["video_id", "text"]}}},
    {"type": "function", "function": {"name": "youtube_my_channel", "description": "Get your own YouTube channel info and stats.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    # Substack tools
    {"type": "function", "function": {"name": "substack_create_draft", "description": "Create a draft post on Substack newsletter.", "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "Post title"}, "body": {"type": "string", "description": "Post body text (plain text, double newlines for paragraphs)"}, "subtitle": {"type": "string", "description": "Post subtitle"}, "audience": {"type": "string", "description": "Audience: 'everyone' (free) or 'only_paid'. Default: 'everyone'"}}, "required": ["title", "body"]}}},
    {"type": "function", "function": {"name": "substack_list_drafts", "description": "List draft posts on Substack.", "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "description": "Maximum drafts to return (default 10)"}}, "required": []}}},
    {"type": "function", "function": {"name": "substack_list_posts", "description": "List published Substack posts.", "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "description": "Maximum posts to return (default 10)"}, "offset": {"type": "integer", "description": "Offset for pagination (default 0)"}}, "required": []}}},
    {"type": "function", "function": {"name": "substack_get_post", "description": "Get a specific Substack post or draft by ID.", "parameters": {"type": "object", "properties": {"post_id": {"type": "string", "description": "The post or draft ID"}}, "required": ["post_id"]}}},
    {"type": "function", "function": {"name": "substack_update_draft", "description": "Update an existing Substack draft.", "parameters": {"type": "object", "properties": {"draft_id": {"type": "string", "description": "The draft ID to update"}, "title": {"type": "string", "description": "New title"}, "body": {"type": "string", "description": "New body text"}, "subtitle": {"type": "string", "description": "New subtitle"}}, "required": ["draft_id"]}}},
    {"type": "function", "function": {"name": "substack_delete_draft", "description": "Delete a Substack draft.", "parameters": {"type": "object", "properties": {"draft_id": {"type": "string", "description": "The draft ID to delete"}}, "required": ["draft_id"]}}},
    {"type": "function", "function": {"name": "substack_get_stats", "description": "Get Substack publication statistics (subscribers, posts, etc.).", "parameters": {"type": "object", "properties": {}, "required": []}}},
    # Shared Folder / File Transfer Tools
    {"type": "function", "function": {"name": "list_shared", "description": "List files in the shared folder. Files placed here are synced to connected clients automatically. To remove files, use run_command with rm.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "share_file", "description": "Copy a file into the shared folder so the client can pull/download it. Use this when the user says 'send this to the client' or 'share this file'.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Path to the file to share"}, "filename": {"type": "string", "description": "Optional name for the file in shared folder (defaults to original name)"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "list_transfers", "description": "List files uploaded by the client (in the transfers folder). Use when user asks 'what did the client send' or 'check uploads'.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_transfer", "description": "Copy a file from the transfers folder (client uploads) to the working directory. Use when user says 'pull that file' or 'grab the upload'.", "parameters": {"type": "object", "properties": {"filename": {"type": "string", "description": "Name of the file in transfers folder"}, "destination": {"type": "string", "description": "Where to copy it (defaults to working directory)"}}, "required": ["filename"]}}},
]

# Add browser automation tools
try:
    from browser import get_browser_tool_definitions
    TOOLS.extend(get_browser_tool_definitions())
except ImportError:
    pass  # Playwright not installed

# Add skill-based tools (social media, etc.)
try:
    from skills import get_skill_manager
    _skill_mgr = get_skill_manager()
    TOOLS.extend(_skill_mgr.get_tool_definitions())
except Exception:
    pass

# Register collaboration tools
try:
    from collab_tools import COLLAB_TOOLS
    TOOLS.extend(COLLAB_TOOLS)
except ImportError:
    pass
