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
                    "end_line": {"type": "integer", "description": "Last line to read (1-indexed, optional)"},
                },
                "required": ["path"],
            },
        },
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
                    "content": {"type": "string", "description": "Content to write to the file"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories on the DGX Spark SERVER (Linux). Cannot access the user's Mac/iPhone. Shows file sizes and types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path on the server to list (defaults to working directory)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_working_directory",
            "description": "Get the current working directory path.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "change_directory",
            "description": "Change the current working directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path to change to"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command. Use for: pip, python, git, rm, mv, cp, etc. Do NOT use rm to remove shared folder files — use the remove_shared tool instead so deletions sync to clients.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The shell command to execute"}},
                "required": ["command"],
            },
        },
    },
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
            "description": "Search the web using DuckDuckGo. Use this for weather, news, prices, current events, or any factual query that needs up-to-date information.",
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
            "name": "generate_image",
            "description": "Generate an image using Flux 1 Dev via local ComfyUI. This is MAUDE's default image path. Saves to shared folder and returns a download URL. Use markdown ![desc](/download/filename.png) to display.",
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
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image_flux2",
            "description": "Generate an image using Flux 2 via Replicate (cloud, paid). Only use when the user explicitly asks for Flux 2 / flux2. Do not use as a fallback for local ComfyUI failures. Saves to shared folder and returns a download URL. Use markdown ![desc](/download/filename.png) to display.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Text description of the image to generate"},
                    "model": {"type": "string", "enum": ["pro", "dev", "klein"], "description": "Flux 2 variant: pro (best quality), dev (open weights), klein (cheapest). Default: pro"},
                    "aspect_ratio": {"type": "string", "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"], "description": "Output aspect ratio. Default: 1:1"},
                    "seed": {"type": "integer", "description": "Seed for reproducibility (-1 for random)"},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_3d",
            "description": "Generate a 3D GLB model from an image using TRELLIS 2 (local, free) or Tripo (cloud, better topology). Returns file path to the GLB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to the input image file"},
                    "engine": {"type": "string", "enum": ["trellis", "tripo", "both"], "description": "Which engine to use: trellis (free, local), tripo (cloud, costs credits), or both in parallel"},
                    "resolution": {"type": "string", "enum": ["512", "1024", "1536"], "description": "TRELLIS resolution (default 1024)"},
                    "seed": {"type": "integer", "description": "Random seed for TRELLIS generation"},
                    "quad": {"type": "boolean", "description": "Enable quad remeshing for Tripo (cleaner topology)"},
                    "face_limit": {"type": "integer", "description": "Target face count for Tripo (e.g. 50000, 100000)"},
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_image_search",
            "description": "Search the web for images. Returns image URLs with markdown display syntax. Use when the user wants to find pictures, photos, or images of something.",
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
    {
        "type": "function",
        "function": {
            "name": "hyperframes_doctor",
            "description": "Run HyperFrames diagnostics for Node.js, FFmpeg, Chrome, Docker, and render readiness. Use before first HyperFrames render or when video rendering fails.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hyperframes_browser_ensure",
            "description": "Install or verify the managed Chrome browser required by HyperFrames local rendering. Use when hyperframes_doctor reports Chrome is missing.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hyperframes_init",
            "description": "Create a new HyperFrames HTML-to-video project under data/hyperframes using the HyperFrames CLI. Use when starting a new programmatic video composition.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Project name, e.g. product-intro"},
                    "example": {"type": "string", "description": "Built-in HyperFrames example/template name. Default: blank"},
                    "tailwind": {"type": "boolean", "description": "Create a Tailwind-enabled project if supported by the CLI."},
                    "install_skills": {"type": "boolean", "description": "Allow HyperFrames to install its AI coding skills during init. Default: false"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hyperframes_lint",
            "description": "Lint a HyperFrames project for missing timing attributes, adapter libraries, media issues, and other structural problems before rendering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_path": {"type": "string", "description": "Path to the HyperFrames project directory"},
                    "verbose": {"type": "boolean", "description": "Include informational lint findings."},
                },
                "required": ["project_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hyperframes_render",
            "description": "Render a HyperFrames project to MP4, MOV, or WebM via the HyperFrames CLI. Can share the finished video through Maude's shared download folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_path": {"type": "string", "description": "Path to the HyperFrames project directory"},
                    "output": {"type": "string", "description": "Optional output path. Defaults to project/renders/<project>-timestamp.<format>"},
                    "format": {"type": "string", "enum": ["mp4", "mov", "webm"], "description": "Output format. Default: mp4"},
                    "fps": {"type": "integer", "enum": [24, 30, 60], "description": "Frame rate. Default: 30"},
                    "quality": {"type": "string", "enum": ["draft", "standard", "high"], "description": "Encoding quality. Use draft for quick previews, high for final delivery."},
                    "docker": {"type": "boolean", "description": "Use Docker mode for deterministic rendering."},
                    "gpu": {"type": "boolean", "description": "Use hardware video encoding when available."},
                    "workers": {"type": "string", "description": "Worker count, e.g. auto, 1, 2, 4. Default: auto"},
                    "share": {"type": "boolean", "description": "Copy the rendered video to Maude's shared download folder. Default: true"},
                    "timeout": {"type": "integer", "description": "Render timeout in seconds. Default: 900"},
                },
                "required": ["project_path"],
            },
        },
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
                    "pattern": {"type": "string", "description": "Text to search for"},
                },
                "required": ["path", "pattern"],
            },
        },
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
                    "pattern": {"type": "string", "description": "Text to search for"},
                },
                "required": ["directory", "pattern"],
            },
        },
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
                    "new_content": {"type": "string", "description": "New content to insert"},
                },
                "required": ["path", "start_line", "end_line", "new_content"],
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
                    "provider": {"type": "string", "description": "Optional: claude, openai, gemini, grok, mistral"},
                },
                "required": ["question"],
            },
        },
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
    # Google Tools
    {
        "type": "function",
        "function": {
            "name": "gmail_list",
            "description": "List recent emails from Gmail. Use query for searching (same syntax as Gmail search).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'from:someone@example.com', 'subject:invoice', 'is:unread')",
                    },
                    "max_results": {"type": "integer", "description": "Maximum emails to return (default 10)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gmail_read",
            "description": "Read a specific email by its message ID.",
            "parameters": {
                "type": "object",
                "properties": {"message_id": {"type": "string", "description": "The Gmail message ID"}},
                "required": ["message_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gmail_send",
            "description": "Send an email via Gmail.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body text"},
                    "cc": {"type": "string", "description": "CC recipients (optional)"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drive_list",
            "description": "List files in Google Drive. Use query for filtering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Drive query (e.g., \"name contains 'report'\")"},
                    "max_results": {"type": "integer", "description": "Maximum files to return (default 20)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drive_search",
            "description": "Search Google Drive for files by name or content.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search term"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drive_read",
            "description": "Read the contents of a file from Google Drive (text files, Google Docs, etc.).",
            "parameters": {
                "type": "object",
                "properties": {"file_id": {"type": "string", "description": "The Google Drive file ID"}},
                "required": ["file_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drive_upload",
            "description": "Upload a local file to Google Drive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "local_path": {"type": "string", "description": "Path to the local file to upload"},
                    "folder_id": {"type": "string", "description": "Optional Drive folder ID to upload into"},
                },
                "required": ["local_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drive_create_folder",
            "description": "Create a new folder in Google Drive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name for the new folder"},
                    "parent_id": {"type": "string", "description": "Optional parent folder ID to create inside"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drive_create_doc",
            "description": "Create a new Google Doc in Google Drive. Use folder_name to place it in a folder by name (auto-resolves ID, creates folder if needed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name for the new document"},
                    "folder_name": {
                        "type": "string",
                        "description": "Folder name to create inside (e.g. 'maude') — resolved automatically",
                    },
                    "folder_id": {
                        "type": "string",
                        "description": "Folder ID to create inside (use folder_name instead if you only know the name)",
                    },
                    "content": {"type": "string", "description": "Optional initial content for the document"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drive_create_sheet",
            "description": "Create a new Google Sheet in Google Drive. Use folder_name to place it in a folder by name (auto-resolves ID, creates folder if needed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name for the new spreadsheet"},
                    "folder_name": {
                        "type": "string",
                        "description": "Folder name to create inside (e.g. 'maude') — resolved automatically",
                    },
                    "folder_id": {
                        "type": "string",
                        "description": "Folder ID to create inside (use folder_name instead if you only know the name)",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drive_update_doc",
            "description": "Write or append content to an existing Google Doc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "The Google Doc ID"},
                    "content": {"type": "string", "description": "The content to write to the document"},
                    "append": {
                        "type": "boolean",
                        "description": "If true, append to existing content. If false (default), replace all content.",
                    },
                },
                "required": ["doc_id", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drive_delete",
            "description": "Delete a file or folder from Google Drive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {"type": "string", "description": "The Google Drive file or folder ID to delete"}
                },
                "required": ["file_id"],
            },
        },
    },
    # Google Sheets tools
    {
        "type": "function",
        "function": {
            "name": "sheets_read",
            "description": "Read data from a Google Sheets spreadsheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "spreadsheet_id": {"type": "string", "description": "The Google Sheets spreadsheet ID"},
                    "range": {
                        "type": "string",
                        "description": "Cell range to read (e.g., 'Sheet1!A1:D10'). Default: 'Sheet1'",
                    },
                },
                "required": ["spreadsheet_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sheets_write",
            "description": "Write data to a Google Sheets spreadsheet (overwrites existing data in range).",
            "parameters": {
                "type": "object",
                "properties": {
                    "spreadsheet_id": {"type": "string", "description": "The Google Sheets spreadsheet ID"},
                    "range": {"type": "string", "description": "Cell range to write to (e.g., 'Sheet1!A1')"},
                    "values": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "string"}},
                        "description": "2D array of values (rows of columns)",
                    },
                },
                "required": ["spreadsheet_id", "range", "values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sheets_append",
            "description": "Append rows to a Google Sheets spreadsheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "spreadsheet_id": {"type": "string", "description": "The Google Sheets spreadsheet ID"},
                    "range": {"type": "string", "description": "Range to append after (e.g., 'Sheet1!A1')"},
                    "values": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "string"}},
                        "description": "2D array of rows to append",
                    },
                },
                "required": ["spreadsheet_id", "range", "values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sheets_create",
            "description": "Create a new Google Sheets spreadsheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Name for the new spreadsheet"},
                    "folder_id": {"type": "string", "description": "Optional Drive folder ID to create inside"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sheets_list_sheets",
            "description": "List all sheet tabs in a Google Sheets spreadsheet.",
            "parameters": {
                "type": "object",
                "properties": {"spreadsheet_id": {"type": "string", "description": "The Google Sheets spreadsheet ID"}},
                "required": ["spreadsheet_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sheets_clear",
            "description": "Clear a range of cells in a Google Sheets spreadsheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "spreadsheet_id": {"type": "string", "description": "The Google Sheets spreadsheet ID"},
                    "range": {"type": "string", "description": "Cell range to clear (e.g., 'Sheet1!A1:D10')"},
                },
                "required": ["spreadsheet_id", "range"],
            },
        },
    },
    # Google Calendar tools
    {
        "type": "function",
        "function": {
            "name": "calendar_list_events",
            "description": "List upcoming Google Calendar events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_results": {"type": "integer", "description": "Maximum events to return (default 10)"},
                    "time_min": {
                        "type": "string",
                        "description": "Start time filter (ISO format, e.g., '2025-01-15T00:00:00Z'). Default: now",
                    },
                    "time_max": {"type": "string", "description": "End time filter (ISO format)"},
                    "calendar_id": {"type": "string", "description": "Calendar ID (default: 'primary')"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_create_event",
            "description": "Create a new Google Calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Event title"},
                    "start": {
                        "type": "string",
                        "description": "Start time (ISO format, e.g., '2025-01-15T10:00:00-05:00')",
                    },
                    "end": {"type": "string", "description": "End time (ISO format)"},
                    "description": {"type": "string", "description": "Event description"},
                    "location": {"type": "string", "description": "Event location"},
                    "calendar_id": {"type": "string", "description": "Calendar ID (default: 'primary')"},
                },
                "required": ["summary", "start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_update_event",
            "description": "Update an existing Google Calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "The event ID to update"},
                    "summary": {"type": "string", "description": "New event title"},
                    "start": {"type": "string", "description": "New start time (ISO format)"},
                    "end": {"type": "string", "description": "New end time (ISO format)"},
                    "description": {"type": "string", "description": "New description"},
                    "location": {"type": "string", "description": "New location"},
                    "calendar_id": {"type": "string", "description": "Calendar ID (default: 'primary')"},
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_delete_event",
            "description": "Delete a Google Calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "The event ID to delete"},
                    "calendar_id": {"type": "string", "description": "Calendar ID (default: 'primary')"},
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_search_events",
            "description": "Search Google Calendar events by text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search text"},
                    "max_results": {"type": "integer", "description": "Maximum events to return (default 10)"},
                    "calendar_id": {"type": "string", "description": "Calendar ID (default: 'primary')"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_list_calendars",
            "description": "List all available Google Calendars.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # Google Slides tools
    {
        "type": "function",
        "function": {
            "name": "slides_get_presentation",
            "description": "Get Google Slides presentation metadata and slide list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "presentation_id": {"type": "string", "description": "The Google Slides presentation ID"}
                },
                "required": ["presentation_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "slides_get_slide",
            "description": "Get text content from a specific slide in a Google Slides presentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "presentation_id": {"type": "string", "description": "The Google Slides presentation ID"},
                    "slide_index": {"type": "integer", "description": "Slide index (0-based). Default: 0"},
                },
                "required": ["presentation_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "slides_create_presentation",
            "description": "Create a new Google Slides presentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Title for the new presentation"},
                    "folder_id": {"type": "string", "description": "Optional Drive folder ID to create inside"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "slides_add_slide",
            "description": "Add a new slide to a Google Slides presentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "presentation_id": {"type": "string", "description": "The Google Slides presentation ID"},
                    "layout": {
                        "type": "string",
                        "description": "Slide layout: BLANK, TITLE, TITLE_AND_BODY, TITLE_AND_TWO_COLUMNS, TITLE_ONLY, SECTION_HEADER. Default: BLANK",
                    },
                },
                "required": ["presentation_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "slides_add_text",
            "description": "Add a text box to a slide in a Google Slides presentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "presentation_id": {"type": "string", "description": "The Google Slides presentation ID"},
                    "slide_id": {"type": "string", "description": "The slide object ID to add text to"},
                    "text": {"type": "string", "description": "The text content to add"},
                    "x": {"type": "number", "description": "X position in points (default 100)"},
                    "y": {"type": "number", "description": "Y position in points (default 100)"},
                    "width": {"type": "number", "description": "Text box width in points (default 400)"},
                    "height": {"type": "number", "description": "Text box height in points (default 300)"},
                },
                "required": ["presentation_id", "slide_id", "text"],
            },
        },
    },
    # Google Contacts tools
    {
        "type": "function",
        "function": {
            "name": "contacts_list",
            "description": "List Google Contacts. Optionally search by name or email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_results": {"type": "integer", "description": "Maximum contacts to return (default 20)"},
                    "query": {"type": "string", "description": "Search query to filter contacts"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "contacts_get",
            "description": "Get detailed info for a single Google Contact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_name": {
                        "type": "string",
                        "description": "Contact resource name (e.g., 'people/c1234567890')",
                    }
                },
                "required": ["resource_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "contacts_create",
            "description": "Create a new Google Contact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "given_name": {"type": "string", "description": "First name"},
                    "family_name": {"type": "string", "description": "Last name"},
                    "email": {"type": "string", "description": "Email address"},
                    "phone": {"type": "string", "description": "Phone number"},
                    "organization": {"type": "string", "description": "Company/organization name"},
                },
                "required": ["given_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "contacts_update",
            "description": "Update an existing Google Contact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_name": {
                        "type": "string",
                        "description": "Contact resource name (e.g., 'people/c1234567890')",
                    },
                    "given_name": {"type": "string", "description": "New first name"},
                    "family_name": {"type": "string", "description": "New last name"},
                    "email": {"type": "string", "description": "New email address"},
                    "phone": {"type": "string", "description": "New phone number"},
                },
                "required": ["resource_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "contacts_delete",
            "description": "Delete a Google Contact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_name": {
                        "type": "string",
                        "description": "Contact resource name (e.g., 'people/c1234567890')",
                    }
                },
                "required": ["resource_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "contacts_search",
            "description": "Search Google Contacts by name or email.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    # YouTube tools
    {
        "type": "function",
        "function": {
            "name": "youtube_search",
            "description": "Search YouTube for videos, channels, or playlists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Maximum results (default 5)"},
                    "video_type": {
                        "type": "string",
                        "description": "Type: 'video', 'channel', or 'playlist'. Default: 'video'",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_get_video",
            "description": "Get detailed info about a YouTube video (title, stats, description, duration).",
            "parameters": {
                "type": "object",
                "properties": {"video_id": {"type": "string", "description": "The YouTube video ID"}},
                "required": ["video_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_get_channel",
            "description": "Get YouTube channel info and stats.",
            "parameters": {
                "type": "object",
                "properties": {"channel_id": {"type": "string", "description": "The YouTube channel ID"}},
                "required": ["channel_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_list_playlists",
            "description": "List YouTube playlists. If no channel_id, lists your own playlists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel_id": {"type": "string", "description": "Channel ID (omit for your own playlists)"},
                    "max_results": {"type": "integer", "description": "Maximum results (default 10)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_get_playlist_items",
            "description": "List videos in a YouTube playlist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_id": {"type": "string", "description": "The playlist ID"},
                    "max_results": {"type": "integer", "description": "Maximum results (default 20)"},
                },
                "required": ["playlist_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_create_playlist",
            "description": "Create a new YouTube playlist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Playlist title"},
                    "description": {"type": "string", "description": "Playlist description"},
                    "privacy": {
                        "type": "string",
                        "description": "Privacy: 'public', 'private', or 'unlisted'. Default: 'private'",
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_add_to_playlist",
            "description": "Add a video to a YouTube playlist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_id": {"type": "string", "description": "The playlist ID"},
                    "video_id": {"type": "string", "description": "The video ID to add"},
                },
                "required": ["playlist_id", "video_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_get_comments",
            "description": "Get comments on a YouTube video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {"type": "string", "description": "The video ID"},
                    "max_results": {"type": "integer", "description": "Maximum comments (default 10)"},
                },
                "required": ["video_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_post_comment",
            "description": "Post a comment on a YouTube video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {"type": "string", "description": "The video ID to comment on"},
                    "text": {"type": "string", "description": "Comment text"},
                },
                "required": ["video_id", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_upload",
            "description": "Upload a video to YouTube. Defaults to public. Can set thumbnail and add to playlist in one call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the video file"},
                    "title": {"type": "string", "description": "Video title"},
                    "description": {"type": "string", "description": "Video description"},
                    "tags": {"type": "string", "description": "Comma-separated tags"},
                    "privacy": {
                        "type": "string",
                        "description": "Privacy: 'public', 'private', or 'unlisted'. Default: 'public'",
                    },
                    "category": {
                        "type": "string",
                        "description": "YouTube category ID. Default: '22' (People & Blogs). Common: '24'=Entertainment, '28'=Science & Tech, '10'=Music",
                    },
                    "thumbnail_path": {"type": "string", "description": "Path to custom thumbnail image"},
                    "playlist_id": {"type": "string", "description": "Playlist ID to add the video to after upload"},
                },
                "required": ["file_path", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_my_channel",
            "description": "Get your own YouTube channel info and stats.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # Substack tools
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
    # GitHub Tools — Pull Requests
    {
        "type": "function",
        "function": {
            "name": "github_list_prs",
            "description": "List pull requests for a GitHub repository. Defaults to the repo in the current directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository in owner/repo format (optional, defaults to current repo)",
                    },
                    "state": {
                        "type": "string",
                        "description": "Filter by state: open, closed, merged, all (default: open)",
                    },
                    "limit": {"type": "integer", "description": "Maximum PRs to return (default 10)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_view_pr",
            "description": "View details of a specific pull request including status checks, review state, and description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer", "description": "The pull request number"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                },
                "required": ["pr_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_create_pr",
            "description": "Create a new pull request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "PR title"},
                    "body": {"type": "string", "description": "PR description/body"},
                    "base": {
                        "type": "string",
                        "description": "Base branch to merge into (default: repo default branch)",
                    },
                    "head": {"type": "string", "description": "Head branch with changes (default: current branch)"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                    "draft": {"type": "boolean", "description": "Create as draft PR (default: false)"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_merge_pr",
            "description": "Merge a pull request. Supports merge, squash, or rebase strategies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer", "description": "The pull request number to merge"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                    "method": {
                        "type": "string",
                        "description": "Merge method: merge, squash, or rebase (default: merge)",
                    },
                    "delete_branch": {
                        "type": "boolean",
                        "description": "Delete the branch after merging (default: true)",
                    },
                },
                "required": ["pr_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_close_pr",
            "description": "Close a pull request without merging.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer", "description": "The pull request number"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                    "comment": {"type": "string", "description": "Optional comment to leave before closing"},
                },
                "required": ["pr_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_pr_diff",
            "description": "View the diff/changes of a pull request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer", "description": "The pull request number"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                },
                "required": ["pr_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_pr_comments",
            "description": "List comments on a pull request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer", "description": "The pull request number"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                },
                "required": ["pr_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_comment_pr",
            "description": "Add a comment to a pull request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer", "description": "The pull request number"},
                    "body": {"type": "string", "description": "Comment text"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                },
                "required": ["pr_number", "body"],
            },
        },
    },
    # GitHub Tools — Issues
    {
        "type": "function",
        "function": {
            "name": "github_list_issues",
            "description": "List issues for a GitHub repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                    "state": {"type": "string", "description": "Filter by state: open, closed, all (default: open)"},
                    "labels": {"type": "string", "description": "Filter by label name"},
                    "limit": {"type": "integer", "description": "Maximum issues to return (default 10)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_view_issue",
            "description": "View details of a specific issue including comments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_number": {"type": "integer", "description": "The issue number"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                },
                "required": ["issue_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_create_issue",
            "description": "Create a new issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Issue title"},
                    "body": {"type": "string", "description": "Issue description"},
                    "labels": {"type": "string", "description": "Comma-separated label names"},
                    "assignee": {"type": "string", "description": "GitHub username to assign"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_close_issue",
            "description": "Close an issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_number": {"type": "integer", "description": "The issue number"},
                    "comment": {"type": "string", "description": "Optional comment to leave before closing"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                },
                "required": ["issue_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_comment_issue",
            "description": "Add a comment to an issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_number": {"type": "integer", "description": "The issue number"},
                    "body": {"type": "string", "description": "Comment text"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                },
                "required": ["issue_number", "body"],
            },
        },
    },
    # GitHub Tools — Repos, Branches, Commits
    {
        "type": "function",
        "function": {
            "name": "github_list_repos",
            "description": "List repositories for a user/org, or your own repos if no owner specified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "GitHub username or org (optional, defaults to authenticated user)",
                    },
                    "limit": {"type": "integer", "description": "Maximum repos to return (default 10)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_view_repo",
            "description": "View detailed information about a repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository in owner/repo format (optional, defaults to current repo)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_list_branches",
            "description": "List branches in a repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                    "limit": {"type": "integer", "description": "Maximum branches to show (default 20)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_list_commits",
            "description": "List recent commits in a repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                    "branch": {"type": "string", "description": "Branch name (default: default branch)"},
                    "limit": {"type": "integer", "description": "Maximum commits to return (default 10)"},
                },
                "required": [],
            },
        },
    },
    # GitHub Tools — CI/CD & Releases
    {
        "type": "function",
        "function": {
            "name": "github_list_runs",
            "description": "List recent GitHub Actions workflow runs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                    "limit": {"type": "integer", "description": "Maximum runs to return (default 10)"},
                    "status": {
                        "type": "string",
                        "description": "Filter by status: queued, in_progress, completed, failure, success",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_view_run",
            "description": "View details of a specific workflow run including job results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "integer", "description": "The workflow run ID"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                },
                "required": ["run_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_rerun",
            "description": "Re-run a GitHub Actions workflow run.",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "integer", "description": "The workflow run ID to re-run"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                    "failed_only": {"type": "boolean", "description": "Only re-run failed jobs (default: false)"},
                },
                "required": ["run_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_list_releases",
            "description": "List releases for a repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                    "limit": {"type": "integer", "description": "Maximum releases to return (default 5)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_create_release",
            "description": "Create a new GitHub release with a tag.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tag": {"type": "string", "description": "Tag name for the release (e.g. v1.0.0)"},
                    "title": {"type": "string", "description": "Release title"},
                    "notes": {"type": "string", "description": "Release notes (auto-generated if omitted)"},
                    "draft": {"type": "boolean", "description": "Create as draft (default: false)"},
                    "prerelease": {"type": "boolean", "description": "Mark as pre-release (default: false)"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                },
                "required": ["tag"],
            },
        },
    },
    # GitHub Tools — Search & Notifications
    {
        "type": "function",
        "function": {
            "name": "github_search",
            "description": "Search GitHub for repositories, issues, pull requests, or code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "type": {
                        "type": "string",
                        "description": "What to search: repos, issues, prs, or code (default: repos)",
                        "enum": ["repos", "issues", "prs", "code"],
                    },
                    "limit": {"type": "integer", "description": "Maximum results (default 10)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_notifications",
            "description": "List unread GitHub notifications.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "Maximum notifications (default 10)"}},
                "required": [],
            },
        },
    },
    # Memory Tools
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save a piece of information to persistent memory. Use this proactively when the user shares facts, preferences, or context you should remember across conversations. Categories: 'fact', 'preference', 'person', 'task'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Short unique identifier for the memory (e.g. 'favorite_language', 'project_deadline', 'wife_name')",
                    },
                    "value": {"type": "string", "description": "The information to remember"},
                    "category": {
                        "type": "string",
                        "description": "Memory category",
                        "enum": ["fact", "preference", "person", "task"],
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
            "description": "Search persistent memory for relevant information. Use when the user references something from a previous conversation, or when you need context about the user, their preferences, or past interactions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for in memory"},
                    "category": {
                        "type": "string",
                        "description": "Optional category filter",
                        "enum": ["fact", "preference", "person", "task", "conversation"],
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
                        "description": "Filter by category",
                        "enum": ["fact", "preference", "person", "task", "conversation"],
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
    # MemPalace — layered long-term memory (L0 identity / L1 essential / L2 on-demand / L3 deep search)
    {
        "type": "function",
        "function": {
            "name": "palace_search",
            "description": "Semantic search over MemPalace drawers (L3 deep search). Use for general-purpose recall when key/value memory doesn't find what you need. Returns ranked snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "wing": {"type": "string", "description": "Optional wing name to narrow the search"},
                    "room": {"type": "string", "description": "Optional room name to narrow the search"},
                    "n_results": {"type": "integer", "description": "Max results (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "palace_recall",
            "description": "Browse MemPalace drawers by wing/room without a query (L2 on-demand retrieval). Use when you want to see everything in a topic area.",
            "parameters": {
                "type": "object",
                "properties": {
                    "wing": {"type": "string", "description": "Wing name"},
                    "room": {"type": "string", "description": "Room name"},
                    "n_results": {"type": "integer", "description": "Max results (default 10)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "palace_status",
            "description": "Show MemPalace stats: palace path, total drawers, and whether L0 identity is configured.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "palace_kg_add_fact",
            "description": "Record a structured fact in the MemPalace knowledge graph as a (subject, predicate, object) triple. Use for durable relational facts like 'Matt works_at NVIDIA' that should survive rewrites.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "Entity the fact is about"},
                    "predicate": {"type": "string", "description": "Relationship (e.g. 'works_at', 'owns', 'prefers')"},
                    "object": {"type": "string", "description": "Target of the relationship"},
                    "confidence": {"type": "number", "description": "0.0–1.0 confidence (default 1.0)"},
                },
                "required": ["subject", "predicate", "object"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "palace_kg_query",
            "description": "Query the MemPalace knowledge graph for all facts involving an entity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "Entity name to look up"},
                    "direction": {
                        "type": "string",
                        "description": "Relationship direction (default 'outgoing')",
                        "enum": ["outgoing", "incoming", "both"],
                    },
                },
                "required": ["entity"],
            },
        },
    },
    # Shared Folder / File Transfer Tools
    {
        "type": "function",
        "function": {
            "name": "list_shared",
            "description": "List files in the shared folder. Files placed here are synced to connected clients automatically. To remove files, use the remove_shared tool (NOT rm — rm won't propagate the deletion to clients).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "share_file",
            "description": "Copy a file into the shared folder so the client can pull/download it. Use this when the user says 'send this to the client' or 'share this file'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to share"},
                    "filename": {
                        "type": "string",
                        "description": "Optional name for the file in shared folder (defaults to original name)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_shared",
            "description": "Remove a file from the shared folder. ALWAYS use this instead of rm for shared folder files — it records the deletion so client sync removes the file locally too, preventing it from being re-synced back.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to remove from the shared folder",
                    },
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_transfers",
            "description": "List files uploaded by the client (in the transfers folder). Use when user asks 'what did the client send' or 'check uploads'.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_transfer",
            "description": "Copy a file from the transfers folder (client uploads) to the working directory. Use when user says 'pull that file' or 'grab the upload'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Name of the file in transfers folder"},
                    "destination": {
                        "type": "string",
                        "description": "Where to copy it (defaults to working directory)",
                    },
                },
                "required": ["filename"],
            },
        },
    },
    # Agent dispatch tools
    {
        "type": "function",
        "function": {
            "name": "run_agent",
            "description": """Dispatch a task to a specialized agent that has tool access (can search the web, read/write files, run commands).
Available agents:
- 'code': Code generation, debugging, refactoring (has file ops + shell)
- 'research': Multi-step web research, gathering info (has web + file read)
- 'writer': Documentation, long-form content (has file ops + web)
- 'reasoning': Complex analysis, planning (has file ops + web + shell)
- 'search': Quick web lookups (has web_search + web_browse)

Use this for tasks that require the agent to DO work (search, read, write) — not just answer from knowledge.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "enum": ["code", "research", "writer", "reasoning", "search"],
                        "description": "Which specialized agent to use",
                    },
                    "task": {"type": "string", "description": "Clear description of what the agent should do"},
                    "context": {
                        "type": "string",
                        "description": "Relevant context (code snippets, file paths, requirements)",
                    },
                },
                "required": ["agent", "task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_agents",
            "description": """Dispatch multiple tasks to agents in parallel. Each agent runs independently with its own tool access. Use this when you need to research/investigate multiple things at once, or split a large task across specialists.

Example: research two topics simultaneously, or have one agent write code while another researches docs.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "agent": {
                                    "type": "string",
                                    "enum": ["code", "research", "writer", "reasoning", "search"],
                                    "description": "Which agent to use",
                                },
                                "task": {"type": "string", "description": "What this agent should do"},
                                "context": {"type": "string", "description": "Optional context for this agent"},
                            },
                            "required": ["agent", "task"],
                        },
                        "description": "Array of agent tasks to run in parallel",
                    }
                },
                "required": ["tasks"],
            },
        },
    },
    # ── Planned Execution ──────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "execute_plan",
            "description": """Execute a multi-stage tool plan. Each stage's tools run in parallel; stages run sequentially. Use this when you need multiple tools across dependent steps — it collapses several round-trips into one call.

Reference earlier results in later stages with $N.M (stage N, tool index M). Example: $0.1 = result from stage 0, second tool.

Example plan — search 3 files in parallel, then edit based on findings:
stages: [
  [{"name": "search_file", "args": {"path": "src/main.py", "pattern": "TODO"}},
   {"name": "search_file", "args": {"path": "src/utils.py", "pattern": "TODO"}},
   {"name": "read_file", "args": {"path": "README.md"}}],
  [{"name": "run_command", "args": {"command": "grep -rn 'FIXME' src/"}}]
]""",
            "parameters": {
                "type": "object",
                "properties": {
                    "stages": {
                        "type": "array",
                        "description": "Ordered list of stages. Each stage is an array of tool calls that run in parallel.",
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "Tool name"},
                                    "args": {"type": "object", "description": "Tool arguments"},
                                },
                                "required": ["name", "args"],
                            },
                        },
                    }
                },
                "required": ["stages"],
            },
        },
    },
    # ── Command Center tools ─────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "system_stats",
            "description": "Get system stats: CPU, RAM, disk usage, GPU temperature/utilization/memory. Use when the user asks about system health, resource usage, or monitoring.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gpu_processes",
            "description": "Show what processes are currently using the GPU and how much VRAM each is consuming.",
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
    {
        "type": "function",
        "function": {
            "name": "session_list",
            "description": "List recent conversation sessions across all channels (CLI, Telegram, etc.) with message counts and timestamps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max sessions to return (default 20, max 50)"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "activity_feed",
            "description": "Show recent activity from the chat sync log — messages across all channels with timestamps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Number of recent activities (default 20, max 100)"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scheduler_status",
            "description": "Show all scheduled MAUDE tasks — cron schedules, run counts, last results, enabled/disabled status.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "node_status",
            "description": "Show connected nodes and service status — Spark services (gateway, llama, telegram), Tailscale peers, remote client heartbeats.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
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

# Register sandbox tools
try:
    from sandbox_manager import SANDBOX_TOOLS

    TOOLS.extend(SANDBOX_TOOLS)
except ImportError:
    pass

# Register forge tools
try:
    from forge import FORGE_TOOLS

    TOOLS.extend(FORGE_TOOLS)
except ImportError:
    pass

# Register browser workflow tools
try:
    from browser_workflows import WORKFLOW_TOOLS

    TOOLS.extend(WORKFLOW_TOOLS)
except ImportError:
    pass

# Register social media posting tools
try:
    from social_posting import SOCIAL_TOOLS

    TOOLS.extend(SOCIAL_TOOLS)
except ImportError:
    pass
