"""Domain-owned tool schemas."""

TOOL_NAMES = {
    "calendar_create_event",
    "calendar_delete_event",
    "calendar_list_calendars",
    "calendar_list_events",
    "calendar_search_events",
    "calendar_update_event",
    "contacts_create",
    "contacts_delete",
    "contacts_get",
    "contacts_list",
    "contacts_search",
    "contacts_update",
    "drive_create_doc",
    "drive_create_folder",
    "drive_create_sheet",
    "drive_delete",
    "drive_list",
    "drive_read",
    "drive_search",
    "drive_update_doc",
    "drive_upload",
    "gmail_list",
    "gmail_read",
    "gmail_send",
    "sheets_append",
    "sheets_clear",
    "sheets_create",
    "sheets_list_sheets",
    "sheets_read",
    "sheets_write",
    "slides_add_slide",
    "slides_add_text",
    "slides_create_presentation",
    "slides_get_presentation",
    "slides_get_slide",
    "youtube_add_to_playlist",
    "youtube_create_playlist",
    "youtube_get_channel",
    "youtube_get_comments",
    "youtube_get_playlist_items",
    "youtube_get_video",
    "youtube_list_playlists",
    "youtube_my_channel",
    "youtube_post_comment",
    "youtube_search",
    "youtube_upload",
}

SCHEMAS = [
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
                        "description": "Search query (e.g., 'from:someone@example.com', "
                        "'subject:invoice', 'is:unread')",
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
            "description": "Create a new Google Doc in Google Drive. Use folder_name to place it in a folder by "
            "name (auto-resolves ID, creates folder if needed).",
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
            "description": "Create a new Google Sheet in Google Drive. Use folder_name to place it in a folder by "
            "name (auto-resolves ID, creates folder if needed).",
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
                        "description": "Slide layout: BLANK, TITLE, TITLE_AND_BODY, "
                        "TITLE_AND_TWO_COLUMNS, TITLE_ONLY, "
                        "SECTION_HEADER. Default: BLANK",
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
            "description": "Upload a video to YouTube using MAUDE's configured Google OAuth credentials. Defaults "
            "to public. Can set thumbnail and add to playlist in one call. Do not require a YouTube "
            "API key in the shell environment.",
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
                        "description": "YouTube category ID. Default: '22' (People & "
                        "Blogs). Common: '24'=Entertainment, "
                        "'28'=Science & Tech, '10'=Music",
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
]


def schemas() -> list[dict]:
    return list(SCHEMAS)
