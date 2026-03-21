"""
Google tools — lazy-import registrations for Gmail, Drive, Sheets,
Calendar, Slides, Contacts, YouTube.
"""

from tool_registry import register_tool

# ── Gmail ──────────────────────────────────────────────────────


@register_tool("gmail_list")
def _dispatch_gmail_list(args):
    from google_tools import gmail_list_messages

    return gmail_list_messages(args.get("query", ""), args.get("max_results", 10))


@register_tool("gmail_read")
def _dispatch_gmail_read(args):
    from google_tools import gmail_read_message

    return gmail_read_message(args.get("message_id", ""))


@register_tool("gmail_send")
def _dispatch_gmail_send(args):
    from google_tools import gmail_send_message

    return gmail_send_message(args.get("to", ""), args.get("subject", ""), args.get("body", ""), args.get("cc"))


# ── Drive ──────────────────────────────────────────────────────


@register_tool("drive_list")
def _dispatch_drive_list(args):
    from google_tools import drive_list_files

    return drive_list_files(args.get("query", ""), args.get("max_results", 20))


@register_tool("drive_search")
def _dispatch_drive_search(args):
    from google_tools import drive_search

    return drive_search(args.get("query", ""))


@register_tool("drive_read")
def _dispatch_drive_read(args):
    from google_tools import drive_read_file

    return drive_read_file(args.get("file_id", ""))


@register_tool("drive_upload")
def _dispatch_drive_upload(args):
    from google_tools import drive_upload_file

    return drive_upload_file(args.get("local_path", ""), args.get("folder_id"))


@register_tool("drive_create_folder")
def _dispatch_drive_create_folder(args):
    from google_tools import drive_create_folder

    return drive_create_folder(args.get("name", ""), args.get("parent_id"))


@register_tool("drive_create_doc")
def _dispatch_drive_create_doc(args):
    from google_tools import drive_create_doc

    return drive_create_doc(
        args.get("name", ""), args.get("folder_id"), args.get("folder_name"), args.get("content", "")
    )


@register_tool("drive_create_sheet")
def _dispatch_drive_create_sheet(args):
    from google_tools import drive_create_sheet

    return drive_create_sheet(args.get("name", ""), args.get("folder_id"), args.get("folder_name"))


@register_tool("drive_update_doc")
def _dispatch_drive_update_doc(args):
    from google_tools import drive_update_doc

    return drive_update_doc(args.get("doc_id", ""), args.get("content", ""), args.get("append", False))


@register_tool("drive_delete")
def _dispatch_drive_delete(args):
    from google_tools import drive_delete_file

    return drive_delete_file(args.get("file_id", ""))


# ── Sheets ─────────────────────────────────────────────────────


@register_tool("sheets_read")
def _dispatch_sheets_read(args):
    from google_tools import sheets_read

    return sheets_read(args.get("spreadsheet_id", ""), args.get("range", "Sheet1"))


@register_tool("sheets_write")
def _dispatch_sheets_write(args):
    from google_tools import sheets_write

    return sheets_write(args.get("spreadsheet_id", ""), args.get("range", ""), args.get("values", []))


@register_tool("sheets_append")
def _dispatch_sheets_append(args):
    from google_tools import sheets_append

    return sheets_append(args.get("spreadsheet_id", ""), args.get("range", ""), args.get("values", []))


@register_tool("sheets_create")
def _dispatch_sheets_create(args):
    from google_tools import sheets_create

    return sheets_create(args.get("title", ""), args.get("folder_id"))


@register_tool("sheets_list_sheets")
def _dispatch_sheets_list_sheets(args):
    from google_tools import sheets_list_sheets

    return sheets_list_sheets(args.get("spreadsheet_id", ""))


@register_tool("sheets_clear")
def _dispatch_sheets_clear(args):
    from google_tools import sheets_clear

    return sheets_clear(args.get("spreadsheet_id", ""), args.get("range", ""))


# ── Calendar ───────────────────────────────────────────────────


@register_tool("calendar_list_events")
def _dispatch_calendar_list_events(args):
    from google_tools import calendar_list_events

    return calendar_list_events(
        args.get("max_results", 10), args.get("time_min"), args.get("time_max"), args.get("calendar_id", "primary")
    )


@register_tool("calendar_create_event")
def _dispatch_calendar_create_event(args):
    from google_tools import calendar_create_event

    return calendar_create_event(
        args.get("summary", ""),
        args.get("start", ""),
        args.get("end", ""),
        args.get("description"),
        args.get("location"),
        args.get("calendar_id", "primary"),
    )


@register_tool("calendar_update_event")
def _dispatch_calendar_update_event(args):
    from google_tools import calendar_update_event

    return calendar_update_event(
        args.get("event_id", ""),
        args.get("summary"),
        args.get("start"),
        args.get("end"),
        args.get("description"),
        args.get("location"),
        args.get("calendar_id", "primary"),
    )


@register_tool("calendar_delete_event")
def _dispatch_calendar_delete_event(args):
    from google_tools import calendar_delete_event

    return calendar_delete_event(args.get("event_id", ""), args.get("calendar_id", "primary"))


@register_tool("calendar_search_events")
def _dispatch_calendar_search_events(args):
    from google_tools import calendar_search_events

    return calendar_search_events(
        args.get("query", ""), args.get("max_results", 10), args.get("calendar_id", "primary")
    )


@register_tool("calendar_list_calendars")
def _dispatch_calendar_list_calendars(args):
    from google_tools import calendar_list_calendars

    return calendar_list_calendars()


# ── Slides ─────────────────────────────────────────────────────


@register_tool("slides_get_presentation")
def _dispatch_slides_get_presentation(args):
    from google_tools import slides_get_presentation

    return slides_get_presentation(args.get("presentation_id", ""))


@register_tool("slides_get_slide")
def _dispatch_slides_get_slide(args):
    from google_tools import slides_get_slide

    return slides_get_slide(args.get("presentation_id", ""), args.get("slide_index", 0))


@register_tool("slides_create_presentation")
def _dispatch_slides_create_presentation(args):
    from google_tools import slides_create_presentation

    return slides_create_presentation(args.get("title", ""), args.get("folder_id"))


@register_tool("slides_add_slide")
def _dispatch_slides_add_slide(args):
    from google_tools import slides_add_slide

    return slides_add_slide(args.get("presentation_id", ""), args.get("layout", "BLANK"))


@register_tool("slides_add_text")
def _dispatch_slides_add_text(args):
    from google_tools import slides_add_text

    return slides_add_text(
        args.get("presentation_id", ""),
        args.get("slide_id", ""),
        args.get("text", ""),
        args.get("x", 100),
        args.get("y", 100),
        args.get("width", 400),
        args.get("height", 300),
    )


# ── Contacts ───────────────────────────────────────────────────


@register_tool("contacts_list")
def _dispatch_contacts_list(args):
    from google_tools import contacts_list

    return contacts_list(args.get("max_results", 20), args.get("query"))


@register_tool("contacts_get")
def _dispatch_contacts_get(args):
    from google_tools import contacts_get

    return contacts_get(args.get("resource_name", ""))


@register_tool("contacts_create")
def _dispatch_contacts_create(args):
    from google_tools import contacts_create

    return contacts_create(
        args.get("given_name", ""),
        args.get("family_name", ""),
        args.get("email"),
        args.get("phone"),
        args.get("organization"),
    )


@register_tool("contacts_update")
def _dispatch_contacts_update(args):
    from google_tools import contacts_update

    return contacts_update(
        args.get("resource_name", ""),
        args.get("given_name"),
        args.get("family_name"),
        args.get("email"),
        args.get("phone"),
    )


@register_tool("contacts_delete")
def _dispatch_contacts_delete(args):
    from google_tools import contacts_delete

    return contacts_delete(args.get("resource_name", ""))


@register_tool("contacts_search")
def _dispatch_contacts_search(args):
    from google_tools import contacts_search

    return contacts_search(args.get("query", ""))


# ── YouTube ────────────────────────────────────────────────────


@register_tool("youtube_search")
def _dispatch_youtube_search(args):
    from google_tools import youtube_search

    return youtube_search(args.get("query", ""), args.get("max_results", 5), args.get("video_type", "video"))


@register_tool("youtube_get_video")
def _dispatch_youtube_get_video(args):
    from google_tools import youtube_get_video

    return youtube_get_video(args.get("video_id", ""))


@register_tool("youtube_get_channel")
def _dispatch_youtube_get_channel(args):
    from google_tools import youtube_get_channel

    return youtube_get_channel(args.get("channel_id", ""))


@register_tool("youtube_list_playlists")
def _dispatch_youtube_list_playlists(args):
    from google_tools import youtube_list_playlists

    return youtube_list_playlists(args.get("channel_id"), args.get("max_results", 10))


@register_tool("youtube_get_playlist_items")
def _dispatch_youtube_get_playlist_items(args):
    from google_tools import youtube_get_playlist_items

    return youtube_get_playlist_items(args.get("playlist_id", ""), args.get("max_results", 20))


@register_tool("youtube_create_playlist")
def _dispatch_youtube_create_playlist(args):
    from google_tools import youtube_create_playlist

    return youtube_create_playlist(args.get("title", ""), args.get("description", ""), args.get("privacy", "private"))


@register_tool("youtube_add_to_playlist")
def _dispatch_youtube_add_to_playlist(args):
    from google_tools import youtube_add_to_playlist

    return youtube_add_to_playlist(args.get("playlist_id", ""), args.get("video_id", ""))


@register_tool("youtube_get_comments")
def _dispatch_youtube_get_comments(args):
    from google_tools import youtube_get_comments

    return youtube_get_comments(args.get("video_id", ""), args.get("max_results", 10))


@register_tool("youtube_post_comment")
def _dispatch_youtube_post_comment(args):
    from google_tools import youtube_post_comment

    return youtube_post_comment(args.get("video_id", ""), args.get("text", ""))


@register_tool("youtube_my_channel")
def _dispatch_youtube_my_channel(args):
    from google_tools import youtube_my_channel

    return youtube_my_channel()
