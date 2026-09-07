"""
Dynamic tool selection — tiered + keyword + session activation.

Tiers (see MAUDE_IMPROVEMENTS_TODO #2):
  always-on  — full schemas every turn (file / shell / search / memory)
  session    — browser / google / media / social — sticky once activated
  rare       — substack / forge / hyperframes — keyword (or explicit activate)

Lazy schemas:
  Full JSON schemas only for always-on + active domains.
  Inactive session/rare domains appear as name + one-line stubs (no params),
  or can be expanded via activate_tool_domain / keyword match / prior tool use.
"""

from __future__ import annotations

import os
import threading
from typing import Iterable

from .tool_defs import TOOLS

# ── Always-on core (full schemas every turn) ──────────────────────
# Intentionally small: read / write / run / search / memory (+ plan/escalation).
_CORE_TOOL_NAMES = {
    # files + shell
    "read_file",
    "write_file",
    "edit_file",
    "list_directory",
    "change_directory",
    "search_file",
    "search_directory",
    "run_command",
    # search
    "web_browse",
    "web_search",
    # memory + scratch
    "save_memory",
    "recall_memory",
    "scratch_set",
    "scratch_add_finding",
    "scratch_show",
    "scratch_clear",
    "palace_search",
    "palace_status",
    # planning / escalation
    "ask_frontier",
    "execute_plan",
    # domain lazy-load controls
    "list_tool_domains",
    "activate_tool_domain",
}

# ── Tool groups activated by keyword detection ────────────────────
_TOOL_GROUPS = {
    "browser": {
        "keywords": [
            "browser",
            "playwright",
            "headless",
            "click",
            "login",
            "log in",
            "sign in",
            "screenshot",
            "navigate to",
            "fill form",
            "open the site",
            "open url",
            "web page",
            "webpage",
            "puppeteer",
            "scrape the page",
            "interact with the page",
        ],
        "tools": {
            "browser_open",
            "browser_snapshot",
            "browser_click",
            "browser_type",
            "browser_navigate",
            "browser_screenshot",
            "browser_extract",
            "browser_fill_form",
            "browser_select",
            "browser_login",
            "browser_check_session",
            "browser_close",
        },
        "tier": "session",
        "description": "Interactive browser automation (open, click, type, login, screenshot)",
    },
    "social": {
        "keywords": [
            "social post",
            "tweet",
            "post to twitter",
            "post to x",
            "post on x",
            "linkedin post",
            "post to linkedin",
            "instagram post",
            "post to instagram",
            "facebook post",
            "post to facebook",
            "social media",
            "post this",
            "publish to social",
        ],
        "tools": {"social_post", "social_x_post_video"},
        "tier": "session",
        "description": "Post to social platforms via browser session",
    },
    "media": {
        "keywords": [
            "generate image",
            "generate an image",
            "create image",
            "create an image",
            "make image",
            "make an image",
            "draw",
            "flux",
            "generate a picture",
            "make a picture",
            "illustration",
            "lora",
            "stillion",
            "view_image",
            "analyze image",
            "describe this image",
            "describe the image",
            "look at this photo",
            "look at this image",
            "find image",
            "find picture",
            "find photo",
            "show me a picture",
            "show me a photo",
            "show me an image",
            "web image",
            "find me a picture",
            "find me an image",
            "find me a photo",
            "search for image",
            "search for photo",
            "generate 3d",
            "3d model",
            "glb",
            "trellis",
        ],
        "tools": {
            "generate_image",
            "generate_image_muse",
            "view_image",
            "web_image_search",
            "generate_3d",
            "share_file",
        },
        "tier": "session",
        "description": "Image gen, vision analysis, web image search, 3D generation",
    },
    # Keep fine-grained groups for keyword precision; same tools as media
    "image_gen": {
        "keywords": [
            "generate image",
            "generate an image",
            "create image",
            "create an image",
            "make image",
            "make an image",
            "draw",
            "flux",
            "generate a picture",
            "make a picture",
            "image of",
            "picture of",
            "illustration",
            "render",
            "lora",
            "stillion",
            "muse image",
            "muse-image",
        ],
        "tools": {"generate_image", "generate_image_muse", "share_file"},
        "tier": "session",
        "description": "Generate images with Flux / ComfyUI",
        "activates": "media",
    },
    "web_image": {
        "keywords": [
            "find image",
            "find picture",
            "find photo",
            "show me a picture",
            "show me a photo",
            "show me an image",
            "photo of",
            "image of",
            "picture of",
            "images of",
            "photos of",
            "pictures of",
            "search for image",
            "search for photo",
            "web image",
            "find me a picture",
            "find me an image",
            "find me a photo",
        ],
        "tools": {"web_image_search"},
        "tier": "session",
        "description": "Search the web for images",
        "activates": "media",
    },
    "vision": {
        "keywords": [
            "view_image",
            "analyze this image",
            "describe this image",
            "describe the photo",
            "what is in this image",
            "what is in this photo",
            "attached image",
            "attached photo",
        ],
        "tools": {"view_image"},
        "tier": "session",
        "description": "Vision / image analysis",
        "activates": "media",
    },
    "shared": {
        "keywords": [
            "shared",
            "transfer",
            "client",
            "pull",
            "push",
            "send to client",
            "grab",
            "fetch",
            "upload",
            "download",
            "sync",
        ],
        "tools": {"list_shared", "share_file", "list_transfers", "get_transfer"},
        "tier": "session",
        "description": "Shared folder and transfer tools",
    },
    "gmail": {
        "keywords": ["gmail", "email", "inbox", "mail"],
        "tools": {"gmail_list", "gmail_read", "gmail_send"},
        "tier": "session",
        "description": "Gmail list / read / send",
        "activates": "google",
    },
    "drive": {
        "keywords": [
            "drive",
            "google doc",
            "google drive",
            "my documents",
            "my files on",
            "cloud files",
            "gdrive",
            "folder",
        ],
        "tools": {
            "drive_list",
            "drive_search",
            "drive_read",
            "drive_upload",
            "drive_create_doc",
            "drive_create_folder",
            "drive_create_sheet",
            "drive_update_doc",
            "drive_delete",
        },
        "tier": "session",
        "description": "Google Drive files and docs",
        "activates": "google",
    },
    "sheets": {
        "keywords": ["sheet", "spreadsheet", "csv", "table", "cells", "rows", "columns"],
        "tools": {
            "sheets_read",
            "sheets_write",
            "sheets_append",
            "sheets_create",
            "sheets_list_sheets",
            "sheets_clear",
        },
        "tier": "session",
        "description": "Google Sheets read/write",
        "activates": "google",
    },
    "calendar": {
        "keywords": ["calendar", "event", "meeting", "schedule", "appointment", "reminder"],
        "tools": {
            "calendar_list_events",
            "calendar_create_event",
            "calendar_update_event",
            "calendar_delete_event",
            "calendar_search_events",
            "calendar_list_calendars",
        },
        "tier": "session",
        "description": "Google Calendar events",
        "activates": "google",
    },
    "slides": {
        "keywords": ["slide", "presentation", "deck", "powerpoint", "ppt"],
        "tools": {
            "slides_get_presentation",
            "slides_get_slide",
            "slides_create_presentation",
            "slides_add_slide",
            "slides_add_text",
        },
        "tier": "session",
        "description": "Google Slides presentations",
        "activates": "google",
    },
    "contacts": {
        "keywords": ["contact", "phone number", "address book", "people"],
        "tools": {
            "contacts_list",
            "contacts_get",
            "contacts_create",
            "contacts_update",
            "contacts_delete",
            "contacts_search",
        },
        "tier": "session",
        "description": "Google Contacts",
        "activates": "google",
    },
    "youtube": {
        "keywords": ["youtube", "playlist", "channel", "subscribe", "upload video", "upload to youtube", "video"],
        "tools": {
            "youtube_search",
            "youtube_get_video",
            "youtube_get_channel",
            "youtube_list_playlists",
            "youtube_create_playlist",
            "youtube_upload",
            "youtube_get_comments",
            "youtube_post_comment",
            "youtube_my_channel",
        },
        "tier": "session",
        "description": "YouTube search, upload, playlists",
        "activates": "google",
    },
    "google": {
        "keywords": ["google"],
        "tools": {
            "gmail_list",
            "gmail_read",
            "gmail_send",
            "drive_list",
            "drive_search",
            "drive_read",
            "drive_upload",
            "drive_create_doc",
            "drive_create_folder",
            "drive_create_sheet",
            "drive_update_doc",
            "drive_delete",
            "sheets_read",
            "sheets_write",
            "sheets_create",
            "calendar_list_events",
            "calendar_create_event",
            "contacts_list",
            "contacts_search",
        },
        "tier": "session",
        "description": "Google Workspace (Gmail, Drive, Sheets, Calendar, Contacts)",
    },
    "hyperframes": {
        "keywords": [
            "hyperframes",
            "hyperframe",
            "html to video",
            "html-to-video",
            "programmatic video",
            "video composition",
            "render video",
            "render mp4",
            "create video",
            "make a video",
            "motion graphics",
        ],
        "tools": {
            "skill_hyperframes",
            "hyperframes_doctor",
            "hyperframes_browser_ensure",
            "hyperframes_init",
            "hyperframes_lint",
            "hyperframes_render",
            "share_file",
        },
        "tier": "rare",
        "description": "HyperFrames HTML-to-video pipeline",
    },
    "substack": {
        "keywords": ["substack", "newsletter", "draft", "publish", "blog post", "article"],
        "tools": {
            "substack_create_draft",
            "substack_list_drafts",
            "substack_list_posts",
            "substack_get_post",
            "substack_update_draft",
            "substack_delete_draft",
            "substack_get_stats",
        },
        "tier": "rare",
        "description": "Substack newsletter drafts and posts",
    },
    "workflow": {
        "keywords": [
            "workflow",
            "monitor",
            "price check",
            "competitor",
            "automate",
            "scheduled browse",
            "recurring",
            "change detection",
            "price monitor",
        ],
        "tools": {
            "workflow_create",
            "workflow_run",
            "workflow_list",
            "workflow_get",
            "workflow_delete",
            "workflow_history",
            "workflow_schedule",
            "workflow_unschedule",
        },
        "tier": "session",
        "description": "Scheduled browser workflows / monitors",
    },
    "memory": {
        "keywords": [
            "remember",
            "recall",
            "forget",
            "memory",
            "memories",
            "you know",
            "do you remember",
            "what do you know",
            "i told you",
            "i mentioned",
            "last time",
            "my preference",
            "my favorite",
            "i like",
            "i prefer",
            "don't forget",
            "keep in mind",
            "note that",
            "fact",
            "facts",
            "who is",
            "what is",
            "tell me about",
            "knowledge",
            "relationship",
            "wing",
            "room",
            "drawer",
            "palace",
        ],
        "tools": {
            "save_memory",
            "recall_memory",
            "list_memories",
            "forget_memory",
            "scratch_set",
            "scratch_add_finding",
            "scratch_show",
            "scratch_clear",
            "palace_recall",
            "palace_kg_add_fact",
            "palace_kg_query",
        },
        "tier": "session",
        "description": "Extended memory / palace tools (core memory always available)",
    },
    "collab": {
        "keywords": [
            "who's online",
            "whos online",
            "mesh status",
            "devices",
            "dispatch",
            "send to spark",
            "send to mac",
            "run on",
            "project",
            "collaboration",
            "activity",
            "task",
            "what are they doing",
            "online",
            "mac",
            "windows",
            "pc",
            "laptop",
            "other machine",
            "other device",
            "remote",
            "cross-machine",
            "mattwell",
            "macbook",
            "on the",
        ],
        "tools": {"mesh_status", "dispatch_task", "create_project", "list_projects", "add_to_project", "list_tasks"},
        "tier": "session",
        "description": "Cross-device collaboration and mesh",
    },
    "github": {
        "keywords": [
            "pull request",
            "pr ",
            "prs",
            "merge",
            "github",
            "repo",
            "issue",
            "branch",
            "commit",
            "release",
            "workflow",
            "ci/cd",
            "actions",
            "notification",
        ],
        "tools": {
            "github_list_prs",
            "github_view_pr",
            "github_create_pr",
            "github_merge_pr",
            "github_close_pr",
            "github_pr_diff",
            "github_pr_comments",
            "github_comment_pr",
            "github_list_issues",
            "github_view_issue",
            "github_create_issue",
            "github_close_issue",
            "github_comment_issue",
            "github_list_repos",
            "github_view_repo",
            "github_list_branches",
            "github_list_commits",
            "github_list_runs",
            "github_view_run",
            "github_rerun",
            "github_list_releases",
            "github_create_release",
            "github_search",
            "github_notifications",
        },
        "tier": "session",
        "description": "GitHub PRs, issues, repos, Actions",
    },
    "agents": {
        "keywords": [
            "research",
            "analyze",
            "investigate",
            "compare",
            "look into",
            "deep dive",
            "comprehensive",
            "parallel",
            "use the .* agent",
            "run agent",
            "dispatch agent",
        ],
        "tools": {"run_agent", "run_agents"},
        "tier": "session",
        "description": "Research / analysis subagents",
    },
    "sandbox": {
        "keywords": [
            "sandbox",
            "container",
            "docker",
            "build me",
            "build a",
            "make me a",
            "develop",
            "coding project",
            "web app",
            "autonomous",
            "work on",
            "dev environment",
        ],
        "tools": {
            "sandbox_exec",
            "sandbox_write_file",
            "sandbox_read_file",
            "sandbox_list",
            "sandbox_status",
            "sandbox_copy_out",
            "sandbox_project",
        },
        "tier": "session",
        "description": "Isolated sandbox / container project tools",
    },
    "forge": {
        "keywords": [
            "forge",
            "build me",
            "build a",
            "create an app",
            "make me a",
            "autonomous",
            "build autonomously",
            "develop",
            "web app",
            "api service",
            "saas",
            "mvp",
            "prototype",
            "ai tool",
        ],
        "tools": {"forge_build", "forge_status", "forge_log"},
        "tier": "rare",
        "description": "Autonomous app forge builder",
    },
    "command_center": {
        "keywords": [
            "system stats",
            "cpu",
            "gpu",
            "temperature",
            "memory usage",
            "ram",
            "disk",
            "vram",
            "what's running",
            "whats running",
            "processes",
            "sessions",
            "activity",
            "recent activity",
            "scheduler",
            "scheduled tasks",
            "nodes",
            "services",
            "status",
            "monitoring",
            "dashboard",
            "command center",
            "health",
            "uptime",
            "resources",
        ],
        "tools": {
            "system_stats",
            "gpu_processes",
            "memory_browse",
            "session_list",
            "activity_feed",
            "scheduler_status",
            "node_status",
        },
        "tier": "session",
        "description": "System monitoring and command center",
    },
    "missions": {
        "keywords": [
            "mission",
            "missions",
            "mission tick",
            "tick",
            "dashboard",
            "blocker",
            "blockers",
            "archive mission",
            "start a mission",
            "objective",
            "success criteria",
            "next action",
        ],
        "tools": {
            "mission_create",
            "mission_list",
            "mission_get",
            "mission_update",
            "mission_log",
            "mission_brief",
            "mission_tick",
            "mission_next",
            "mission_dashboard",
            "mission_blockers",
            "mission_archive",
            "mission_history",
        },
        "tier": "session",
        "description": "Long-running mission board tools",
    },
}

# Session-tier group names (sticky once activated for a session)
_SESSION_TIER_GROUPS = {
    name for name, g in _TOOL_GROUPS.items() if g.get("tier") == "session"
}
# Rare-tier (keyword or explicit activate; not sticky by default)
_RARE_TIER_GROUPS = {
    name for name, g in _TOOL_GROUPS.items() if g.get("tier") == "rare"
}

# Canonical domain names exposed via list/activate (dedupe aliases)
_DOMAIN_CANONICAL = {
    "image_gen": "media",
    "web_image": "media",
    "vision": "media",
    "gmail": "google",
    "drive": "google",
    "sheets": "google",
    "calendar": "google",
    "slides": "google",
    "contacts": "google",
    "youtube": "google",
}

# Build lookup: tool name -> tool definition
_TOOL_BY_NAME = {t["function"]["name"]: t for t in TOOLS}

# tool name -> set of group names that contain it
_TOOL_TO_GROUPS: dict[str, set[str]] = {}
for _gname, _g in _TOOL_GROUPS.items():
    for _tname in _g["tools"]:
        _TOOL_TO_GROUPS.setdefault(_tname, set()).add(_gname)

# ── Session activation state ──────────────────────────────────────
_session_lock = threading.Lock()
_session_domains: dict[str, set[str]] = {}  # session_id -> set of domain/group names


def _default_session_id() -> str:
    return os.environ.get("MAUDE_SESSION_ID", "default")


def _lazy_stubs_enabled() -> bool:
    """Whether to include per-domain one-liner stub tools (default off).

    Discovery is always available via list_tool_domains / activate_tool_domain
    (always-on). Set MAUDE_LAZY_TOOL_STUBS=1 to also emit domain_* stub tools
    for inactive domains.
    """
    return os.environ.get("MAUDE_LAZY_TOOL_STUBS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _canonical_domain(name: str) -> str:
    return _DOMAIN_CANONICAL.get(name, name)


def activate_domain(domain: str, session_id: str | None = None) -> list[str]:
    """Activate a domain/group for the session. Returns list of activated names."""
    sid = session_id or _default_session_id()
    domain = (domain or "").strip().lower().replace("-", "_")
    if not domain:
        return []

    groups: set[str] = set()
    # Accept tool name as a way to activate its group(s)
    if domain in _TOOL_TO_GROUPS:
        for g in _TOOL_TO_GROUPS[domain]:
            groups.add(g)
            groups.add(_canonical_domain(g))
            act = _TOOL_GROUPS.get(g, {}).get("activates")
            if act:
                groups.add(_canonical_domain(act))
    elif domain in _TOOL_GROUPS:
        groups.add(domain)
        groups.add(_canonical_domain(domain))
        act = _TOOL_GROUPS[domain].get("activates")
        if act:
            groups.add(_canonical_domain(act))
    else:
        for gname in _TOOL_GROUPS:
            if gname.startswith(domain) or domain.startswith(gname):
                groups.add(gname)
                groups.add(_canonical_domain(gname))
        if not groups:
            return []

    with _session_lock:
        active = _session_domains.setdefault(sid, set())
        active.update(groups)

    return sorted(groups)


def get_session_domains(session_id: str | None = None) -> set[str]:
    sid = session_id or _default_session_id()
    with _session_lock:
        return set(_session_domains.get(sid, set()))


def clear_session_domains(session_id: str | None = None) -> None:
    """Clear sticky domain activations (tests / session reset)."""
    sid = session_id or _default_session_id()
    with _session_lock:
        _session_domains.pop(sid, None)


def list_domain_catalog() -> list[dict]:
    """Canonical domains with tier + one-line description + tool names."""
    seen: set[str] = set()
    catalog = []
    # Prefer canonical parents first
    preferred_order = [
        "browser",
        "google",
        "media",
        "social",
        "github",
        "collab",
        "agents",
        "workflow",
        "command_center",
        "missions",
        "shared",
        "sandbox",
        "memory",
        "substack",
        "forge",
        "hyperframes",
    ]
    for name in preferred_order:
        if name not in _TOOL_GROUPS or name in seen:
            continue
        g = _TOOL_GROUPS[name]
        seen.add(name)
        catalog.append(
            {
                "name": name,
                "tier": g.get("tier", "session"),
                "description": g.get("description", name),
                "tools": sorted(g["tools"]),
            }
        )
    # Any remaining non-alias groups
    for name, g in sorted(_TOOL_GROUPS.items()):
        if name in seen or name in _DOMAIN_CANONICAL:
            continue
        catalog.append(
            {
                "name": name,
                "tier": g.get("tier", "session"),
                "description": g.get("description", name),
                "tools": sorted(g["tools"]),
            }
        )
    return catalog


def _one_line(text: str, max_len: int = 120) -> str:
    line = " ".join((text or "").split())
    if len(line) > max_len:
        return line[: max_len - 1] + "…"
    return line


def _domain_stub(domain: dict) -> dict:
    """One lazy discovery entry per inactive domain (name + one-liner)."""
    tools_preview = ", ".join(domain["tools"][:6])
    if len(domain["tools"]) > 6:
        tools_preview += ", …"
    desc = (
        f"[lazy domain:{domain['name']}] {domain['description']}. "
        f"Tools: {tools_preview}. "
        f"Call activate_tool_domain(domain=\"{domain['name']}\") to load full schemas."
    )
    return {
        "type": "function",
        "function": {
            "name": f"domain_{domain['name']}",
            "description": _one_line(desc, max_len=220),
            "parameters": {
                "type": "object",
                "properties": {
                    "activate": {
                        "type": "boolean",
                        "description": "Set true to activate this domain (same as activate_tool_domain)",
                    }
                },
                "required": [],
            },
        },
    }


def _groups_matching_message(msg_lower: str) -> set[str]:
    matched: set[str] = set()
    for name, group in _TOOL_GROUPS.items():
        for kw in group["keywords"]:
            if kw in msg_lower:
                matched.add(name)
                act = group.get("activates")
                if act:
                    matched.add(act)
                break
    return matched


def _groups_from_tool_names(tool_names: Iterable[str]) -> set[str]:
    groups: set[str] = set()
    for name in tool_names:
        for g in _TOOL_TO_GROUPS.get(name, ()):
            groups.add(_canonical_domain(g))
            act = _TOOL_GROUPS.get(g, {}).get("activates")
            if act:
                groups.add(_canonical_domain(act))
    return groups


def _tool_names_from_messages(messages: list | None) -> set[str]:
    """Collect tool names recently used in conversation (for sticky reactivation)."""
    if not messages:
        return set()
    names: set[str] = set()
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                n = fn.get("name") if isinstance(fn, dict) else None
                if n:
                    names.add(n)
    return names


def _tools_for_groups(group_names: set[str]) -> set[str]:
    names: set[str] = set()
    for g in group_names:
        group = _TOOL_GROUPS.get(g)
        if group:
            names.update(group["tools"])
        # If canonical parent, also pull alias group tools already merged
    return names


def select_active_tool_names(
    message: str,
    *,
    session_id: str | None = None,
    messages: list | None = None,
    sticky: bool = True,
) -> set[str]:
    """Resolve which tool names get full schemas this turn."""
    sid = session_id or _default_session_id()
    msg_lower = (message or "").lower()

    active_names = set(_CORE_TOOL_NAMES)
    matched_groups = _groups_matching_message(msg_lower)

    # History-based: tools already used in this conversation
    history_tools = _tool_names_from_messages(messages)
    history_groups = _groups_from_tool_names(history_tools)

    # Sticky session domains
    session_groups = get_session_domains(sid) if sticky else set()

    all_groups = set(matched_groups) | set(history_groups) | set(session_groups)

    # Canonicalize
    all_groups = {_canonical_domain(g) for g in all_groups} | all_groups

    # Persist sticky activation for session-tier groups (and rare if explicitly matched)
    if sticky and (matched_groups or history_groups):
        to_stick = set()
        for g in matched_groups | history_groups:
            cg = _canonical_domain(g)
            # Session tier always sticks; rare sticks only if matched this turn
            if cg in _SESSION_TIER_GROUPS or g in _SESSION_TIER_GROUPS:
                to_stick.add(cg)
            elif g in _RARE_TIER_GROUPS or cg in _RARE_TIER_GROUPS:
                # rare: stick for this session once keyword-matched (so multi-step rare workflows work)
                to_stick.add(cg)
            else:
                to_stick.add(cg)
            # also stick the concrete group if it has its own tools
            if g in _TOOL_GROUPS:
                to_stick.add(g)
        if to_stick:
            with _session_lock:
                active = _session_domains.setdefault(sid, set())
                active.update(to_stick)

    active_names.update(_tools_for_groups(all_groups))
    active_names.update(history_tools)

    # Flux 2 is cloud/paid — only when explicitly requested
    if "flux 2" in msg_lower or "flux2" in msg_lower:
        active_names.add("generate_image_flux2")
    if "muse image" in msg_lower or "muse-image" in msg_lower:
        active_names.add("generate_image_muse")

    return active_names


def get_tools_for_message(
    message: str,
    session_id: str | None = None,
    messages: list | None = None,
    *,
    lazy: bool | None = None,
    sticky: bool = True,
) -> list:
    """Return tools relevant to the user's message.

    Always includes core tools with full schemas.
    Adds specialized groups on keyword match, prior tool use, or session sticky
    activation. Inactive domains appear as one name+one-liner stub each when
    MAUDE_LAZY_TOOL_STUBS is enabled (default), so the model can discover and
    activate them without paying full schema cost every turn.
    """
    sid = session_id or _default_session_id()
    full_names = select_active_tool_names(
        message, session_id=sid, messages=messages, sticky=sticky
    )

    use_lazy = _lazy_stubs_enabled() if lazy is None else lazy
    active_domains = get_session_domains(sid)
    # Also treat keyword-matched groups this turn as active (even before stick)
    matched = _groups_matching_message((message or "").lower())
    active_domains = set(active_domains) | {_canonical_domain(g) for g in matched} | matched

    result: list = []
    seen: set[str] = set()
    for t in TOOLS:
        name = t["function"]["name"]
        if name in full_names:
            result.append(t)
            seen.add(name)

    # Domain control tools may not be in TOOLS — append synthetic defs
    for name in ("list_tool_domains", "activate_tool_domain"):
        if name in full_names and name not in seen:
            syn = _DOMAIN_CONTROL_TOOLS.get(name)
            if syn:
                result.append(syn)
                seen.add(name)

    # Per-domain lazy stubs for inactive session/rare domains
    if use_lazy:
        for domain in list_domain_catalog():
            dname = domain["name"]
            if dname in active_domains or _canonical_domain(dname) in active_domains:
                continue
            # Skip if any of this domain's tools already have full schemas
            domain_tools = set(domain["tools"])
            if domain_tools & full_names:
                continue
            stub = _domain_stub(domain)
            sname = stub["function"]["name"]
            if sname not in seen:
                result.append(stub)
                seen.add(sname)

    return result


def payload_stats(tools: list | None = None, message: str = "hello") -> dict:
    """Helper for tests/metrics: count tools and JSON payload size."""
    import json

    tools = tools if tools is not None else get_tools_for_message(message)
    payload = len(json.dumps(tools))
    lazy_count = sum(
        1
        for t in tools
        if (t.get("function", {}).get("description") or "").startswith("[lazy")
        or (t.get("function", {}).get("name") or "").startswith("domain_")
    )
    full_count = len(tools) - lazy_count
    return {
        "tool_count": len(tools),
        "full_schema_count": full_count,
        "lazy_stub_count": lazy_count,
        "payload_chars": payload,
    }


# ── Domain control tool definitions (always-on) ───────────────────
_DOMAIN_CONTROL_TOOLS = {
    "list_tool_domains": {
        "type": "function",
        "function": {
            "name": "list_tool_domains",
            "description": (
                "List tool domains not in the always-on set, with tier and one-line "
                "description. Use activate_tool_domain to load full schemas for a domain."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "include_tools": {
                        "type": "boolean",
                        "description": "If true, include tool name lists per domain (default false)",
                    }
                },
                "required": [],
            },
        },
    },
    "activate_tool_domain": {
        "type": "function",
        "function": {
            "name": "activate_tool_domain",
            "description": (
                "Load full tool schemas for a domain for the rest of this session. "
                "Session: browser, google, media, social, github, collab, agents, "
                "workflow, sandbox, missions, command_center, shared, memory. "
                "Rare: substack, forge, hyperframes. "
                "Call list_tool_domains for one-line descriptions of each."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Domain name to activate (e.g. browser, google, media)",
                    }
                },
                "required": ["domain"],
            },
        },
    },
}


def _domain_index_description() -> str:
    """Compact names + one-liners for lazy schema discovery (always-on)."""
    parts = []
    for d in list_domain_catalog():
        parts.append(f"{d['name']}[{d['tier']}]: {d['description']}")
    return (
        "List tool domains not in the always-on set (names + one-liners). "
        "Use activate_tool_domain(domain=...) to load full schemas. "
        "Domains — " + " | ".join(parts)
    )


# Refresh list_tool_domains description with live domain one-liners
_DOMAIN_CONTROL_TOOLS["list_tool_domains"]["function"]["description"] = _domain_index_description()


def _handle_list_tool_domains(arguments: dict) -> str:
    import json

    include_tools = bool(arguments.get("include_tools", False))
    sid = _default_session_id()
    active = get_session_domains(sid)
    catalog = list_domain_catalog()
    lines = ["Tool domains (full schemas load on keyword match or activate_tool_domain):", ""]
    for d in catalog:
        status = "ACTIVE" if d["name"] in active or _canonical_domain(d["name"]) in active else "inactive"
        lines.append(f"- {d['name']} [{d['tier']}] ({status}): {d['description']}")
        if include_tools:
            lines.append(f"    tools: {', '.join(d['tools'][:12])}{'…' if len(d['tools']) > 12 else ''}")
    lines.append("")
    lines.append(f"Always-on core tools ({len(_CORE_TOOL_NAMES)}): {', '.join(sorted(_CORE_TOOL_NAMES))}")
    if include_tools:
        return json.dumps({"domains": catalog, "active": sorted(active), "core": sorted(_CORE_TOOL_NAMES)}, indent=2)
    return "\n".join(lines)


def _handle_activate_tool_domain(arguments: dict) -> str:
    domain = arguments.get("domain", "")
    activated = activate_domain(domain)
    if not activated:
        available = ", ".join(d["name"] for d in list_domain_catalog())
        return f"Unknown domain '{domain}'. Available: {available}"
    tools = sorted(_tools_for_groups(set(activated)))
    # Only list tools that actually exist
    tools = [t for t in tools if t in _TOOL_BY_NAME]
    return (
        f"Activated domain(s): {', '.join(activated)}. "
        f"Full schemas now available for: {', '.join(tools) if tools else '(no registered tools yet)'}. "
        "Call the domain tools on the next step."
    )


def handle_domain_stub_call(name: str, arguments: dict | None = None) -> str | None:
    """If name is a domain_* lazy stub, activate that domain and return a message.

    Returns None if name is not a domain stub.
    """
    if not name.startswith("domain_"):
        return None
    domain = name[len("domain_") :]
    args = arguments or {}
    # Always activate when the model calls the stub (activate flag optional)
    if args.get("activate") is False:
        d = next((x for x in list_domain_catalog() if x["name"] == domain), None)
        if not d:
            return f"Unknown domain stub '{name}'"
        return (
            f"Domain '{domain}': {d['description']}. "
            f"Tools: {', '.join(d['tools'])}. "
            f"Call activate_tool_domain(domain=\"{domain}\") or re-call with activate=true."
        )
    activated = activate_domain(domain)
    if not activated:
        return f"Unknown domain '{domain}'"
    tools = sorted(t for t in _tools_for_groups(set(activated)) if t in _TOOL_BY_NAME)
    return (
        f"Activated domain(s): {', '.join(activated)}. "
        f"Full schemas now available for: {', '.join(tools) if tools else '(none registered)'}. "
        "Call the domain tools on the next step."
    )


def register_domain_control_tools() -> None:
    """Register list/activate handlers and inject schemas into TOOLS if missing."""
    from tool_registry import register_tool

    @register_tool("list_tool_domains")
    def _list(args):
        return _handle_list_tool_domains(args or {})

    @register_tool("activate_tool_domain")
    def _activate(args):
        return _handle_activate_tool_domain(args or {})

    # Ensure TOOLS list includes domain control schemas for catalog consumers
    # (refresh descriptions if already present so domain index stays current)
    by_name = {t["function"]["name"]: t for t in TOOLS}
    for name, defn in _DOMAIN_CONTROL_TOOLS.items():
        if name in by_name:
            by_name[name]["function"]["description"] = defn["function"]["description"]
            _TOOL_BY_NAME[name] = by_name[name]
        else:
            TOOLS.append(defn)
            _TOOL_BY_NAME[name] = defn


# Auto-register on import (safe if tool_registry present)
try:
    register_domain_control_tools()
except Exception:
    pass
