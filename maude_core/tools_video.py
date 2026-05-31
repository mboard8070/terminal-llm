"""
Video content tools for pre-publish review artifacts.
"""

from __future__ import annotations

import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tool_registry import register_tool

from .paths import resolve_path
from .tools_shared import SHARED_DIR

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_.-]+")
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _slug(value: str, fallback: str = "video") -> str:
    slug = _SAFE_NAME_RE.sub("-", (value or "").strip()).strip(".-").lower()
    return slug[:80] or fallback


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "ok", "passed", "pass"}
    return bool(value)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _check_line(label: str, passed: bool, detail: str = "") -> str:
    status = "PASS" if passed else "FAIL"
    suffix = f" - {detail}" if detail else ""
    return f"- [{status}] {label}{suffix}"


def _artifact_dir(project_path: str | None) -> Path:
    if project_path:
        project_dir = resolve_path(project_path)
        if project_dir.exists() and project_dir.is_dir():
            out = project_dir / "reviews"
            out.mkdir(parents=True, exist_ok=True)
            return out
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    return SHARED_DIR


@register_tool("video_pre_publish_checklist")
def _dispatch_video_pre_publish_checklist(args: dict[str, Any]) -> str:
    """Create a required pre-publish checklist artifact for video work."""

    project_path = _clean_text(args.get("project_path"))
    video_path = _clean_text(args.get("video_path") or args.get("file_path") or args.get("media_path"))
    platform = _clean_text(args.get("platform") or "unspecified")
    title = _clean_text(args.get("title"))
    description = _clean_text(args.get("description") or args.get("caption"))
    link_url = _clean_text(args.get("link_url") or args.get("url"))
    link_placement = _clean_text(args.get("link_placement"))
    cta_text = _clean_text(args.get("cta_text"))
    timing_notes = _clean_text(args.get("timing_notes"))
    render_review = _clean_text(args.get("render_review"))
    tags = _clean_text(args.get("tags") or args.get("hashtags"))
    privacy = _clean_text(args.get("privacy") or args.get("visibility"))

    raw_url_in_video = _as_bool(args.get("raw_url_in_video"), False)
    media_attached = _as_bool(args.get("media_attached"), bool(video_path))
    transition_timing_ok = _as_bool(args.get("transition_timing_ok"), False)
    text_readable = _as_bool(args.get("text_readable"), False)
    no_placeholders = _as_bool(args.get("no_placeholders"), False)
    render_reviewed = _as_bool(args.get("render_reviewed"), bool(render_review))
    link_in_description = _as_bool(
        args.get("link_in_description"), bool(link_url and link_url in description) or "description" in link_placement.lower() or "caption" in link_placement.lower()
    )

    failures = []
    if not video_path:
        failures.append("video_path is required")
    if not title:
        failures.append("title is required")
    if not description:
        failures.append("description/caption is required")
    if not cta_text:
        failures.append("designed CTA text is required")
    if link_url and not link_in_description:
        failures.append("link_url must be placed in the description/caption, not only on screen")
    if raw_url_in_video:
        failures.append("raw URLs must not be burned into the video frame")
    if not transition_timing_ok:
        failures.append("transition/screen-wipe timing has not been confirmed")
    if not text_readable:
        failures.append("text readability has not been confirmed")
    if not no_placeholders:
        failures.append("placeholder/broken asset check has not been confirmed")
    if not render_reviewed:
        failures.append("rendered MP4 playback review is required")
    if not media_attached:
        failures.append("media attachment is not confirmed")

    approved = not failures
    out_dir = _artifact_dir(project_path or None)
    filename = f"prepublish-{_slug(title or Path(video_path).stem)}-{int(datetime.now(UTC).timestamp())}.md"
    artifact_path = out_dir / filename

    lines = [
        "# Video Pre-Publish Checklist",
        "",
        f"Created: {_now()}",
        f"Status: {'APPROVED' if approved else 'BLOCKED'}",
        f"Platform: {platform}",
        f"Video: {video_path or '(missing)'}",
        "",
        "## Publish Metadata",
        "",
        f"- Title: {title or '(missing)'}",
        f"- Description/Caption: {description or '(missing)'}",
        f"- Link URL: {link_url or '(none)'}",
        f"- Link placement: {link_placement or '(missing)'}",
        f"- On-screen CTA: {cta_text or '(missing)'}",
        f"- Tags/Hashtags: {tags or '(none)'}",
        f"- Privacy/Visibility: {privacy or '(unspecified)'}",
        "",
        "## Quality Checks",
        "",
        _check_line("Media attached", media_attached),
        _check_line("Screen wipe / transition timing confirmed", transition_timing_ok, timing_notes),
        _check_line("Rendered MP4 playback reviewed", render_reviewed, render_review),
        _check_line("On-screen text readable", text_readable),
        _check_line("No placeholders, broken assets, or missing media", no_placeholders),
        _check_line("No raw URL burned into video frame", not raw_url_in_video),
        _check_line("Clickable link is in description/caption", link_in_description if link_url else True),
        "",
        "## Blocking Issues",
        "",
    ]
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- None")
    lines.append("")

    payload = "\n".join(lines)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=artifact_path.parent, delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    tmp_path.replace(artifact_path)

    result = f"Video pre-publish checklist {'approved' if approved else 'blocked'}: {artifact_path}"
    if failures:
        result += "\n\nBlocking issues:\n" + "\n".join(f"- {failure}" for failure in failures)
    return result
