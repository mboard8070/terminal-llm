"""
Social media posting skill for MAUDE.

Supports posting to Twitter/X, LinkedIn, and Bluesky with optional image
attachments and scheduled posting via MAUDE's ProactiveScheduler.
"""

import os
import json
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import requests

from skills import skill


# ---------------------------------------------------------------------------
# Platform character limits
# ---------------------------------------------------------------------------
CHAR_LIMITS: Dict[str, int] = {
    "twitter": 280,
    "linkedin": 3000,
    "bluesky": 300,
}

PLATFORM_DISPLAY_NAMES: Dict[str, str] = {
    "twitter": "Twitter/X",
    "linkedin": "LinkedIn",
    "bluesky": "Bluesky",
}

# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------

def _get_key(name: str) -> Optional[str]:
    """Retrieve an API key from env vars or MAUDE's key manager (if available)."""
    value = os.environ.get(name)
    if value:
        return value

    # Attempt MAUDE key manager integration
    try:
        from key_manager import get_key  # type: ignore
        return get_key(name)
    except Exception:
        return None


def _twitter_configured() -> Tuple[bool, Optional[str]]:
    """Check whether Twitter credentials are present.

    Returns (configured, error_message).
    """
    required = [
        "TWITTER_API_KEY",
        "TWITTER_API_SECRET",
        "TWITTER_ACCESS_TOKEN",
        "TWITTER_ACCESS_SECRET",
    ]
    missing = [k for k in required if not _get_key(k)]
    if missing:
        return False, (
            "Twitter/X is not configured. Set the following environment variables "
            "(or add them to MAUDE's key manager):\n"
            + "\n".join(f"  - {k}" for k in missing)
            + "\n\nYou can create these at https://developer.twitter.com/en/portal"
        )
    return True, None


def _linkedin_configured() -> Tuple[bool, Optional[str]]:
    """Check whether LinkedIn credentials are present."""
    if not _get_key("LINKEDIN_ACCESS_TOKEN"):
        return False, (
            "LinkedIn is not configured. Set the following environment variable "
            "(or add it to MAUDE's key manager):\n"
            "  - LINKEDIN_ACCESS_TOKEN\n\n"
            "Generate a token via a LinkedIn OAuth2 flow at "
            "https://www.linkedin.com/developers/apps"
        )
    return True, None


def _bluesky_configured() -> Tuple[bool, Optional[str]]:
    """Check whether Bluesky credentials are present."""
    required = ["BLUESKY_HANDLE", "BLUESKY_APP_PASSWORD"]
    missing = [k for k in required if not _get_key(k)]
    if missing:
        return False, (
            "Bluesky is not configured. Set the following environment variables "
            "(or add them to MAUDE's key manager):\n"
            + "\n".join(f"  - {k}" for k in missing)
            + "\n\nCreate an app password at https://bsky.app/settings/app-passwords"
        )
    return True, None


_PLATFORM_CHECK = {
    "twitter": _twitter_configured,
    "linkedin": _linkedin_configured,
    "bluesky": _bluesky_configured,
}

# ---------------------------------------------------------------------------
# Content validation
# ---------------------------------------------------------------------------

def _validate_content(content: str, platform: str) -> Tuple[str, List[str]]:
    """Validate and potentially truncate content for *platform*.

    Returns (final_content, warnings).
    """
    warnings: List[str] = []
    limit = CHAR_LIMITS.get(platform)
    if limit is None:
        return content, warnings

    if len(content) > limit:
        warnings.append(
            f"Content exceeds {PLATFORM_DISPLAY_NAMES[platform]} limit of "
            f"{limit} characters ({len(content)} chars). "
            f"Truncating to {limit} characters."
        )
        content = content[: limit - 1] + "\u2026"  # ellipsis
    return content, warnings


def _validate_image(image_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Validate an image path if provided.

    Returns (resolved_path_or_None, error_or_None).
    """
    if not image_path:
        return None, None

    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        return None, f"Image not found: {image_path}"
    if not path.is_file():
        return None, f"Image path is not a file: {image_path}"

    mime, _ = mimetypes.guess_type(str(path))
    if mime and not mime.startswith("image/"):
        return None, f"File does not appear to be an image (detected {mime}): {image_path}"

    return str(path), None


# ---------------------------------------------------------------------------
# Twitter/X posting (tweepy + API v2)
# ---------------------------------------------------------------------------

def _post_twitter(content: str, image_path: Optional[str] = None) -> str:
    """Redirect Twitter/X posting to browser-based social_post tool.

    X's API requires a paid tier for posting. Use social_post instead,
    which posts via browser automation using saved login cookies.
    """
    return (
        "Twitter/X API posting is disabled (requires paid API tier).\n"
        "Use the social_post tool instead — it posts via browser automation "
        "using your saved login session:\n"
        '  social_post(platform="x", content="...", image_path="...")\n'
        "If not logged in, run browser_login('x') first."
    )


# ---------------------------------------------------------------------------
# LinkedIn posting (requests + OAuth2)
# ---------------------------------------------------------------------------

def _get_linkedin_person_urn(access_token: str) -> Tuple[Optional[str], Optional[str]]:
    """Fetch the authenticated user's LinkedIn person URN.

    Returns (urn, error).
    """
    url = "https://api.linkedin.com/v2/userinfo"
    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 401:
            return None, (
                "LinkedIn access token is expired or invalid. "
                "Please generate a new token."
            )
        resp.raise_for_status()
        data = resp.json()
        sub = data.get("sub")
        if not sub:
            return None, "Could not determine LinkedIn member ID from /userinfo."
        return f"urn:li:person:{sub}", None
    except requests.RequestException as exc:
        return None, f"Error fetching LinkedIn profile: {exc}"


def _upload_linkedin_image(
    access_token: str, person_urn: str, image_path: str
) -> Tuple[Optional[str], Optional[str]]:
    """Upload an image to LinkedIn and return the asset URN.

    Returns (asset_urn, error).
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
    }

    # Step 1: register upload
    register_url = "https://api.linkedin.com/v2/assets?action=registerUpload"
    register_body = {
        "registerUploadRequest": {
            "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
            "owner": person_urn,
            "serviceRelationships": [
                {
                    "relationshipType": "OWNER",
                    "identifier": "urn:li:userGeneratedContent",
                }
            ],
        }
    }

    try:
        resp = requests.post(register_url, headers=headers, json=register_body, timeout=30)
        resp.raise_for_status()
        upload_info = resp.json()

        upload_url = (
            upload_info["value"]["uploadMechanism"]
            ["com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest"]["uploadUrl"]
        )
        asset = upload_info["value"]["asset"]

        # Step 2: upload binary
        mime_type, _ = mimetypes.guess_type(image_path)
        with open(image_path, "rb") as f:
            upload_headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": mime_type or "application/octet-stream",
            }
            resp = requests.put(upload_url, headers=upload_headers, data=f, timeout=60)
            resp.raise_for_status()

        return asset, None

    except requests.RequestException as exc:
        return None, f"Error uploading image to LinkedIn: {exc}"
    except (KeyError, TypeError) as exc:
        return None, f"Unexpected LinkedIn upload response structure: {exc}"


def _post_linkedin(content: str, image_path: Optional[str] = None) -> str:
    """Post to LinkedIn via the REST API.

    Returns a human-readable result string.
    """
    configured, err = _linkedin_configured()
    if not configured:
        return err

    access_token = _get_key("LINKEDIN_ACCESS_TOKEN")

    person_urn, err = _get_linkedin_person_urn(access_token)
    if err:
        return err

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
    }

    # Build the share payload
    share_content: Dict = {
        "shareCommentary": {"text": content},
        "shareMediaCategory": "NONE",
    }

    if image_path:
        asset_urn, img_err = _upload_linkedin_image(access_token, person_urn, image_path)
        if img_err:
            return img_err

        share_content["shareMediaCategory"] = "IMAGE"
        share_content["media"] = [
            {
                "status": "READY",
                "media": asset_urn,
            }
        ]

    body = {
        "author": person_urn,
        "lifecycleState": "PUBLISHED",
        "specificContent": {"com.linkedin.ugc.ShareContent": share_content},
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
    }

    try:
        resp = requests.post(
            "https://api.linkedin.com/v2/ugcPosts",
            headers=headers,
            json=body,
            timeout=30,
        )
        if resp.status_code == 401:
            return (
                "LinkedIn access token is expired or invalid. "
                "Please generate a new token."
            )
        resp.raise_for_status()
        post_id = resp.json().get("id", "unknown")
        return (
            f"Posted to LinkedIn successfully.\n"
            f"Post ID: {post_id}"
        )
    except requests.RequestException as exc:
        return f"Error posting to LinkedIn: {exc}"


# ---------------------------------------------------------------------------
# Bluesky posting (AT Protocol via requests)
# ---------------------------------------------------------------------------

_BLUESKY_API = "https://bsky.social/xrpc"


def _bluesky_create_session() -> Tuple[Optional[Dict], Optional[str]]:
    """Create an authenticated Bluesky session.

    Returns (session_dict, error).
    """
    handle = _get_key("BLUESKY_HANDLE")
    app_password = _get_key("BLUESKY_APP_PASSWORD")

    try:
        resp = requests.post(
            f"{_BLUESKY_API}/com.atproto.server.createSession",
            json={"identifier": handle, "password": app_password},
            timeout=15,
        )
        if resp.status_code == 401:
            return None, (
                "Bluesky authentication failed. Check your BLUESKY_HANDLE and "
                "BLUESKY_APP_PASSWORD values."
            )
        resp.raise_for_status()
        return resp.json(), None
    except requests.RequestException as exc:
        return None, f"Error connecting to Bluesky: {exc}"


def _bluesky_upload_image(
    session: Dict, image_path: str
) -> Tuple[Optional[Dict], Optional[str]]:
    """Upload an image blob to Bluesky.

    Returns (blob_ref, error).
    """
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/jpeg"

    try:
        with open(image_path, "rb") as f:
            image_data = f.read()

        # Bluesky enforces a 1 MB limit on image uploads
        if len(image_data) > 1_000_000:
            return None, (
                "Image exceeds Bluesky's 1 MB upload limit. "
                "Please use a smaller or more compressed image."
            )

        resp = requests.post(
            f"{_BLUESKY_API}/com.atproto.repo.uploadBlob",
            headers={
                "Authorization": f"Bearer {session['accessJwt']}",
                "Content-Type": mime_type,
            },
            data=image_data,
            timeout=30,
        )
        resp.raise_for_status()
        blob = resp.json().get("blob")
        if not blob:
            return None, "Bluesky image upload succeeded but returned no blob reference."
        return blob, None
    except requests.RequestException as exc:
        return None, f"Error uploading image to Bluesky: {exc}"


def _post_bluesky(content: str, image_path: Optional[str] = None) -> str:
    """Post to Bluesky via the AT Protocol.

    Returns a human-readable result string.
    """
    configured, err = _bluesky_configured()
    if not configured:
        return err

    session, err = _bluesky_create_session()
    if err:
        return err

    # Build the post record
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
    record: Dict = {
        "$type": "app.bsky.feed.post",
        "text": content,
        "createdAt": now,
    }

    # Attach image embed if provided
    if image_path:
        blob, img_err = _bluesky_upload_image(session, image_path)
        if img_err:
            return img_err

        record["embed"] = {
            "$type": "app.bsky.embed.images",
            "images": [
                {
                    "alt": "",
                    "image": blob,
                }
            ],
        }

    try:
        resp = requests.post(
            f"{_BLUESKY_API}/com.atproto.repo.createRecord",
            headers={"Authorization": f"Bearer {session['accessJwt']}"},
            json={
                "repo": session["did"],
                "collection": "app.bsky.feed.post",
                "record": record,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        uri = data.get("uri", "")

        # Build a web-friendly URL from the AT URI
        # at://did:plc:xxx/app.bsky.feed.post/rkey -> https://bsky.app/profile/handle/post/rkey
        handle = _get_key("BLUESKY_HANDLE")
        rkey = uri.rsplit("/", 1)[-1] if "/" in uri else ""
        web_url = f"https://bsky.app/profile/{handle}/post/{rkey}" if rkey else uri

        return (
            f"Posted to Bluesky successfully.\n"
            f"URI: {uri}\n"
            f"URL: {web_url}"
        )
    except requests.RequestException as exc:
        return f"Error posting to Bluesky: {exc}"


# ---------------------------------------------------------------------------
# Dispatcher map
# ---------------------------------------------------------------------------

_PLATFORM_POSTERS = {
    "twitter": _post_twitter,
    "linkedin": _post_linkedin,
    "bluesky": _post_bluesky,
}

# ---------------------------------------------------------------------------
# Scheduling helper
# ---------------------------------------------------------------------------

def _schedule_post(
    platform: str, content: str, schedule_expr: str, image_path: Optional[str] = None
) -> str:
    """Schedule a social media post via MAUDE's ProactiveScheduler."""
    try:
        from scheduler import get_scheduler
    except ImportError:
        return (
            "Scheduler module is not available. Cannot schedule posts.\n"
            "Post immediately by omitting the 'schedule' parameter."
        )

    scheduler = get_scheduler()

    # The scheduler works with cron expressions and prompts.  We encode the
    # post request as a MAUDE prompt that will invoke this skill at the
    # scheduled time.
    escaped_content = content.replace('"', '\\"')
    prompt_parts = [
        f'Use the post_social skill to post to {platform}.',
        f'Content: "{escaped_content}"',
    ]
    if image_path:
        prompt_parts.append(f'Image: "{image_path}"')

    prompt = " ".join(prompt_parts)
    task_name = f"Social post ({platform})"

    # Resolve shorthand schedule strings the scheduler already supports
    # (e.g. "@evening", "@daily"), plus a simple "in N hours" pattern.
    cron_expr = schedule_expr.strip()

    # Handle "in N hours/minutes" patterns
    import re

    relative_match = re.match(
        r"^in\s+(\d+)\s+(hour|hours|minute|minutes|min|mins)$",
        cron_expr,
        re.IGNORECASE,
    )
    if relative_match:
        amount = int(relative_match.group(1))
        unit = relative_match.group(2).lower()

        from datetime import timedelta

        if unit.startswith("hour"):
            target = datetime.now() + timedelta(hours=amount)
        else:
            target = datetime.now() + timedelta(minutes=amount)

        # Convert to a one-shot cron: minute hour day month *
        cron_expr = f"{target.minute} {target.hour} {target.day} {target.month} *"

    result = scheduler.schedule(
        name=task_name,
        cron=cron_expr,
        prompt=prompt,
    )
    return f"Post scheduled.\n{result}"


# ---------------------------------------------------------------------------
# Primary skill: post_social
# ---------------------------------------------------------------------------

@skill(
    name="post_social",
    description="Post to social media via API (LinkedIn, Bluesky only — for X/Twitter use the social_post tool instead)",
    version="1.1.0",
    author="MAUDE",
    triggers=["linkedin post", "bluesky post", "social media api"],
    parameters={
        "type": "object",
        "properties": {
            "platform": {
                "type": "string",
                "enum": ["twitter", "linkedin", "bluesky", "all"],
                "description": "Which platform to post to (for X/Twitter, prefer social_post tool)",
            },
            "content": {
                "type": "string",
                "description": "The post content/text",
            },
            "image_path": {
                "type": "string",
                "description": "Optional path to an image to attach",
            },
            "schedule": {
                "type": "string",
                "description": "Optional: schedule for later (e.g., 'in 2 hours', '@evening')",
            },
        },
        "required": ["platform", "content"],
    },
)
def post_social(
    platform: str,
    content: str,
    image_path: Optional[str] = None,
    schedule: Optional[str] = None,
) -> str:
    """Post content to one or more social media platforms."""

    platform = platform.strip().lower()
    if platform not in ("twitter", "linkedin", "bluesky", "all"):
        return (
            f"Unknown platform '{platform}'. "
            "Supported: twitter, linkedin, bluesky, all"
        )

    # Validate image (once, shared across platforms)
    resolved_image, img_err = _validate_image(image_path)
    if img_err:
        return img_err

    # If scheduling, hand off to the scheduler
    if schedule:
        return _schedule_post(platform, content, schedule, resolved_image)

    # Determine target platforms
    platforms = (
        ["twitter", "linkedin", "bluesky"] if platform == "all" else [platform]
    )

    results: List[str] = []
    all_warnings: List[str] = []

    for plat in platforms:
        display = PLATFORM_DISPLAY_NAMES.get(plat, plat)

        # Validate content length for this platform
        validated_content, warnings = _validate_content(content, plat)
        all_warnings.extend(warnings)

        poster = _PLATFORM_POSTERS.get(plat)
        if poster is None:
            results.append(f"[{display}] Internal error: no poster registered.")
            continue

        result = poster(validated_content, resolved_image)
        results.append(f"[{display}]\n{result}")

    output_parts: List[str] = []

    if all_warnings:
        output_parts.append("Warnings:\n" + "\n".join(f"  - {w}" for w in all_warnings))

    output_parts.append("\n---\n".join(results))

    return "\n\n".join(output_parts)


# ---------------------------------------------------------------------------
# Secondary skill: social_status
# ---------------------------------------------------------------------------

@skill(
    name="social_status",
    description="Show which social media platforms are configured and authenticated",
    version="1.0.0",
    author="MAUDE",
    triggers=["social status", "social config", "social setup"],
    parameters={
        "type": "object",
        "properties": {},
    },
)
def social_status() -> str:
    """Display configuration status for each supported social media platform."""

    lines: List[str] = ["Social Media Configuration Status:", ""]

    for plat in ("twitter", "linkedin", "bluesky"):
        display = PLATFORM_DISPLAY_NAMES[plat]
        limit = CHAR_LIMITS[plat]
        check_fn = _PLATFORM_CHECK[plat]
        configured, err = check_fn()

        if configured:
            lines.append(f"  [OK]  {display} (char limit: {limit})")
        else:
            lines.append(f"  [--]  {display} (not configured)")
            # Indent the setup instructions beneath the platform name
            for instruction_line in (err or "").split("\n"):
                lines.append(f"        {instruction_line}")

    lines.append("")
    lines.append(
        "Use the post_social skill to post content, or provide a 'schedule' "
        "parameter to schedule a post for later."
    )

    return "\n".join(lines)
