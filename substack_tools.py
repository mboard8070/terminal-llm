"""
Substack Tools for MAUDE - Draft and manage Substack newsletter posts.

Uses the unofficial Substack API with session-based authentication.

Requires SUBSTACK_PUBLICATION_URL, SUBSTACK_EMAIL, and SUBSTACK_PASSWORD in .env
"""

import os
import re
import json
import requests
from typing import Optional


# Configuration from environment
SUBSTACK_URL = os.environ.get("SUBSTACK_PUBLICATION_URL", "").rstrip("/")
SUBSTACK_EMAIL = os.environ.get("SUBSTACK_EMAIL", "")
SUBSTACK_PASSWORD = os.environ.get("SUBSTACK_PASSWORD", "")

# Cached authenticated session
_session: Optional[requests.Session] = None


# ─────────────────────────────────────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_setup() -> str:
    """Check if Substack credentials are configured. Returns 'OK' or an error message."""
    if not SUBSTACK_URL:
        return "Error: SUBSTACK_PUBLICATION_URL not set in environment. Add it to your .env file."
    if not SUBSTACK_EMAIL:
        return "Error: SUBSTACK_EMAIL not set in environment. Add it to your .env file."
    if not SUBSTACK_PASSWORD:
        return "Error: SUBSTACK_PASSWORD not set in environment. Add it to your .env file."
    return "OK"


def _get_session() -> requests.Session:
    """
    Get an authenticated requests.Session for the Substack API.

    Authenticates by POSTing to https://substack.com/api/v1/login with email/password.
    The response sets session cookies that are reused on subsequent requests.
    The session is cached at module level so authentication only happens once.
    """
    global _session

    if _session is not None:
        return _session

    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "MAUDE/1.0 (Substack Integration)",
    })

    login_url = "https://substack.com/api/v1/login"
    login_payload = {
        "email": SUBSTACK_EMAIL,
        "password": SUBSTACK_PASSWORD,
        "for_pub": "",
    }

    try:
        resp = session.post(login_url, json=login_payload, timeout=30)
    except requests.RequestException as e:
        raise ConnectionError(f"Failed to connect to Substack login endpoint: {e}")

    if resp.status_code != 200:
        # Try to extract a useful error message from the response
        try:
            error_data = resp.json()
            error_msg = error_data.get("error", error_data.get("errors", resp.text))
        except (ValueError, KeyError):
            error_msg = resp.text[:500]
        raise RuntimeError(
            f"Substack login failed (HTTP {resp.status_code}): {error_msg}"
        )

    # Check for error in the JSON response body
    try:
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"Substack login error: {data['error']}")
    except ValueError:
        pass  # Non-JSON 200 response is acceptable if cookies were set

    _session = session
    return _session


def _body_to_prosemirror(body: str) -> dict:
    """
    Convert a plain-text body string into Substack's ProseMirror JSON format.

    Splits the text on double newlines to form paragraphs. Each paragraph
    becomes a ProseMirror paragraph node. Empty paragraphs are preserved
    as empty paragraph nodes (visual spacing).
    """
    paragraphs = body.split("\n\n")
    content = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph:
            # A paragraph may contain single newlines (soft breaks).
            # Split on single newlines and interleave with hardBreak nodes.
            lines = paragraph.split("\n")
            inline_content = []
            for i, line in enumerate(lines):
                if line:
                    inline_content.append({"type": "text", "text": line})
                if i < len(lines) - 1:
                    inline_content.append({"type": "hardBreak"})
            content.append({
                "type": "paragraph",
                "content": inline_content if inline_content else [{"type": "text", "text": ""}],
            })
        else:
            # Empty paragraph for spacing
            content.append({"type": "paragraph"})

    if not content:
        content.append({"type": "paragraph"})

    return {"type": "doc", "content": content}


def _strip_html(html_text: str) -> str:
    """Strip HTML tags from text and return plain text."""
    if not html_text:
        return ""
    # Remove HTML tags
    clean = re.sub(r"<[^>]+>", "", html_text)
    # Decode common HTML entities
    clean = clean.replace("&amp;", "&")
    clean = clean.replace("&lt;", "<")
    clean = clean.replace("&gt;", ">")
    clean = clean.replace("&quot;", '"')
    clean = clean.replace("&#39;", "'")
    clean = clean.replace("&nbsp;", " ")
    # Collapse whitespace
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _format_date(date_str: str) -> str:
    """Format an ISO date string to a more readable form, or return as-is."""
    if not date_str:
        return "N/A"
    try:
        # Truncate to just the date+time portion
        return date_str[:19].replace("T", " ")
    except Exception:
        return date_str


# ─────────────────────────────────────────────────────────────────────────────
# Public API Functions
# ─────────────────────────────────────────────────────────────────────────────

def substack_create_draft(title: str, body: str, subtitle: str = "", audience: str = "everyone") -> str:
    """
    Create a new draft post on Substack.

    Args:
        title: The post title.
        body: The post body as plain text. Double newlines create paragraph breaks.
        subtitle: Optional subtitle for the post.
        audience: 'everyone' (free) or 'only_paid' (paid subscribers only).

    Returns:
        String with the draft ID and edit URL, or an error message.
    """
    status = _check_setup()
    if status != "OK":
        return status

    try:
        session = _get_session()

        # Convert body text to ProseMirror JSON format
        draft_body = _body_to_prosemirror(body)

        payload = {
            "draft_title": title,
            "draft_subtitle": subtitle,
            "draft_body": json.dumps(draft_body),
            "audience": audience,
            "type": "newsletter",
        }

        url = f"{SUBSTACK_URL}/api/v1/drafts/"
        resp = session.post(url, json=payload, timeout=30)

        if resp.status_code not in (200, 201):
            try:
                error_detail = resp.json()
            except ValueError:
                error_detail = resp.text[:500]
            return f"Error creating draft (HTTP {resp.status_code}): {error_detail}"

        data = resp.json()
        draft_id = data.get("id", "unknown")
        edit_url = f"{SUBSTACK_URL}/publish/post/{draft_id}"

        output = [
            "Draft created successfully!",
            f"  Title: {title}",
        ]
        if subtitle:
            output.append(f"  Subtitle: {subtitle}")
        output.extend([
            f"  Audience: {audience}",
            f"  Draft ID: {draft_id}",
            f"  Edit URL: {edit_url}",
        ])

        return "\n".join(output)

    except Exception as e:
        return f"Error creating draft: {e}"


def substack_list_drafts(limit: int = 10) -> str:
    """
    List existing draft posts on Substack.

    Args:
        limit: Maximum number of drafts to return (default 10).

    Returns:
        Formatted string with draft titles, IDs, dates, and edit URLs.
    """
    status = _check_setup()
    if status != "OK":
        return status

    try:
        session = _get_session()

        url = f"{SUBSTACK_URL}/api/v1/drafts/?limit={limit}"
        resp = session.get(url, timeout=30)

        if resp.status_code != 200:
            try:
                error_detail = resp.json()
            except ValueError:
                error_detail = resp.text[:500]
            return f"Error listing drafts (HTTP {resp.status_code}): {error_detail}"

        drafts = resp.json()

        if not drafts:
            return "No drafts found."

        # The response may be a list or a dict with a list inside
        if isinstance(drafts, dict):
            drafts = drafts.get("drafts", drafts.get("results", []))

        if not drafts:
            return "No drafts found."

        output = [f"Found {len(drafts)} draft(s):", ""]
        for draft in drafts:
            draft_id = draft.get("id", "unknown")
            title = draft.get("draft_title", draft.get("title", "(untitled)"))
            subtitle = draft.get("draft_subtitle", draft.get("subtitle", ""))
            created = _format_date(draft.get("draft_created_at", draft.get("created_at", "")))
            edit_url = f"{SUBSTACK_URL}/publish/post/{draft_id}"

            output.append(f"Title: {title}")
            if subtitle:
                output.append(f"  Subtitle: {subtitle}")
            output.append(f"  Draft ID: {draft_id}")
            output.append(f"  Created: {created}")
            output.append(f"  Edit URL: {edit_url}")
            output.append("")

        return "\n".join(output)

    except Exception as e:
        return f"Error listing drafts: {e}"


def substack_list_posts(limit: int = 10, offset: int = 0) -> str:
    """
    List published posts on Substack.

    Args:
        limit: Maximum number of posts to return (default 10).
        offset: Number of posts to skip (for pagination).

    Returns:
        Formatted string with post titles, dates, audiences, and URLs.
    """
    status = _check_setup()
    if status != "OK":
        return status

    try:
        session = _get_session()

        url = f"{SUBSTACK_URL}/api/v1/posts?limit={limit}&offset={offset}"
        resp = session.get(url, timeout=30)

        if resp.status_code != 200:
            try:
                error_detail = resp.json()
            except ValueError:
                error_detail = resp.text[:500]
            return f"Error listing posts (HTTP {resp.status_code}): {error_detail}"

        posts = resp.json()

        # Handle both list and dict response formats
        if isinstance(posts, dict):
            posts = posts.get("posts", posts.get("results", []))

        if not posts:
            return "No published posts found."

        output = [f"Found {len(posts)} published post(s):", ""]
        for post in posts:
            post_id = post.get("id", "unknown")
            title = post.get("title", "(untitled)")
            subtitle = post.get("subtitle", "")
            slug = post.get("slug", "")
            audience = post.get("audience", "everyone")
            post_date = _format_date(post.get("post_date", post.get("published_at", "")))
            canonical_url = post.get("canonical_url", "")
            if not canonical_url and slug:
                canonical_url = f"{SUBSTACK_URL}/p/{slug}"

            output.append(f"Title: {title}")
            if subtitle:
                output.append(f"  Subtitle: {subtitle}")
            output.append(f"  Post ID: {post_id}")
            output.append(f"  Date: {post_date}")
            output.append(f"  Audience: {audience}")
            if slug:
                output.append(f"  Slug: {slug}")
            output.append(f"  URL: {canonical_url}")
            output.append("")

        return "\n".join(output)

    except Exception as e:
        return f"Error listing posts: {e}"


def substack_get_post(post_id: str) -> str:
    """
    Get a specific post or draft by ID.

    Args:
        post_id: The post/draft ID to retrieve.

    Returns:
        Formatted string with post details including title, body text (truncated),
        audience, publish date, and any available stats.
    """
    status = _check_setup()
    if status != "OK":
        return status

    try:
        session = _get_session()

        url = f"{SUBSTACK_URL}/api/v1/posts/{post_id}"
        resp = session.get(url, timeout=30)

        if resp.status_code != 200:
            try:
                error_detail = resp.json()
            except ValueError:
                error_detail = resp.text[:500]
            return f"Error getting post (HTTP {resp.status_code}): {error_detail}"

        post = resp.json()

        title = post.get("title", post.get("draft_title", "(untitled)"))
        subtitle = post.get("subtitle", post.get("draft_subtitle", ""))
        audience = post.get("audience", "everyone")
        post_date = _format_date(
            post.get("post_date", post.get("published_at", post.get("draft_created_at", "")))
        )
        slug = post.get("slug", "")
        canonical_url = post.get("canonical_url", "")
        if not canonical_url and slug:
            canonical_url = f"{SUBSTACK_URL}/p/{slug}"

        # Extract body text
        body_html = post.get("body_html", post.get("body", ""))
        body_text = _strip_html(body_html)
        if len(body_text) > 2000:
            body_text = body_text[:2000] + "... [truncated]"

        output = [
            f"Title: {title}",
        ]
        if subtitle:
            output.append(f"Subtitle: {subtitle}")
        output.extend([
            f"Post ID: {post_id}",
            f"Date: {post_date}",
            f"Audience: {audience}",
        ])
        if slug:
            output.append(f"Slug: {slug}")
        if canonical_url:
            output.append(f"URL: {canonical_url}")

        # Stats (if available on published posts)
        reactions = post.get("reactions", {})
        if reactions:
            heart_count = reactions.get("\u2764", 0)
            if heart_count:
                output.append(f"Likes: {heart_count}")

        comment_count = post.get("comment_count")
        if comment_count is not None:
            output.append(f"Comments: {comment_count}")

        # Substack may also include these stats
        for stat_key in ("email_sent_count", "email_open_count", "web_view_count"):
            val = post.get(stat_key)
            if val is not None:
                label = stat_key.replace("_", " ").title()
                output.append(f"{label}: {val}")

        output.append("")
        output.append("Body:")
        output.append(body_text if body_text else "(no body content)")

        return "\n".join(output)

    except Exception as e:
        return f"Error getting post: {e}"


def substack_update_draft(draft_id: str, title: str = None, body: str = None, subtitle: str = None) -> str:
    """
    Update an existing draft post on Substack.

    Only the fields that are provided (not None) will be updated.

    Args:
        draft_id: The ID of the draft to update.
        title: New title (optional).
        body: New body text (optional). Double newlines create paragraph breaks.
        subtitle: New subtitle (optional).

    Returns:
        Confirmation string or error message.
    """
    status = _check_setup()
    if status != "OK":
        return status

    try:
        session = _get_session()

        payload = {}
        if title is not None:
            payload["draft_title"] = title
        if subtitle is not None:
            payload["draft_subtitle"] = subtitle
        if body is not None:
            draft_body = _body_to_prosemirror(body)
            payload["draft_body"] = json.dumps(draft_body)

        if not payload:
            return "Error: No fields provided to update. Specify at least one of: title, body, subtitle."

        url = f"{SUBSTACK_URL}/api/v1/drafts/{draft_id}"
        resp = session.put(url, json=payload, timeout=30)

        if resp.status_code not in (200, 201):
            try:
                error_detail = resp.json()
            except ValueError:
                error_detail = resp.text[:500]
            return f"Error updating draft (HTTP {resp.status_code}): {error_detail}"

        data = resp.json()
        updated_title = data.get("draft_title", data.get("title", "(untitled)"))
        edit_url = f"{SUBSTACK_URL}/publish/post/{draft_id}"

        updated_fields = []
        if title is not None:
            updated_fields.append("title")
        if subtitle is not None:
            updated_fields.append("subtitle")
        if body is not None:
            updated_fields.append("body")

        output = [
            "Draft updated successfully!",
            f"  Title: {updated_title}",
            f"  Draft ID: {draft_id}",
            f"  Updated fields: {', '.join(updated_fields)}",
            f"  Edit URL: {edit_url}",
        ]

        return "\n".join(output)

    except Exception as e:
        return f"Error updating draft: {e}"


def substack_delete_draft(draft_id: str) -> str:
    """
    Delete a draft post on Substack.

    Args:
        draft_id: The ID of the draft to delete.

    Returns:
        Confirmation string or error message.
    """
    status = _check_setup()
    if status != "OK":
        return status

    try:
        session = _get_session()

        url = f"{SUBSTACK_URL}/api/v1/drafts/{draft_id}"
        resp = session.delete(url, timeout=30)

        if resp.status_code not in (200, 204):
            try:
                error_detail = resp.json()
            except ValueError:
                error_detail = resp.text[:500]
            return f"Error deleting draft (HTTP {resp.status_code}): {error_detail}"

        return f"Draft deleted successfully. (Draft ID: {draft_id})"

    except Exception as e:
        return f"Error deleting draft: {e}"


def substack_get_stats() -> str:
    """
    Get publication stats from Substack.

    Returns:
        Formatted string with publication name, subscriber count, post count,
        and other available statistics.
    """
    status = _check_setup()
    if status != "OK":
        return status

    try:
        session = _get_session()

        url = f"{SUBSTACK_URL}/api/v1/publication"
        resp = session.get(url, timeout=30)

        if resp.status_code != 200:
            try:
                error_detail = resp.json()
            except ValueError:
                error_detail = resp.text[:500]
            return f"Error getting publication stats (HTTP {resp.status_code}): {error_detail}"

        pub = resp.json()

        name = pub.get("name", "Unknown")
        subdomain = pub.get("subdomain", "")
        pub_id = pub.get("id", "unknown")
        author_name = pub.get("author_name", "")
        custom_domain = pub.get("custom_domain", "")
        base_url = pub.get("base_url", SUBSTACK_URL)

        output = [
            "Substack Publication Stats",
            "=" * 40,
            f"Name: {name}",
            f"Publication ID: {pub_id}",
        ]

        if subdomain:
            output.append(f"Subdomain: {subdomain}")
        if author_name:
            output.append(f"Author: {author_name}")
        if custom_domain:
            output.append(f"Custom Domain: {custom_domain}")
        output.append(f"URL: {base_url}")

        # Subscriber / follower counts (field names may vary)
        for key in ("subscriber_count", "active_subscriber_count", "free_subscriber_count",
                     "paid_subscriber_count", "follower_count", "email_subscriber_count"):
            val = pub.get(key)
            if val is not None:
                label = key.replace("_", " ").title()
                output.append(f"{label}: {val}")

        # Post count
        post_count = pub.get("post_count")
        if post_count is not None:
            output.append(f"Post Count: {post_count}")

        # Other interesting stats
        for key in ("has_posts", "has_podcast", "paid_enabled", "stripe_connected",
                     "default_coupon_pct_off"):
            val = pub.get(key)
            if val is not None:
                label = key.replace("_", " ").title()
                output.append(f"{label}: {val}")

        return "\n".join(output)

    except Exception as e:
        return f"Error getting publication stats: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# CLI for testing
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    # Reload env vars from .env in case they weren't loaded
    load_dotenv(override=True)

    # Re-read after dotenv load
    SUBSTACK_URL = os.environ.get("SUBSTACK_PUBLICATION_URL", "").rstrip("/")
    SUBSTACK_EMAIL = os.environ.get("SUBSTACK_EMAIL", "")
    SUBSTACK_PASSWORD = os.environ.get("SUBSTACK_PASSWORD", "")

    if "--test" in sys.argv:
        print("Substack Tools - Connection Test")
        print("=" * 40)

        status = _check_setup()
        print(f"Config check: {status}")

        if status == "OK":
            print("\nAttempting login...")
            try:
                session = _get_session()
                print("Login successful!")
            except Exception as e:
                print(f"Login failed: {e}")
                sys.exit(1)

            print("\nGetting publication stats...")
            print(substack_get_stats())

            print("\nListing published posts...")
            print(substack_list_posts(limit=3))

            print("\nListing drafts...")
            print(substack_list_drafts(limit=3))

    elif "--create-test-draft" in sys.argv:
        print("Creating a test draft...")
        result = substack_create_draft(
            title="Test Draft from MAUDE",
            body="This is a test paragraph.\n\nThis is a second paragraph.\n\nAnd a third one with some detail.",
            subtitle="Automated test post",
        )
        print(result)

    else:
        print("Substack Tools for MAUDE")
        print("=" * 40)
        print("Usage:")
        print("  python substack_tools.py --test               # Test connection and list posts")
        print("  python substack_tools.py --create-test-draft   # Create a test draft")
        print("")
        print("Available functions:")
        print("  substack_create_draft(title, body, subtitle, audience)")
        print("  substack_list_drafts(limit)")
        print("  substack_list_posts(limit, offset)")
        print("  substack_get_post(post_id)")
        print("  substack_update_draft(draft_id, title, body, subtitle)")
        print("  substack_delete_draft(draft_id)")
        print("  substack_get_stats()")
