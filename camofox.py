"""
Camofox browser backend for MAUDE — anti-detection Firefox via REST API.

Camofox-browser wraps Camoufox (Firefox fork with C++ fingerprint spoofing)
behind a REST API. When CAMOFOX_URL is set, MAUDE routes browser operations
through this module instead of local Playwright.

Setup:
    # Docker (recommended)
    docker run -d -p 9377:9377 --name camofox jo-inc/camofox-browser

    # Or npm
    git clone https://github.com/jo-inc/camofox-browser && cd camofox-browser
    npm install && npm start

Then set CAMOFOX_URL=http://localhost:9377 in your environment.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

import requests

from maude_bootstrap import ensure_local_maude

ensure_local_maude()

from maude.config import runtime_paths

try:
    from maude_core import log
except ImportError:
    def log(msg: str):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_TIMEOUT = 30
_SNAPSHOT_MAX_CHARS = 12_000
SCREENSHOT_DIR = runtime_paths().screenshots_dir


def get_camofox_url() -> str:
    return os.environ.get("CAMOFOX_URL", "").rstrip("/")


def is_camofox_mode() -> bool:
    return bool(get_camofox_url())


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

_user_id: str | None = None
_tab_id: str | None = None
_ref_map: dict[str, dict] = {}  # @eN → {role, name} from last snapshot

# Same regex as browser.py for annotating interactive elements
_ROLE_LINE_RE = re.compile(
    r"^(\s*-\s+)("
    r"link|button|textbox|checkbox|radio|combobox|searchbox|"
    r"slider|switch|tab|menuitem|option|spinbutton|treeitem"
    r')(\s+"[^"]*")?(.*)$'
)


def _ensure_session():
    global _user_id
    if not _user_id:
        _user_id = f"maude_{uuid.uuid4().hex[:10]}"


def _post(path: str, body: dict, timeout: int = _TIMEOUT) -> dict:
    resp = requests.post(f"{get_camofox_url()}{path}", json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _get(path: str, params: dict = None, timeout: int = _TIMEOUT) -> dict:
    resp = requests.get(f"{get_camofox_url()}{path}", params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _delete(path: str, timeout: int = _TIMEOUT) -> dict:
    resp = requests.delete(f"{get_camofox_url()}{path}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────


def check_available() -> bool:
    """Verify the Camofox server is reachable."""
    try:
        resp = requests.get(f"{get_camofox_url()}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Browser operations — same interface as BrowserSession methods
# ─────────────────────────────────────────────────────────────────────────────


def open_page(url: str) -> str:
    """Navigate to URL via Camofox. Creates tab if needed."""
    global _tab_id
    _ensure_session()

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        if not _tab_id:
            data = _post("/tabs", {
                "userId": _user_id,
                "sessionKey": "maude_default",
                "url": url,
            }, timeout=60)
            _tab_id = data.get("tabId")
        else:
            data = _post(f"/tabs/{_tab_id}/navigate", {
                "userId": _user_id,
                "url": url,
            }, timeout=60)

        title = data.get("title", "(no title)")
        final_url = data.get("url", url)
        return f"Opened: {title}\nURL: {final_url}"

    except requests.ConnectionError:
        return (
            f"Error: Cannot connect to Camofox at {get_camofox_url()}. "
            "Start with: docker run -d -p 9377:9377 --name camofox jo-inc/camofox-browser"
        )
    except Exception as e:
        return f"Error navigating: {e}"


def snapshot(max_chars: int = _SNAPSHOT_MAX_CHARS) -> str:
    """Get accessibility tree snapshot with @eN refs."""
    global _ref_map
    if not _tab_id:
        return "Error: No browser session. Use browser_open first."

    try:
        data = _get(f"/tabs/{_tab_id}/snapshot", {"userId": _user_id})
        raw = data.get("snapshot", "")

        # Annotate interactive elements with @eN refs (same logic as browser.py)
        _ref_map.clear()
        ref_counter = 0
        annotated_lines = []

        for line in raw.split("\n"):
            m = _ROLE_LINE_RE.match(line)
            if m:
                ref_counter += 1
                ref = f"@e{ref_counter}"
                indent, role, name_part, rest = m.groups()
                name_clean = name_part.strip().strip('"') if name_part else ""
                _ref_map[ref] = {"role": role, "name": name_clean}
                annotated_lines.append(f"{indent}{role}{name_part or ''} [{ref}]{rest}")
            else:
                annotated_lines.append(line)

        snapshot_text = "\n".join(annotated_lines)
        if len(snapshot_text) > max_chars:
            snapshot_text = snapshot_text[:max_chars] + "\n\n[... truncated ...]"

        header = f"Interactive elements: {ref_counter}\n\n"
        return header + snapshot_text

    except Exception as e:
        return f"Error getting snapshot: {e}"


def click(selector: str) -> str:
    """Click element by @eN ref or text."""
    if not _tab_id:
        return "Error: No browser session. Use browser_open first."

    try:
        if selector.startswith("@e"):
            # Use Camofox ref-based click
            clean_ref = selector.lstrip("@")
            data = _post(f"/tabs/{_tab_id}/click", {
                "userId": _user_id,
                "ref": clean_ref,
            })
            return f"Clicked '{selector}'.\nURL: {data.get('url', '')}"
        else:
            # For CSS/text selectors, get snapshot, find matching ref, click it
            return "Error: Camofox requires @eN refs. Run browser_snapshot first to get element refs."

    except Exception as e:
        return f"Error clicking '{selector}': {e}"


def type_text(selector: str, text: str) -> str:
    """Type text into element by @eN ref."""
    if not _tab_id:
        return "Error: No browser session. Use browser_open first."

    try:
        if selector.startswith("@e"):
            clean_ref = selector.lstrip("@")
            _post(f"/tabs/{_tab_id}/type", {
                "userId": _user_id,
                "ref": clean_ref,
                "text": text,
            })
            return f"Typed '{text[:50]}{'...' if len(text) > 50 else ''}' into '{selector}'."
        else:
            return "Error: Camofox requires @eN refs. Run browser_snapshot first to get element refs."

    except Exception as e:
        return f"Error typing into '{selector}': {e}"


def navigate(url: str) -> str:
    """Navigate to URL in existing session."""
    return open_page(url)


def take_screenshot() -> str:
    """Take screenshot and save to disk."""
    if not _tab_id:
        return "Error: No browser session. Use browser_open first."

    try:
        resp = requests.get(
            f"{get_camofox_url()}/tabs/{_tab_id}/screenshot",
            params={"userId": _user_id},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()

        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        import time
        path = SCREENSHOT_DIR / f"camofox_{int(time.time())}.png"
        path.write_bytes(resp.content)
        return f"Screenshot saved: {path}"

    except Exception as e:
        return f"Error taking screenshot: {e}"


def close() -> str:
    """Close the Camofox session."""
    global _tab_id, _user_id
    try:
        if _user_id:
            _delete(f"/sessions/{_user_id}")
        _tab_id = None
        _user_id = None
        _ref_map.clear()
        return "Browser session closed."
    except Exception as e:
        _tab_id = None
        _user_id = None
        _ref_map.clear()
        return f"Session closed (with warning: {e})"
