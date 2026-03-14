"""
Social Media Posting for MAUDE — posts via the running BrowserSession.

Flow: browser_login('x') → user logs in via VNC → browser stays open →
social_post('x', 'content') navigates the SAME browser to compose & publish.

No new browser instance. No cookie expiration. Session stays alive as long
as the browser is running on Spark.
"""

from __future__ import annotations

import time
import random
from pathlib import Path
from typing import Optional

try:
    from maude_core import log
except ImportError:
    def log(msg: str):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SCREENSHOT_DIR = Path.home() / ".config" / "maude" / "screenshots"


def _human_pause(lo: float = 1.0, hi: float = 3.0):
    time.sleep(random.uniform(lo, hi))


def _type_human(page, text: str, delay_lo: int = 30, delay_hi: int = 80):
    for ch in text:
        page.keyboard.type(ch, delay=random.randint(delay_lo, delay_hi))
        if random.random() < 0.04:
            time.sleep(random.uniform(0.2, 0.6))


def _first_match(page, selectors: list):
    for sel in selectors:
        try:
            loc = page.locator(sel)
            if loc.count() > 0:
                return loc.first
        except Exception:
            continue
    return None


def _screenshot(page, label: str = "social") -> Optional[str]:
    try:
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        path = SCREENSHOT_DIR / f"{label}_{ts}.png"
        page.screenshot(path=str(path))
        return str(path)
    except Exception:
        return None


def _get_page(platform: str = None):
    """Get the page for a platform from the running BrowserSession.

    If a platform-specific tab exists (from browser_login), uses that tab.
    Otherwise falls back to the default page. This keeps each platform's
    background JS running in its own tab to maintain sessions.
    """
    from browser import _get_session
    session = _get_session()
    if not session.is_active:
        return None, (
            "No browser running. Run browser_login('<platform>') first "
            "and keep the browser open after logging in."
        )
    session._touch()  # reset inactivity timer

    # Try platform-specific tab first
    if platform:
        page = session.get_platform_page(platform)
        if page is not None:
            return page, None

    # Fall back to default page
    return session._page, None


# ─────────────────────────────────────────────────────────────────────────────
# Login checks
# ─────────────────────────────────────────────────────────────────────────────

_LOGIN_SELECTORS = {
    "x": [
        '[data-testid="SideNav_AccountSwitcher_Button"]',
        '[data-testid="AppTabBar_Home_Link"]',
        'a[aria-label="Profile"]',
    ],
    "linkedin": [
        'img.global-nav__me-photo',
        '.feed-identity-module',
        'button:has(img.global-nav__me-photo)',
    ],
    "facebook": [
        '[aria-label="Your profile"]',
        '[aria-label="Account"]',
        '[aria-label="Account controls and settings"]',
    ],
    "instagram": [
        'svg[aria-label="New post"]',
        '[aria-label="New post"]',
        'svg[aria-label="Create"]',
        '[aria-label="Create"]',
        'a[href="/direct/inbox/"]',
    ],
}


def _is_logged_in(page, platform: str) -> bool:
    for sel in _LOGIN_SELECTORS.get(platform, []):
        try:
            if page.locator(sel).count() > 0:
                return True
        except Exception:
            continue
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Platform posters — all operate on the existing page
# ─────────────────────────────────────────────────────────────────────────────

def _post_x(page, content: str, image_path: Optional[str]) -> str:
    page.goto("https://x.com/home", wait_until="domcontentloaded", timeout=30_000)
    _human_pause(2, 4)

    if not _is_logged_in(page, "x"):
        return "Error: Not logged in to X. Run browser_login('x') and keep the browser open."

    compose = _first_match(page, [
        '[data-testid="tweetTextarea_0"]',
        'div[role="textbox"][data-testid="tweetTextarea_0"]',
        '[aria-label="Post text"]',
        'div.public-DraftEditor-content',
    ])

    if compose is None:
        btn = _first_match(page, [
            '[data-testid="SideNav_NewTweet_Button"]',
            'a[aria-label="Post"]',
            'a[href="/compose/post"]',
        ])
        if btn:
            btn.click()
            _human_pause(1, 2)
            compose = _first_match(page, [
                '[data-testid="tweetTextarea_0"]',
                '[aria-label="Post text"]',
                'div.public-DraftEditor-content',
            ])

    if compose is None:
        _screenshot(page, "x_compose_fail")
        return "Error: Could not find the compose area on X."

    compose.click()
    _human_pause(0.5, 1.0)
    _type_human(page, content)
    _human_pause(1, 2)

    if image_path:
        fi = _first_match(page, [
            'input[data-testid="fileInput"]',
            'input[type="file"][accept*="image"]',
        ])
        if fi:
            fi.set_input_files(image_path)
            _human_pause(2, 4)
        else:
            return "Error: Could not find file input on X for image upload."

    post_btn = _first_match(page, [
        '[data-testid="tweetButtonInline"]',
        '[data-testid="tweetButton"]',
    ])
    if not post_btn or not post_btn.is_enabled():
        _screenshot(page, "x_post_btn_fail")
        return "Error: Post button not found or disabled on X."

    post_btn.click()
    _human_pause(3, 5)

    ss = _screenshot(page, "x_posted")
    return f"Posted to X successfully.{f' Screenshot: {ss}' if ss else ''}"


def _post_linkedin(page, content: str, image_path: Optional[str]) -> str:
    page.goto("https://www.linkedin.com/feed/", wait_until="domcontentloaded", timeout=30_000)
    _human_pause(2, 4)

    if not _is_logged_in(page, "linkedin"):
        return "Error: Not logged in to LinkedIn. Run browser_login('linkedin') and keep the browser open."

    start_btn = _first_match(page, [
        'button.share-box-feed-entry__trigger',
        'button:has-text("Start a post")',
        '[aria-label="Text"]',
    ])
    if not start_btn:
        _screenshot(page, "linkedin_start_fail")
        return "Error: Could not find 'Start a post' on LinkedIn."

    start_btn.click()
    _human_pause(1, 2)

    editor = _first_match(page, [
        'div.ql-editor[data-placeholder]',
        '[role="textbox"][aria-label*="editor"]',
        '.ql-editor',
        'div[contenteditable="true"]',
    ])
    if not editor:
        _screenshot(page, "linkedin_editor_fail")
        return "Error: Could not find LinkedIn post editor."

    editor.click()
    _human_pause(0.5, 1.0)
    _type_human(page, content, 25, 60)
    _human_pause(1, 2)

    if image_path:
        try:
            media_btn = _first_match(page, [
                '[aria-label="Add media"]',
                'button[aria-label="Add media"]',
                'button:has(li-icon[type="image"])',
            ])
            if media_btn:
                media_btn.click()
                _human_pause(1, 2)
            fi = page.locator('input[type="file"]').first
            fi.set_input_files(image_path)
            _human_pause(3, 5)

            # LinkedIn opens an image Editor with Back/Next buttons.
            # Click "Next" (possibly twice) to get past it to the post view.
            for _ in range(3):
                next_btn = _first_match(page, [
                    'button:has-text("Next")',
                    '[aria-label="Next"]',
                    'button.share-box-footer__primary-btn:has-text("Next")',
                ])
                if next_btn:
                    next_btn.click()
                    _human_pause(1, 3)
                else:
                    break

            # After editor, LinkedIn may show a "Done" button
            done_btn = _first_match(page, [
                'button:has-text("Done")',
                '[aria-label="Done"]',
            ])
            if done_btn:
                done_btn.click()
                _human_pause(1, 2)

        except Exception:
            log("LinkedIn image upload failed — continuing without image")

    post_btn = _first_match(page, [
        'button.share-actions__primary-action',
        'button:has-text("Post")',
        '[aria-label="Post"]',
    ])
    if not post_btn:
        _screenshot(page, "linkedin_post_fail")
        return "Error: Could not click the Post button on LinkedIn."

    post_btn.click()
    _human_pause(3, 5)

    ss = _screenshot(page, "linkedin_posted")
    return f"Posted to LinkedIn successfully.{f' Screenshot: {ss}' if ss else ''}"


def _post_facebook(page, content: str, image_path: Optional[str]) -> str:
    page.goto("https://www.facebook.com/", wait_until="domcontentloaded", timeout=30_000)
    _human_pause(2, 4)

    # Dismiss any popups (Remember Password, notifications, etc.)
    for dismiss_sel in [
        'button:has-text("Not Now")',
        'button:has-text("Not now")',
        '[aria-label="Close"]',
        '[aria-label="Decline optional cookies"]',
    ]:
        dismiss = _first_match(page, [dismiss_sel])
        if dismiss:
            try:
                dismiss.click()
                _human_pause(0.5, 1)
            except Exception:
                pass

    if not _is_logged_in(page, "facebook"):
        return "Error: Not logged in to Facebook. Run browser_login('facebook') and keep the browser open."

    trigger = _first_match(page, [
        '[aria-label="Create a post"]',
        '[role="button"]:has-text("What\'s on your mind")',
        'span:has-text("What\'s on your mind")',
    ])
    if not trigger:
        _screenshot(page, "facebook_compose_fail")
        return "Error: Could not find Facebook compose trigger."

    trigger.click()
    _human_pause(1, 3)

    editor = _first_match(page, [
        'div[contenteditable="true"][role="textbox"]',
        '[aria-label*="What\'s on your mind"]',
        'div[contenteditable="true"][data-lexical-editor="true"]',
    ])
    if not editor:
        _screenshot(page, "facebook_editor_fail")
        return "Error: Could not find Facebook post editor."

    editor.click()
    _human_pause(0.5, 1.0)
    _type_human(page, content, 25, 60)
    _human_pause(1, 2)

    if image_path:
        try:
            # Click the Photo/video button in the compose dialog
            photo_btn = _first_match(page, [
                '[aria-label="Photo/video"]',
                '[aria-label="Photo/Video"]',
                '[aria-label="Photo/video "]',
                'div[role="button"]:has-text("Photo/video")',
                'div[role="button"]:has-text("Photo/Video")',
            ])
            if photo_btn:
                photo_btn.click()
                _human_pause(1, 2)

            # Try multiple file input selectors — Facebook changes these
            fi = None
            for fi_sel in [
                'input[type="file"][accept*="image"]',
                'input[type="file"][accept*="video"]',
                'input[type="file"]',
            ]:
                loc = page.locator(fi_sel)
                if loc.count() > 0:
                    fi = loc.first
                    break

            if fi:
                fi.set_input_files(image_path)
                _human_pause(3, 5)
            else:
                log("Facebook: no file input found — continuing without image")
        except Exception as e:
            log(f"Facebook image upload failed: {e} — continuing without image")

    post_btn = _first_match(page, [
        '[aria-label="Post"]',
        'div[aria-label="Post"]',
    ])
    if not post_btn:
        _screenshot(page, "facebook_post_fail")
        return "Error: Could not click Facebook Post button."

    post_btn.click()
    _human_pause(3, 5)

    ss = _screenshot(page, "facebook_posted")
    return f"Posted to Facebook successfully.{f' Screenshot: {ss}' if ss else ''}"


def _post_instagram(page, content: str, image_path: str) -> str:
    page.goto("https://www.instagram.com/", wait_until="domcontentloaded", timeout=30_000)
    _human_pause(2, 4)

    if not _is_logged_in(page, "instagram"):
        return "Error: Not logged in to Instagram. Run browser_login('instagram') and keep the browser open."

    # Instagram has "Create" and/or "Post" in the sidebar — click the outer
    # container (a/div), not the inner SVG, so the click actually registers.
    # Try the direct /create/select/... URL as a fallback.
    opened = False
    for sel in [
        'a:has(span:has-text("Post"))',
        'a:has(span:has-text("Create"))',
        'div[role="button"]:has(span:has-text("Create"))',
        '[aria-label="New post"]',
        '[aria-label="Create"]',
        'a[href*="/create"]',
    ]:
        btn = _first_match(page, [sel])
        if btn:
            btn.click()
            _human_pause(2, 3)
            # Check if the create dialog appeared (file input or "Select from computer")
            if page.locator('input[type="file"]').count() > 0:
                opened = True
                break
            if _first_match(page, ['button:has-text("Select from computer")', 'button:has-text("Select From Computer")']):
                opened = True
                break

    if not opened:
        # Last resort: navigate directly to the create page
        page.goto("https://www.instagram.com/create/select/", wait_until="domcontentloaded", timeout=15_000)
        _human_pause(2, 4)
        if page.locator('input[type="file"]').count() == 0 and not _first_match(page, ['button:has-text("Select from computer")']):
            _screenshot(page, "instagram_newpost_fail")
            return "Error: Could not open Instagram create dialog."

    # Click "Select from computer" if visible (exposes the file input)
    select_btn = _first_match(page, [
        'button:has-text("Select from computer")',
        'button:has-text("Select From Computer")',
        'button:has-text("Select from")',
    ])
    if select_btn:
        select_btn.click()
        _human_pause(1, 2)

    # Upload the image via the file input
    try:
        fi = page.locator('input[type="file"]').first
        fi.set_input_files(image_path)
        _human_pause(3, 5)
    except Exception as e:
        _screenshot(page, "instagram_upload_fail")
        return f"Error uploading image to Instagram: {e}"

    # Click through Crop → Filter → Caption steps (Next button, up to 3 times)
    for _ in range(3):
        next_btn = _first_match(page, [
            'button:has-text("Next")',
            '[aria-label="Next"]',
            'div[role="button"]:has-text("Next")',
        ])
        if next_btn:
            next_btn.click()
            _human_pause(1, 3)
        else:
            break

    # Write caption
    caption_box = _first_match(page, [
        '[aria-label="Write a caption..."]',
        '[aria-label="Write a caption\u2026"]',
        'textarea[aria-label*="caption"]',
        'div[role="textbox"]',
        '[contenteditable="true"]',
    ])
    if caption_box:
        caption_box.click()
        _human_pause(0.5, 1.0)
        _type_human(page, content, 25, 60)
        _human_pause(1, 2)

    share_btn = _first_match(page, [
        'button:has-text("Share")',
        '[aria-label="Share"]',
        'div[role="button"]:has-text("Share")',
    ])
    if not share_btn:
        _screenshot(page, "instagram_share_fail")
        return "Error: Could not find Instagram Share button."

    share_btn.click()
    _human_pause(5, 8)

    ss = _screenshot(page, "instagram_posted")
    return f"Posted to Instagram successfully.{f' Screenshot: {ss}' if ss else ''}"


# ─────────────────────────────────────────────────────────────────────────────
# Public tool function
# ─────────────────────────────────────────────────────────────────────────────

_POSTERS = {
    "x": _post_x,
    "linkedin": _post_linkedin,
    "facebook": _post_facebook,
    "instagram": _post_instagram,
}


def social_post(platform: str, content: str, image_path: str = None) -> str:
    """Post content using the running browser session.

    Requires browser_login('<platform>') first — keep the browser open.
    The same browser that you logged into is used to post.

    All Playwright operations are dispatched to the BrowserSession's worker
    thread — the gateway runs each tool call in a new thread, but Playwright
    requires all page operations on the thread that created the browser.
    """
    platform = platform.lower().strip()
    if platform == "twitter":
        platform = "x"

    if platform not in _POSTERS:
        return (
            f"Error: Unsupported platform '{platform}'. "
            f"Supported: x, linkedin, facebook, instagram"
        )

    if platform == "instagram" and not image_path:
        return "Error: Instagram requires an image. Provide image_path."

    if image_path:
        p = Path(image_path)
        if not p.exists():
            return f"Error: Image not found: {image_path}"
        image_path = str(p.resolve())

    from browser import _get_session
    session = _get_session()

    def _do_post():
        page, err = _get_page(platform)
        if err:
            return err
        try:
            return _POSTERS[platform](page, content, image_path)
        except Exception as e:
            _screenshot(page, f"{platform}_error")
            return f"Error posting to {platform}: {e}"

    return session._run_on_worker(_do_post)


# ─────────────────────────────────────────────────────────────────────────────
# Tool definitions (OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────

SOCIAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "social_post",
            "description": (
                "Post content to a social media platform (X, LinkedIn, Facebook, Instagram) "
                "using the running browser session. Requires browser_login('<platform>') first — "
                "the browser must be kept open after logging in. The same live browser is used "
                "to navigate and post, so sessions never expire. Instagram requires an image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "platform": {
                        "type": "string",
                        "enum": ["x", "linkedin", "facebook", "instagram"],
                        "description": "Target platform to post to."
                    },
                    "content": {
                        "type": "string",
                        "description": "The text content of the post (tweet, status update, caption)."
                    },
                    "image_path": {
                        "type": "string",
                        "description": (
                            "Optional absolute path to an image file to attach. "
                            "Required for Instagram."
                        )
                    }
                },
                "required": ["platform", "content"]
            }
        }
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher (prefix-based registration)
# ─────────────────────────────────────────────────────────────────────────────

def execute_social_tool(name: str, args: dict) -> str:
    dispatch = {
        "social_post": lambda a: social_post(
            a.get("platform", ""),
            a.get("content", ""),
            a.get("image_path"),
        ),
    }
    handler = dispatch.get(name)
    if handler is None:
        return f"Error: Unknown social tool '{name}'. Available: {', '.join(dispatch.keys())}"
    try:
        return handler(args)
    except Exception as e:
        return f"Error executing {name}: {e}"
