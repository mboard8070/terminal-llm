"""
Social Media Posting for MAUDE — browser-based posting to X, LinkedIn, Facebook, Instagram.

Uses Playwright with persistent browser data dir so cookies from browser_login are reused.
X requires VNC/visible mode (bot detection blocks headless).
LinkedIn, Facebook, Instagram try headless first with VNC fallback.
"""

from __future__ import annotations

import os
import time
import random
import shutil
from pathlib import Path
from typing import Optional

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

try:
    from maude_core import log
except ImportError:
    def log(msg: str):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BROWSER_DATA_DIR = Path.home() / ".config" / "maude" / "browser_data"
SCREENSHOT_DIR = Path.home() / ".config" / "maude" / "screenshots"

# Anti-detection stealth script injected on every page load
STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
window.chrome = window.chrome || { runtime: {} };
const _origQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (p) => (
    p.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : _origQuery(p)
);
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _human_pause(lo: float = 1.0, hi: float = 3.0):
    """Random pause between actions."""
    time.sleep(random.uniform(lo, hi))


def _type_human(page, text: str, delay_lo: int = 30, delay_hi: int = 80):
    """Type text character-by-character with human-like cadence."""
    for ch in text:
        page.keyboard.type(ch, delay=random.randint(delay_lo, delay_hi))
        if random.random() < 0.04:  # occasional thinking pause
            time.sleep(random.uniform(0.2, 0.6))


def _first_match(page, selectors: list):
    """Return the first Playwright locator that has at least one hit, or None."""
    for sel in selectors:
        try:
            loc = page.locator(sel)
            if loc.count() > 0:
                return loc.first
        except Exception:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SocialPoster — manages its own Playwright instance
# ─────────────────────────────────────────────────────────────────────────────

class SocialPoster:
    """One-shot poster: start browser → post → close."""

    def __init__(self):
        self._pw = None
        self._ctx = None
        self._page = None

    # ── browser lifecycle ─────────────────────────────────────────────────

    def _start(self, headless: bool = True):
        """Launch persistent Chromium context sharing MAUDE's cookie store."""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            )

        BROWSER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

        # Clean stale lock files left by a previous crash
        for stale in ("SingletonLock", "SingletonSocket", "SingletonCookie"):
            p = BROWSER_DATA_DIR / stale
            if p.exists():
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)

        # If visible mode requested and no display → start VNC
        display = os.environ.get("DISPLAY")
        if not headless and not display:
            try:
                from vnc_session import get_vnc_session
                vnc = get_vnc_session()
                result = vnc.start()
                if result.startswith("Error"):
                    raise RuntimeError(result)
                os.environ["DISPLAY"] = vnc.display
                log(f"VNC started on {vnc.display}")
            except ImportError:
                raise RuntimeError(
                    "No display available and vnc_session not found. "
                    "Install VNC or run from a graphical session."
                )

        args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
        ]
        if not headless:
            args += ["--disable-gpu", "--disable-software-rasterizer"]

        self._pw = sync_playwright().start()
        self._ctx = self._pw.chromium.launch_persistent_context(
            user_data_dir=str(BROWSER_DATA_DIR),
            headless=headless,
            args=args,
            viewport={"width": 1280, "height": 900},
            ignore_https_errors=True,
            java_script_enabled=True,
        )

        self._page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
        self._page.set_default_timeout(30_000)
        self._page.add_init_script(STEALTH_JS)

    def _stop(self):
        """Close browser and Playwright — safe to call multiple times."""
        try:
            if self._ctx:
                self._ctx.close()
        except Exception:
            pass
        try:
            if self._pw:
                self._pw.stop()
        except Exception:
            pass
        self._pw = self._ctx = self._page = None

    def _screenshot(self, label: str = "social") -> Optional[str]:
        """Save a debugging screenshot, return path or None."""
        try:
            ts = int(time.time())
            path = SCREENSHOT_DIR / f"{label}_{ts}.png"
            self._page.screenshot(path=str(path))
            return str(path)
        except Exception:
            return None

    # ── login checks ──────────────────────────────────────────────────────

    def _is_logged_in(self, platform: str) -> bool:
        """Quick DOM probe to see if we're authenticated."""
        p = self._page
        checks = {
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
                'a[href="/direct/inbox/"]',
            ],
        }
        for sel in checks.get(platform, []):
            try:
                if p.locator(sel).count() > 0:
                    return True
            except Exception:
                continue
        return False

    # ── platform posters ──────────────────────────────────────────────────

    def _post_x(self, content: str, image_path: Optional[str]) -> str:
        page = self._page

        page.goto("https://x.com/home", wait_until="domcontentloaded", timeout=30_000)
        _human_pause(2, 4)

        if not self._is_logged_in("x"):
            return "Error: Not logged in to X. Run browser_login('x') first."

        # Find the compose box on the home timeline
        compose = _first_match(page, [
            '[data-testid="tweetTextarea_0"]',
            'div[role="textbox"][data-testid="tweetTextarea_0"]',
            '[aria-label="Post text"]',
            'div.public-DraftEditor-content',
        ])

        # If not found, try opening the compose dialog
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
            self._screenshot("x_compose_fail")
            return "Error: Could not find the compose area on X."

        compose.click()
        _human_pause(0.5, 1.0)
        _type_human(page, content)
        _human_pause(1, 2)

        # Attach image
        if image_path:
            try:
                fi = _first_match(page, [
                    'input[data-testid="fileInput"]',
                    'input[type="file"][accept*="image"]',
                ])
                if fi:
                    fi.set_input_files(image_path)
                    _human_pause(2, 4)
                else:
                    return "Error: Could not find file input on X for image upload."
            except Exception as e:
                return f"Error uploading image to X: {e}"

        # Click Post
        post_btn = _first_match(page, [
            '[data-testid="tweetButtonInline"]',
            '[data-testid="tweetButton"]',
        ])
        if not post_btn or not post_btn.is_enabled():
            self._screenshot("x_post_btn_fail")
            return "Error: Post button not found or disabled on X."

        post_btn.click()
        _human_pause(3, 5)

        ss = self._screenshot("x_posted")
        return f"Posted to X successfully.{f' Screenshot: {ss}' if ss else ''}"

    def _post_linkedin(self, content: str, image_path: Optional[str]) -> str:
        page = self._page

        page.goto("https://www.linkedin.com/feed/", wait_until="domcontentloaded", timeout=30_000)
        _human_pause(2, 4)

        if not self._is_logged_in("linkedin"):
            return "Error: Not logged in to LinkedIn. Run browser_login('linkedin') first."

        # Open the post composer
        start_btn = _first_match(page, [
            'button.share-box-feed-entry__trigger',
            'button:has-text("Start a post")',
            '[aria-label="Text"]',
        ])
        if not start_btn:
            self._screenshot("linkedin_start_fail")
            return "Error: Could not find 'Start a post' on LinkedIn."

        start_btn.click()
        _human_pause(1, 2)

        # Find the rich text editor
        editor = _first_match(page, [
            'div.ql-editor[data-placeholder]',
            '[role="textbox"][aria-label*="editor"]',
            '.ql-editor',
            'div[contenteditable="true"]',
        ])
        if not editor:
            self._screenshot("linkedin_editor_fail")
            return "Error: Could not find LinkedIn post editor."

        editor.click()
        _human_pause(0.5, 1.0)
        _type_human(page, content, 25, 60)
        _human_pause(1, 2)

        # Attach image
        if image_path:
            try:
                media_btn = _first_match(page, [
                    '[aria-label="Add media"]',
                    'button[aria-label="Add media"]',
                    'button:has(li-icon[type="image"])' ,
                ])
                if media_btn:
                    media_btn.click()
                    _human_pause(1, 2)
                fi = page.locator('input[type="file"]').first
                fi.set_input_files(image_path)
                _human_pause(3, 5)
            except Exception:
                log("LinkedIn image upload failed — continuing without image")

        # Click Post
        post_btn = _first_match(page, [
            'button.share-actions__primary-action',
            'button:has-text("Post")',
            '[aria-label="Post"]',
        ])
        if not post_btn:
            self._screenshot("linkedin_post_fail")
            return "Error: Could not click the Post button on LinkedIn."

        post_btn.click()
        _human_pause(3, 5)

        ss = self._screenshot("linkedin_posted")
        return f"Posted to LinkedIn successfully.{f' Screenshot: {ss}' if ss else ''}"

    def _post_facebook(self, content: str, image_path: Optional[str]) -> str:
        page = self._page

        page.goto("https://www.facebook.com/", wait_until="domcontentloaded", timeout=30_000)
        _human_pause(2, 4)

        if not self._is_logged_in("facebook"):
            return "Error: Not logged in to Facebook. Run browser_login('facebook') first."

        # Open composer
        trigger = _first_match(page, [
            '[aria-label="Create a post"]',
            '[role="button"]:has-text("What\'s on your mind")',
            'span:has-text("What\'s on your mind")',
        ])
        if not trigger:
            self._screenshot("facebook_compose_fail")
            return "Error: Could not find Facebook compose trigger."

        trigger.click()
        _human_pause(1, 3)

        # Editor
        editor = _first_match(page, [
            'div[contenteditable="true"][role="textbox"]',
            '[aria-label*="What\'s on your mind"]',
            'div[contenteditable="true"][data-lexical-editor="true"]',
        ])
        if not editor:
            self._screenshot("facebook_editor_fail")
            return "Error: Could not find Facebook post editor."

        editor.click()
        _human_pause(0.5, 1.0)
        _type_human(page, content, 25, 60)
        _human_pause(1, 2)

        # Attach image
        if image_path:
            try:
                photo_btn = _first_match(page, [
                    '[aria-label="Photo/video"]',
                    '[aria-label="Photo/Video"]',
                ])
                if photo_btn:
                    photo_btn.click()
                    _human_pause(1, 2)
                fi = page.locator('input[type="file"][accept*="image"]').first
                fi.set_input_files(image_path)
                _human_pause(3, 5)
            except Exception:
                log("Facebook image upload failed — continuing without image")

        # Post
        post_btn = _first_match(page, [
            '[aria-label="Post"]',
            'div[aria-label="Post"]',
        ])
        if not post_btn:
            self._screenshot("facebook_post_fail")
            return "Error: Could not click Facebook Post button."

        post_btn.click()
        _human_pause(3, 5)

        ss = self._screenshot("facebook_posted")
        return f"Posted to Facebook successfully.{f' Screenshot: {ss}' if ss else ''}"

    def _post_instagram(self, content: str, image_path: str) -> str:
        page = self._page

        page.goto("https://www.instagram.com/", wait_until="domcontentloaded", timeout=30_000)
        _human_pause(2, 4)

        if not self._is_logged_in("instagram"):
            return "Error: Not logged in to Instagram. Run browser_login('instagram') first."

        # Click "New post"
        new_post = _first_match(page, [
            'svg[aria-label="New post"]',
            '[aria-label="New post"]',
            'a[href="/create/style/"]',
        ])
        if not new_post:
            self._screenshot("instagram_newpost_fail")
            return "Error: Could not find Instagram 'New post' button."

        new_post.click()
        _human_pause(1, 3)

        # Upload image
        try:
            fi = page.locator('input[type="file"]').first
            fi.set_input_files(image_path)
            _human_pause(3, 5)
        except Exception as e:
            self._screenshot("instagram_upload_fail")
            return f"Error uploading image to Instagram: {e}"

        # Navigate through the creation flow — hit "Next" twice (crop → filters → caption)
        for step in range(2):
            next_btn = _first_match(page, [
                'button:has-text("Next")',
                '[aria-label="Next"]',
                'div[role="button"]:has-text("Next")',
            ])
            if next_btn:
                next_btn.click()
                _human_pause(1, 3)

        # Write caption
        caption_box = _first_match(page, [
            '[aria-label="Write a caption..."]',
            '[aria-label="Write a caption\u2026"]',
            'textarea[aria-label*="caption"]',
            'div[role="textbox"]',
        ])
        if caption_box:
            caption_box.click()
            _human_pause(0.5, 1.0)
            _type_human(page, content, 25, 60)
            _human_pause(1, 2)

        # Share
        share_btn = _first_match(page, [
            'button:has-text("Share")',
            '[aria-label="Share"]',
            'div[role="button"]:has-text("Share")',
        ])
        if not share_btn:
            self._screenshot("instagram_share_fail")
            return "Error: Could not find Instagram Share button."

        share_btn.click()
        _human_pause(5, 8)

        ss = self._screenshot("instagram_posted")
        return f"Posted to Instagram successfully.{f' Screenshot: {ss}' if ss else ''}"

    # ── main entry ────────────────────────────────────────────────────────

    def post(self, platform: str, content: str, image_path: Optional[str] = None) -> str:
        """Post content to a social media platform via browser automation."""
        platform = platform.lower().strip()
        if platform == "twitter":
            platform = "x"

        if platform not in ("x", "linkedin", "facebook", "instagram"):
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

        # Close any active BrowserSession so we can use the same data dir
        try:
            from browser import _get_session
            session = _get_session()
            if session.is_active:
                session.close()
                log("Closed existing BrowserSession for social posting")
        except Exception:
            pass

        # X must use visible/VNC mode — bot detection blocks headless
        use_headless = platform != "x"

        posters = {
            "x": self._post_x,
            "linkedin": self._post_linkedin,
            "facebook": self._post_facebook,
            "instagram": self._post_instagram,
        }

        try:
            self._start(headless=use_headless)
            result = posters[platform](content, image_path)

            # If headless posting failed (not a login issue), retry with VNC
            if (use_headless
                    and result.startswith("Error")
                    and "Not logged in" not in result):
                log(f"Headless posting to {platform} failed, retrying with VNC...")
                self._stop()
                _human_pause(1, 2)
                self._start(headless=False)
                result = posters[platform](content, image_path)

            return result

        except Exception as e:
            self._screenshot(f"{platform}_error")
            return f"Error posting to {platform}: {e}"
        finally:
            self._stop()


# ─────────────────────────────────────────────────────────────────────────────
# Public tool function
# ─────────────────────────────────────────────────────────────────────────────

def social_post(platform: str, content: str, image_path: str = None) -> str:
    """Post content to a social media platform using browser automation.

    Requires a prior browser_login session for the target platform.
    X uses VNC mode (bot detection). Others try headless first.
    """
    return SocialPoster().post(platform, content, image_path)


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
                "via browser automation. Requires a prior browser_login session for the target "
                "platform. X always uses visible/VNC mode; others try headless first. "
                "Instagram requires an image."
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
    """Dispatch social_* tool calls."""
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
