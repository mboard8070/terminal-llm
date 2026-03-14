"""
Browser Automation for MAUDE - Full interactive browser control via Playwright.

Extends MAUDE's existing web_view screenshot capability to full interactive
browser automation: clicking, typing, form filling, navigation, and more.

Uses Playwright's sync API to match MAUDE's synchronous tool execution model.
"""

from __future__ import annotations

import os
import base64
import signal
import time
import threading
from pathlib import Path
from typing import Optional, Dict, List

try:
    from playwright.sync_api import sync_playwright, Browser, Page, Playwright, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Import MAUDE core for logging and vision model access
try:
    from maude_core import log, VISION_URL, VISION_MODEL
    from openai import OpenAI
except ImportError:
    def log(msg: str):
        pass
    VISION_URL = "http://localhost:11434/v1"
    VISION_MODEL = "llava:7b"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

HEADLESS = os.environ.get("MAUDE_BROWSER_HEADLESS", "true").lower() in ("true", "1", "yes")
BROWSER_DATA_DIR = Path.home() / ".config" / "maude" / "browser_data"
ACTION_TIMEOUT_MS = 30_000      # 30 seconds per action
INACTIVITY_TIMEOUT = 3600       # 1 hour auto-close (supports social posting between uses)
LOGIN_INACTIVITY_TIMEOUT = 3600  # 1 hour for login sessions
SCREENSHOT_DIR = Path.home() / ".config" / "maude" / "screenshots"

# Common login URLs — lets users say browser_login("x") instead of the full URL
SOCIAL_LOGIN_URLS = {
    "x": "https://x.com/i/flow/login",
    "twitter": "https://x.com/i/flow/login",
    "linkedin": "https://www.linkedin.com/login",
    "instagram": "https://www.instagram.com/accounts/login/",
    "facebook": "https://www.facebook.com/login",
    "github": "https://github.com/login",
    "reddit": "https://www.reddit.com/login",
    "youtube": "https://accounts.google.com/ServiceLogin?service=youtube",
    "google": "https://accounts.google.com/ServiceLogin",
    "tiktok": "https://www.tiktok.com/login",
    "pinterest": "https://www.pinterest.com/login/",
    "bluesky": "https://bsky.app/",
}


# ─────────────────────────────────────────────────────────────────────────────
# BrowserSession
# ─────────────────────────────────────────────────────────────────────────────

class BrowserSession:
    """Manages a persistent Playwright browser and page with lazy initialization."""

    def __init__(self):
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
        self._platform_pages: Dict[str, Page] = {}  # platform → tab (for multi-login)
        self._last_activity: float = 0.0
        self._lock = threading.Lock()
        self._inactivity_timer: Optional[threading.Timer] = None

    @property
    def is_active(self) -> bool:
        """Check whether a browser session is currently running."""
        return self._page is not None and not self._page.is_closed()

    def _ensure_dirs(self):
        """Create required directories."""
        BROWSER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    def _ensure_browser(self):
        """Lazy-init: start the browser and open a page if not already running."""
        if self.is_active:
            self._touch()
            return

        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                "Playwright is not installed. Run: pip install playwright && playwright install chromium"
            )

        self._ensure_dirs()
        self._kill_orphaned_chrome()
        log("Starting browser session...")

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(BROWSER_DATA_DIR),
            headless=HEADLESS,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
            viewport={"width": 1280, "height": 900},
            ignore_https_errors=True,
            java_script_enabled=True,
        )

        # Use the first page in the context or create one
        if self._browser.pages:
            self._page = self._browser.pages[0]
        else:
            self._page = self._browser.new_page()

        self._page.set_default_timeout(ACTION_TIMEOUT_MS)
        self._touch()
        log("Browser session started.")

    def _touch(self):
        """Record activity and reset the inactivity auto-close timer."""
        self._last_activity = time.time()
        self._reset_inactivity_timer()

    def _reset_inactivity_timer(self):
        """Cancel any existing timer and start a new one."""
        if self._inactivity_timer is not None:
            self._inactivity_timer.cancel()
        self._inactivity_timer = threading.Timer(INACTIVITY_TIMEOUT, self._auto_close)
        self._inactivity_timer.daemon = True
        self._inactivity_timer.start()

    def _auto_close(self):
        """Automatically close the browser after inactivity."""
        if self._last_activity and (time.time() - self._last_activity) >= INACTIVITY_TIMEOUT:
            log("Browser auto-closing after 5 minutes of inactivity.")
            self.close()

    # ── Selector resolution ──────────────────────────────────────────────────

    def _resolve_element(self, selector: str, page: Page):
        """
        Smart selector resolution.  Tries strategies in order:
          1. CSS selector
          2. Text content (case-insensitive)
          3. aria-label
        Returns a Playwright Locator.
        """
        # 1. Try as CSS
        try:
            loc = page.locator(selector)
            if loc.count() > 0:
                return loc.first
        except Exception:
            pass

        # 2. Try by visible text (case-insensitive via regex)
        try:
            loc = page.get_by_text(selector, exact=False)
            if loc.count() > 0:
                return loc.first
        except Exception:
            pass

        # 3. Try by aria-label
        try:
            loc = page.get_by_label(selector, exact=False)
            if loc.count() > 0:
                return loc.first
        except Exception:
            pass

        # 4. Try by placeholder
        try:
            loc = page.get_by_placeholder(selector, exact=False)
            if loc.count() > 0:
                return loc.first
        except Exception:
            pass

        # 5. Try by role with name
        try:
            loc = page.get_by_role("button", name=selector)
            if loc.count() > 0:
                return loc.first
        except Exception:
            pass

        try:
            loc = page.get_by_role("link", name=selector)
            if loc.count() > 0:
                return loc.first
        except Exception:
            pass

        return None

    def _page_summary(self) -> str:
        """Return a compact summary of the current page state."""
        if not self.is_active:
            return "No active page."
        page = self._page
        title = page.title() or "(no title)"
        url = page.url
        return f"Page: {title}\nURL: {url}"

    # ── Public actions ───────────────────────────────────────────────────────

    def open(self, url: str) -> str:
        """Open a URL and return title + text summary."""
        with self._lock:
            try:
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url

                self._ensure_browser()
                log(f"Navigating to {url}")

                self._page.goto(url, wait_until="domcontentloaded", timeout=ACTION_TIMEOUT_MS)
                # Give JS time to settle
                self._page.wait_for_load_state("networkidle", timeout=10_000)

                title = self._page.title() or "(no title)"
                # Extract a text snapshot
                text = self._page.inner_text("body") or ""
                text = _clean_text(text, limit=3000)

                self._touch()
                return f"Opened: {title}\nURL: {self._page.url}\n\nPage content:\n{text}"

            except PlaywrightTimeout:
                title = self._page.title() or "(no title)" if self.is_active else "(timeout)"
                return f"Page partially loaded (timeout).\nTitle: {title}\nURL: {url}\nTip: Try browser_extract() to read current content."
            except Exception as e:
                return f"Error opening {url}: {e}"

    def click(self, selector: str) -> str:
        """Click an element identified by selector or text."""
        with self._lock:
            try:
                if not self.is_active:
                    return "Error: No browser session. Use browser_open first."

                log(f"Clicking: {selector}")
                element = self._resolve_element(selector, self._page)

                if element is None:
                    return (
                        f"Error: Could not find element matching '{selector}'.\n"
                        f"Tip: Use browser_extract() to see current page structure, "
                        f"or try a different selector."
                    )

                # Scroll into view and click
                element.scroll_into_view_if_needed(timeout=5_000)
                element.click(timeout=ACTION_TIMEOUT_MS)

                # Wait briefly for any navigation or DOM change
                self._page.wait_for_load_state("domcontentloaded", timeout=5_000)

                self._touch()
                summary = self._page_summary()
                return f"Clicked '{selector}'.\n{summary}"

            except PlaywrightTimeout:
                self._touch()
                return f"Click on '{selector}' timed out. The element may be obscured or the page is loading."
            except Exception as e:
                return f"Error clicking '{selector}': {e}"

    def type_text(self, selector: str, text: str) -> str:
        """Type text into an input element."""
        with self._lock:
            try:
                if not self.is_active:
                    return "Error: No browser session. Use browser_open first."

                log(f"Typing into: {selector}")

                if selector.lower() == "active":
                    # Type into currently focused element
                    self._page.keyboard.type(text, delay=30)
                    self._touch()
                    return f"Typed {len(text)} characters into the active element."

                element = self._resolve_element(selector, self._page)
                if element is None:
                    return (
                        f"Error: Could not find input matching '{selector}'.\n"
                        f"Tip: Try 'active' to type into the currently focused element."
                    )

                # Clear existing content, then type
                element.click(timeout=5_000)
                element.fill("")
                element.type(text, delay=30)

                self._touch()
                return f"Typed '{text[:50]}{'...' if len(text) > 50 else ''}' into '{selector}'."

            except PlaywrightTimeout:
                self._touch()
                return f"Typing into '{selector}' timed out."
            except Exception as e:
                return f"Error typing into '{selector}': {e}"

    def navigate(self, url: str) -> str:
        """Navigate to a new URL in the existing session."""
        with self._lock:
            try:
                if not self.is_active:
                    return "Error: No browser session. Use browser_open first."

                if not url.startswith(("http://", "https://")):
                    url = "https://" + url

                log(f"Navigating to {url}")
                self._page.goto(url, wait_until="domcontentloaded", timeout=ACTION_TIMEOUT_MS)

                try:
                    self._page.wait_for_load_state("networkidle", timeout=10_000)
                except PlaywrightTimeout:
                    pass  # Acceptable, page may have streaming content

                title = self._page.title() or "(no title)"
                self._touch()
                return f"Navigated to: {title}\nURL: {self._page.url}"

            except PlaywrightTimeout:
                self._touch()
                return f"Navigation to {url} timed out. Page may still be loading."
            except Exception as e:
                return f"Error navigating to {url}: {e}"

    def screenshot(self) -> str:
        """Take a screenshot and analyze it with the vision model."""
        with self._lock:
            try:
                if not self.is_active:
                    return "Error: No browser session. Use browser_open first."

                log("Taking screenshot...")
                screenshot_bytes = self._page.screenshot(full_page=False)
                base64_image = base64.b64encode(screenshot_bytes).decode("utf-8")
                log(f"Screenshot captured ({len(screenshot_bytes) // 1024}KB)")

                # Save to disk for reference
                ts = int(time.time())
                screenshot_path = SCREENSHOT_DIR / f"browser_{ts}.png"
                screenshot_path.write_bytes(screenshot_bytes)

                # Analyze with LLaVA vision model
                url = self._page.url
                title = self._page.title() or "(no title)"
                prompt = (
                    f"This is a screenshot of a web page. "
                    f"Title: {title}. URL: {url}. "
                    f"Describe the layout, visible content, and any interactive elements "
                    f"(buttons, forms, links, menus) you can see."
                )

                log("Analyzing screenshot with vision model...")
                try:
                    vision_client = OpenAI(base_url=VISION_URL, api_key="not-needed")
                    response = vision_client.chat.completions.create(
                        model=VISION_MODEL,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                            ]
                        }],
                        max_tokens=1024,
                        temperature=0.2
                    )
                    analysis = response.choices[0].message.content
                    log("Vision analysis complete.")
                except Exception as ve:
                    analysis = f"(Vision model unavailable: {ve})\nScreenshot saved to {screenshot_path}"

                self._touch()
                return (
                    f"Screenshot of {title} ({url})\n"
                    f"Saved: {screenshot_path}\n\n"
                    f"Visual analysis:\n{analysis}"
                )

            except Exception as e:
                return f"Error taking screenshot: {e}"

    def extract(self, selector: str = None) -> str:
        """Extract text content from the page or a specific element."""
        with self._lock:
            try:
                if not self.is_active:
                    return "Error: No browser session. Use browser_open first."

                title = self._page.title() or "(no title)"
                url = self._page.url

                if selector:
                    log(f"Extracting text from: {selector}")
                    element = self._resolve_element(selector, self._page)
                    if element is None:
                        return f"Error: Could not find element matching '{selector}'."
                    text = element.inner_text()
                    source = f"Element '{selector}'"
                else:
                    log("Extracting full page text...")
                    text = self._page.inner_text("body") or ""
                    source = "Full page"

                text = _clean_text(text, limit=10_000)

                self._touch()
                return (
                    f"Page: {title}\nURL: {url}\nSource: {source}\n\n"
                    f"Content ({len(text)} chars):\n{text}"
                )

            except Exception as e:
                return f"Error extracting content: {e}"

    def fill_form(self, fields: dict) -> str:
        """Fill multiple form fields at once."""
        with self._lock:
            try:
                if not self.is_active:
                    return "Error: No browser session. Use browser_open first."

                log(f"Filling form ({len(fields)} fields)...")
                results = []

                for selector, value in fields.items():
                    try:
                        element = self._resolve_element(selector, self._page)
                        if element is None:
                            results.append(f"  MISS: '{selector}' - element not found")
                            continue

                        element.click(timeout=5_000)
                        element.fill("")
                        element.type(str(value), delay=20)
                        results.append(f"  OK: '{selector}' = '{str(value)[:40]}'")

                    except Exception as fe:
                        results.append(f"  FAIL: '{selector}' - {fe}")

                self._touch()
                filled = sum(1 for r in results if r.startswith("  OK"))
                return (
                    f"Form fill complete: {filled}/{len(fields)} fields filled.\n"
                    + "\n".join(results)
                )

            except Exception as e:
                return f"Error filling form: {e}"

    def select_option(self, selector: str, value: str) -> str:
        """Select a dropdown option by value, label, or index."""
        with self._lock:
            try:
                if not self.is_active:
                    return "Error: No browser session. Use browser_open first."

                log(f"Selecting '{value}' in '{selector}'")
                element = self._resolve_element(selector, self._page)

                if element is None:
                    return f"Error: Could not find dropdown matching '{selector}'."

                # Try selecting by value first, then by label
                try:
                    element.select_option(value=value, timeout=ACTION_TIMEOUT_MS)
                    method = "value"
                except Exception:
                    try:
                        element.select_option(label=value, timeout=ACTION_TIMEOUT_MS)
                        method = "label"
                    except Exception:
                        # Last resort: try as visible text index
                        try:
                            element.select_option(index=int(value), timeout=ACTION_TIMEOUT_MS)
                            method = "index"
                        except (ValueError, Exception):
                            return (
                                f"Error: Could not select '{value}' in '{selector}'. "
                                f"Try using the exact option value, label, or numeric index."
                            )

                self._touch()
                return f"Selected '{value}' (by {method}) in '{selector}'.\n{self._page_summary()}"

            except Exception as e:
                return f"Error selecting option: {e}"

    @staticmethod
    def _kill_orphaned_chrome():
        """Kill any Chrome/Chromium processes using our browser data directory.

        When the gateway restarts, the in-memory BrowserSession singleton is lost
        but the Chrome process it spawned keeps running — orphaned. This prevents
        a new browser from acquiring the profile lock.
        """
        import subprocess as _sp
        data_dir = str(BROWSER_DATA_DIR)
        try:
            # Find chrome/chromium processes whose command line references our data dir
            result = _sp.run(
                ["pgrep", "-f", data_dir],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return  # No orphaned processes

            pids = [int(p) for p in result.stdout.strip().split("\n") if p.strip()]
            for pid in pids:
                if pid == os.getpid():
                    continue  # Don't kill ourselves
                try:
                    os.kill(pid, signal.SIGTERM)
                    log(f"Killed orphaned Chrome process {pid}")
                except (ProcessLookupError, PermissionError):
                    pass

            # Brief wait for processes to exit
            if pids:
                time.sleep(1.0)

        except Exception as e:
            log(f"Warning: orphan cleanup failed: {e}")

    def _resolve_platform(self, url: str) -> Optional[str]:
        """Return platform key if url matches a known social login shorthand."""
        key = url.lower().strip()
        if key in SOCIAL_LOGIN_URLS:
            return key
        if key == "twitter":
            return "x"
        return None

    def get_platform_page(self, platform: str) -> Optional[Page]:
        """Return the open tab for a platform, or None if not tracked / closed."""
        page = self._platform_pages.get(platform)
        if page is not None and not page.is_closed():
            return page
        self._platform_pages.pop(platform, None)
        return None

    def login(self, url: str) -> str:
        """Launch a VISIBLE browser for manual login. Cookies persist for future headless use.

        If the browser is already running (e.g. logged into another platform),
        opens a NEW TAB instead of tearing down the session. This keeps all
        previous login tabs alive so background JS can maintain sessions.

        If no local display is available, auto-starts a VNC session with noVNC
        so the user can interact via a web browser on any device.
        """
        with self._lock:
            try:
                if not PLAYWRIGHT_AVAILABLE:
                    raise RuntimeError(
                        "Playwright is not installed. Run: pip install playwright && playwright install chromium"
                    )

                # Resolve shorthand names like "x", "linkedin"
                platform = self._resolve_platform(url)
                resolved = SOCIAL_LOGIN_URLS.get(url.lower().strip())
                if resolved:
                    log(f"Resolved '{url}' → {resolved}")
                    url = resolved
                elif not url.startswith(("http://", "https://")):
                    url = "https://" + url

                self._ensure_dirs()

                # ── Browser already running? Open a new tab ──────────────
                if self.is_active:
                    log(f"Browser active — opening new tab for {url}")
                    new_page = self._browser.new_page()
                    new_page.set_default_timeout(ACTION_TIMEOUT_MS)
                    new_page.goto(url, wait_until="domcontentloaded", timeout=ACTION_TIMEOUT_MS)

                    title = new_page.title() or "(no title)"
                    if platform:
                        self._platform_pages[platform] = new_page
                    # Keep _page pointing at the newest tab for general use
                    self._page = new_page
                    self._touch()

                    tab_list = ", ".join(self._platform_pages.keys()) or "(none)"
                    return (
                        f"New tab opened for login.\n"
                        f"Page: {title}\n"
                        f"URL: {new_page.url}\n\n"
                        f"Log in manually in the new tab.\n"
                        f"Open tabs: {tab_list}\n"
                        f"All previous login sessions are still active."
                    )

                # ── Fresh launch ─────────────────────────────────────────

                # Kill any orphaned Chrome from a previous gateway process
                self._kill_orphaned_chrome()

                # Clean up stale profile state that blocks visible launch
                import shutil
                for stale in ("SingletonLock", "SingletonSocket", "SingletonCookie",
                              "ShaderCache", "GrShaderCache", "GraphiteDawnCache",
                              "BrowserMetrics"):
                    p = BROWSER_DATA_DIR / stale
                    if p.exists():
                        if p.is_dir():
                            shutil.rmtree(p, ignore_errors=True)
                        else:
                            p.unlink(missing_ok=True)

                # Detect if we have a local display or need VNC
                local_display = os.environ.get("DISPLAY")
                vnc_url = None
                needs_vnc = not local_display

                if local_display and local_display.startswith(":"):
                    # DISPLAY was set (possibly from a previous login) — verify X is alive
                    try:
                        from vnc_session import get_vnc_session
                        if not get_vnc_session().is_active:
                            needs_vnc = True
                            local_display = None
                    except ImportError:
                        pass

                if needs_vnc:
                    # No local display — start VNC session
                    try:
                        from vnc_session import get_vnc_session
                        vnc = get_vnc_session()
                        result = vnc.start()

                        if result.startswith("Error"):
                            return result

                        vnc_url = result
                        local_display = vnc.display
                        os.environ["DISPLAY"] = local_display
                        log(f"Using VNC display {local_display}, noVNC at {vnc_url}")
                    except ImportError:
                        return (
                            "Error: No display available and vnc_session module not found.\n"
                            "Either run from a graphical session or install VNC:\n"
                            "  sudo apt-get install -y xvfb x11vnc novnc python3-websockify"
                        )

                log(f"Starting VISIBLE browser for login to {url} on {local_display}")

                # Ensure DISPLAY is set before Playwright launches Chromium
                os.environ["DISPLAY"] = local_display

                self._playwright = sync_playwright().start()
                self._browser = self._playwright.chromium.launch_persistent_context(
                    user_data_dir=str(BROWSER_DATA_DIR),
                    headless=False,  # VISIBLE for manual interaction
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--no-sandbox",
                        "--disable-gpu",
                        "--disable-software-rasterizer",
                    ],
                    viewport={"width": 1280, "height": 900},
                    ignore_https_errors=True,
                    java_script_enabled=True,
                )

                if self._browser.pages:
                    self._page = self._browser.pages[0]
                else:
                    self._page = self._browser.new_page()

                self._page.set_default_timeout(ACTION_TIMEOUT_MS)
                self._page.goto(url, wait_until="domcontentloaded", timeout=ACTION_TIMEOUT_MS)

                title = self._page.title() or "(no title)"
                if platform:
                    self._platform_pages[platform] = self._page
                self._last_activity = time.time()

                # Longer timeout for login sessions
                if self._inactivity_timer is not None:
                    self._inactivity_timer.cancel()
                self._inactivity_timer = threading.Timer(LOGIN_INACTIVITY_TIMEOUT, self._auto_close)
                self._inactivity_timer.daemon = True
                self._inactivity_timer.start()

                # Build response based on whether we're using VNC or local display
                if vnc_url:
                    try:
                        from vnc_session import get_vnc_session
                        all_urls = get_vnc_session().get_all_urls()
                    except Exception:
                        all_urls = [vnc_url]

                    url_lines = "\n".join(f"  {u}" for u in all_urls)
                    return (
                        f"Browser opened via VNC for login.\n"
                        f"Page: {title}\n"
                        f"URL: {self._page.url}\n\n"
                        f"IMPORTANT: Show the user EXACTLY these noVNC links (do NOT modify or substitute the URLs):\n"
                        f"{url_lines}\n\n"
                        f"Log in manually. Leave the browser open — do NOT close it.\n"
                        f"You can log into more accounts with browser_login('<platform>').\n"
                        f"Each gets its own tab. Sessions stay alive as long as tabs are open."
                    )
                else:
                    return (
                        f"Browser opened in VISIBLE mode for login.\n"
                        f"Page: {title}\n"
                        f"URL: {self._page.url}\n\n"
                        f"Log in manually in the browser window.\n"
                        f"Leave the browser open — do NOT close it.\n"
                        f"You can log into more accounts with browser_login('<platform>').\n"
                        f"Each gets its own tab. Sessions stay alive as long as tabs are open."
                    )

            except Exception as e:
                return f"Error opening login browser: {e}"

    def check_session(self, url: str, logged_in_selector: str) -> str:
        """Check if a saved login session is still valid for a site."""
        with self._lock:
            try:
                # Resolve shorthand names
                resolved = SOCIAL_LOGIN_URLS.get(url.lower().strip())
                if resolved:
                    # For session checks, go to the main site, not the login page
                    main_urls = {
                        "x": "https://x.com/home",
                        "twitter": "https://x.com/home",
                        "linkedin": "https://www.linkedin.com/feed/",
                        "instagram": "https://www.instagram.com/",
                        "facebook": "https://www.facebook.com/",
                        "github": "https://github.com/",
                        "reddit": "https://www.reddit.com/",
                        "youtube": "https://www.youtube.com/",
                        "google": "https://myaccount.google.com/",
                        "tiktok": "https://www.tiktok.com/foryou",
                        "pinterest": "https://www.pinterest.com/",
                        "bluesky": "https://bsky.app/",
                    }
                    url = main_urls.get(url.lower().strip(), resolved)

                if not url.startswith(("http://", "https://")):
                    url = "https://" + url

                self._ensure_browser()
                log(f"Checking session at {url} for '{logged_in_selector}'")

                self._page.goto(url, wait_until="domcontentloaded", timeout=ACTION_TIMEOUT_MS)
                try:
                    self._page.wait_for_load_state("networkidle", timeout=10_000)
                except PlaywrightTimeout:
                    pass

                current_url = self._page.url
                title = self._page.title() or "(no title)"

                # Check for the logged-in indicator
                element = self._resolve_element(logged_in_selector, self._page)

                self._touch()

                if element is not None:
                    return (
                        f"Session VALID — found '{logged_in_selector}' on page.\n"
                        f"Page: {title}\n"
                        f"URL: {current_url}"
                    )
                else:
                    # Check if we got redirected to a login page
                    login_indicators = ["login", "signin", "sign-in", "sign_in", "auth"]
                    redirected_to_login = any(ind in current_url.lower() for ind in login_indicators)
                    hint = " (redirected to login page)" if redirected_to_login else ""

                    return (
                        f"Session EXPIRED — '{logged_in_selector}' not found{hint}.\n"
                        f"Page: {title}\n"
                        f"URL: {current_url}\n"
                        f"Use browser_login to re-authenticate."
                    )

            except Exception as e:
                return f"Error checking session: {e}"

    def close(self) -> str:
        """Close the browser session and clean up."""
        with self._lock:
            return self._close_internal()

    def _close_internal(self) -> str:
        """Internal close without acquiring the lock (called from auto-close)."""
        try:
            if self._inactivity_timer is not None:
                self._inactivity_timer.cancel()
                self._inactivity_timer = None

            if self._browser is not None:
                try:
                    self._browser.close()
                except Exception:
                    pass
                self._browser = None

            if self._playwright is not None:
                try:
                    self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None

            self._page = None
            self._platform_pages.clear()
            self._last_activity = 0.0

            # Stop VNC session if one was started for login
            vnc_stopped = False
            try:
                from vnc_session import get_vnc_session
                vnc = get_vnc_session()
                if vnc.is_active:
                    vnc.stop()
                    vnc_stopped = True
                    # Clear stale DISPLAY so next login restarts VNC
                    display = os.environ.get("DISPLAY", "")
                    if display == vnc.display:
                        del os.environ["DISPLAY"]
                    log("VNC session stopped.")
            except ImportError:
                pass

            log("Browser session closed.")
            msg = "Browser session closed. Login cookies saved."
            if vnc_stopped:
                msg += " VNC session stopped."
            return msg

        except Exception as e:
            return f"Error closing browser: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_session: Optional[BrowserSession] = None


def _get_session() -> BrowserSession:
    """Get or create the global browser session."""
    global _session
    if _session is None:
        _session = BrowserSession()
    return _session


# ─────────────────────────────────────────────────────────────────────────────
# Public tool functions
# ─────────────────────────────────────────────────────────────────────────────

def browser_open(url: str) -> str:
    """Open a URL in a headless Chromium browser. Returns page title and text summary."""
    return _get_session().open(url)


def browser_click(selector: str) -> str:
    """Click an element by CSS selector, text content, or aria-label."""
    return _get_session().click(selector)


def browser_type(selector: str, text: str) -> str:
    """Type text into an input or textarea. Use selector='active' for the focused element."""
    return _get_session().type_text(selector, text)


def browser_navigate(url: str) -> str:
    """Navigate to a new URL in the current browser session."""
    return _get_session().navigate(url)


def browser_screenshot() -> str:
    """Take a screenshot and return a visual description via the LLaVA vision model."""
    return _get_session().screenshot()


def browser_extract(selector: str = None) -> str:
    """Extract text content from the page or a specific CSS selector. Limited to 10,000 chars."""
    return _get_session().extract(selector)


def browser_fill_form(fields: dict) -> str:
    """Fill multiple form fields at once. Keys are selectors, values are text to type."""
    return _get_session().fill_form(fields)


def browser_select(selector: str, value: str) -> str:
    """Select a dropdown option by value, label, or index."""
    return _get_session().select_option(selector, value)


def browser_login(url: str) -> str:
    """Open a VISIBLE browser for manual login. Saves session for future headless use.
    Accepts shorthand names: x, linkedin, instagram, facebook, github, reddit, etc."""
    return _get_session().login(url)


def browser_check_session(url: str, logged_in_selector: str) -> str:
    """Check if a saved login session is still valid. Returns VALID or EXPIRED."""
    return _get_session().check_session(url, logged_in_selector)


def browser_close() -> str:
    """Close the browser session and free resources."""
    return _get_session().close()


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _clean_text(text: str, limit: int = 10_000) -> str:
    """Clean extracted page text: collapse whitespace, strip blank lines, truncate."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned.append(stripped)
    result = "\n".join(cleaned)
    if len(result) > limit:
        result = result[:limit] + "\n\n... (content truncated)"
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definitions (OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────

def get_browser_tool_definitions() -> list:
    """Return OpenAI-compatible tool definitions for all browser tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "browser_open",
                "description": (
                    "Open a URL in a headless Chromium browser. Returns the page title "
                    "and a text summary. Starts a persistent browser session that "
                    "maintains cookies and state across calls."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to open (e.g. 'https://example.com')"
                        }
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "browser_click",
                "description": (
                    "Click an element on the current page. Accepts a CSS selector, "
                    "visible text, or aria-label. Smart resolution tries multiple "
                    "strategies to find the element."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selector": {
                            "type": "string",
                            "description": (
                                "CSS selector (e.g. '#submit-btn', '.nav-link'), "
                                "visible text (e.g. 'Sign In'), or aria-label"
                            )
                        }
                    },
                    "required": ["selector"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "browser_type",
                "description": (
                    "Type text into an input field or textarea on the current page. "
                    "Use selector='active' to type into the currently focused element."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selector": {
                            "type": "string",
                            "description": (
                                "CSS selector, placeholder text, label, or 'active' "
                                "for the currently focused element"
                            )
                        },
                        "text": {
                            "type": "string",
                            "description": "The text to type into the field"
                        }
                    },
                    "required": ["selector", "text"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "browser_navigate",
                "description": (
                    "Navigate to a new URL in the current browser session. "
                    "Preserves cookies and login state from previous pages."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to navigate to"
                        }
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "browser_screenshot",
                "description": (
                    "Take a screenshot of the current browser page and analyze it "
                    "with the LLaVA vision model. Returns a description of the visual "
                    "layout, content, and interactive elements."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "browser_extract",
                "description": (
                    "Extract text content from the current page or a specific element. "
                    "Limited to 10,000 characters. Use with no selector for full page "
                    "text, or pass a CSS selector for a specific section."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selector": {
                            "type": "string",
                            "description": (
                                "Optional CSS selector to extract from a specific element. "
                                "Omit for full page text."
                            )
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "browser_fill_form",
                "description": (
                    "Fill multiple form fields at once. Each key in the fields object "
                    "is a CSS selector (or label/placeholder), and each value is the "
                    "text to type into that field."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fields": {
                            "type": "object",
                            "description": (
                                "Object mapping selectors to values. "
                                "Example: {\"#username\": \"alice\", \"#password\": \"secret\"}"
                            ),
                            "additionalProperties": {"type": "string"}
                        }
                    },
                    "required": ["fields"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "browser_select",
                "description": (
                    "Select an option from a dropdown/select element. "
                    "Tries matching by value, then label, then numeric index."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selector": {
                            "type": "string",
                            "description": "CSS selector for the <select> element"
                        },
                        "value": {
                            "type": "string",
                            "description": "The option value, visible label, or numeric index to select"
                        }
                    },
                    "required": ["selector", "value"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "browser_login",
                "description": (
                    "Open a VISIBLE (non-headless) browser window for manual login to a website. "
                    "Accepts shorthand names like 'x', 'linkedin', 'instagram', 'facebook', "
                    "'github', 'reddit', 'google', 'tiktok', 'pinterest', 'bluesky' or a full URL. "
                    "Log in manually, then close the browser to save the session. "
                    "Future headless browser operations will use the saved cookies."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": (
                                "Site to log into — shorthand name (e.g. 'x', 'linkedin', 'instagram') "
                                "or full URL (e.g. 'https://example.com/login')"
                            )
                        }
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "browser_check_session",
                "description": (
                    "Check if a saved login session is still valid for a website. "
                    "Opens the site headlessly and looks for a logged-in indicator element. "
                    "Returns VALID if found, EXPIRED if not."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": (
                                "Site to check — shorthand name (e.g. 'x', 'linkedin') "
                                "or full URL"
                            )
                        },
                        "logged_in_selector": {
                            "type": "string",
                            "description": (
                                "CSS selector, text, or aria-label that indicates a logged-in state "
                                "(e.g. 'nav[aria-label=\"Primary\"]', 'Home', 'Profile')"
                            )
                        }
                    },
                    "required": ["url", "logged_in_selector"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "browser_close",
                "description": (
                    "Close the browser session and free resources. "
                    "The session will also auto-close after 5 minutes of inactivity."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Tool Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def execute_browser_tool(name: str, args: dict) -> str:
    """
    Dispatch a browser tool call by name.

    Args:
        name: The tool name (e.g. 'browser_open')
        args: Dictionary of arguments for the tool

    Returns:
        String result of the tool execution.
    """
    dispatch = {
        "browser_open":          lambda a: browser_open(a.get("url", "")),
        "browser_click":         lambda a: browser_click(a.get("selector", "")),
        "browser_type":          lambda a: browser_type(a.get("selector", ""), a.get("text", "")),
        "browser_navigate":      lambda a: browser_navigate(a.get("url", "")),
        "browser_screenshot":    lambda a: browser_screenshot(),
        "browser_extract":       lambda a: browser_extract(a.get("selector")),
        "browser_fill_form":     lambda a: browser_fill_form(a.get("fields", {})),
        "browser_select":        lambda a: browser_select(a.get("selector", ""), a.get("value", "")),
        "browser_login":         lambda a: browser_login(a.get("url", "")),
        "browser_check_session": lambda a: browser_check_session(a.get("url", ""), a.get("logged_in_selector", "")),
        "browser_close":         lambda a: browser_close(),
    }

    handler = dispatch.get(name)
    if handler is None:
        return f"Error: Unknown browser tool '{name}'. Available: {', '.join(dispatch.keys())}"

    try:
        return handler(args)
    except Exception as e:
        return f"Error executing {name}: {e}"
