"""
Browser Automation for MAUDE - Full interactive browser control via Playwright.

Extends MAUDE's existing web_view screenshot capability to full interactive
browser automation: clicking, typing, form filling, navigation, and more.

Uses Playwright's sync API to match MAUDE's synchronous tool execution model.
"""

from __future__ import annotations

import os
import base64
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
INACTIVITY_TIMEOUT = 300        # 5 minutes auto-close
SCREENSHOT_DIR = Path.home() / ".config" / "maude" / "screenshots"


# ─────────────────────────────────────────────────────────────────────────────
# BrowserSession
# ─────────────────────────────────────────────────────────────────────────────

class BrowserSession:
    """Manages a persistent Playwright browser and page with lazy initialization."""

    def __init__(self):
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
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
            self._last_activity = 0.0

            log("Browser session closed.")
            return "Browser session closed."

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
        "browser_open":       lambda a: browser_open(a.get("url", "")),
        "browser_click":      lambda a: browser_click(a.get("selector", "")),
        "browser_type":       lambda a: browser_type(a.get("selector", ""), a.get("text", "")),
        "browser_navigate":   lambda a: browser_navigate(a.get("url", "")),
        "browser_screenshot": lambda a: browser_screenshot(),
        "browser_extract":    lambda a: browser_extract(a.get("selector")),
        "browser_fill_form":  lambda a: browser_fill_form(a.get("fields", {})),
        "browser_select":     lambda a: browser_select(a.get("selector", ""), a.get("value", "")),
        "browser_close":      lambda a: browser_close(),
    }

    handler = dispatch.get(name)
    if handler is None:
        return f"Error: Unknown browser tool '{name}'. Available: {', '.join(dispatch.keys())}"

    try:
        return handler(args)
    except Exception as e:
        return f"Error executing {name}: {e}"
