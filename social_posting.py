"""
Social Media Posting for MAUDE — posts via the running BrowserSession.

Flow: browser_login('x') → user logs in via VNC → browser stays open →
social_post('x', 'content') navigates the SAME browser to compose & publish.

No new browser instance. No cookie expiration. Session stays alive as long
as the browser is running on Spark.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

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


def _locator_count(page, selector: str) -> int:
    try:
        return page.locator(selector).count()
    except Exception:
        return 0


def _is_video_media(path: str | None) -> bool:
    if not path:
        return False
    return Path(path).suffix.lower() in {".mp4", ".mov", ".m4v", ".webm"}


def _latest_shared_video(max_age_hours: int = 48) -> str | None:
    shared_dir = Path(__file__).resolve().parent / "shared"
    if not shared_dir.exists():
        return None

    cutoff = time.time() - (max_age_hours * 3600)
    videos = []
    for ext in ("*.mp4", "*.mov", "*.m4v", "*.webm"):
        videos.extend(shared_dir.glob(ext))
    videos = [p for p in videos if p.is_file() and p.stat().st_mtime >= cutoff]
    if not videos:
        return None
    return str(max(videos, key=lambda p: p.stat().st_mtime).resolve())


def _x_media_state(page) -> dict:
    try:
        return page.evaluate(
            """
            () => {
                const dialogs = Array.from(document.querySelectorAll('div[role="dialog"]'));
                const root = dialogs[dialogs.length - 1] || document;
                const text = root.innerText || '';
                const hasPreview = !!root.querySelector(
                    '[data-testid="attachments"], [data-testid="videoPlayer"], video, [aria-label*="Remove media"], [aria-label*="Remove video"], [aria-label*="Remove image"], [aria-label*="Edit media"]'
                );
                const processingTexts = [
                    'Preparing media',
                    'Processing media',
                    'Uploading media',
                    'Uploading video',
                    'Processing video',
                    'Transcoding video'
                ];
                const processing = processingTexts.some(value => text.includes(value));
                const errors = [
                    'The media could not be played.',
                    'The media could not be uploaded.',
                    'Your video file could not be processed.',
                    'Unsupported video format',
                    'upload failed',
                    'Upload failed',
                    'file is too large',
                    'File is too large',
                    'video is too long',
                    'Video is too long'
                ];
                const error = errors.find(value => text.includes(value)) || null;
                const postButton = root.querySelector('[data-testid="tweetButtonInline"], [data-testid="tweetButton"]');
                const postEnabled = !!postButton && !postButton.disabled && postButton.getAttribute('aria-disabled') !== 'true';
                return { hasPreview, processing, error, postEnabled };
            }
            """
        )
    except Exception:
        return {"hasPreview": False, "processing": False, "error": None, "postEnabled": False}


def _x_wait_for_media_ready(page, media_path: str) -> str | None:
    # X's video composer DOM changes frequently. Treat explicit media errors as fatal,
    # but do not require a specific preview selector before checking the Post button.
    settle_s = 20 if _is_video_media(media_path) else 5
    for elapsed in range(settle_s):
        state = _x_media_state(page)
        if state.get("error"):
            return f"Error: X media upload failed: {state['error']}"
        if state.get("hasPreview") and not state.get("processing"):
            return None
        time.sleep(1)
    state = _x_media_state(page)
    log(f"social_post x: media settle complete, continuing to post button wait, state={state}")
    return None


def _x_wait_for_post_button(page, root, has_media: bool):
    timeout_s = 300 if has_media else 60
    selectors = [
        '[data-testid="tweetButtonInline"]',
        '[data-testid="tweetButton"]',
    ]
    last_state = None
    for elapsed in range(timeout_s):
        if has_media:
            last_state = _x_media_state(page)
            if last_state.get("error"):
                return None, f"Error: X media upload failed: {last_state['error']}"
            if elapsed and elapsed % 30 == 0:
                log(f"social_post x: waiting for post button, state={last_state}")
        post_btn = _first_match(root, selectors)
        try:
            if post_btn and post_btn.is_enabled():
                return post_btn, None
        except Exception:
            pass
        time.sleep(1)
    if has_media:
        return None, f"Error: Post button not found or disabled on X after media upload. Last media state: {last_state}"
    return None, "Error: Post button not found or disabled on X."


def _x_verify_post_media(page, is_video: bool) -> bool:
    selector = (
        '[data-testid="videoPlayer"], article video'
        if is_video
        else 'article [data-testid="tweetPhoto"], article img[src*="twimg.com/media"]'
    )
    deadline = time.time() + 45
    while time.time() < deadline:
        if _locator_count(page, selector) > 0:
            return True
        time.sleep(1)
    return False


def _x_composer_root_from_compose(compose):
    try:
        handle = compose.element_handle(timeout=5_000)
        if not handle:
            return None
        root_handle = handle.evaluate_handle(
            """
            node => {
                let el = node;
                for (let i = 0; i < 24 && el; i++, el = el.parentElement) {
                    const fileInputs = el.querySelectorAll('input[type="file"]');
                    const postButton = el.querySelector('[data-testid="tweetButtonInline"], [data-testid="tweetButton"]');
                    if (fileInputs.length && postButton) return el;
                }
                return node.closest('[role="dialog"]') || node.closest('main') || document.body;
            }
            """
        )
        return root_handle.as_element()
    except Exception:
        return None


def _x_select_media(page, root, media_path: str) -> bool:
    selectors = [
        'input[data-testid="fileInput"]',
        'input[type="file"][accept*="video"]',
        'input[type="file"][accept*="image"]',
        'input[type="file"]',
    ]
    for sel in selectors:
        try:
            loc = root.locator(sel) if hasattr(root, "locator") else None
            count = loc.count() if loc else 0
            for idx in range(count):
                candidate = loc.nth(idx)
                accept = candidate.get_attribute("accept") or ""
                if "image" in accept or "video" in accept or sel == 'input[type="file"]':
                    candidate.set_input_files(media_path)
                    return True
        except Exception:
            continue
    return False


def _screenshot(page, label: str = "social") -> str | None:
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
            "No browser running. Run browser_login('<platform>') first and keep the browser open after logging in."
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
        "img.global-nav__me-photo",
        ".feed-identity-module",
        "button:has(img.global-nav__me-photo)",
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
        'a:has(span:text-is("Create"))',
        'a:has(span:text-is("Post"))',
        'span:text-is("Home")',  # sidebar nav present = logged in
    ],
    "reddit": [
        '[data-testid="reddit-user-nav"]',
        'button[aria-label*="profile" i]',
        'a[href*="/user/"]',
    ],
    "tiktok": [
        '[data-e2e="upload-btn"]',
        'a[href*="/upload"]',
        '[data-e2e="profile-icon"]',
    ],
    "bluesky": [
        '[data-testid="composeButton"]',
        'button[aria-label*="New post" i]',
        'button[aria-label*="compose" i]',
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


def _post_x(page, content: str, image_path: str | None) -> str:
    page.goto("https://x.com/compose/post", wait_until="domcontentloaded", timeout=30_000)
    _human_pause(2, 4)

    if not _is_logged_in(page, "x"):
        return "Error: Not logged in to X. Run browser_login('x') and keep the browser open."

    dialog = _first_match(page, ['div[role="dialog"]'])
    root = dialog if dialog else page

    compose = _first_match(
        root,
        [
            '[data-testid="tweetTextarea_0"]',
            'div[role="textbox"][data-testid="tweetTextarea_0"]',
            '[aria-label="Post text"]',
            "div.public-DraftEditor-content",
        ],
    )

    if compose is None:
        _screenshot(page, "x_compose_fail")
        return "Error: Could not find the compose area on X."

    composer_root = _x_composer_root_from_compose(compose) or root

    compose.click()
    _human_pause(0.5, 1.0)
    _type_human(page, content)
    _human_pause(1, 2)

    if image_path:
        media_path = str(Path(image_path).expanduser())
        if not Path(media_path).exists():
            return f"Error: Media file not found: {media_path}"

        if _x_select_media(page, composer_root, media_path):
            _screenshot(page, "x_media_selected")
            media_error = _x_wait_for_media_ready(page, media_path)
            if media_error:
                _screenshot(page, "x_media_attach_fail")
                return media_error
            state = _x_media_state(page)
            if not state.get("hasPreview"):
                _screenshot(page, "x_media_no_preview")
                return f"Error: X accepted the file input, but no media preview appeared. State: {state}"
            _screenshot(page, "x_media_ready")
            _human_pause(2, 4)
        else:
            return "Error: Could not find the composer file input on X for media upload."

    post_btn, post_error = _x_wait_for_post_button(page, composer_root, bool(image_path))
    if post_error:
        _screenshot(page, "x_post_btn_fail")
        return post_error

    post_btn.click()
    _human_pause(3, 5)

    if image_path and not _x_verify_post_media(page, _is_video_media(image_path)):
        _screenshot(page, "x_post_verify_missing_media")
        return "Posted to X, but MAUDE could not verify the uploaded media afterward. Check the latest x_post_verify_missing_media screenshot."

    ss = _screenshot(page, "x_posted")
    return f"Posted to X successfully.{f' Screenshot: {ss}' if ss else ''}"

def _post_linkedin(page, content: str, image_path: str | None) -> str:
    page.goto("https://www.linkedin.com/feed/", wait_until="domcontentloaded", timeout=30_000)
    _human_pause(2, 4)

    if not _is_logged_in(page, "linkedin"):
        return "Error: Not logged in to LinkedIn. Run browser_login('linkedin') and keep the browser open."

    start_btn = _first_match(
        page,
        [
            "button.share-box-feed-entry__trigger",
            'button:has-text("Start a post")',
            '[aria-label="Text"]',
        ],
    )
    if not start_btn:
        _screenshot(page, "linkedin_start_fail")
        return "Error: Could not find 'Start a post' on LinkedIn."

    start_btn.click()
    _human_pause(1, 2)

    editor = _first_match(
        page,
        [
            "div.ql-editor[data-placeholder]",
            '[role="textbox"][aria-label*="editor"]',
            ".ql-editor",
            'div[contenteditable="true"]',
        ],
    )
    if not editor:
        _screenshot(page, "linkedin_editor_fail")
        return "Error: Could not find LinkedIn post editor."

    editor.click()
    _human_pause(0.5, 1.0)
    _type_human(page, content, 25, 60)
    _human_pause(1, 2)

    if image_path:
        try:
            media_btn = _first_match(
                page,
                [
                    '[aria-label="Add media"]',
                    'button[aria-label="Add media"]',
                    'button:has(li-icon[type="image"])',
                ],
            )
            if media_btn:
                media_btn.click()
                _human_pause(1, 2)
            fi = page.locator('input[type="file"]').first
            fi.set_input_files(image_path)
            _human_pause(3, 5)

            # LinkedIn opens an image Editor with Back/Next buttons.
            # Click "Next" (possibly twice) to get past it to the post view.
            for _ in range(3):
                next_btn = _first_match(
                    page,
                    [
                        'button:has-text("Next")',
                        '[aria-label="Next"]',
                        'button.share-box-footer__primary-btn:has-text("Next")',
                    ],
                )
                if next_btn:
                    next_btn.click()
                    _human_pause(1, 3)
                else:
                    break

            # After editor, LinkedIn may show a "Done" button
            done_btn = _first_match(
                page,
                [
                    'button:has-text("Done")',
                    '[aria-label="Done"]',
                ],
            )
            if done_btn:
                done_btn.click()
                _human_pause(1, 2)

        except Exception:
            log("LinkedIn image upload failed — continuing without image")

    post_btn = _first_match(
        page,
        [
            "button.share-actions__primary-action",
            'button:has-text("Post")',
            '[aria-label="Post"]',
        ],
    )
    if not post_btn:
        _screenshot(page, "linkedin_post_fail")
        return "Error: Could not click the Post button on LinkedIn."

    post_btn.click()
    _human_pause(3, 5)

    ss = _screenshot(page, "linkedin_posted")
    return f"Posted to LinkedIn successfully.{f' Screenshot: {ss}' if ss else ''}"


def _post_facebook(page, content: str, image_path: str | None) -> str:
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

    # Step 1: Open compose dialog via "What's on your mind?"
    trigger = _first_match(
        page,
        [
            '[aria-label="Create a post"]',
            '[role="button"]:has-text("What\'s on your mind")',
            'span:has-text("What\'s on your mind")',
        ],
    )
    if not trigger:
        _screenshot(page, "facebook_compose_fail")
        return "Error: Could not find Facebook compose trigger."

    trigger.click()
    _human_pause(2, 3)

    # Step 2: Upload image FIRST if provided (image must go before text
    # so the compose dialog doesn't lose focus when the image loads in)
    if image_path:
        # Facebook keeps hidden file inputs in the DOM. First check if one
        # exists already; if not, click the Photo/video icon in "Add to your
        # post" to make it appear.
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

        if not fi:
            # Click the Photo/video icon in "Add to your post" bar.
            # Use JS to find it reliably — the icons have no stable aria-labels.
            try:
                page.evaluate("""
                    () => {
                        // Strategy 1: aria-label on the icon
                        const byLabel = document.querySelector(
                            '[aria-label="Photo/video"], [aria-label="Photo/Video"]'
                        );
                        if (byLabel) {
                            const btn = byLabel.closest('[role="button"]') || byLabel;
                            btn.click();
                            return true;
                        }
                        // Strategy 2: find "Add to your post" text, click first icon sibling
                        const spans = document.querySelectorAll('span');
                        for (const s of spans) {
                            if (s.textContent.trim() === 'Add to your post') {
                                // The icons are siblings of the span's container
                                const row = s.parentElement;
                                if (row) {
                                    const icons = row.querySelectorAll(
                                        '[role="button"], [tabindex="0"], div[class] > div[class]'
                                    );
                                    for (const icon of icons) {
                                        // Skip the text span itself
                                        if (icon.contains(s)) continue;
                                        icon.click();
                                        return true;
                                    }
                                }
                            }
                        }
                        return false;
                    }
                """)
                _human_pause(2, 3)
            except Exception:
                pass

            # Re-check for file input
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
            try:
                fi.set_input_files(image_path)
                _human_pause(3, 5)
            except Exception as e:
                _screenshot(page, "facebook_upload_fail")
                return f"Error uploading image to Facebook: {e}"
        else:
            _screenshot(page, "facebook_no_file_input")
            return "Error: Could not find file input on Facebook. Image upload failed."

    # Step 3: Type the content (after image so compose doesn't reflow).
    # Facebook's Lexical editor needs a real Playwright click to activate —
    # plain JS focus/click doesn't trigger React's event handlers.
    editor_focused = False

    # Try 1: Playwright force-click on the editor (bypasses actionability)
    for sel in [
        'div[contenteditable="true"][role="textbox"]',
        '[aria-label*="What\'s on your mind"]',
        'div[contenteditable="true"][data-lexical-editor="true"]',
    ]:
        try:
            loc = page.locator(sel)
            if loc.count() > 0:
                loc.first.click(force=True, timeout=5000)
                editor_focused = True
                _human_pause(0.5, 1.0)
                break
        except Exception:
            continue

    # Try 2: Click by coordinates — the text area is near the top of the
    # compose dialog, roughly at the "What's on your mind?" placeholder
    if not editor_focused:
        try:
            dialog = page.locator('div[role="dialog"]').first
            box = dialog.bounding_box()
            if box:
                # Click near the top-center of the dialog (where the text area is)
                page.mouse.click(box["x"] + box["width"] / 2, box["y"] + 100)
                editor_focused = True
                _human_pause(0.5, 1.0)
        except Exception:
            pass

    # Try 3: Use Tab to cycle focus into the editor
    if not editor_focused:
        try:
            page.keyboard.press("Tab")
            _human_pause(0.3, 0.5)
            page.keyboard.press("Tab")
            _human_pause(0.3, 0.5)
        except Exception:
            pass

    _type_human(page, content, 25, 60)
    _human_pause(1, 2)

    # Step 4: Click Post (also force in case of overlay)
    _human_pause(1, 2)
    post_btn = _first_match(
        page,
        [
            '[aria-label="Post"]',
            'div[aria-label="Post"]',
            'button:has-text("Post")',
        ],
    )
    if not post_btn:
        _screenshot(page, "facebook_post_fail")
        return "Error: Could not find Facebook Post button."

    try:
        post_btn.click(force=True)
    except Exception:
        # JS fallback for Post button
        try:
            page.evaluate("""
                () => {
                    const btn = document.querySelector('[aria-label="Post"]');
                    if (btn) { btn.click(); return; }
                    const divs = document.querySelectorAll('div[role="button"]');
                    for (const d of divs) {
                        if (d.textContent.trim() === 'Post') { d.click(); return; }
                    }
                }
            """)
        except Exception:
            _screenshot(page, "facebook_post_fail")
            return "Error: Could not click Facebook Post button."
    _human_pause(5, 8)

    ss = _screenshot(page, "facebook_posted")
    return f"Posted to Facebook successfully.{f' Screenshot: {ss}' if ss else ''}"


def _post_instagram(page, content: str, image_path: str) -> str:
    page.goto("https://www.instagram.com/", wait_until="domcontentloaded", timeout=30_000)
    _human_pause(2, 4)

    if not _is_logged_in(page, "instagram"):
        return "Error: Not logged in to Instagram. Run browser_login('instagram') and keep the browser open."

    # Instagram sidebar has separate "Create" and "Post" items.
    # "Create" opens a general creation menu; "Post" may go directly to post flow.
    # We need to open the creation dialog that shows a file upload.
    # Strategy: try multiple selectors, after each click wait and check for the
    # dialog (indicated by a file input, "Select from computer", or a drag-drop area).
    opened = False

    # Selectors ordered by specificity — "Post" sidebar item first since it's
    # the most direct route, then "Create", then aria-labels, then href fallback.
    create_selectors = [
        # "Post" sidebar link (Instagram added this as a direct post shortcut)
        'a:has(span:text-is("Post"))',
        'div[role="button"]:has(span:text-is("Post"))',
        # "Create" sidebar link
        'a:has(span:text-is("Create"))',
        'div[role="button"]:has(span:text-is("Create"))',
        # SVG/aria-label variants
        '[aria-label="New post"]',
        '[aria-label="Create"]',
        'svg[aria-label="New post"]',
        'svg[aria-label="Create"]',
        # href-based
        'a[href*="/create"]',
    ]

    def _dialog_appeared() -> bool:
        """Check if the create/upload dialog is visible."""
        try:
            if page.locator('input[type="file"]').count() > 0:
                return True
        except Exception:
            pass
        if _first_match(
            page,
            [
                'button:has-text("Select from computer")',
                'button:has-text("Select From Computer")',
                'button:has-text("Select from")',
                # Drag-and-drop area text that appears in the create dialog
                'text="Drag photos and videos here"',
                'text="Drag and drop photos and videos here"',
            ],
        ):
            return True
        # Check for the dialog/modal container itself
        return bool(
            _first_match(
                page,
                [
                    'div[role="dialog"]:has(input[type="file"])',
                    'div[role="dialog"]:has-text("Select from computer")',
                    'div[role="dialog"]:has-text("Create new post")',
                ],
            )
        )

    for sel in create_selectors:
        btn = _first_match(page, [sel])
        if btn:
            try:
                btn.click()
            except Exception:
                continue
            _human_pause(2, 3)

            if _dialog_appeared():
                opened = True
                break

            # "Create" might open a sub-menu (Post, Reel, Story, etc.).
            # Look for a "Post" option inside the menu and click it.
            submenu_post = _first_match(
                page,
                [
                    'text="Post"',
                    'button:has-text("Post")',
                    'div[role="menuitem"]:has-text("Post")',
                    'a:has-text("Post")',
                    '[role="dialog"] span:text-is("Post")',
                ],
            )
            if submenu_post:
                try:
                    submenu_post.click()
                except Exception:
                    continue
                _human_pause(2, 3)
                if _dialog_appeared():
                    opened = True
                    break

    if not opened:
        # JavaScript fallback: find the create/post link by visible text
        try:
            page.evaluate("""
                () => {
                    const spans = document.querySelectorAll('span');
                    for (const s of spans) {
                        if (s.textContent.trim() === 'Create' || s.textContent.trim() === 'Post') {
                            const clickable = s.closest('a') || s.closest('[role="button"]') || s.parentElement;
                            if (clickable) { clickable.click(); return true; }
                        }
                    }
                    return false;
                }
            """)
            _human_pause(2, 3)
            if _dialog_appeared():
                opened = True
        except Exception:
            pass

    if not opened:
        # Last resort: navigate directly to create URLs
        for url in [
            "https://www.instagram.com/create/select/",
            "https://www.instagram.com/create/style/",
        ]:
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=15_000)
                _human_pause(2, 4)
                if _dialog_appeared():
                    opened = True
                    break
            except Exception:
                continue

    if not opened:
        _screenshot(page, "instagram_newpost_fail")
        return "Error: Could not open Instagram create dialog. The UI may have changed."

    # Click "Select from computer" if visible (exposes the hidden file input)
    select_btn = _first_match(
        page,
        [
            'button:has-text("Select from computer")',
            'button:has-text("Select From Computer")',
            'button:has-text("Select from")',
        ],
    )
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
        next_btn = _first_match(
            page,
            [
                'button:has-text("Next")',
                '[aria-label="Next"]',
                'div[role="button"]:has-text("Next")',
            ],
        )
        if next_btn:
            next_btn.click(force=True)
            _human_pause(2, 4)
        else:
            break

    # Write caption
    caption_box = _first_match(
        page,
        [
            '[aria-label="Write a caption..."]',
            '[aria-label="Write a caption\u2026"]',
            'textarea[aria-label*="caption"]',
            'div[role="textbox"]',
            '[contenteditable="true"]',
        ],
    )
    if caption_box:
        caption_box.click()
        _human_pause(0.5, 1.0)
        _type_human(page, content, 25, 60)
        _human_pause(1, 2)

    share_btn = _first_match(
        page,
        [
            'button:has-text("Share")',
            '[aria-label="Share"]',
            'div[role="button"]:has-text("Share")',
        ],
    )
    if not share_btn:
        _screenshot(page, "instagram_share_fail")
        return "Error: Could not find Instagram Share button."

    share_btn.click(force=True)
    _human_pause(5, 8)

    ss = _screenshot(page, "instagram_posted")
    return f"Posted to Instagram successfully.{f' Screenshot: {ss}' if ss else ''}"


# ─────────────────────────────────────────────────────────────────────────────
# Reddit
# ─────────────────────────────────────────────────────────────────────────────


def _post_reddit(page, content: str, image_path: str | None, subreddit: str = None) -> str:
    """Post to Reddit. If subreddit is given, posts there; otherwise posts to the user's profile."""
    if subreddit:
        sub = subreddit.lstrip("r/").lstrip("/")
        page.goto(f"https://www.reddit.com/r/{sub}/submit", wait_until="domcontentloaded", timeout=30_000)
    else:
        page.goto("https://www.reddit.com/submit", wait_until="domcontentloaded", timeout=30_000)
    _human_pause(2, 4)

    # Check login
    login_indicators = [
        'button:has-text("Log In")',
        'a[href*="login"]',
    ]
    for sel in login_indicators:
        try:
            if page.locator(sel).count() > 0 and page.locator('[data-testid="reddit-user-nav"]').count() == 0:
                return "Error: Not logged in to Reddit. Run browser_login('reddit') first."
        except Exception:
            continue

    _human_pause(1, 2)

    # Reddit's new editor — try the title field first
    title_field = _first_match(page, [
        'textarea[placeholder*="title" i]',
        'input[placeholder*="title" i]',
        '[data-testid="post-title"]',
        'div[contenteditable="true"][role="textbox"]:first-of-type',
    ])

    # Split content: first line is title, rest is body
    lines = content.strip().split("\n", 1)
    title = lines[0].strip()
    body = lines[1].strip() if len(lines) > 1 else ""

    if title_field:
        title_field.click(timeout=5_000)
        _type_human(page, title)
        _human_pause(0.5, 1.5)

    # Body text
    if body:
        body_field = _first_match(page, [
            'div[contenteditable="true"][role="textbox"]',
            'textarea[placeholder*="text" i]',
            '.DraftEditor-root',
            '[data-testid="post-content"]',
        ])
        if body_field:
            body_field.click(timeout=5_000)
            _human_pause(0.3, 0.8)
            _type_human(page, body)

    # Image upload
    if image_path:
        file_input = _first_match(page, [
            'input[type="file"]',
            'input[accept*="image"]',
        ])
        if file_input:
            file_input.set_input_files(image_path)
            _human_pause(3, 5)

    _human_pause(1, 2)

    # Post button
    post_btn = _first_match(page, [
        'button:has-text("Post")',
        'button[type="submit"]:has-text("Post")',
        '[data-testid="post-submit-button"]',
    ])
    if not post_btn:
        _screenshot(page, "reddit_post_fail")
        return "Error: Could not find Reddit Post button."

    post_btn.click(timeout=10_000)
    _human_pause(3, 5)

    ss = _screenshot(page, "reddit_posted")
    return f"Posted to Reddit successfully.{f' Screenshot: {ss}' if ss else ''}"


# ─────────────────────────────────────────────────────────────────────────────
# TikTok
# ─────────────────────────────────────────────────────────────────────────────


def _post_tiktok(page, content: str, image_path: str | None) -> str:
    """Post to TikTok. Requires a video file (image_path should be a video)."""
    if not image_path:
        return "Error: TikTok requires a video file. Provide image_path pointing to a video (.mp4, .mov, .webm)."

    page.goto("https://www.tiktok.com/upload", wait_until="domcontentloaded", timeout=30_000)
    _human_pause(3, 5)

    # Check login
    login_check = _first_match(page, [
        'button:has-text("Log in")',
        '[data-e2e="top-login-button"]',
    ])
    profile_check = _first_match(page, [
        '[data-e2e="upload-btn"]',
        'div[class*="upload"]',
        'input[type="file"]',
    ])
    if login_check and not profile_check:
        return "Error: Not logged in to TikTok. Run browser_login('tiktok') first."

    # File upload
    file_input = _first_match(page, [
        'input[type="file"][accept*="video"]',
        'input[type="file"]',
    ])
    if not file_input:
        _screenshot(page, "tiktok_upload_fail")
        return "Error: Could not find TikTok file upload input."

    file_input.set_input_files(image_path)
    _human_pause(5, 10)  # TikTok processes the video

    # Caption
    caption_field = _first_match(page, [
        'div[contenteditable="true"][data-text="true"]',
        'div[contenteditable="true"]',
        '[data-e2e="caption-input"]',
        'div[class*="caption"] div[contenteditable]',
    ])
    if caption_field:
        caption_field.click(timeout=5_000)
        # Clear default text
        page.keyboard.press("Control+a")
        _human_pause(0.2, 0.5)
        _type_human(page, content)
        _human_pause(1, 2)

    # Post button
    post_btn = _first_match(page, [
        'button:has-text("Post")',
        '[data-e2e="post-button"]',
        'button[class*="post-button"]',
    ])
    if not post_btn:
        _screenshot(page, "tiktok_post_fail")
        return "Error: Could not find TikTok Post button."

    post_btn.click(timeout=10_000)
    _human_pause(5, 8)

    ss = _screenshot(page, "tiktok_posted")
    return f"Posted to TikTok successfully.{f' Screenshot: {ss}' if ss else ''}"


# ─────────────────────────────────────────────────────────────────────────────
# Bluesky
# ─────────────────────────────────────────────────────────────────────────────


def _post_bluesky(page, content: str, image_path: str | None) -> str:
    """Post to Bluesky (bsky.app)."""
    page.goto("https://bsky.app/", wait_until="domcontentloaded", timeout=30_000)
    _human_pause(2, 4)

    # Check login — look for compose button or "Sign in" button
    signin_check = _first_match(page, [
        'button:has-text("Sign in")',
        'a:has-text("Sign in")',
    ])
    compose_check = _first_match(page, [
        '[data-testid="composeButton"]',
        'button[aria-label*="New post" i]',
        'button[aria-label*="compose" i]',
    ])
    if signin_check and not compose_check:
        return "Error: Not logged in to Bluesky. Run browser_login('bluesky') first."

    # Click compose
    if compose_check:
        compose_check.click(timeout=5_000)
        _human_pause(1, 2)

    # Type into the compose box
    editor = _first_match(page, [
        'div[contenteditable="true"][role="textbox"]',
        'div[contenteditable="true"]',
        '[data-testid="composerTextInput"]',
    ])
    if not editor:
        _screenshot(page, "bluesky_compose_fail")
        return "Error: Could not find Bluesky compose editor."

    editor.click(timeout=5_000)
    _human_pause(0.3, 0.8)
    _type_human(page, content)
    _human_pause(1, 2)

    # Image upload
    if image_path:
        file_input = _first_match(page, [
            'input[type="file"][accept*="image"]',
            'input[type="file"]',
        ])
        if file_input:
            file_input.set_input_files(image_path)
            _human_pause(2, 4)
        else:
            # Try the image button then the file input
            img_btn = _first_match(page, [
                'button[aria-label*="photo" i]',
                'button[aria-label*="image" i]',
                'button[aria-label*="gallery" i]',
            ])
            if img_btn:
                img_btn.click(timeout=5_000)
                _human_pause(1, 2)
                file_input = page.locator('input[type="file"]').first
                file_input.set_input_files(image_path)
                _human_pause(2, 4)

    # Post button
    post_btn = _first_match(page, [
        'button:has-text("Post")',
        '[data-testid="composerPublishButton"]',
        'button[aria-label*="publish" i]',
    ])
    if not post_btn:
        _screenshot(page, "bluesky_post_fail")
        return "Error: Could not find Bluesky Post button."

    post_btn.click(timeout=10_000)
    _human_pause(3, 5)

    ss = _screenshot(page, "bluesky_posted")
    return f"Posted to Bluesky successfully.{f' Screenshot: {ss}' if ss else ''}"


# ─────────────────────────────────────────────────────────────────────────────
# Public tool function
# ─────────────────────────────────────────────────────────────────────────────

_POSTERS = {
    "x": _post_x,
    "linkedin": _post_linkedin,
    "facebook": _post_facebook,
    "instagram": _post_instagram,
    "reddit": _post_reddit,
    "tiktok": _post_tiktok,
    "bluesky": _post_bluesky,
}


def social_post(platform: str, content: str, image_path: str = None, **kwargs) -> str:
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

    if not image_path:
        image_path = kwargs.get("media_path") or kwargs.get("video_path")

    expects_media = bool(kwargs.get("expect_media"))
    if platform == "x" and not image_path and expects_media:
        image_path = _latest_shared_video()
        if image_path:
            log(f"social_post x: auto-attaching latest shared video: {image_path}")

    if platform not in _POSTERS:
        return f"Error: Unsupported platform '{platform}'. Supported: {', '.join(_POSTERS)}"

    if platform == "instagram" and not image_path:
        return "Error: Instagram requires an image. Provide image_path or media_path."
    if platform == "tiktok" and not image_path:
        return "Error: TikTok requires a video. Provide video_path, media_path, or image_path."
    if expects_media and not image_path:
        return "Error: This post was requested with media, but no media_path/video_path/image_path was provided."

    if image_path:
        p = Path(image_path).expanduser()
        if not p.exists():
            return f"Error: Media file not found: {image_path}"
        image_path = str(p.resolve())

    from browser import _get_session

    session = _get_session()

    def _do_post():
        page, err = _get_page(platform)
        if err:
            return err
        try:
            poster = _POSTERS[platform]
            # Reddit accepts an extra subreddit kwarg
            if platform == "reddit":
                return poster(page, content, image_path, subreddit=kwargs.get("subreddit"))
            return poster(page, content, image_path)
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
                "Post content to a social media platform using the running browser session. "
                "Requires browser_login('<platform>') first — keep the browser open after login. "
                "Instagram requires an image. TikTok requires a video file. "
                "X, LinkedIn, Facebook, Reddit, and Bluesky can attach images or videos when media_path/video_path/image_path is provided. "
                "For Reddit, first line of content is the title, rest is the body. "
                "IMPORTANT: When the user asks to post WITH an image or video, you MUST pass media_path, video_path, or image_path. Do not make a text-only post if media was requested."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "platform": {
                        "type": "string",
                        "enum": ["x", "linkedin", "facebook", "instagram", "reddit", "tiktok", "bluesky"],
                        "description": "Target platform to post to.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The text content of the post. For Reddit: first line is title, rest is body.",
                    },
                    "image_path": {
                        "type": "string",
                        "description": (
                            "Path to image/video to attach. Use for any platform when attaching media. "
                            "Required for Instagram images and TikTok videos; also valid for X videos."
                        ),
                    },
                    "media_path": {
                        "type": "string",
                        "description": "Alias for image_path. Preferred for X/LinkedIn/Facebook/Reddit/Bluesky media attachments, including videos.",
                    },
                    "video_path": {
                        "type": "string",
                        "description": "Alias for image_path when the attachment is a video, especially for X or TikTok.",
                    },
                    "expect_media": {
                        "type": "boolean",
                        "description": "Set true when the user explicitly requested an image or video attachment; the tool will fail instead of posting text-only if no path is supplied.",
                    },
                    "subreddit": {
                        "type": "string",
                        "description": "Reddit only — subreddit to post to (e.g. 'python'). Omit for user profile.",
                    },
                },
                "required": ["platform", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "social_x_post_video",
            "description": (
                "Post a video to X/Twitter using the running browser session. "
                "Use this instead of browser tools whenever the user asks to upload/post a video to X. "
                "If video_path/media_path/image_path is omitted, the latest shared MP4/MOV/WebM is attached automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The text content of the X post.",
                    },
                    "video_path": {
                        "type": "string",
                        "description": "Path to the video file. Optional; latest shared video is used if omitted.",
                    },
                    "media_path": {
                        "type": "string",
                        "description": "Alias for video_path.",
                    },
                    "image_path": {
                        "type": "string",
                        "description": "Alias for video_path, retained for compatibility.",
                    },
                },
                "required": ["content"],
            },
        },
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
            media_path=a.get("media_path"),
            video_path=a.get("video_path"),
            expect_media=a.get("expect_media"),
            subreddit=a.get("subreddit"),
        ),
        "social_x_post_video": lambda a: social_post(
            "x",
            a.get("content", ""),
            a.get("image_path"),
            media_path=a.get("media_path"),
            video_path=a.get("video_path"),
            expect_media=True,
        ),
    }
    handler = dispatch.get(name)
    if handler is None:
        return f"Error: Unknown social tool '{name}'. Available: {', '.join(dispatch.keys())}"
    try:
        return handler(args)
    except Exception as e:
        return f"Error executing {name}: {e}"
