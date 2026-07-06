"""
Web and vision tool implementations — 4 tools (all cacheable).

Vision routes through the active model's native multimodal support
(Claude, Mistral Large) when available, falling back to local LLaVA.
"""

import os

from openai import OpenAI

from maude_core import config as _config
from maude_core.config import (
    VISION_CAPABLE_MODELS,
    VISION_FALLBACK_KEY_ENV,
    VISION_FALLBACK_MODEL,
    VISION_FALLBACK_URL,
    VISION_MODEL,
    VISION_MODEL_ROUTES,
    VISION_URL,
)
from maude_core.log import log
from maude_core.paths import resolve_path
from tool_registry import register_tool


def _vision_analyze(base64_image: str, mime_type: str, prompt: str) -> str:
    """Send an image to the best available vision model and return the analysis.

    Uses the active model's native multimodal vision when supported (Claude,
    Mistral Large), otherwise falls back to local LLaVA.
    """
    active_model = _config.MODEL

    if active_model in VISION_CAPABLE_MODELS:
        route = VISION_MODEL_ROUTES[active_model]
        api_key = os.environ.get(route["api_key_env"], "")
        if not api_key:
            log(f"No API key for {active_model}, falling back to LLaVA")
            return _vision_llava(base64_image, mime_type, prompt)

        if route["provider"] == "anthropic":
            return _vision_anthropic(active_model, api_key, base64_image, mime_type, prompt)
        else:
            # Mistral / OpenAI-compatible
            return _vision_openai_compat(active_model, route["base_url"], api_key, base64_image, mime_type, prompt)

    # Fallback: Nemotron Nano VL via OpenRouter (free), then LLaVA as last resort
    or_key = os.environ.get(VISION_FALLBACK_KEY_ENV, "")
    if or_key:
        return _vision_openai_compat(
            VISION_FALLBACK_MODEL, VISION_FALLBACK_URL, or_key, base64_image, mime_type, prompt
        )
    return _vision_llava(base64_image, mime_type, prompt)


def _vision_anthropic(model: str, api_key: str, base64_image: str, mime_type: str, prompt: str) -> str:
    """Analyze image via Anthropic Claude API (native vision)."""
    import anthropic

    log(f"Analyzing with {model} (Anthropic vision)...")
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": base64_image,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return response.content[0].text


def _vision_openai_compat(
    model: str, base_url: str, api_key: str, base64_image: str, mime_type: str, prompt: str
) -> str:
    """Analyze image via OpenAI-compatible API (Mistral, OpenRouter)."""
    log(f"Analyzing with {model} (multimodal vision)...")
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                ],
            }
        ],
        max_tokens=1024,
        temperature=0.2,
    )
    return response.choices[0].message.content


def _vision_llava(base64_image: str, mime_type: str, prompt: str) -> str:
    """Analyze image via local LLaVA (Ollama fallback)."""
    log("Analyzing with LLaVA (local fallback)...")
    client = OpenAI(base_url=VISION_URL, api_key="not-needed")
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                ],
            }
        ],
        max_tokens=1024,
        temperature=0.2,
    )
    return response.choices[0].message.content


def tool_web_browse(url: str) -> str:
    """Fetch and parse web page content."""
    import requests

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return "Error: beautifulsoup4 not installed"

    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        log(f"Fetching {url}")

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside", "header", "noscript", "iframe"]):
            tag.decompose()

        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", {"class": ["content", "post", "article", "main"]})
        )
        if main_content:
            text = main_content.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = "\n".join(lines)

        if len(text) > 15000:
            text = text[:15000] + "\n\n... (content truncated)"

        log(f"Retrieved {len(text)} chars from {url}")
        return f"Content from {url}:\n\n{text}"

    except requests.exceptions.Timeout:
        return f"Error: Request timed out for {url}"
    except requests.exceptions.RequestException as e:
        return f"Error fetching {url}: {e}"
    except Exception as e:
        return f"Error parsing {url}: {e}"


def tool_web_search(query: str, num_results: int = 5) -> str:
    """Search the web using DuckDuckGo."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "Error: ddgs not installed"

    try:
        num_results = min(max(1, num_results), 10)
        log(f"Searching: {query}")

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))

        if not results:
            return f"No results found for: {query}"

        output = f"Search results for: {query}\n\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. {r.get('title', 'No title')}\n"
            output += f"   URL: {r.get('href', 'No URL')}\n"
            output += f"   {r.get('body', 'No description')}\n\n"

        log(f"Found {len(results)} results")
        return output

    except Exception as e:
        return f"Error searching: {e}"


def tool_web_view(url: str, question: str = None) -> str:
    """Screenshot a webpage and analyze it with the active vision model."""
    import base64

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "Error: playwright not installed"

    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        log(f"Capturing screenshot of {url}")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1024, "height": 768})
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(1000)
            screenshot_bytes = page.screenshot(full_page=False)
            browser.close()

        base64_image = base64.b64encode(screenshot_bytes).decode("utf-8")
        log(f"Screenshot captured ({len(screenshot_bytes) // 1024}KB)")

        if question:
            prompt = f"This is a screenshot of {url}. {question}"
        else:
            prompt = f"This is a screenshot of {url}. Describe what you see."

        analysis = _vision_analyze(base64_image, "image/png", prompt)
        log("Vision analysis complete")

        result = f"Visual analysis of {url}:\n\n{analysis}"
        return result

    except Exception as e:
        error_msg = str(e)
        if "playwright" in error_msg.lower():
            return f"Error: Playwright issue. Run: playwright install chromium\n{e}"
        return f"Error viewing webpage: {e}"


def tool_view_image(path: str, question: str = None) -> str:
    """Analyze a local image file with the active vision model."""
    import base64

    try:
        file_path = resolve_path(path)
        if not file_path.exists():
            return f"Error: Image not found: {file_path}"
        if not file_path.is_file():
            return f"Error: Not a file: {file_path}"

        ext = file_path.suffix.lower()
        if ext not in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
            return f"Error: Unsupported format: {ext}"

        if file_path.stat().st_size > 20_000_000:
            return "Error: Image too large (>20MB)"

        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(ext, "image/png")

        log(f"Reading image {file_path}")

        with open(file_path, "rb") as f:
            image_bytes = f.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        log(f"Image loaded ({len(image_bytes) // 1024}KB)")

        if question:
            prompt = f"This is an image from {file_path.name}. {question}"
        else:
            prompt = f"This is an image from {file_path.name}. Describe what you see."

        analysis = _vision_analyze(base64_image, mime_type, prompt)
        log("Vision analysis complete")

        return f"Analysis of {file_path.name}:\n\n{analysis}"

    except Exception as e:
        return f"Error analyzing image: {e}"


# ── Registry wrappers ──────────────────────────────────────────


@register_tool("web_browse", cacheable=True)
def _dispatch_web_browse(args):
    return tool_web_browse(args.get("url", ""))


@register_tool("web_search", cacheable=True)
def _dispatch_web_search(args):
    return tool_web_search(args.get("query", ""), args.get("num_results", 5))


@register_tool("web_view", cacheable=True)
def _dispatch_web_view(args):
    return tool_web_view(args.get("url", ""), args.get("question"))


@register_tool("view_image", cacheable=True)
def _dispatch_view_image(args):
    return tool_view_image(args.get("path", ""), args.get("question"))


def tool_web_image_search(query: str, num_results: int = 5) -> str:
    """Search the web for images using DuckDuckGo."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "Error: ddgs not installed"

    try:
        num_results = min(max(1, num_results), 10)
        log(f"Image search: {query}")

        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=num_results))

        if not results:
            return f"No image results found for: {query}"

        # Prefer HTTPS URLs
        output = f"Image search results for: {query}\n\n"
        for i, r in enumerate(results, 1):
            img_url = r.get("image", "")
            title = r.get("title", "Image")
            source = r.get("source", "")
            if img_url.startswith("http://"):
                https_url = "https://" + img_url[7:]
                img_url = https_url
            output += f"{i}. {title}\n"
            output += f"   ![{title}]({img_url})\n"
            if source:
                output += f"   Source: {source}\n"
            output += "\n"

        log(f"Found {len(results)} image results")
        return output

    except Exception as e:
        return f"Error searching images: {e}"


@register_tool("web_image_search", cacheable=True)
def _dispatch_web_image_search(args):
    return tool_web_image_search(args.get("query", ""), args.get("num_results", 5))
