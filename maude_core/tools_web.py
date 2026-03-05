"""
Web and vision tool implementations — 4 tools (all cacheable).
"""

from openai import OpenAI

from tool_registry import register_tool
from .log import log
from .paths import resolve_path
from .config import VISION_URL, VISION_MODEL


def tool_web_browse(url: str) -> str:
    """Fetch and parse web page content."""
    import requests
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return "Error: beautifulsoup4 not installed"

    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        log(f"Fetching {url}")

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'header', 'noscript', 'iframe']):
            tag.decompose()

        main_content = soup.find('main') or soup.find('article') or soup.find('div', {'class': ['content', 'post', 'article', 'main']})
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)

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
    """Screenshot a webpage and analyze it with LLaVA."""
    import base64

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "Error: playwright not installed"

    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        log(f"Capturing screenshot of {url}")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1024, 'height': 768})
            page.goto(url, wait_until='networkidle', timeout=30000)
            page.wait_for_timeout(1000)
            screenshot_bytes = page.screenshot(full_page=False)
            browser.close()

        base64_image = base64.b64encode(screenshot_bytes).decode('utf-8')
        log(f"Screenshot captured ({len(screenshot_bytes) // 1024}KB)")

        if question:
            prompt = f"This is a screenshot of {url}. {question}"
        else:
            prompt = f"This is a screenshot of {url}. Describe what you see."

        log("Analyzing with LLaVA...")

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
        log("Vision analysis complete")

        result = f"Visual analysis of {url}:\n\n{analysis}"
        return result

    except Exception as e:
        error_msg = str(e)
        if "playwright" in error_msg.lower():
            return f"Error: Playwright issue. Run: playwright install chromium\n{e}"
        elif "connect" in error_msg.lower():
            return f"Error: Cannot connect to vision model at {VISION_URL}\n{e}"
        return f"Error viewing webpage: {e}"


def tool_view_image(path: str, question: str = None) -> str:
    """Analyze a local image file with LLaVA."""
    import base64

    try:
        file_path = resolve_path(path)
        if not file_path.exists():
            return f"Error: Image not found: {file_path}"
        if not file_path.is_file():
            return f"Error: Not a file: {file_path}"

        ext = file_path.suffix.lower()
        if ext not in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
            return f"Error: Unsupported format: {ext}"

        if file_path.stat().st_size > 20_000_000:
            return f"Error: Image too large (>20MB)"

        mime_types = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                      '.gif': 'image/gif', '.webp': 'image/webp'}
        mime_type = mime_types.get(ext, 'image/png')

        log(f"Reading image {file_path}")

        with open(file_path, 'rb') as f:
            image_bytes = f.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        log(f"Image loaded ({len(image_bytes) // 1024}KB)")

        if question:
            prompt = f"This is an image from {file_path.name}. {question}"
        else:
            prompt = f"This is an image from {file_path.name}. Describe what you see."

        log("Analyzing with LLaVA...")

        vision_client = OpenAI(base_url=VISION_URL, api_key="not-needed")
        response = vision_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                ]
            }],
            max_tokens=1024,
            temperature=0.2
        )

        analysis = response.choices[0].message.content
        log("Vision analysis complete")

        return f"Analysis of {file_path.name}:\n\n{analysis}"

    except Exception as e:
        if "connect" in str(e).lower():
            return f"Error: Cannot connect to vision model at {VISION_URL}"
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
