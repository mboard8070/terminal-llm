#!/usr/bin/env python3
"""
MAUDE Telegram bot.
Uses the same model as chat_local.py with web search capability.
"""

import asyncio
import json
import os
import sys
from dotenv import load_dotenv

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

load_dotenv()

from openai import OpenAI
from channels import get_gateway, IncomingMessage
from channels.telegram import create_telegram_channel

# Use same model config as chat_local.py
LOCAL_URL = os.environ.get("LLM_SERVER_URL", "http://localhost:30000/v1")
MODEL = os.environ.get("MAUDE_MODEL", "nemotron")

# System prompt
SYSTEM_PROMPT = f"""You are MAUDE, a local AI assistant running {MODEL}.

Be brief and direct. Keep responses concise for mobile/Telegram.

TOOLS:
- web_search: Search the web. Results include snippets - often enough to answer without browsing.
- web_browse: Fetch a URL. Use sparingly - many sites have messy HTML.

STRATEGY:
1. Start with web_search - the snippets often contain the answer
2. If snippets are sufficient, respond directly without browsing
3. Only use web_browse for specific pages with clean content
4. If web_browse returns junk, use search snippets instead
5. For weather: prefer wttr.in or search snippets over weather.com"""

# Tools - just web search and browse
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo. Use for weather, news, prices, current info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_browse",
            "description": "Fetch and read content from a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                },
                "required": ["url"]
            }
        }
    }
]


def tool_web_search(query: str) -> str:
    """Search the web using DuckDuckGo."""
    try:
        from ddgs import DDGS
        print(f">>> Searching: {query}", flush=True)
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return f"No results for: {query}"
        output = f"Search results for: {query}\n\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. {r.get('title', 'No title')}\n"
            output += f"   {r.get('body', '')[:150]}\n\n"
        return output
    except Exception as e:
        return f"Search error: {e}"


def tool_web_browse(url: str) -> str:
    """Fetch and parse web page content."""
    try:
        import requests
        from bs4 import BeautifulSoup

        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        print(f">>> Fetching: {url}", flush=True)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove junk elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'header', 'noscript', 'iframe', 'form', 'button']):
            tag.decompose()

        # Try to find main content
        main = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        if main:
            text = main.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)

        # Clean up whitespace
        text = ' '.join(text.split())

        # Limit length for model context
        if len(text) > 1500:
            text = text[:1500] + "..."

        return f"Content from {url}:\n{text}" if text else f"No readable content from {url}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


def execute_tool(name: str, args: dict) -> str:
    """Execute a tool."""
    if name == "web_search":
        return tool_web_search(args.get("query", ""))
    elif name == "web_browse":
        return tool_web_browse(args.get("url", ""))
    return f"Unknown tool: {name}"


def get_client():
    return OpenAI(base_url=LOCAL_URL, api_key="not-needed")


async def maude_callback(msg: IncomingMessage) -> str:
    """Handle incoming Telegram message."""
    print(f">>> {msg.username}: {msg.text}", flush=True)

    try:
        client = get_client()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": msg.text}
        ]

        # Allow up to 3 tool iterations
        for _ in range(3):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1024
            )

            assistant_msg = response.choices[0].message

            # No tool calls - return response
            if not assistant_msg.tool_calls:
                reply = assistant_msg.content or ""
                if not reply.strip():
                    reply = "I found some information but couldn't summarize it. Try rephrasing your question."
                print(f">>> MAUDE: {reply[:100]}...", flush=True)
                return reply

            # Handle tool calls
            messages.append({
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in assistant_msg.tool_calls
                ]
            })

            for tc in assistant_msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = execute_tool(tc.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })

        return "Still processing... try again in a moment."

    except Exception as e:
        print(f">>> Error: {e}", flush=True)
        return f"Error: {str(e)[:100]}"


async def main():
    print("=" * 50, flush=True)
    print("MAUDE Telegram Bot", flush=True)
    print("=" * 50, flush=True)

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("ERROR: TELEGRAM_BOT_TOKEN not set!", flush=True)
        sys.exit(1)

    print(f"Model: {MODEL}", flush=True)
    print(f"Server: {LOCAL_URL}", flush=True)

    try:
        client = get_client()
        client.models.list()
        print("LLM: Connected", flush=True)
    except Exception as e:
        print(f"LLM: Not available ({e})", flush=True)

    print("Tools: web_search, web_browse", flush=True)

    gateway = get_gateway()
    gateway.set_maude_callback(maude_callback)

    telegram = create_telegram_channel(token)
    if not telegram:
        print("ERROR: Failed to create Telegram channel", flush=True)
        sys.exit(1)

    gateway.register(telegram)

    try:
        await telegram.connect()
    except Exception as e:
        print(f"ERROR: Failed to connect: {e}", flush=True)
        sys.exit(1)

    if gateway.authorized:
        print("\nPaired users:", flush=True)
        for key, auth in gateway.authorized.items():
            print(f"  - {auth.username} ({auth.channel})", flush=True)
    else:
        code = gateway.generate_pairing_code()
        print(f"\nPairing Code: {code}", flush=True)

    print("\nReady.\n", flush=True)

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)
        await telegram.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
