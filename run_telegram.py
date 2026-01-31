#!/usr/bin/env python3
"""
MAUDE Telegram bot.
Uses shared maude_core for full tool access.
"""

import asyncio
import json
import sys
from dotenv import load_dotenv

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

load_dotenv()

import os
from openai import OpenAI
from channels import get_gateway, IncomingMessage
from channels.telegram import create_telegram_channel

# Use shared MAUDE core
import maude_core
from maude_core import TOOLS, execute_tool, reset_rate_limits, append_chat_log, read_chat_log_since

# Get config from maude_core
LOCAL_URL = maude_core.LOCAL_URL
MODEL = maude_core.MODEL

# Set up logging callback
def telegram_log(message: str):
    print(f">>> {message}", flush=True)

maude_core.set_log_callback(telegram_log)

# System prompt - now with full tool access
SYSTEM_PROMPT = f"""You are MAUDE, a local AI assistant running {MODEL}.

Be brief and direct. Keep responses concise for mobile/Telegram.

TOOLS AVAILABLE:
- web_search: Search the web (use first for current info)
- web_browse: Fetch URL content
- read_file, write_file, edit_file: File operations
- search_file, search_directory: Search in files
- list_directory, change_directory: Navigate filesystem
- run_command: Execute shell commands
- view_image, web_view: Vision analysis (LLaVA)
- ask_frontier: Escalate to cloud AI if needed

STRATEGY:
1. For current info (weather, news): use web_search first
2. For file tasks: use file tools
3. Keep responses concise for mobile"""


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

        reset_rate_limits()  # Reset per-turn limits

        # Allow up to 5 tool iterations
        for _ in range(5):
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
                # Log for sync (after processing complete)
                append_chat_log("telegram", "user", msg.text)
                append_chat_log("telegram", "assistant", reply)
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
                print(f">>> [{tc.function.name}]", flush=True)
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
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


async def sync_cli_messages(gateway):
    """Watch for CLI messages and push to Telegram."""
    from channels import OutgoingMessage
    import os

    # Start from end of file
    log_path = os.path.expanduser("~/.config/maude/chat_sync.jsonl")
    try:
        position = os.path.getsize(log_path) if os.path.exists(log_path) else 0
    except:
        position = 0

    while True:
        await asyncio.sleep(2)
        try:
            entries, new_position = read_chat_log_since(position)
            for entry in entries:
                if entry.get("channel") == "cli":
                    role = entry.get("role", "")
                    content = entry.get("content", "")
                    if role == "user":
                        text = f"[cli] YOU: {content}"
                    elif role == "assistant":
                        text = f"[cli] MAUDE: {content}"
                    else:
                        continue
                    await gateway.broadcast(OutgoingMessage(text=text))
            position = new_position
        except Exception:
            pass


async def main(standalone: bool = True):
    """Run Telegram bot. If standalone=False, runs quietly as background service."""
    if standalone:
        print("=" * 50, flush=True)
        print("MAUDE Telegram Bot", flush=True)
        print("=" * 50, flush=True)

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        if standalone:
            print("ERROR: TELEGRAM_BOT_TOKEN not set!", flush=True)
            sys.exit(1)
        return

    if standalone:
        print(f"Model: {MODEL}", flush=True)
        print(f"Server: {LOCAL_URL}", flush=True)

        try:
            client = get_client()
            client.models.list()
            print("LLM: Connected", flush=True)
        except Exception as e:
            print(f"LLM: Not available ({e})", flush=True)

        tool_names = [t["function"]["name"] for t in TOOLS]
        print(f"Tools: {', '.join(tool_names[:5])}... ({len(tool_names)} total)", flush=True)

    gateway = get_gateway()
    gateway.set_maude_callback(maude_callback)

    telegram = create_telegram_channel(token)
    if not telegram:
        if standalone:
            print("ERROR: Failed to create Telegram channel", flush=True)
            sys.exit(1)
        return

    gateway.register(telegram)

    try:
        await telegram.connect()
    except Exception as e:
        if standalone:
            print(f"ERROR: Failed to connect: {e}", flush=True)
            sys.exit(1)
        return

    if standalone:
        if gateway.authorized:
            print("\nPaired users:", flush=True)
            for key, auth in gateway.authorized.items():
                print(f"  - {auth.username} ({auth.channel})", flush=True)
        else:
            code = gateway.generate_pairing_code()
            print(f"\nPairing Code: {code}", flush=True)

        print("\nReady.\n", flush=True)

    # Start sync task for CLI messages
    sync_task = asyncio.create_task(sync_cli_messages(gateway))

    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        sync_task.cancel()
        await telegram.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
