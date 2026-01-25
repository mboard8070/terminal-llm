#!/usr/bin/env python3
"""
MAUDE Telegram Service - Full AI Integration.

Runs the Telegram bot with full MAUDE AI processing and proactive scheduling.
Designed to run as a systemd service.

Usage:
    python maude_telegram_service.py

Set TELEGRAM_BOT_TOKEN in .env or environment.
"""

import asyncio
import os
import sys
import signal
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Ensure unbuffered output for logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Load environment before other imports
load_dotenv()

from openai import OpenAI
from channels import get_gateway, IncomingMessage, OutgoingMessage
from channels.telegram import create_telegram_channel
from scheduler import get_scheduler
from skills import get_skill_manager
from memory import MaudeMemory

# Configuration
LOCAL_URL = os.environ.get("LLM_SERVER_URL", "http://localhost:30000/v1")
MODEL = "nemotron"
NUM_CTX = int(os.environ.get("MAUDE_NUM_CTX", "32768"))
SERVICE_NAME = "MAUDE Telegram Service"

# Global state
gateway = None
scheduler = None
memory = None
client = None
shutdown_event = None


def log(msg: str):
    """Log with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def create_llm_client():
    """Create connection to local LLM."""
    return OpenAI(base_url=LOCAL_URL, api_key="not-needed")


def get_system_prompt(user_query: str = "") -> str:
    """Build system prompt with memory context."""
    base_prompt = """You are MAUDE, an AI assistant responding via Telegram.
Be concise but helpful. You're chatting with a user on their phone.

You have access to:
- Memory of past conversations
- Skills for weather, stocks, notes, calculations, etc.
- Web search and browsing capabilities

Keep responses mobile-friendly (shorter than terminal mode).
Use markdown sparingly - Telegram supports basic formatting.

Important: You're in messaging mode, not terminal mode.
Don't reference file operations unless asked."""

    # Add memory context if available
    if memory and user_query:
        memory_context = memory.get_context_for_prompt(user_query)
        if memory_context:
            base_prompt += "\n\n" + memory_context

    return base_prompt


def get_tools():
    """Get available tools for the LLM."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information",
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
                "name": "remember",
                "description": "Save important information to memory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Memory key/name"},
                        "value": {"type": "string", "description": "Information to remember"}
                    },
                    "required": ["key", "value"]
                }
            }
        }
    ]

    # Add skill tools
    skill_manager = get_skill_manager()
    tools.extend(skill_manager.get_tool_definitions())

    return tools


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool call."""
    log(f"Tool: {name}")
    log(f"  -> Args: {arguments}")

    if name == "web_search":
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(arguments.get("query", ""), max_results=5))
            if not results:
                return "No results found."
            output = ""
            for r in results[:5]:
                output += f"- {r.get('title', '')}: {r.get('body', '')}\n"
            return output
        except Exception as e:
            return f"Search error: {e}"

    elif name == "remember":
        if memory:
            key = arguments.get("key", "")
            value = arguments.get("value", "")
            memory.store(key, value, category="user")
            return f"Remembered: {key}"
        return "Memory not available"

    elif name.startswith("skill_"):
        skill_name = name[6:]
        result = get_skill_manager().execute_skill(skill_name, **arguments)
        log(f"  -> Skill result ({len(result) if result else 0} chars): {result[:100] if result else 'None'}...")
        return result

    return f"Unknown tool: {name}"


async def process_message(msg: IncomingMessage) -> str:
    """Process a message through MAUDE AI."""
    global client, memory

    if not client:
        return "MAUDE is starting up, please try again in a moment."

    log(f"Processing: {msg.text[:50]}... from {msg.username}")

    # Build conversation
    messages = [
        {"role": "system", "content": get_system_prompt(msg.text)},
        {"role": "user", "content": msg.text}
    ]

    try:
        # Get available tools
        tools = get_tools()

        # Call LLM with tools
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
            extra_body={"num_ctx": NUM_CTX}
        )

        assistant_msg = response.choices[0].message

        # Handle tool calls
        if assistant_msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_msg.tool_calls
                ]
            })

            # Execute tools
            for tc in assistant_msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except:
                    args = {}
                result = execute_tool(tc.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })

            # Get final response
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                extra_body={"num_ctx": NUM_CTX}
            )
            result = response.choices[0].message.content

        else:
            result = assistant_msg.content

        # Save to memory
        if memory:
            session_id = f"telegram_{msg.user_id}"
            memory.save_message(session_id, "user", msg.text)
            memory.save_message(session_id, "assistant", result or "")

        log(f"Response: {len(result or '')} chars")
        log(f"Response text: {result[:200] if result else 'None'}...")
        return result or "I couldn't generate a response."

    except Exception as e:
        log(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return f"Sorry, I encountered an error: {str(e)[:100]}"


async def process_scheduled_prompt(prompt: str) -> str:
    """Process a scheduled task prompt."""
    # Create a fake incoming message for consistency
    msg = IncomingMessage(
        channel="scheduler",
        channel_id="system",
        user_id="scheduler",
        username="Scheduler",
        text=prompt,
        timestamp=datetime.now().isoformat()
    )
    return await process_message(msg)


async def main():
    """Main service loop."""
    global gateway, scheduler, memory, client, shutdown_event

    log(f"Starting {SERVICE_NAME}")

    # Check for token
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        log("ERROR: TELEGRAM_BOT_TOKEN not set!")
        sys.exit(1)

    # Initialize components
    log("Initializing memory...")
    memory = MaudeMemory()

    log("Initializing LLM client...")
    try:
        client = create_llm_client()
        # Quick test
        client.models.list()
        log("LLM connection OK")
    except Exception as e:
        log(f"WARNING: LLM not available: {e}")
        log("Bot will respond with 'AI unavailable' until LLM is running")
        client = None

    # Initialize channels
    log("Initializing Telegram channel...")
    gateway = get_gateway()
    gateway.set_maude_callback(process_message)

    telegram = create_telegram_channel(token)
    if not telegram:
        log("ERROR: Failed to create Telegram channel")
        sys.exit(1)

    gateway.register(telegram)

    # Connect to Telegram
    try:
        await telegram.connect()
        log("Telegram connected")
    except Exception as e:
        log(f"ERROR: Telegram connection failed: {e}")
        sys.exit(1)

    # Show paired users
    if gateway.authorized:
        log(f"Paired users: {len(gateway.authorized)}")
        for key, auth in gateway.authorized.items():
            log(f"  - {auth.username} ({auth.channel})")
    else:
        code = gateway.generate_pairing_code()
        log(f"No paired users. Pairing code: {code}")

    # Initialize scheduler
    log("Initializing scheduler...")
    scheduler = get_scheduler()
    scheduler.set_maude_callback(process_scheduled_prompt)
    scheduler.set_gateway(gateway)
    await scheduler.start()

    task_count = len([t for t in scheduler.tasks.values() if t.enabled])
    log(f"Scheduler started with {task_count} tasks")

    # Setup graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        log(f"Received signal {sig}, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    log("=" * 50)
    log("MAUDE Telegram Service running")
    log("=" * 50)

    # Main loop
    try:
        await shutdown_event.wait()
    except asyncio.CancelledError:
        pass

    # Cleanup
    log("Shutting down...")
    await scheduler.stop()
    await telegram.disconnect()
    log("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
