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
from maude_core import TOOLS, execute_tool, reset_rate_limits, append_chat_log, read_chat_log_since, get_tools_for_message, fast_dispatch
from maude_core.tools_plan import PARALLEL_SAFE

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
- Web: web_search, web_browse, web_view, view_image
- Files: read_file, write_file, edit_file, search_file, search_directory, list_directory, change_directory, run_command
- Cloud AI: ask_frontier (escalate to Claude/Gemini)
- Gmail: gmail_list, gmail_read, gmail_send
- Google Drive: drive_list, drive_search, drive_read, drive_upload, drive_create_doc, drive_create_sheet, drive_update_doc, drive_delete
- Sheets: sheets_read, sheets_write, sheets_append, sheets_create, sheets_list_sheets, sheets_clear
- Calendar: calendar_list_events, calendar_create_event, calendar_update_event, calendar_delete_event, calendar_search_events
- Slides: slides_get_presentation, slides_get_slide, slides_create_presentation, slides_add_slide, slides_add_text
- Contacts: contacts_list, contacts_get, contacts_create, contacts_update, contacts_delete, contacts_search
- YouTube: youtube_search, youtube_get_video, youtube_get_channel, youtube_list_playlists, youtube_create_playlist, youtube_get_comments, youtube_post_comment, youtube_my_channel
- Substack: substack_create_draft, substack_list_drafts, substack_list_posts, substack_get_post, substack_update_draft, substack_delete_draft, substack_get_stats
- Browser: browser_open, browser_click, browser_type, browser_navigate, browser_screenshot, browser_extract, browser_fill_form, browser_select, browser_close
- Social media: skill_post_social (Twitter/X, LinkedIn, Bluesky), skill_social_status
- Scheduling: schedule_task
- Planning: execute_plan — run a multi-stage tool plan in one call. Stages run sequentially, tools within a stage run in parallel. Use $N.M to reference earlier results.

EFFICIENCY: When you need multiple pieces of information, call ALL tools in one response. Batch reads, searches, etc. Only chain sequentially when a later call depends on an earlier result.

STRATEGY:
1. For current info: use web_search first
2. For file tasks: use file tools
3. For Google/YouTube/Substack: use the appropriate tools
4. Keep responses concise for mobile"""


def _escalate_to_frontier(user_question: str) -> str:
    """Escalate a question to Claude/Gemini when the local model can't handle it."""
    try:
        from frontier import ask_frontier, list_available_providers, RateLimitError

        available = list_available_providers()
        if not available:
            return "I wasn't able to handle that locally and no cloud models are configured."

        priority = ["claude", "gemini", "openai", "grok", "mistral"]
        errors = []
        for provider in priority:
            if provider not in available:
                continue
            try:
                print(f">>> Escalating to {provider}...", flush=True)
                response = ask_frontier(
                    query=user_question,
                    provider_name=provider,
                    system_prompt="You are MAUDE, a capable AI assistant. Answer the user's question directly and helpfully. Be concise — this is for Telegram."
                )
                print(f">>> ({response.provider} — {response.input_tokens}+{response.output_tokens} tokens, ${response.cost_usd:.4f})", flush=True)
                return response.content
            except RateLimitError:
                errors.append(f"{provider}: rate limited")
                continue
            except Exception as e:
                errors.append(f"{provider}: {e}")
                continue

        return f"I tried escalating to cloud models but they're all unavailable ({', '.join(errors)})."

    except ImportError:
        return "I wasn't able to handle that request. Could you give me more detail?"


def get_client():
    return OpenAI(base_url=LOCAL_URL, api_key="not-needed")


async def maude_callback(msg: IncomingMessage) -> str:
    """Handle incoming Telegram message."""
    print(f">>> {msg.username}: {msg.text}", flush=True)

    try:
        # Fast path: direct tool dispatch without LLM reasoning
        try:
            result = fast_dispatch(msg.text)
            if result:
                tool_name, args, tool_result = result
                print(f">>> → {tool_name}", flush=True)
                client = get_client()
                summary_messages = [
                    {"role": "system", "content": "You are MAUDE. A tool was already called. Summarize the result concisely for mobile/Telegram."},
                    {"role": "user", "content": msg.text},
                    {"role": "user", "content": f"Tool result:\n{tool_result[:2000]}\n\nSummarize briefly."}
                ]
                try:
                    summary = client.chat.completions.create(
                        model=MODEL, messages=summary_messages,
                        temperature=0.2, max_tokens=512
                    )
                    reply = summary.choices[0].message.content or tool_result[:1000]
                except Exception:
                    reply = tool_result[:1000]
                append_chat_log("telegram", "user", msg.text)
                append_chat_log("telegram", "assistant", reply)
                return reply
        except Exception:
            pass

        # Auto-route: check if a subagent should handle this directly
        try:
            from auto_router import route_message
            decision = route_message(msg.text)
            if decision.subagent and decision.confidence >= 0.5:
                print(f">>> routing → {decision.subagent} ({decision.intent}, {decision.confidence:.0%})", flush=True)
                from execution import execute_subagent
                reply = execute_subagent(
                    decision.subagent,
                    msg.text,
                    prefer_cloud=decision.prefer_cloud
                )
                if reply and not reply.startswith("Error:"):
                    append_chat_log("telegram", "user", msg.text)
                    append_chat_log("telegram", "assistant", reply)
                    return reply
        except ImportError:
            pass
        except Exception:
            pass  # Fall through to main loop

        client = get_client()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": msg.text}
        ]

        reset_rate_limits()  # Reset per-turn limits
        active_tools = get_tools_for_message(msg.text)

        # Allow up to 5 tool iterations
        for _ in range(5):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=active_tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1024
            )

            assistant_msg = response.choices[0].message

            # No tool calls - return response
            if not assistant_msg.tool_calls:
                reply = assistant_msg.content or ""
                if not reply.strip():
                    reply = _escalate_to_frontier(msg.text)
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

            # Parse tool calls
            parsed_tcs = []
            for tc in assistant_msg.tool_calls:
                print(f">>> [{tc.function.name}]", flush=True)
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                parsed_tcs.append((tc.id, tc.function.name, args))

            # Split into parallel-safe and sequential
            par = [(tid, fn, a) for tid, fn, a in parsed_tcs if fn in PARALLEL_SAFE]
            seq = [(tid, fn, a) for tid, fn, a in parsed_tcs if fn not in PARALLEL_SAFE]

            # Run parallel-safe tools concurrently
            if len(par) > 1:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                par_results = {}
                with ThreadPoolExecutor(max_workers=min(len(par), 6)) as pool:
                    futures = {
                        pool.submit(execute_tool, fn, a): tid
                        for tid, fn, a in par
                    }
                    for future in as_completed(futures):
                        par_results[futures[future]] = future.result()
                for tid, fn, a in par:
                    messages.append({"role": "tool", "tool_call_id": tid, "content": par_results[tid]})
            elif len(par) == 1:
                tid, fn, a = par[0]
                messages.append({"role": "tool", "tool_call_id": tid, "content": execute_tool(fn, a)})

            # Run mutating tools sequentially
            for tid, fn, a in seq:
                result = execute_tool(fn, a)
                messages.append({"role": "tool", "tool_call_id": tid, "content": result})

        # Local model exhausted tool iterations — escalate to frontier
        return _escalate_to_frontier(msg.text)

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

    # Start proactive heartbeat
    heartbeat_task = None
    try:
        from heartbeat import get_heartbeat
        hb = get_heartbeat()
        if hb.enabled:
            hb.set_tool_executor(execute_tool)
            hb.set_gateway(gateway)
            heartbeat_task = asyncio.create_task(hb.start())
            if standalone:
                print("Heartbeat: enabled", flush=True)
    except ImportError:
        pass
    except Exception:
        pass

    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        if heartbeat_task:
            heartbeat_task.cancel()
        sync_task.cancel()
        await telegram.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
