#!/usr/bin/env python3
"""
MAUDE - Terminal LLM Chat powered by local LLMs
"""

import sys
import os
import asyncio
import threading
from dotenv import load_dotenv
load_dotenv()

import time
import json
from pathlib import Path
from typing import Optional
from openai import OpenAI

# Textual TUI framework
from textual.app import App, ComposeResult
from textual.widgets import Input, RichLog, Static, Footer
from textual.containers import Container, Vertical
from textual.binding import Binding
from textual import work
from rich.text import Text
from rich.panel import Panel
import pyfiglet

# MAUDE core - shared tools
import maude_core
from maude_core import TOOLS, execute_tool, reset_rate_limits, append_chat_log, read_chat_log_since, get_tools_for_message, fast_dispatch
import conversation_sync
from collab import get_hub as get_collab_hub

# Voice mode
from voice import VoiceMode, VoiceConfig, check_voice_dependencies

# Minimal imports
from keys import KeyManager

# Global reference to app for output
_app = None

# Track last MAUDE response for /copy command
_last_response = ""


class TUIConsole:
    """Console that writes to Textual RichLog when app is running, otherwise stdout."""
    def __init__(self):
        self.width = 80

    def print(self, *args, end="\n", **kwargs):
        text = " ".join(str(a) for a in args)
        if _app and hasattr(_app, 'output_log'):
            try:
                _app.call_from_thread(_app.write_output, text)
            except:
                print(text, end=end)
        else:
            print(text, end=end)

    def input(self, prompt=""):
        return input(prompt)

    def clear(self):
        if not _app:
            os.system('clear' if os.name != 'nt' else 'cls')


console = TUIConsole()


class VoiceController:
    """Manages voice mode state and lifecycle for TUI integration."""

    def __init__(self, app: 'MaudeApp'):
        self.app = app
        self.voice_mode: Optional[VoiceMode] = None
        self._active = False

    def create_maude_callback(self):
        """Create a synchronous callback that wraps chat() for voice mode."""
        def callback(text: str) -> str:
            # Add user message to shared history
            self.app.messages.append({"role": "user", "content": text})

            # Call chat() - this handles all tool execution
            response = chat(self.app.client, self.app.messages)

            if response:
                # Add to shared history
                self.app.messages.append({"role": "assistant", "content": response})
                # Log for sync
                append_chat_log("voice", "user", text)
                append_chat_log("voice", "assistant", response)
                return response
            return "I encountered an error processing your request."

        return callback

    async def initialize(self):
        """Initialize voice mode components."""
        if self.voice_mode is None:
            self.voice_mode = VoiceMode()

            # Set UI callbacks for TUI integration
            self.voice_mode.set_ui_callbacks(
                on_status=lambda s: self.app.call_from_thread(self.app.update_voice_status, s),
                on_transcription=lambda t: self.app.call_from_thread(
                    self.app.write_output, f"\n[bold green]YOU (voice) >[/bold green] {t}"
                ),
                on_response=lambda r: self.app.call_from_thread(
                    self.app.write_output, f"[bold magenta]MAUDE:[/bold magenta] {r}"
                )
            )

            await self.voice_mode.initialize()
        self._active = True

    async def run_single(self):
        """Run a single voice listen/respond cycle."""
        if not self._active:
            await self.initialize()

        self.app.call_from_thread(self.app.update_voice_status, "Listening...")
        text = await self.voice_mode.listen()

        if text:
            self.app.call_from_thread(
                self.app.write_output, f"\n[bold green]YOU (voice) >[/bold green] {text}"
            )
            self.app.call_from_thread(self.app.update_voice_status, "Processing...")

            callback = self.create_maude_callback()
            response = callback(text)

            self.app.call_from_thread(self.app.update_voice_status, "Speaking...")
            await self.voice_mode.speak(response)

        self.app.call_from_thread(self.app.update_voice_status, "")

    async def run_talk_mode(self):
        """Run continuous voice conversation mode."""
        if not self._active:
            await self.initialize()

        callback = self.create_maude_callback()
        await self.voice_mode.talk_mode(callback)
        self.app.call_from_thread(self.app.update_voice_status, "")

    def stop(self):
        """Stop voice mode."""
        if self.voice_mode:
            self.voice_mode.stop_talk_mode()
        self._active = False
        if self.app:
            self.app.call_from_thread(self.app.update_voice_status, "")


# Set up logging callback for maude_core to use TUI console
def tui_log(message: str):
    console.print(f"[dim cyan]  -> {message}[/dim cyan]")

maude_core.set_log_callback(tui_log)

# Get config from maude_core
LOCAL_URL = maude_core.LOCAL_URL
MODEL = maude_core.MODEL
NUM_CTX = maude_core.NUM_CTX

# Color palette for animation - fire gradient
COLORS = ["red", "bright_red", "orange1", "orange3", "yellow", "bright_yellow"]

def create_client():
    """Create local API client."""
    return OpenAI(
        base_url=LOCAL_URL,
        api_key="not-needed"
    )


def fire_text(text: str, offset: int = 0) -> Text:
    """Create fire-colored text (red -> orange -> yellow gradient)."""
    result = Text()
    for i, char in enumerate(text):
        if char.strip():
            color = COLORS[(i + offset) % len(COLORS)]
            result.append(char, style=f"bold {color}")
        else:
            result.append(char)
    return result


def get_banner_with_mech():
    """Generate banner text."""
    banner = pyfiglet.figlet_format("MAUDE", font="banner3")
    banner_lines = banner.rstrip('\n').split('\n')
    return banner_lines


def animate_banner():
    """Display animated MAUDE banner."""
    banner_lines = get_banner_with_mech()

    # Animate the banner with fire colors
    with Live(console=console, refresh_per_second=12, transient=True) as live:
        for frame in range(25):
            content = Text()

            # Add animated MAUDE banner
            for line in banner_lines:
                content.append_text(fire_text(line, frame))
                content.append('\n')

            live.update(Align.center(content))
            time.sleep(0.06)

    # Final static banner with fire gradient
    final_content = Text()
    for line in banner_lines:
        final_content.append_text(fire_text(line, 24))
        final_content.append('\n')

    console.print(Align.center(final_content))
    console.print()


def print_separator():
    """Print a styled separator."""
    console.print("─" * console.width, style="dim cyan")


def _escalate_to_frontier(user_question: str) -> str:
    """Escalate a question to Claude/Gemini when the local model gets stuck."""
    try:
        from frontier import ask_frontier, list_available_providers, RateLimitError

        available = list_available_providers()
        if not available:
            return "I wasn't able to handle that locally and no cloud models are configured. Could you give me more detail so I can try a different approach?"

        # Try providers in priority order: Claude first, then Gemini, then others
        priority = ["claude", "gemini", "openai", "grok", "mistral"]
        errors = []
        for provider in priority:
            if provider not in available:
                continue
            try:
                console.print(f"[dim cyan]  Asking {provider}...[/dim cyan]")
                response = ask_frontier(
                    query=user_question,
                    provider_name=provider,
                    system_prompt="You are MAUDE, a capable AI assistant. Answer the user's question directly and helpfully. Be concise."
                )
                console.print(f"[dim cyan]  ({response.provider} — {response.input_tokens}+{response.output_tokens} tokens, ${response.cost_usd:.4f})[/dim cyan]")
                return response.content
            except RateLimitError as e:
                errors.append(f"{provider}: rate limited")
                continue
            except Exception as e:
                errors.append(f"{provider}: {e}")
                continue

        return f"I tried escalating to cloud models but they're all unavailable right now ({', '.join(errors)}). Could you try again in a moment?"

    except ImportError:
        return "I wasn't able to handle that locally. Could you give me more detail so I can try a different approach?"


def chat(client, messages: list):
    """Send chat request to local LLM with tool support."""
    global show_thinking

    max_tool_iterations = 15
    tool_iteration = 0
    recent_tool_calls = []
    consecutive_duplicates = 0
    reset_rate_limits()  # Reset per-turn limits in maude_core

    # Cloud models: gateway handles tools, just send a plain request
    if MODEL in _CLOUD_MODELS:
        try:
            start_time = time.time()
            # Strip any tool_calls/tool messages from history (local-only)
            clean_msgs = [m for m in messages if "tool_calls" not in m and m.get("role") != "tool"]
            # Stream for TUI typewriter, non-stream for headless
            use_stream = _app and hasattr(_app, 'stream_token')
            response = client.chat.completions.create(
                model=MODEL,
                messages=clean_msgs,
                temperature=0.2,
                max_tokens=4096,
                timeout=300,
                stream=use_stream,
            )
            if use_stream:
                full_content = ""
                token_count = 0
                prompt_tokens = 0
                first = True
                for chunk in response:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        _app.stream_token(delta.content, is_first=first, prefix="MAUDE: ")
                        full_content += delta.content
                        first = False
                    # Some providers send usage in the final chunk
                    if hasattr(chunk, 'usage') and chunk.usage:
                        token_count = chunk.usage.completion_tokens or 0
                        prompt_tokens = chunk.usage.prompt_tokens or 0
                elapsed_time = time.time() - start_time
                if full_content:
                    _app.call_from_thread(_app.write_output, f"[dim]{prompt_tokens}+{token_count} tokens in {elapsed_time:.1f}s[/dim]")
            else:
                elapsed_time = time.time() - start_time
                msg = response.choices[0].message
                full_content = msg.content or ""
                if full_content:
                    token_count = response.usage.completion_tokens if response.usage else 0
                    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                    console.print(f"[bold magenta]MAUDE:[/bold magenta] {full_content}")
                    console.print(f"[dim]{prompt_tokens}+{token_count} tokens in {elapsed_time:.1f}s[/dim]")
            return full_content
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return None

    # Get relevant tools based on user's latest message
    user_msg = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    active_tools = get_tools_for_message(user_msg)

    while True:
        tool_iteration += 1
        if tool_iteration > max_tool_iterations:
            console.print("[dim yellow](max tool iterations reached — escalating to frontier model...)[/dim yellow]")
            user_question = next(
                (m["content"] for m in reversed(messages) if m.get("role") == "user"),
                "Help me with my request"
            )
            return _escalate_to_frontier(user_question)
        try:
            start_time = time.time()

            # Reduce max_tokens after tool results to prevent reasoning loops
            has_tool_results = any(m.get("role") == "tool" for m in messages)
            max_tokens = 2048 if has_tool_results else 4096

            # Stream when TUI is available (typewriter effect), non-stream otherwise
            use_stream = _app and hasattr(_app, 'stream_token')
            kwargs = dict(
                model=MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens,
                tools=active_tools,
                tool_choice="auto",
                timeout=120,
                stream=use_stream,
            )
            # num_ctx is local-only (llama-server); skip for cloud models
            if MODEL not in _CLOUD_MODELS:
                kwargs["extra_body"] = {"num_ctx": NUM_CTX}
            response = client.chat.completions.create(**kwargs)

            if use_stream:
                # Streaming path — use existing stream_token for typewriter
                full_content = ""
                tool_calls_data = {}
                tool_calls_acc = {}  # index -> {id, name, arguments}
                first = True
                token_count = 0
                prompt_tokens = 0
                for chunk in response:
                    choice = chunk.choices[0] if chunk.choices else None
                    if not choice:
                        if hasattr(chunk, 'usage') and chunk.usage:
                            token_count = getattr(chunk.usage, 'completion_tokens', 0) or 0
                            prompt_tokens = getattr(chunk.usage, 'prompt_tokens', 0) or 0
                        continue
                    delta = choice.delta
                    if not delta:
                        continue
                    if delta.content:
                        _app.stream_token(delta.content, is_first=first, prefix="MAUDE: ")
                        full_content += delta.content
                        first = False
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            i = tc.index
                            if i not in tool_calls_acc:
                                tool_calls_acc[i] = {"id": "", "name": "", "arguments": ""}
                            if tc.id:
                                tool_calls_acc[i]["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    tool_calls_acc[i]["name"] += tc.function.name
                                if tc.function.arguments:
                                    tool_calls_acc[i]["arguments"] += tc.function.arguments
                elapsed_time = time.time() - start_time
                if full_content:
                    _app.call_from_thread(_app.write_output, f"[dim]{prompt_tokens}+{token_count} tokens in {elapsed_time:.1f}s[/dim]")
                # Build tool_calls_data from accumulated stream chunks
                for tc in tool_calls_acc.values():
                    if tc["id"]:
                        tool_calls_data[tc["id"]] = {"name": tc["name"], "arguments": tc["arguments"]}
            else:
                # Non-streaming fallback (headless / no TUI)
                elapsed_time = time.time() - start_time
                msg = response.choices[0].message
                full_content = msg.content or ""
                if full_content:
                    token_count = response.usage.completion_tokens if response.usage else 0
                    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                    console.print(f"[bold magenta]MAUDE:[/bold magenta] {full_content}")
                    console.print(f"[dim]{token_count} tokens in {elapsed_time:.1f}s[/dim]")
                tool_calls_data = {}
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls_data[tc.id] = {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }

            # Handle tool calls
            if tool_calls_data:
                # Add assistant message with tool calls to history
                messages.append({
                    "role": "assistant",
                    "content": full_content or "",
                    "tool_calls": [
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": tc_data["name"],
                                "arguments": tc_data["arguments"]
                            }
                        }
                        for tc_id, tc_data in tool_calls_data.items()
                    ]
                })

                # Execute each tool call
                for tc_id, tc_data in tool_calls_data.items():
                    func_name = tc_data["name"]
                    try:
                        func_args = json.loads(tc_data["arguments"])
                    except json.JSONDecodeError:
                        func_args = {}

                    # Check for duplicate tool calls (same tool + same args)
                    call_signature = (func_name, json.dumps(func_args, sort_keys=True))
                    if call_signature in recent_tool_calls:
                        consecutive_duplicates += 1
                        console.print(f"[dim yellow]  [{func_name}] (skipped duplicate)[/dim yellow]")
                        result = "(Already called with same arguments - see previous result. STOP retrying and respond to the user with the best answer you can based on information gathered so far.)"
                        # Break out if too many duplicates in a row
                        if consecutive_duplicates >= 3:
                            console.print("[dim yellow](escalating to frontier model...)[/dim yellow]")
                            # Extract the user's original question from messages
                            user_question = next(
                                (m["content"] for m in reversed(messages) if m.get("role") == "user"),
                                "Help me with my request"
                            )
                            return _escalate_to_frontier(user_question)
                    else:
                        consecutive_duplicates = 0  # Reset on successful new call
                        recent_tool_calls.append(call_signature)
                        # Args preview — show first key arg
                        args_preview = json.dumps(func_args, ensure_ascii=False)
                        if len(args_preview) > 60:
                            args_preview = args_preview[:60] + "..."
                        console.print(f"[dim yellow]  [{func_name}] {args_preview}[/dim yellow]")
                        tool_start = time.time()
                        result = execute_tool(func_name, func_args)
                        tool_elapsed = time.time() - tool_start
                        # Result preview
                        preview = (result or "")[:80].replace("\n", " ").strip()
                        if len(result or "") > 80:
                            preview += "..."
                        console.print(f"[dim]    → {preview} ({tool_elapsed:.1f}s)[/dim]")

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result
                    })

                # Continue loop to get next response
                continue

            # No tool calls - we're done
            return full_content

        except Exception as e:
            error_msg = str(e)
            # If tools aren't supported, retry without them
            if "tool" in error_msg.lower() or "function" in error_msg.lower():
                console.print("[dim yellow]Tools not supported, falling back to basic chat...[/dim yellow]")
                try:
                    fb_kwargs = dict(
                        model=MODEL,
                        messages=[m for m in messages if "tool_calls" not in m and m.get("role") != "tool"],
                        temperature=0.2,
                        max_tokens=1024,
                    )
                    if MODEL not in _CLOUD_MODELS:
                        fb_kwargs["extra_body"] = {"num_ctx": NUM_CTX}
                    response = client.chat.completions.create(**fb_kwargs)
                    msg = response.choices[0].message
                    if msg.content:
                        console.print(f"[bold magenta]MAUDE:[/bold magenta] {msg.content}")
                    return msg.content
                except Exception as e2:
                    console.print(f"[red]Error: {e2}[/red]")
                    return None
            else:
                console.print(f"[red]Error: {e}[/red]")
                console.print("[dim]Is the server running? Start with: ./start_server.sh[/dim]")
                return None


AVAILABLE_MODELS = {
    "nemotron": "nemotron",
    "llava": "llava",
    "mistral": "mistral-large-latest",
    "codestral": "codestral-latest",
    "devstral": "devstral-2512",
    "devstral-small": "devstral-small-latest",
    "devstral-medium": "devstral-medium-latest",
    "claude": "claude-opus-4-20250514",
    "sonnet": "claude-sonnet-4-20250514",
}

# Cloud models route through the gateway's HTTP mirror (same port as local)
GATEWAY_URL = "http://localhost:30080/v1"

_CLOUD_MODELS = {"mistral-large-latest", "codestral-latest", "devstral-2512", "devstral-small-latest", "devstral-medium-latest", "claude-opus-4-20250514", "claude-sonnet-4-20250514"}

def _switch_model(name: str) -> str:
    """Switch the active model at runtime, routing cloud models via gateway."""
    global MODEL
    key = name.lower().strip()
    if key not in AVAILABLE_MODELS:
        return f"Unknown model: {name}\nAvailable: {', '.join(AVAILABLE_MODELS.keys())}"
    MODEL = AVAILABLE_MODELS[key]
    maude_core.MODEL = MODEL

    # Point the client at the right endpoint
    if MODEL in _CLOUD_MODELS:
        base_url = GATEWAY_URL
    else:
        base_url = LOCAL_URL
    if _app and hasattr(_app, 'client'):
        _app.client = OpenAI(base_url=base_url, api_key="not-needed")

    return f"Switched to {key} ({MODEL}) via {'gateway' if MODEL in _CLOUD_MODELS else 'local'}"


def handle_command(cmd: str) -> str:
    """Handle slash commands. Returns response or None if not a command."""
    if not cmd.startswith("/"):
        return None

    parts = cmd[1:].strip().split(maxsplit=10)
    if not parts:
        return None

    command = parts[0].lower()

    if command == "help":
        return """MAUDE Commands:

/help              - Show this help
/model             - Show current model configuration
/model switch NAME - Switch model (nemotron, llava, mistral, codestral, devstral, claude, sonnet)
/copy         - Copy last response to file (~/.config/maude/last_response.txt)
/copymode     - Show how to copy text in tmux
/voice start  - Single voice listen/respond
/voice talk   - Continuous voice conversation
/voice stop   - Stop voice mode
/voice config - Show voice configuration
/voice deps   - Check voice dependencies

Tools available:
- search_directory, search_file, read_file, write_file, edit_file
- list_directory, change_directory, get_working_directory
- run_command (shell)
- web_browse, web_search, web_view
- view_image
- ask_frontier (escalate to cloud AI, if configured)

Say "quit" to exit."""
    elif command == "model":
        if len(parts) >= 3 and parts[1].lower() == "switch":
            return _switch_model(parts[2])
        elif len(parts) >= 2 and parts[1].lower() == "switch":
            return "Usage: /model switch <name>\nAvailable: nemotron, llava, mistral, codestral, devstral, devstral-small, devstral-medium, claude, sonnet"
        from frontier import list_available_providers
        frontier_providers = list_available_providers()
        frontier_info = ", ".join(frontier_providers) if frontier_providers else "none configured"
        return f"""Model Configuration:

Active:       {MODEL}
Server:       {LOCAL_URL}
Context:      {NUM_CTX} tokens

Available models:
  nemotron            local (llama-server)
  llava               LLaVA (local, vision)
  mistral             Mistral Large (cloud)
  codestral           Codestral (cloud, code)
  devstral            Devstral 2 (cloud, code agent)
  devstral-small      Devstral Small (cloud, code light)
  devstral-medium     Devstral Medium (cloud, code mid)
  claude              Claude Opus (cloud)
  sonnet              Claude Sonnet (cloud)

Vision:       {maude_core.VISION_MODEL}
Vision URL:   {maude_core.VISION_URL}

Frontier:     {frontier_info}

Switch model: /model switch <name>"""
    elif command == "copy":
        global _last_response
        if not _last_response:
            return "No response to copy yet."
        # Try clipboard first
        import shutil
        copied = False
        for clip_cmd in ["xclip -selection clipboard", "xsel --clipboard", "pbcopy", "wl-copy"]:
            binary = clip_cmd.split()[0]
            if shutil.which(binary):
                try:
                    subprocess.run(clip_cmd.split(), input=_last_response.encode(), timeout=5)
                    copied = True
                    break
                except Exception:
                    continue
        # Always save to file too
        copy_path = Path.home() / ".config" / "maude" / "last_response.txt"
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        copy_path.write_text(_last_response)
        if copied:
            return "Copied to clipboard!"
        return f"Saved to: {copy_path}\n(Install xclip for direct clipboard: sudo apt install xclip)\n\nTip: Hold Shift + click-drag to select text in the TUI."
    elif command == "voice":
        subcommand = parts[1].lower() if len(parts) > 1 else ""

        if subcommand == "deps":
            deps = check_voice_dependencies()
            lines = ["Voice Dependencies:"]
            for dep, available in deps.items():
                status = "[green]OK[/green]" if available else "[red]MISSING[/red]"
                lines.append(f"  {dep}: {status}")
            return "\n".join(lines)

        elif subcommand == "config":
            return """Voice Configuration:

Backend: whisper_local (Whisper STT + TTS)
TTS Provider: piper (fallback: espeak)
Whisper Model: base

Use /voice start for single interaction
Use /voice talk for continuous conversation"""

        elif subcommand in ["start", "talk", "stop"]:
            # Return special signal for async handling
            return f"__VOICE_ACTION__{subcommand}"

        else:
            return """Voice Commands:

/voice start  - Single voice listen/respond
/voice talk   - Continuous voice conversation
/voice stop   - Stop voice mode
/voice config - Show voice configuration
/voice deps   - Check voice dependencies

Say "stop", "exit", or "quit" during talk mode to end."""
    else:
        return f"Unknown command: /{command}\nType /help for available commands."


SYSTEM_PROMPT = f"""You are MAUDE, a local AI assistant running {MODEL}.

STYLE: Be brief. Action over explanation. Use tools proactively.

TOOLS AVAILABLE:
- File ops: read_file, write_file, edit_file, search_file, search_directory, list_directory, change_directory, get_working_directory
- Shell: run_command (git, pip, python, etc.)
- Web: web_search, web_browse, web_view, view_image (LLaVA vision)
- Cloud AI: ask_frontier (escalate to Claude/Gemini), send_to_claude (delegate to Claude Code)
- Gmail: gmail_list, gmail_read, gmail_send
- Google Drive: drive_list, drive_search, drive_read, drive_upload, drive_create_folder, drive_create_doc, drive_create_sheet, drive_update_doc, drive_delete
- Google Sheets: sheets_read, sheets_write, sheets_append, sheets_create, sheets_list_sheets, sheets_clear
- Google Calendar: calendar_list_events, calendar_create_event, calendar_update_event, calendar_delete_event, calendar_search_events, calendar_list_calendars
- Google Slides: slides_get_presentation, slides_get_slide, slides_create_presentation, slides_add_slide, slides_add_text
- Google Contacts: contacts_list, contacts_get, contacts_create, contacts_update, contacts_delete, contacts_search
- YouTube: youtube_search, youtube_get_video, youtube_get_channel, youtube_list_playlists, youtube_get_playlist_items, youtube_create_playlist, youtube_add_to_playlist, youtube_get_comments, youtube_post_comment, youtube_my_channel
- Substack: substack_create_draft, substack_list_drafts, substack_list_posts, substack_get_post, substack_update_draft, substack_delete_draft, substack_get_stats
- Browser: browser_open, browser_click, browser_type, browser_navigate, browser_screenshot, browser_extract, browser_fill_form, browser_select, browser_close
- Social media: skill_post_social (Twitter/X, LinkedIn, Bluesky), skill_social_status
- Scheduling: schedule_task

DELEGATION TO CLAUDE:
Use send_to_claude when user says "ask Claude", "delegate to Claude", or for complex multi-file tasks.

FILE EDITING WORKFLOW:
1. search_directory to find which file contains the code
2. read_file with line range to see context
3. edit_file to replace lines

Confirm before destructive operations."""


class MaudeApp(App):
    """MAUDE TUI Application with fixed input at bottom."""

    CSS = """
    #output {
        height: 1fr;
        border: solid cyan;
        padding: 0 1;
        scrollbar-gutter: stable;
    }
    #status {
        height: 1;
        padding: 0 1;
    }
    #input-container {
        height: auto;
        max-height: 3;
        padding: 0 1;
    }
    Input {
        border: solid green;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "handle_interrupt", "Quit/Stop Voice"),
    ]

    def action_handle_interrupt(self):
        """Handle Ctrl+C: stop voice during voice mode, exit otherwise."""
        if self._voice_active and self.voice_controller:
            self.voice_controller.stop()
            self._voice_active = False
            self.write_output("\n[dim yellow]Voice mode interrupted.[/dim yellow]")
            self.update_voice_status("")
        else:
            self.exit()

    def __init__(self):
        super().__init__()
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.conv_id = str(__import__('uuid').uuid4())
        self.conv_title = ""
        self._collab_activity = ""
        self.client = None
        self.spinner_frame = 0
        self.spinner_timer = None
        self.thinking_line_count = 0
        # Voice mode
        self.voice_controller: Optional[VoiceController] = None
        self._voice_active = False
        # For sync: start from end of log file
        import os
        log_path = os.path.expanduser("~/.config/maude/chat_sync.jsonl")
        try:
            self.sync_position = os.path.getsize(log_path) if os.path.exists(log_path) else 0
        except:
            self.sync_position = 0

    def compose(self) -> ComposeResult:
        yield RichLog(id="output", wrap=True, highlight=True, markup=True)
        yield Static("", id="status")
        with Container(id="input-container"):
            yield Input(placeholder="Type message... (quit to exit)", id="user-input")

    def on_mount(self):
        global _app
        _app = self

        # Load API keys
        km = KeyManager()
        km.load_all_keys()

        # Create client
        try:
            self.client = create_client()
        except Exception as e:
            self.write_output(f"[red]Failed to connect: {e}[/red]")
            return

        self.output_log = self.query_one("#output", RichLog)
        self.input_widget = self.query_one("#user-input", Input)
        self.banner_lines = get_banner_with_mech()
        self.banner_frame = 0

        # Start banner animation
        self.animate_banner()

        # Start presence heartbeat for collaboration
        self._start_collab_heartbeat()

    def animate_banner(self):
        """Animate the fire-colored banner."""
        if self.banner_frame < 25:
            # Clear and redraw banner with new frame
            self.output_log.clear()
            for line in self.banner_lines:
                self.output_log.write(fire_text(line, self.banner_frame))
            self.banner_frame += 1
            self.set_timer(0.06, self.animate_banner)
        else:
            # Animation done, show final banner + info
            self.output_log.clear()
            for line in self.banner_lines:
                self.output_log.write(fire_text(line, 24))
            self.write_output("")
            self.write_output(f"[dim]{MODEL} | Files | Shell | Web | Vision[/dim]")
            self.write_output("[dim]Type /help for commands, 'quit' to exit[/dim]\n")
            self.input_widget.focus()
            # Start sync polling
            self.check_telegram_messages()

    def _start_collab_heartbeat(self):
        """Send presence heartbeat every 30s in background."""
        import threading
        def _loop():
            hub = get_collab_hub()
            while True:
                try:
                    hub.heartbeat(
                        client_id=f"tui-{hub.presence._clients and 'main' or 'main'}",
                        client_type="tui",
                        activity=self._collab_activity or "idle",
                        conversation_id=self.conv_id,
                    )
                except Exception:
                    pass
                time.sleep(30)
        threading.Thread(target=_loop, daemon=True).start()

    def write_output(self, text):
        """Write to the output log."""
        if hasattr(self, 'output_log'):
            self.output_log.write(text)

    def stream_token(self, token: str, is_first: bool = False, prefix: str = ""):
        """Append a streaming token to the output log.
        Must be called from a worker thread (uses call_from_thread).

        On the first token we write a new Text to the RichLog and stash it
        along with its starting line index.  On subsequent tokens we delete
        the Strips produced by the previous render and re-write the grown
        Text, giving a live typewriter effect.
        """
        if not hasattr(self, 'output_log'):
            return

        if is_first:
            self._stream_text = Text()
            if prefix:
                self._stream_text.append(prefix, style="bold magenta")
            self._stream_mark = None

        self._stream_text.append(token)

        text_snapshot = self._stream_text.copy()
        mark = self._stream_mark

        def _render():
            log = self.output_log
            if mark is not None:
                del log.lines[mark:]
            self._stream_mark = len(log.lines)
            log.write(text_snapshot, scroll_end=True)

        self.call_from_thread(_render)

    def update_voice_status(self, status: str):
        """Update status bar for voice mode."""
        try:
            status_widget = self.query_one("#status", Static)
            if status:
                status_widget.update(Text(f"🎤 {status}", style="cyan"))
            else:
                status_widget.update("")
        except:
            pass

    def check_telegram_messages(self):
        """Check for Telegram messages and display them."""
        try:
            entries, self.sync_position = read_chat_log_since(self.sync_position)
            for entry in entries:
                if entry.get("channel") == "telegram":
                    role = entry.get("role", "")
                    content = entry.get("content", "")
                    if role == "user":
                        self.write_output(f"\n[dim cyan][telegram][/dim cyan] [bold blue]USER >[/bold blue] {content}")
                    elif role == "assistant":
                        self.write_output(f"[dim cyan][telegram][/dim cyan] [bold magenta]MAUDE:[/bold magenta] {content}")
        except:
            pass
        # Check again in 2 seconds
        self.set_timer(2.0, self.check_telegram_messages)

    async def on_input_submitted(self, event: Input.Submitted):
        """Handle user input."""
        user_input = event.value.strip()
        self.input_widget.clear()

        if not user_input:
            return

        # Show user message
        self.write_output(f"\n[bold green]YOU >[/bold green] {user_input}")

        # Exit commands
        if user_input.lower() in ("quit", "exit", "bye", "goodbye"):
            self.write_output("\n[dim magenta]MAUDE signing off.[/dim magenta]")
            self.exit()
            return

        # Slash commands
        if user_input.startswith("/"):
            result = handle_command(user_input)
            if result:
                # Check for voice action signals
                if result.startswith("__VOICE_ACTION__"):
                    action = result.replace("__VOICE_ACTION__", "")
                    if action == "start":
                        self.write_output("[dim]Starting voice mode...[/dim]")
                        self.voice_start_worker()
                    elif action == "talk":
                        self.write_output("[dim]Starting talk mode... Say 'stop' to end.[/dim]")
                        self.voice_talk_worker()
                    elif action == "stop":
                        if self.voice_controller:
                            self.voice_controller.stop()
                            self.write_output("[dim]Voice mode stopped.[/dim]")
                        else:
                            self.write_output("[dim]Voice mode not active.[/dim]")
                else:
                    self.write_output(f"\n{result}")
            return

        # Process with LLM in background
        self.process_message(user_input)

    def start_spinner(self):
        """Start the thinking spinner animation."""
        self.spinner_frame = 0
        self.update_spinner()

    def update_spinner(self):
        """Update spinner animation frame."""
        spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        color = "bright_cyan" if self.spinner_frame % 2 else "cyan"
        spinner = spinners[self.spinner_frame % len(spinners)]

        status = self.query_one("#status", Static)
        status.update(Text(f"{spinner} thinking...", style=color))

        self.spinner_frame += 1
        self.spinner_timer = self.set_timer(0.1, self.update_spinner)

    def stop_spinner(self):
        """Stop the thinking spinner."""
        if self.spinner_timer:
            self.spinner_timer.stop()
            self.spinner_timer = None
        status = self.query_one("#status", Static)
        status.update("")

    @work(thread=True)
    def process_message(self, user_input: str):
        """Process message with LLM in background thread."""
        self.messages.append({"role": "user", "content": user_input})

        # Start spinner
        self.call_from_thread(self.start_spinner)

        # Fast path: direct tool dispatch without LLM reasoning
        response = None
        try:
            result = fast_dispatch(user_input)
            if result:
                tool_name, args, tool_result = result
                console.print(f"[dim cyan]  → {tool_name}[/dim cyan]")
                # Give the LLM just the result to summarize (no tool definitions needed)
                summary_messages = [
                    {"role": "system", "content": "You are MAUDE. The user asked a question and a tool was already called. Summarize the result concisely."},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": f"I used {tool_name} and got this result:"},
                    {"role": "user", "content": f"Tool result:\n{tool_result[:3000]}\n\nSummarize this for me."}
                ]
                try:
                    summary = self.client.chat.completions.create(
                        model=MODEL,
                        messages=summary_messages,
                        temperature=0.2,
                        max_tokens=1024,
                        extra_body={"num_ctx": NUM_CTX}
                    )
                    response = summary.choices[0].message.content
                except Exception:
                    response = tool_result[:2000]
        except Exception as e:
            console.print(f"[dim yellow]  fast dispatch skipped: {e}[/dim yellow]")

        # Auto-route: check if a subagent should handle this directly
        if not response:
            try:
                from auto_router import route_message
                decision = route_message(user_input, self.messages[-10:])
                if decision.subagent and decision.confidence >= 0.5:
                    cloud_tag = " [cloud]" if decision.prefer_cloud else ""
                    console.print(f"[dim cyan]  route: {decision.intent} → {decision.subagent} ({decision.confidence:.0%}){cloud_tag}[/dim cyan]")
                    from execution import execute_subagent
                    response = execute_subagent(
                        decision.subagent,
                        user_input,
                        prefer_cloud=decision.prefer_cloud
                    )
            except ImportError:
                pass
            except Exception as e:
                console.print(f"[dim yellow]  routing skipped: {e}[/dim yellow]")

        # Fall back to main tool-calling chat loop
        if not response:
            response = chat(self.client, self.messages)

        # Stop spinner
        self.call_from_thread(self.stop_spinner)

        if response:
            global _last_response
            _last_response = response
            self.messages.append({"role": "assistant", "content": response})
            # Log for sync (after processing complete)
            append_chat_log("cli", "user", user_input)
            append_chat_log("cli", "assistant", response)
            # Sync to gateway for cross-device history
            if not self.conv_title:
                self.conv_title = conversation_sync.generate_title(user_input)
            conversation_sync.save_conversation(
                self.conv_id, self.conv_title, MODEL, self.messages
            )
            # Emit activity event
            self._collab_activity = f"chatting: {user_input[:40]}"
            try:
                get_collab_hub().emit(
                    "chat", f"Asked about: {user_input[:50]}",
                    data={"model": MODEL},
                    client_id="tui-main",
                    conversation_id=self.conv_id,
                )
            except Exception:
                pass

    @work(thread=True)
    def voice_start_worker(self):
        """Single voice interaction in background thread."""
        self._voice_active = True
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Initialize voice controller if needed
            if self.voice_controller is None:
                self.voice_controller = VoiceController(self)

            loop.run_until_complete(self.voice_controller.run_single())
        except Exception as e:
            self.call_from_thread(self.write_output, f"[red]Voice error: {e}[/red]")
        finally:
            loop.close()
            self._voice_active = False

    @work(thread=True)
    def voice_talk_worker(self):
        """Continuous voice mode in background thread."""
        self._voice_active = True
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Initialize voice controller if needed
            if self.voice_controller is None:
                self.voice_controller = VoiceController(self)

            loop.run_until_complete(self.voice_controller.run_talk_mode())
        except Exception as e:
            self.call_from_thread(self.write_output, f"[red]Voice error: {e}[/red]")
        finally:
            loop.close()
            self._voice_active = False
            self.call_from_thread(self.write_output, "[dim]Talk mode ended.[/dim]")


def run_telegram_in_background():
    """Run Telegram bot in a background thread."""
    try:
        from run_telegram import main as telegram_main
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(telegram_main(standalone=False))
    except ImportError:
        pass  # Telegram deps not installed
    except Exception:
        pass  # Silently continue - TUI keeps working


def run_transcription_server():
    """Run transcription server in a background thread."""
    try:
        from fastapi import FastAPI, UploadFile, File
        from fastapi.responses import JSONResponse
        import uvicorn
        import tempfile

        app = FastAPI()
        whisper_model = None
        whisper_type = None

        def get_whisper():
            nonlocal whisper_model, whisper_type
            if whisper_model is not None:
                return whisper_model, whisper_type

            try:
                from faster_whisper import WhisperModel
                whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
                whisper_type = "faster"
            except:
                try:
                    import whisper
                    whisper_model = whisper.load_model("base")
                    whisper_type = "original"
                except:
                    raise RuntimeError("No Whisper available")
            return whisper_model, whisper_type

        @app.get("/health")
        def health():
            return {"status": "ok", "service": "transcription"}

        @app.post("/transcribe")
        async def transcribe(audio: UploadFile = File(...)):
            model, wtype = get_whisper()

            suffix = ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                content = await audio.read()
                f.write(content)
                temp_path = f.name

            try:
                if wtype == "faster":
                    segments, _ = model.transcribe(temp_path)
                    text = " ".join(seg.text for seg in segments).strip()
                else:
                    result = model.transcribe(temp_path)
                    text = result["text"].strip()
                return JSONResponse({"text": text, "success": True})
            finally:
                import os
                os.unlink(temp_path)

        # Run on port 30001
        uvicorn.run(app, host="0.0.0.0", port=30001, log_level="warning")

    except ImportError:
        pass  # FastAPI not installed
    except Exception:
        pass  # Silently continue


def main():
    # Start transcription server in background
    transcription_thread = threading.Thread(target=run_transcription_server, daemon=True)
    transcription_thread.start()

    # Check if Telegram should be enabled
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")

    if telegram_token:
        # Start Telegram bot in background thread
        telegram_thread = threading.Thread(target=run_telegram_in_background, daemon=True)
        telegram_thread.start()

    # Start proactive heartbeat in background
    try:
        from heartbeat import get_heartbeat
        hb = get_heartbeat()
        if hb.enabled:
            hb.set_tool_executor(execute_tool)

            def _run_heartbeat():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(hb.start())
                loop.run_forever()

            heartbeat_thread = threading.Thread(target=_run_heartbeat, daemon=True)
            heartbeat_thread.start()
    except ImportError:
        pass
    except Exception:
        pass  # Heartbeat is optional

    app = MaudeApp()
    app.run()


if __name__ == "__main__":
    main()
