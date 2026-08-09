#!/usr/bin/env python3
"""
MAUDE - Terminal LLM Chat powered by local LLMs
"""

import asyncio
import os
import threading

from dotenv import load_dotenv

load_dotenv()
# Also load variables.env (API keys used by gateway and subagents)
_vars_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "variables.env")
if os.path.exists(_vars_env):
    load_dotenv(_vars_env, override=False)

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import ClassVar

import pyfiglet
from openai import OpenAI
from rich.align import Align
from rich.live import Live
from rich.text import Text
from textual import work

# Textual TUI framework
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import Input, RichLog, Static

import conversation_sync

# MAUDE core - shared tools
import maude_core
from collab import get_hub as get_collab_hub

# Minimal imports
from keys import KeyManager
from maude_core import (
    append_chat_log,
    execute_tool,
    fast_dispatch,
    get_tools_for_message,
    read_chat_log_since,
    reset_rate_limits,
)
from maude_core.tools_plan import PARALLEL_SAFE

# Voice mode
from voice import VoiceMode, check_voice_dependencies

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
        if _app and hasattr(_app, "output_log"):
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
            os.system("clear" if os.name != "nt" else "cls")


console = TUIConsole()


class VoiceController:
    """Manages voice mode state and lifecycle for TUI integration."""

    def __init__(self, app: "MaudeApp"):
        self.app = app
        self.voice_mode: VoiceMode | None = None
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
                ),
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
            self.app.call_from_thread(self.app.write_output, f"\n[bold green]YOU (voice) >[/bold green] {text}")
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
    """Create API client, routing cloud models through gateway."""
    base_url = GATEWAY_URL if MODEL in _CLOUD_MODELS else LOCAL_URL
    return OpenAI(base_url=base_url, api_key="not-needed")


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
    banner_lines = banner.rstrip("\n").split("\n")
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
                content.append("\n")

            live.update(Align.center(content))
            time.sleep(0.06)

    # Final static banner with fire gradient
    final_content = Text()
    for line in banner_lines:
        final_content.append_text(fire_text(line, 24))
        final_content.append("\n")

    console.print(Align.center(final_content))
    console.print()


def print_separator():
    """Print a styled separator."""
    console.print("─" * console.width, style="dim cyan")


def _escalate_to_frontier(user_question: str) -> str:
    """Escalate a question to Claude/Gemini when the local model gets stuck."""
    try:
        from frontier import RateLimitError, ask_frontier, list_available_providers

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
                    system_prompt="You are MAUDE, a capable AI assistant. Answer the user's question directly and helpfully. Be concise.",
                )
                console.print(
                    f"[dim cyan]  ({response.provider} — {response.input_tokens}+{response.output_tokens} tokens, ${response.cost_usd:.4f})[/dim cyan]"
                )
                return response.content
            except RateLimitError:
                errors.append(f"{provider}: rate limited")
                continue
            except Exception as e:
                errors.append(f"{provider}: {e}")
                continue

        return f"I tried escalating to cloud models but they're all unavailable right now ({', '.join(errors)}). Could you try again in a moment?"

    except ImportError:
        return "I wasn't able to handle that locally. Could you give me more detail so I can try a different approach?"


def _compact_tool_result(name: str, result: str) -> str:
    """Truncate tool results to prevent context bloat across loop iterations."""
    if not result:
        return result
    n = len(result)
    if name in ("write_file", "edit_file", "change_directory", "get_working_directory"):
        return result
    if name == "read_file" and n > 3000:
        lines = result.split("\n")
        if len(lines) > 100:
            head = "\n".join(lines[:80])
            tail = "\n".join(lines[-20:])
            return f"{head}\n\n... ({len(lines) - 100} lines omitted) ...\n\n{tail}"
        return result[:3000] + f"\n... (truncated, {n} chars total)"
    if name == "run_command" and n > 3000:
        return result[:2000] + f"\n\n... ({n - 2800} chars omitted) ...\n\n" + result[-800:]
    if name == "list_directory" and n > 2000:
        lines = result.split("\n")
        if len(lines) > 65:
            return "\n".join(lines[:65]) + f"\n... ({len(lines) - 65} more entries)"
        return result[:2000] + "\n... (truncated)"
    if n > 4000:
        return result[:3500] + f"\n... (truncated, {n} chars total)"
    return result


def chat(client, messages: list):
    """Send chat request to local LLM with tool support."""
    global show_thinking

    max_tool_iterations = 40
    tool_iteration = 0
    recent_tool_calls = []
    consecutive_duplicates = 0
    reset_rate_limits()  # Reset per-turn limits in maude_core

    # Cloud models: gateway handles tools, stream with trace support
    if MODEL in _CLOUD_MODELS:
        try:
            start_time = time.time()
            clean_msgs = [m for m in messages if "tool_calls" not in m and m.get("role") != "tool"]
            use_stream = _app and hasattr(_app, "stream_token")

            if use_stream:
                # Raw SSE stream — captures both tool traces and content
                import httpx

                base_url = client.base_url if hasattr(client, "base_url") else GATEWAY_URL
                url = f"{str(base_url).rstrip('/')}/chat/completions"
                payload = {
                    "model": MODEL,
                    "messages": clean_msgs,
                    "temperature": 0.2,
                    "max_tokens": 4096,
                    "stream": True,
                }
                full_content = ""
                token_count = 0
                prompt_tokens = 0
                first = True

                with (
                    httpx.Client(timeout=300) as http,
                    http.stream("POST", url, json=payload, headers={"Authorization": "Bearer not-needed"}) as resp,
                ):
                    if resp.status_code >= 400:
                        resp.read()
                        raise Exception(f"Gateway returned {resp.status_code}: {resp.text[:200]}")
                    buf = ""
                    running_tasks = {}
                    for text_chunk in resp.iter_text():
                        buf += text_chunk
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            # Tool trace comments from gateway
                            if line.startswith(": trace "):
                                try:
                                    trace = json.loads(line[8:])
                                    ttype = trace.get("type", "")
                                    if ttype == "tool_call":
                                        tname = trace.get("name", "?")
                                        targs = trace.get("args", "")
                                        task = trace.get("task", "")
                                        # Try to extract the most useful arg value
                                        arg_hint = ""
                                        try:
                                            parsed = json.loads(targs) if targs else {}
                                            if isinstance(parsed, dict):
                                                for k in (
                                                    "command",
                                                    "query",
                                                    "path",
                                                    "local_path",
                                                    "name",
                                                    "file_id",
                                                    "content",
                                                    "doc_id",
                                                    "url",
                                                ):
                                                    if k in parsed:
                                                        arg_hint = str(parsed[k])
                                                        if len(arg_hint) > 55:
                                                            arg_hint = arg_hint[:55] + "…"
                                                        break
                                        except (json.JSONDecodeError, TypeError):
                                            if targs and targs != "{}" and len(targs) <= 60:
                                                arg_hint = targs
                                        console.print(f"[bold cyan]  ╭─ [/bold cyan][bold white]{task or tname}[/bold white]")
                                        running_tasks[tname] = task or tname
                                        if task:
                                            console.print(f"[cyan]  │[/cyan]  [dim]{tname}[/dim]")
                                        if arg_hint:
                                            console.print(f"[cyan]  │[/cyan]  [dim]{arg_hint}[/dim]")
                                    elif ttype == "parallel_start":
                                        pcount = trace.get("count", 0)
                                        console.print(f"[dim cyan]  ⚡ {pcount} tools in parallel[/dim cyan]")
                                    elif ttype == "tool_result":
                                        tname = trace.get("name", "")
                                        preview = trace.get("preview", "")
                                        elapsed = trace.get("elapsed", 0)
                                        status_color = "green" if not preview.startswith("Error") else "red"
                                        console.print(
                                            f"[cyan]  ╰─[/cyan] [{status_color}]{preview}[/{status_color}] [dim]({elapsed:.1f}s)[/dim]"
                                        )
                                        running_tasks.pop(tname, None)
                                    elif ttype == "context_trim":
                                        removed = trace.get("removed", 0)
                                        if removed:
                                            console.print(
                                                f"[dim cyan]  · context trimmed ({removed} messages)[/dim cyan]"
                                            )
                                    elif ttype == "keepalive":
                                        tname = trace.get("name", "")
                                        elapsed = trace.get("elapsed", 0)
                                        label = running_tasks.get(tname, tname or "task")
                                        console.print(f"[dim cyan]  ⠿ still working: {label} ({elapsed:.1f}s)[/dim cyan]")
                                    elif ttype == "llm_call":
                                        prompt_tokens += trace.get("prompt_tokens", 0)
                                        token_count += trace.get("completion_tokens", 0)
                                    elif ttype == "error":
                                        err_msg = trace.get("message", "unknown error")
                                        console.print(f"[red]  ✗ {err_msg}[/red]")
                                except (json.JSONDecodeError, KeyError):
                                    pass
                                continue
                            # Normal SSE data lines
                            if line.startswith("data: "):
                                data_str = line[6:]
                                if data_str.strip() == "[DONE]":
                                    continue
                                try:
                                    chunk = json.loads(data_str)
                                    choices = chunk.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            _app.stream_token(content, is_first=first, prefix="MAUDE: ")
                                            full_content += content
                                            first = False
                                    usage = chunk.get("usage")
                                    if usage:
                                        token_count = usage.get("completion_tokens", 0) or 0
                                        prompt_tokens = usage.get("prompt_tokens", 0) or 0
                                except json.JSONDecodeError:
                                    pass

                elapsed_time = time.time() - start_time
                if full_content:
                    _app.call_from_thread(
                        _app.write_output, f"[dim]{prompt_tokens}+{token_count} tokens in {elapsed_time:.1f}s[/dim]"
                    )
            else:
                # Non-streaming fallback (headless / no TUI)
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=clean_msgs,
                    temperature=0.2,
                    max_tokens=4096,
                    timeout=300,
                    stream=False,
                )
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

    _tool_step = [0]  # mutable counter for tool step display

    while True:
        tool_iteration += 1
        if tool_iteration > max_tool_iterations:
            console.print("[dim yellow](max tool iterations reached — escalating to frontier model...)[/dim yellow]")
            user_question = next(
                (m["content"] for m in reversed(messages) if m.get("role") == "user"), "Help me with my request"
            )
            return _escalate_to_frontier(user_question)
        try:
            start_time = time.time()

            # Reduce max_tokens after tool results to prevent reasoning loops
            has_tool_results = any(m.get("role") == "tool" for m in messages)
            max_tokens = 2048 if has_tool_results else 4096

            # Stream when TUI is available (typewriter effect), non-stream otherwise
            use_stream = _app and hasattr(_app, "stream_token")
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
                        if hasattr(chunk, "usage") and chunk.usage:
                            token_count = getattr(chunk.usage, "completion_tokens", 0) or 0
                            prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
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
                                    # Show tool name immediately as it streams in
                                    _tool_step[0] += 1
                                    console.print(
                                        f"[bold cyan]  ╭─ [/bold cyan][bold white]{tool_calls_acc[i]['name']}[/bold white]"
                                    )
                                if tc.function.arguments:
                                    tool_calls_acc[i]["arguments"] += tc.function.arguments
                elapsed_time = time.time() - start_time
                if full_content:
                    _app.call_from_thread(
                        _app.write_output, f"[dim]{prompt_tokens}+{token_count} tokens in {elapsed_time:.1f}s[/dim]"
                    )
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
                        tool_calls_data[tc.id] = {"name": tc.function.name, "arguments": tc.function.arguments}

            # Handle tool calls
            if tool_calls_data:
                # Add assistant message with tool calls to history
                messages.append(
                    {
                        "role": "assistant",
                        "content": full_content or "",
                        "tool_calls": [
                            {
                                "id": tc_id,
                                "type": "function",
                                "function": {"name": tc_data["name"], "arguments": tc_data["arguments"]},
                            }
                            for tc_id, tc_data in tool_calls_data.items()
                        ],
                    }
                )

                # Parse all tool calls and check for duplicates
                # Tools that only read state can run in parallel; tools that
                # mutate state (write/edit/run/change) must run sequentially.
                parsed_calls = []  # [(tc_id, func_name, func_args)]
                for tc_id, tc_data in tool_calls_data.items():
                    func_name = tc_data["name"]
                    raw_args = tc_data.get("arguments", "{}")
                    if isinstance(raw_args, dict):
                        func_args = raw_args
                    elif isinstance(raw_args, str):
                        try:
                            func_args = json.loads(raw_args)
                        except (json.JSONDecodeError, ValueError):
                            func_args = {}
                    else:
                        func_args = {}

                    # Check for duplicate tool calls (same tool + same args)
                    call_signature = (func_name, json.dumps(func_args, sort_keys=True))
                    if call_signature in recent_tool_calls:
                        consecutive_duplicates += 1
                        console.print("[cyan]  ╰─[/cyan] [dim yellow]skipped (duplicate call)[/dim yellow]")
                        result = "(Already called with same arguments - see previous result. STOP retrying and respond to the user with the best answer you can based on information gathered so far.)"
                        messages.append({"role": "tool", "tool_call_id": tc_id, "content": result})
                        if consecutive_duplicates >= 3:
                            console.print("[dim yellow](escalating to frontier model...)[/dim yellow]")
                            user_question = next(
                                (m["content"] for m in reversed(messages) if m.get("role") == "user"),
                                "Help me with my request",
                            )
                            return _escalate_to_frontier(user_question)
                    else:
                        consecutive_duplicates = 0
                        recent_tool_calls.append(call_signature)
                        parsed_calls.append((tc_id, func_name, func_args))

                # Split into parallel-safe and sequential calls
                parallel_batch = [(tc_id, fn, args) for tc_id, fn, args in parsed_calls if fn in PARALLEL_SAFE]
                sequential_batch = [(tc_id, fn, args) for tc_id, fn, args in parsed_calls if fn not in PARALLEL_SAFE]

                def _exec_one(tc_id, func_name, func_args):
                    """Execute a single tool call, return (tc_id, func_name, result, elapsed)."""
                    t0 = time.time()
                    res = execute_tool(func_name, func_args)
                    return tc_id, func_name, res, time.time() - t0

                def _print_tool_status(func_name, func_args, result, elapsed):
                    """Print arg hint and result preview for a tool call."""
                    arg_hint = ""
                    for key in (
                        "command",
                        "query",
                        "path",
                        "local_path",
                        "name",
                        "file_id",
                        "content",
                        "doc_id",
                        "url",
                    ):
                        if key in func_args:
                            val = str(func_args[key])
                            if len(val) > 55:
                                val = val[:55] + "…"
                            arg_hint = val
                            break
                    if arg_hint:
                        console.print(f"[cyan]  │[/cyan]  [dim]{arg_hint}[/dim]")
                    result_str = result or ""
                    preview = result_str[:80].replace("\n", " ").strip()
                    if len(result_str) > 80:
                        preview += "…"
                    status_color = "green" if not result_str.startswith("Error") else "red"
                    console.print(
                        f"[cyan]  ╰─[/cyan] [{status_color}]{preview}[/{status_color}] [dim]({elapsed:.1f}s)[/dim]"
                    )

                # Run parallel-safe tools concurrently
                if len(parallel_batch) > 1:
                    console.print(f"[dim cyan]  ⚡ {len(parallel_batch)} tools in parallel[/dim cyan]")
                    parallel_results = {}  # tc_id -> (func_name, result, elapsed)
                    with ThreadPoolExecutor(max_workers=min(len(parallel_batch), 6)) as pool:
                        futures = {
                            pool.submit(_exec_one, tc_id, fn, args): (tc_id, fn, args)
                            for tc_id, fn, args in parallel_batch
                        }
                        for future in as_completed(futures):
                            tc_id, func_name, result, elapsed = future.result()
                            _, _, func_args = futures[future]
                            parallel_results[tc_id] = (func_name, func_args, result, elapsed)
                    # Append results in original order
                    for tc_id, _fn, _args in parallel_batch:
                        func_name, func_args, result, elapsed = parallel_results[tc_id]
                        _print_tool_status(func_name, func_args, result, elapsed)
                        messages.append(
                            {"role": "tool", "tool_call_id": tc_id, "content": _compact_tool_result(func_name, result)}
                        )
                elif len(parallel_batch) == 1:
                    # Single parallel-safe tool — no pool overhead
                    tc_id, func_name, func_args = parallel_batch[0]
                    tc_id, func_name, result, elapsed = _exec_one(tc_id, func_name, func_args)
                    _print_tool_status(func_name, func_args, result, elapsed)
                    messages.append(
                        {"role": "tool", "tool_call_id": tc_id, "content": _compact_tool_result(func_name, result)}
                    )

                # Run state-mutating tools sequentially (order matters)
                for tc_id, func_name, func_args in sequential_batch:
                    tc_id, func_name, result, elapsed = _exec_one(tc_id, func_name, func_args)
                    _print_tool_status(func_name, func_args, result, elapsed)
                    messages.append(
                        {"role": "tool", "tool_call_id": tc_id, "content": _compact_tool_result(func_name, result)}
                    )

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
    "nemotron-super": "nemotron-super",
    "nemotron-a3b": "nemotron-a3b",
    "a3b": "nemotron-a3b",
    "mistral": "mistral-large-latest",
    "codestral": "codestral-latest",
    "devstral": "devstral-2512",
    "devstral-small": "devstral-small-latest",
    "devstral-medium": "devstral-medium-latest",
    "openai": os.environ.get("MAUDE_OPENAI_MODEL", "gpt-4o"),
    "codex": "codex-cli",
    "gemma4": "gemma-4-31b",
    "gemma": "gemma-4-31b",
    "claude": "claude-opus-4-20250514",
    "sonnet": "claude-sonnet-4-20250514",
}

# Cloud models route through the gateway's HTTP mirror (same port as local).
# Also includes non-nemotron local models (e.g. gemma-4-31b on port 30013),
# since LOCAL_URL points at nemotron's port — the gateway knows the right route.
GATEWAY_URL = "http://localhost:30080/v1"

_CLOUD_MODELS = {
    "nemotron-super",
    "nemotron-a3b",
    "mistral-large-latest",
    "codestral-latest",
    "devstral-2512",
    "devstral-small-latest",
    "devstral-medium-latest",
    "codex-cli",
    "gemma-4-31b",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
}

# Resolve short model name (e.g. "mistral") to full ID (e.g. "mistral-large-latest")
if MODEL in AVAILABLE_MODELS and MODEL not in _CLOUD_MODELS:
    MODEL = AVAILABLE_MODELS[MODEL]
    maude_core.MODEL = MODEL


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
    if _app and hasattr(_app, "client"):
        _app.client = OpenAI(base_url=base_url, api_key="not-needed")

    return f"Switched to {key} ({MODEL}) via {'gateway' if MODEL in _CLOUD_MODELS else 'local'}"


def _use_model_for_message(model_name: str):
    """Temporarily switch to a model for a single message. Returns the previous MODEL for restoration."""
    global MODEL
    prev = MODEL
    key = model_name.lower().strip()
    if key in AVAILABLE_MODELS:
        full_id = AVAILABLE_MODELS[key]
        if full_id != MODEL:
            MODEL = full_id
            maude_core.MODEL = MODEL
            if MODEL in _CLOUD_MODELS:
                base_url = GATEWAY_URL
            else:
                base_url = LOCAL_URL
            if _app and hasattr(_app, "client"):
                _app.client = OpenAI(base_url=base_url, api_key="not-needed")
    return prev


def _restore_model(prev_model: str):
    """Restore model after per-message auto-switch."""
    global MODEL
    if prev_model != MODEL:
        MODEL = prev_model
        maude_core.MODEL = MODEL
        if MODEL in _CLOUD_MODELS:
            base_url = GATEWAY_URL
        else:
            base_url = LOCAL_URL
        if _app and hasattr(_app, "client"):
            _app.client = OpenAI(base_url=base_url, api_key="not-needed")


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
/model switch NAME - Switch model (nemotron, mistral, codestral, devstral, claude, sonnet)
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
            return "Usage: /model switch <name>\nAvailable: nemotron, mistral, codestral, devstral, devstral-small, devstral-medium, openai, codex, claude, sonnet"
        from frontier import list_available_providers

        frontier_providers = list_available_providers()
        frontier_info = ", ".join(frontier_providers) if frontier_providers else "none configured"
        return f"""Model Configuration:

Active:       {MODEL}
Server:       {LOCAL_URL}
Context:      {NUM_CTX} tokens

Available models:
  nemotron            local (llama-server)
  mistral             Mistral Large (cloud, vision)
  codestral           Codestral (cloud, code)
  devstral            Devstral 2 (cloud, code agent)
  devstral-small      Devstral Small (cloud, code light)
  devstral-medium     Devstral Medium (cloud, code mid)
  openai              OpenAI (cloud)
  codex               OpenAI/Codex route (cloud, code)
  claude              Claude Opus (cloud, vision)
  sonnet              Claude Sonnet (cloud, vision)

Vision:       native multimodal (active model) / LLaVA fallback

Frontier:     {frontier_info}

Switch model: /model switch <name>"""
    elif command == "copy":
        global _last_response
        if not _last_response:
            return "No response to copy yet."
        # Try clipboard first
        import shutil
        import subprocess

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


SYSTEM_PROMPT = """You are MAUDE, a local AI assistant running on Matt's DGX Spark.

STYLE: Be brief. Action over explanation. Use tools proactively.

EFFICIENCY: When you need multiple pieces of information, call ALL the tools you need in a single response rather than one at a time. For example, if you need to read 3 files, call read_file 3 times in one response — don't read one, wait, then read the next. Similarly, batch web_search, gmail_read, drive_read, github calls, etc. This dramatically reduces round-trips. Only chain tools sequentially when a later call truly depends on an earlier result.

CRITICAL RULES — FOLLOW THESE EXACTLY:
1. DO the work. NEVER tell the user to run commands themselves. You have run_command — use it.
2. VERIFY your work. After writing code: run the build/compile step and check for errors. After starting a server: curl it and confirm it responds. After creating files: check they exist.
3. FIX errors yourself. If a build fails, read the error, fix the code, and rebuild. Do NOT just report the error and stop.
4. NEVER give tutorials or step-by-step instructions for things you can do yourself. If the user asks for a URL, start the server and give them the URL. Don't explain how to start it.
5. NEVER guess system info. Use tools to check IPs, ports, running processes, etc.
6. When serving web apps: bind to 0.0.0.0, check which ports are free first (ss -tlnp), and give the Tailscale URL (run: tailscale ip -4) directly.
7. Complete the ENTIRE task. Don't stop halfway and describe what remains. If you're building a site, it should be built, running, and accessible before you respond.
8. DO NOT ask the user for permission at every step. When given a task, figure it out and do it. Use the tools and information you already have. Only ask the user when you genuinely cannot proceed without information you have no way to obtain (e.g. a password, a personal preference with no reasonable default). "Should I proceed?" and "Do you want me to...?" are almost never appropriate — just do the work.

TOOLS AVAILABLE:
- File ops: read_file, write_file, edit_file, search_file, search_directory, list_directory, change_directory, get_working_directory
- Shell: run_command (git, pip, python, etc.)
- Web: web_search, web_browse, web_view, view_image (native multimodal vision)
  IMPORTANT: Do NOT use web_search unless the user explicitly asks you to search/look something up, or you genuinely need current external information (news, prices, docs) to answer. For tasks like scheduling, posting, file operations, coding, or using existing tools — just do the task directly. Never web search as a first step.
- Cloud AI: ask_frontier (escalate to Claude/Gemini), send_to_claude (delegate to Claude Code)
- Gmail: gmail_list, gmail_read, gmail_send
- Google Drive: drive_list, drive_search, drive_read, drive_upload, drive_create_folder, drive_create_doc, drive_create_sheet, drive_update_doc, drive_delete
- Google Sheets: sheets_read, sheets_write, sheets_append, sheets_create, sheets_list_sheets, sheets_clear
- Google Calendar: calendar_list_events, calendar_create_event, calendar_update_event, calendar_delete_event, calendar_search_events, calendar_list_calendars
- Google Slides: slides_get_presentation, slides_get_slide, slides_create_presentation, slides_add_slide, slides_add_text
- Google Contacts: contacts_list, contacts_get, contacts_create, contacts_update, contacts_delete, contacts_search
- YouTube: youtube_search, youtube_get_video, youtube_get_channel, youtube_list_playlists, youtube_get_playlist_items, youtube_create_playlist, youtube_add_to_playlist, youtube_upload, youtube_get_comments, youtube_post_comment, youtube_my_channel
  youtube_upload defaults to PUBLIC for the user's Shorts workflow.
- Substack: substack_create_draft, substack_list_drafts, substack_list_posts, substack_get_post, substack_update_draft, substack_delete_draft, substack_get_stats
- Browser: browser_open, browser_snapshot, browser_click, browser_type, browser_navigate, browser_screenshot, browser_extract, browser_fill_form, browser_select, browser_close
  browser_snapshot returns an accessibility tree with interactive elements tagged [@e1], [@e2], etc. Pass these refs to browser_click or browser_type for precise interaction. Use browser_snapshot instead of browser_screenshot when you need to click or type — it's faster and more reliable.
  You can interact with websites using browser_open → browser_snapshot → browser_click/type. For social posting, especially X/Twitter posts with images or videos, use social_post instead of raw browser clicks; raw browser posting does not reliably attach media.
- Browser Login: browser_login (opens visible browser via VNC for manual login — accepts shorthand: "x", "linkedin", "instagram", "facebook", "github", "reddit", "tiktok", "bluesky", "youtube", "google", "pinterest" or any URL). browser_check_session (verify if saved login is still valid)
- Browser Workflows: workflow_create, workflow_run, workflow_list, workflow_get, workflow_delete, workflow_history, workflow_schedule, workflow_unschedule
IMPORTANT: When the user asks to "log in", "login", "sign in" to any website or social media, USE browser_login — do NOT give text instructions. The tool handles VNC automatically for remote access.
- Social media posting: social_post (browser-based — X, LinkedIn, Facebook, Instagram, Reddit, TikTok, Bluesky). Uses saved browser_login sessions. For Reddit, first line of content is the title; use subreddit param. When posting with an image or video on any platform, including X, you MUST pass media_path/video_path/image_path and set expect_media=true; never make a text-only post when media was requested. TikTok requires a video path.
- Social media API (fallback): skill_post_social (requires API keys — only use if social_post fails), skill_social_status
- Scheduling: schedule_task
- Planning: execute_plan — run a multi-stage tool plan in one call. Define stages (each an array of tool calls); tools within a stage run in parallel, stages run sequentially. Use $N.M to reference results from stage N, tool M. Use this when you can foresee multiple steps ahead (e.g. read 3 files, then edit based on findings). This saves round-trips — prefer it over calling tools one at a time when the full plan is known upfront.

CROSS-MACHINE (collab):
- mesh_status: Show online devices with client_id and platform
- dispatch_task: Run a shell command on ANY client device (Mac, Windows, Linux).
  Use target= with hostname, client_id, or platform name (e.g. "windows", "macos").
  Example: dispatch_task(prompt="ls ~/Desktop", target="macos", capability="SHELL")
  Example: dispatch_task(prompt="dir Desktop", target="windows", capability="SHELL")
  The command runs on the target client and the result is returned.
- list_tasks: Check task status and results
IMPORTANT: When the user asks to do something "on the mac/windows/pc", use dispatch_task with the right target. Do NOT say you can't access other machines — you CAN via dispatch_task.

PERSISTENT MEMORY:
- save_memory: Proactively save facts, preferences, and context the user shares (e.g. names, projects, preferences)
- recall_memory: Search memory when the user references past conversations or asks "do you remember"
- list_memories, forget_memory: Manage stored memories
IMPORTANT: Save memories proactively when the user shares personal info, preferences, or project context. Relevant memories are automatically loaded into your context each turn.

DELEGATION TO CLAUDE:
Use send_to_claude when user says "ask Claude", "delegate to Claude", or for complex multi-file tasks.

FILE EDITING WORKFLOW:
1. search_directory to find which file contains the code
2. read_file with line range to see context
3. edit_file to replace lines

Confirm before destructive operations."""


class MaudeApp(App):
    """MAUDE TUI Application with fixed input at bottom."""

    ALLOW_SELECT = True

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

    BINDINGS: ClassVar[list] = [
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
        self.conv_id = str(__import__("uuid").uuid4())
        self.conv_title = ""
        self._collab_activity = ""
        self.client = None
        self.spinner_frame = 0
        self.spinner_timer = None
        self.thinking_line_count = 0
        # Voice mode
        self.voice_controller: VoiceController | None = None
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
            self.write_output(f"[dim grey50]{MODEL}[/dim grey50]")
            self.write_output("")
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
                        client_id=f"tui-{(hub.presence._clients and 'main') or 'main'}",
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
        if hasattr(self, "output_log"):
            self.output_log.write(text)

    def stream_token(self, token: str, is_first: bool = False, prefix: str = ""):
        """Append a streaming token to the output log.
        Must be called from a worker thread (uses call_from_thread).

        On the first token we write a new Text to the RichLog and stash it
        along with its starting line index.  On subsequent tokens we delete
        the Strips produced by the previous render and re-write the grown
        Text, giving a live typewriter effect.
        """
        if not hasattr(self, "output_log"):
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
                        self.write_output(
                            f"[dim cyan][telegram][/dim cyan] [bold magenta]MAUDE:[/bold magenta] {content}"
                        )
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

    def _inject_memory_context(self, user_input: str):
        """Inject relevant memories and best-practice guides into the system prompt for this turn."""
        extra_sections = []

        # Current model identity — rebuilt each turn so /model switch is reflected immediately.
        # Tell the LLM who it is so it doesn't guess or try to read random config files.
        current_model = maude_core.MODEL
        alias = next((k for k, v in AVAILABLE_MODELS.items() if v == current_model), None)
        model_line = f"CURRENT MODEL: You are running as `{current_model}`"
        if alias and alias != current_model:
            model_line += f" (alias: {alias})"
        model_line += ". If the user asks which model you are, answer directly from this — do NOT read files or call tools to find out."
        extra_sections.append(model_line)

        # Memory context
        try:
            from maude_core.memory_utils import get_memory

            mem = get_memory()
            if mem:
                context = mem.get_context_for_prompt(user_input, max_memories=5)
                if context:
                    extra_sections.append(context)
        except Exception:
            pass  # Memory unavailable — proceed without context

        # MemPalace context — layered long-term memory (L3 semantic search)
        try:
            from maude_core.mempalace_utils import get_palace_context_for_prompt

            palace_ctx = get_palace_context_for_prompt(user_input, n_results=5)
            if palace_ctx:
                extra_sections.append(palace_ctx)
        except Exception:
            pass  # Palace unavailable — proceed without context

        # Best-practice guides — inject relevant guide based on user input keywords
        guide = self._match_guide(user_input)
        if guide:
            extra_sections.append(guide)

        if extra_sections:
            self.messages[0] = {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + "\n\n".join(extra_sections)}
        else:
            # Reset to base prompt (no stale context from prior turns)
            self.messages[0] = {"role": "system", "content": SYSTEM_PROMPT}

    @staticmethod
    def _match_guide(user_input: str) -> str | None:
        """Match user input to a best-practice guide and return its content."""
        import re

        query = user_input.lower()
        guides_dir = Path(__file__).parent / "guides"
        if not guides_dir.exists():
            return None

        # Keyword patterns mapped to guide filenames
        guide_triggers = {
            "coding-best-practices.md": re.compile(
                r"\b(code|coding|program|refactor|debug|function|class|test|lint|security|api|backend|endpoint|script|module|package|bug|error handling)\b"
            ),
            "website-design-best-practices.md": re.compile(
                r"\b(website|web\s*site|web\s*page|web\s*app|landing\s*page|responsive|layout|navigation|navbar|footer|homepage|frontend|html|css|tailwind|seo|accessibility|a11y|mobile\s*first)\b"
            ),
            "graphic-design-best-practices.md": re.compile(
                r"\b(graphic\s*design|logo|brand|icon|illustration|composition|typography|font|visual\s*design|poster|banner|flyer|mockup|design\s*system|ui\s*design)\b"
            ),
            "color-theory.md": re.compile(
                r"\b(color|colour|palette|complementary|analogous|triadic|monochromatic|hex|rgb|hsl|saturation|hue|contrast\s*ratio|dark\s*mode|theme|warm\s*color|cool\s*color)\b"
            ),
            "writing-best-practices.md": re.compile(
                r"\b(writ(e|ing|ten)|copy|blog\s*post|article|essay|documentation|readme|tone|voice|editing|proofread|grammar|headline|content\s*strategy|technical\s*writing|copywriting)\b"
            ),
            "api-design-best-practices.md": re.compile(
                r"\b(api|rest|endpoint|route|status\s*code|pagination|jwt|bearer|oauth|openapi|swagger|crud|http\s*method|json\s*response|rate\s*limit)\b"
            ),
            "prompt-engineering-best-practices.md": re.compile(
                r"\b(prompt|system\s*prompt|few.?shot|chain.?of.?thought|tool\s*description|function\s*calling|sub.?agent|delegat|instruct\s*the\s*model|ask\s*(claude|frontier|gemini))\b"
            ),
            "image-generation-best-practices.md": re.compile(
                r"\b(generate\s*(an?\s*)?image|image\s*gen|flux|stable\s*diffusion|dall.?e|midjourney|art\s*style|photo\s*prompt|negative\s*prompt|aspect\s*ratio|composition|portrait\s*photo|landscape\s*photo|product\s*photo|render|illustration)\b"
            ),
            "cybersecurity-best-practices.md": re.compile(
                r"\b(security|secure|vulnerab|exploit|injection|xss|csrf|auth(entication|orization)|encrypt|hash|password|firewall|tls|ssl|certificate|hardening|pentest|malware|phishing|incident\s*response|secrets?\s*manag|container\s*security|audit)\b"
            ),
            "marketing-best-practices.md": re.compile(
                r"\b(marketing|campaign|seo|social\s*media\s*strategy|email\s*marketing|newsletter|funnel|conversion|lead\s*gen|brand\s*voice|copywriting|landing\s*page|paid\s*ads|retarget|cta|call\s*to\s*action|content\s*marketing|hashtag|engagement\s*rate|click\s*rate|a/?b\s*test)\b"
            ),
            "web-design-patterns.md": re.compile(
                r"\b(modal|dialog|toast|notification|skeleton|loading\s*state|spinner|empty\s*state|infinite\s*scroll|pagination|drag.?and.?drop|sidebar|tab(s|bed)|breadcrumb|command\s*palette|dashboard|card\s*layout|dark\s*mode|design\s*token|form\s*design|wizard|stepper|table\s*design|data\s*table|micro.?interaction|component|ui\s*pattern|ux\s*pattern)\b"
            ),
        }

        matched_guides = []
        for filename, pattern in guide_triggers.items():
            if pattern.search(query):
                guide_path = guides_dir / filename
                if guide_path.exists():
                    matched_guides.append(guide_path)

        if not matched_guides:
            return None

        # Load matched guides (limit to 2 to avoid bloating the prompt)
        sections = []
        for guide_path in matched_guides[:2]:
            try:
                content = guide_path.read_text()
                sections.append(f"## Reference Guide\n{content}")
            except Exception:
                continue

        return "\n\n".join(sections) if sections else None

    @work(thread=True)
    def process_message(self, user_input: str):
        """Process message with LLM in background thread."""
        self.messages.append({"role": "user", "content": user_input})

        # Inject relevant memories into system prompt
        self._inject_memory_context(user_input)

        # Start spinner
        self.call_from_thread(self.start_spinner)

        # Fast path: direct tool dispatch without LLM reasoning
        response = None
        fast_dispatch_response = False
        try:
            result = fast_dispatch(user_input)
            if result:
                tool_name, _args, tool_result = result
                console.print(f"[dim cyan]  → {tool_name}[/dim cyan]")
                # Give the LLM just the result to summarize (no tool definitions needed)
                summary_messages = [
                    {
                        "role": "system",
                        "content": "You are MAUDE. The user asked a question and a tool was already called. Summarize the result concisely.",
                    },
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": f"I used {tool_name} and got this result:"},
                    {"role": "user", "content": f"Tool result:\n{tool_result[:3000]}\n\nSummarize this for me."},
                ]
                try:
                    sum_kwargs = dict(
                        model=MODEL,
                        messages=summary_messages,
                        temperature=0.2,
                        max_tokens=1024,
                    )
                    if MODEL not in _CLOUD_MODELS:
                        sum_kwargs["extra_body"] = {"num_ctx": NUM_CTX}
                    summary = self.client.chat.completions.create(**sum_kwargs)
                    response = summary.choices[0].message.content
                except Exception:
                    response = tool_result[:2000]
                fast_dispatch_response = True
        except Exception as e:
            console.print(f"[dim yellow]  fast dispatch skipped: {e}[/dim yellow]")

        # Auto-route: detect intent, switch model if appropriate, route to subagent
        prev_model = None
        subagent_response = False
        if not response:
            try:
                from auto_router import route_message

                decision = route_message(user_input, self.messages[-10:])

                # Per-message model auto-switch (DISABLED — models like codestral
                # don't support tool calling, which breaks task execution)
                # if decision.prefer_model:
                #     prev_model = _use_model_for_message(decision.prefer_model)
                #     if prev_model != MODEL:
                #         console.print(f"[dim cyan]  model: {decision.prefer_model} (auto)[/dim cyan]")

                # Route to subagent if confident
                if decision.subagent and decision.confidence >= 0.5:
                    cloud_tag = " [cloud]" if decision.prefer_cloud else ""
                    console.print(
                        f"[dim cyan]  route: {decision.intent} → {decision.subagent} ({decision.confidence:.0%}){cloud_tag}[/dim cyan]"
                    )
                    from execution import execute_subagent

                    response = execute_subagent(decision.subagent, user_input, prefer_cloud=decision.prefer_cloud)
                    subagent_response = True
            except ImportError:
                pass
            except Exception as e:
                console.print(f"[dim yellow]  routing skipped: {e}[/dim yellow]")

        # Display non-streamed responses (fast_dispatch or subagent)
        if response and (fast_dispatch_response or subagent_response):
            self.call_from_thread(self.write_output, f"[bold magenta]MAUDE:[/bold magenta] {response}")

        # Fall back to main tool-calling chat loop
        if not response:
            response = chat(self.client, self.messages)

        # Restore model after per-message auto-switch
        if prev_model is not None:
            _restore_model(prev_model)

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
            conversation_sync.save_conversation(self.conv_id, self.conv_title, MODEL, self.messages)
            # Emit activity event
            self._collab_activity = f"chatting: {user_input[:40]}"
            try:
                get_collab_hub().emit(
                    "chat",
                    f"Asked about: {user_input[:50]}",
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
        import tempfile

        import uvicorn
        from fastapi import FastAPI, File, UploadFile
        from fastapi.responses import JSONResponse

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
                except Exception as exc:
                    raise RuntimeError("No Whisper available") from exc
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
