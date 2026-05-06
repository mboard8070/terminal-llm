#!/usr/bin/env python3
"""
MAUDE Lite — Terminal chat with full tool visibility and native copy/paste.

Uses Rich for formatted output, prompt_toolkit for input, and raw httpx SSE
for streaming so we can display gateway tool traces (like Claude Code does).

Normal terminal scrollback — select text and copy like any other CLI.
"""

import json
import sys
import threading
import time
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.panel import Panel

import conversation_sync
from collab import get_hub as get_collab_hub

# ── Config ────────────────────────────────────────────────────────────────

env_path = Path(__file__).parent / "variables.env"
load_dotenv(env_path)

GATEWAY_URL = "http://localhost:30080/v1"
HISTORY_FILE = Path.home() / ".config" / "maude" / "chat_lite_history"

MODELS = {
    "nemotron": "nemotron",
    "nemotron-super": "nemotron-super",
    "llava": "llava",
    "mistral": "mistral-large-latest",
    "codestral": "codestral-latest",
    "devstral": "devstral-2512",
    "devstral-small": "devstral-small-latest",
    "devstral-medium": "devstral-medium-latest",
    "openai": "openai",
    "codex": "codex",
    "gemma4": "gemma-4-31b",
    "gemma": "gemma-4-31b",
    "claude": "claude-opus-4-20250514",
    "sonnet": "claude-sonnet-4-20250514",
}

DEFAULT_MODEL = "nemotron-super"

# ── Globals ───────────────────────────────────────────────────────────────

console = Console()
_last_response = ""

# ── Streaming with tool traces ────────────────────────────────────────────


def stream_response(messages: list, model_id: str) -> str:
    """Stream via raw httpx SSE to capture gateway tool traces."""
    global _last_response

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 4096,
        "stream": True,
    }

    full_content = ""
    first_token = True
    prompt_tokens = 0
    token_count = 0
    start_time = time.time()
    thinking = True

    # Show thinking spinner until first content arrives
    spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    spinner_idx = [0]
    stop_spinner = threading.Event()

    def _spinner():
        while not stop_spinner.is_set():
            frame = spinner_chars[spinner_idx[0] % len(spinner_chars)]
            print(f"\r  {frame} thinking...", end="", flush=True)
            spinner_idx[0] += 1
            stop_spinner.wait(0.1)
        # Clear spinner line
        print("\r" + " " * 30 + "\r", end="", flush=True)

    spinner_thread = threading.Thread(target=_spinner, daemon=True)

    try:
        with (
            httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as http,
            http.stream(
                "POST",
                f"{GATEWAY_URL}/chat/completions",
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as resp,
        ):
            if resp.status_code >= 400:
                error_text = resp.read().decode()
                console.print(f"\n[red]Gateway error {resp.status_code}: {error_text[:200]}[/red]\n")
                return ""

            spinner_thread.start()
            buf = ""

            for raw_bytes in resp.iter_bytes():
                buf += raw_bytes.decode("utf-8", errors="replace")

                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    # ── Tool trace comments from gateway ──
                    if line.startswith(": trace "):
                        try:
                            trace = json.loads(line[8:])
                            ttype = trace.get("type", "")

                            if ttype == "tool_call":
                                # Stop spinner on first tool call
                                if thinking:
                                    stop_spinner.set()
                                    thinking = False

                                tname = trace.get("name", "?")
                                targs = trace.get("args", "")
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
                                            "platform",
                                            "workflow_id",
                                        ):
                                            if k in parsed:
                                                arg_hint = str(parsed[k])
                                                if len(arg_hint) > 60:
                                                    arg_hint = arg_hint[:60] + "…"
                                                break
                                except (json.JSONDecodeError, TypeError):
                                    if targs and targs != "{}" and len(targs) <= 60:
                                        arg_hint = targs

                                console.print(f"  [bold cyan]╭─[/bold cyan] [bold white]{tname}[/bold white]")
                                if arg_hint:
                                    console.print(f"  [cyan]│[/cyan]  [dim]{arg_hint}[/dim]")

                            elif ttype == "tool_result":
                                tname = trace.get("name", "")
                                preview = trace.get("preview", "")
                                elapsed = trace.get("elapsed", 0)
                                color = "green" if not preview.startswith("Error") else "red"
                                console.print(
                                    f"  [cyan]╰─[/cyan] [{color}]{preview}[/{color}] [dim]({elapsed:.1f}s)[/dim]"
                                )

                            elif ttype == "llm_call":
                                prompt_tokens += trace.get("prompt_tokens", 0)
                                token_count += trace.get("completion_tokens", 0)

                            elif ttype == "error":
                                if thinking:
                                    stop_spinner.set()
                                    thinking = False
                                err_msg = trace.get("message", "unknown error")
                                console.print(f"  [red]✗ {err_msg}[/red]")

                        except (json.JSONDecodeError, KeyError):
                            pass
                        continue

                    # ── Keepalive pings ──
                    if line.startswith(": keepalive") or line.startswith(": ping"):
                        continue

                    # ── Normal SSE data lines ──
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
                                    if first_token:
                                        stop_spinner.set()
                                        thinking = False
                                        console.print()
                                        console.print("[bold magenta]MAUDE[/bold magenta]")
                                        first_token = False
                                    print(content, end="", flush=True)
                                    full_content += content

                            usage = chunk.get("usage")
                            if usage:
                                token_count = usage.get("completion_tokens", 0) or 0
                                prompt_tokens = usage.get("prompt_tokens", 0) or 0
                        except json.JSONDecodeError:
                            pass

        # Stop spinner if still running
        stop_spinner.set()

        elapsed = time.time() - start_time

        if full_content:
            print()  # newline after streamed content
            console.print(f"[dim]{prompt_tokens}+{token_count} tokens · {elapsed:.1f}s[/dim]")
            console.print()

        _last_response = full_content
        return full_content

    except httpx.ConnectError:
        stop_spinner.set()
        console.print("\n[red]Cannot connect to gateway at localhost:30080. Is it running?[/red]\n")
        return ""
    except Exception as e:
        stop_spinner.set()
        console.print(f"\n[red]Error: {e}[/red]\n")
        return ""


# ── Commands ──────────────────────────────────────────────────────────────


def handle_command(cmd: str, messages: list, current_model: list, conv_id: list, conv_title: list) -> bool:
    """Handle slash commands. Returns True if handled."""
    parts = cmd.strip().split()
    command = parts[0].lower()

    if command in ("/quit", "/exit", "/q"):
        console.print("[dim]Goodbye.[/dim]")
        sys.exit(0)

    if command == "/clear":
        messages.clear()
        messages.append(_system_prompt())
        conv_id[0] = str(uuid.uuid4())
        conv_title[0] = ""
        console.print("[dim]Conversation cleared.[/dim]\n")
        return True

    if command == "/model":
        if len(parts) > 2 and parts[1] == "switch" and parts[2] in MODELS:
            current_model[0] = parts[2]
            console.print(f"[dim]Model: {parts[2]} ({MODELS[parts[2]]})[/dim]\n")
        elif len(parts) > 1 and parts[1] == "switch":
            console.print(f"[dim]Usage: /model switch <name>\nAvailable: {', '.join(MODELS.keys())}[/dim]\n")
        else:
            console.print(f"[dim]Usage: /model switch <name>\nAvailable: {', '.join(MODELS.keys())}[/dim]\n")
        return True

    if command == "/copy":
        if not _last_response:
            console.print("[dim]Nothing to copy.[/dim]\n")
            return True
        for clip_cmd in ["xclip -selection clipboard", "xsel --clipboard", "pbcopy", "wl-copy"]:
            try:
                import subprocess

                proc = subprocess.Popen(
                    clip_cmd.split(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                proc.communicate(input=_last_response.encode())
                if proc.returncode == 0:
                    console.print("[dim]Copied to clipboard.[/dim]\n")
                    return True
            except Exception:
                continue
        p = Path.home() / ".config" / "maude" / "last_response.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_last_response)
        console.print(f"[dim]Saved to {p}[/dim]\n")
        return True

    if command == "/help":
        console.print(
            Panel(
                "[bold]/quit[/bold]          Exit\n"
                "[bold]/clear[/bold]         Clear conversation\n"
                "[bold]/model switch[/bold] <name>  Switch model\n"
                "[bold]/copy[/bold]          Copy last response\n"
                "[bold]/help[/bold]          This help",
                title="Commands",
                border_style="dim",
            )
        )
        console.print()
        return True

    return False


def _system_prompt() -> dict:
    """Minimal system prompt — the gateway injects the full one."""
    return {
        "role": "system",
        "content": "You are MAUDE, a capable AI assistant. Be concise and helpful.",
    }


# ── Key Bindings ──────────────────────────────────────────────────────────


def make_keybindings():
    """prompt_toolkit keybindings: Enter submits, Alt+Enter for newline."""
    kb = KeyBindings()

    @kb.add("escape", "enter")
    def _(event):
        event.current_buffer.insert_text("\n")

    return kb


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    session = PromptSession(
        history=FileHistory(str(HISTORY_FILE)),
        auto_suggest=AutoSuggestFromHistory(),
        key_bindings=make_keybindings(),
        multiline=False,
        enable_history_search=True,
    )

    messages = [_system_prompt()]
    current_model = [DEFAULT_MODEL]
    conv_id = [str(uuid.uuid4())]
    conv_title = [""]
    collab_activity = ["idle"]

    # Heartbeat
    def _heartbeat():
        hub = get_collab_hub()
        while True:
            try:
                hub.heartbeat("cli-lite", "cli", collab_activity[0], conv_id[0])
            except Exception:
                pass
            time.sleep(30)

    threading.Thread(target=_heartbeat, daemon=True).start()

    # Banner
    console.print()
    console.print(
        Panel.fit(
            "[bold magenta]MAUDE[/bold magenta]\n"
            "\n"
            "[dim]Files · Shell · Web · Gmail · Drive · Calendar · YouTube · Substack\n"
            "Browser · Social · Vision · Voice · Cross-machine[/dim]\n"
            "\n"
            f"[dim grey50]/model switch <name>  /clear  /copy  /help[/dim grey50]",
            border_style="magenta",
        )
    )
    console.print()

    while True:
        try:
            user_input = session.prompt(
                HTML("<ansigreen><b>maude</b></ansigreen> <ansigray>›</ansigray> "),
            ).strip()

            if not user_input:
                continue

            if user_input.startswith("/") and handle_command(user_input, messages, current_model, conv_id, conv_title):
                continue

            messages.append({"role": "user", "content": user_input})
            collab_activity[0] = f"chatting: {user_input[:40]}"

            response = stream_response(messages, MODELS[current_model[0]])

            if response:
                messages.append({"role": "assistant", "content": response})
                if not conv_title[0]:
                    conv_title[0] = conversation_sync.generate_title(user_input)
                conversation_sync.save_conversation(
                    conv_id[0],
                    conv_title[0],
                    current_model[0],
                    messages,
                )
                try:
                    get_collab_hub().emit(
                        "chat",
                        f"Asked about: {user_input[:50]}",
                        data={"model": current_model[0]},
                        client_id="cli-lite",
                        conversation_id=conv_id[0],
                    )
                except Exception:
                    pass

        except KeyboardInterrupt:
            console.print("\n[dim]Ctrl+C — /quit to exit[/dim]\n")
        except EOFError:
            break


if __name__ == "__main__":
    main()
