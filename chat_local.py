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
from maude_core import TOOLS, execute_tool, reset_rate_limits, append_chat_log, read_chat_log_since

# Minimal imports
from keys import KeyManager

# Global reference to app for output
_app = None


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


def chat(client, messages: list):
    """Send chat request to local LLM with tool support."""
    global show_thinking

    max_tool_iterations = 15
    tool_iteration = 0
    recent_tool_calls = []
    reset_rate_limits()  # Reset per-turn limits in maude_core

    while True:
        tool_iteration += 1
        if tool_iteration > max_tool_iterations:
            console.print("[dim yellow](max tool iterations reached)[/dim yellow]")
            break
        try:
            start_time = time.time()

            # Non-streaming request for TUI compatibility
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=4096,
                tools=TOOLS,
                tool_choice="auto",
                extra_body={"num_ctx": NUM_CTX}
            )

            elapsed_time = time.time() - start_time
            msg = response.choices[0].message
            full_content = msg.content or ""

            # Show response
            if full_content:
                console.print(f"[bold magenta]MAUDE:[/bold magenta] {full_content}")
                token_count = response.usage.completion_tokens if response.usage else 0
                console.print(f"[dim]{token_count} tokens in {elapsed_time:.1f}s[/dim]")

            # Collect tool calls
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
                        console.print(f"[dim yellow]  [{func_name}] (skipped duplicate)[/dim yellow]")
                        result = "(Already called with same arguments - see previous result)"
                    else:
                        recent_tool_calls.append(call_signature)
                        console.print(f"[dim yellow]  [{func_name}][/dim yellow]")
                        result = execute_tool(func_name, func_args)

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
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=[m for m in messages if "tool_calls" not in m and m.get("role") != "tool"],
                        temperature=0.2,
                        max_tokens=1024,
                        extra_body={"num_ctx": NUM_CTX}
                    )
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

/help    - Show this help
/model   - Show current model configuration

Tools available:
- search_directory, search_file, read_file, write_file, edit_file
- list_directory, change_directory, get_working_directory
- run_command (shell)
- web_browse, web_search, web_view
- view_image
- ask_frontier (escalate to cloud AI, if configured)

Say "quit" to exit."""
    elif command == "model":
        from frontier import list_available_providers
        frontier_providers = list_available_providers()
        frontier_info = ", ".join(frontier_providers) if frontier_providers else "none configured"
        return f"""Model Configuration:

Local LLM:    {MODEL}
Server:       {LOCAL_URL}
Context:      {NUM_CTX} tokens

Vision:       {VISION_MODEL}
Vision URL:   {VISION_URL}

Frontier:     {frontier_info}

Set MAUDE_MODEL env var to use a different local model."""
    else:
        return f"Unknown command: /{command}\nType /help for available commands."


SYSTEM_PROMPT = f"""You are MAUDE, a local coding assistant running {MODEL}.

STYLE: Be brief. Action over explanation.

TOOLS:
- search_directory(directory, pattern): Search across all files in a project - use when you don't know which file
- search_file(path, pattern): Search within a single file
- read_file(path, start_line, end_line): Read file with line numbers
- write_file(path, content): Create/overwrite file
- edit_file(path, start_line, end_line, new_content): Edit specific lines
- list_directory, change_directory, get_working_directory
- run_command: Shell commands (git, pip, python, etc.)
- web_browse, web_search: Web access
- web_view, view_image: Visual analysis (LLaVA)
- ask_frontier(question, context, provider): Escalate to cloud AI (optional, if configured)

ESCALATION TO FRONTIER (optional):
If frontier APIs are configured, use ask_frontier when:
- Complex multi-step reasoning or architecture decisions needed
- You're uncertain about the correct approach
- Deep domain expertise required (security, performance, algorithms)
If no frontier APIs are available, do your best with local capabilities.

FILE EDITING WORKFLOW:
1. search_directory to find which file contains the code
2. read_file with line range to see context
3. edit_file to replace lines

Use tools proactively. Confirm before destructive operations."""


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
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.client = None
        self.spinner_frame = 0
        self.spinner_timer = None
        self.thinking_line_count = 0
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

    def write_output(self, text):
        """Write to the output log."""
        if hasattr(self, 'output_log'):
            self.output_log.write(text)

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

        response = chat(self.client, self.messages)

        # Stop spinner
        self.call_from_thread(self.stop_spinner)

        if response:
            self.messages.append({"role": "assistant", "content": response})
            # Log for sync (after processing complete)
            append_chat_log("cli", "user", user_input)
            append_chat_log("cli", "assistant", response)


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


def main():
    # Check if Telegram should be enabled
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")

    if telegram_token:
        # Start Telegram bot in background thread
        telegram_thread = threading.Thread(target=run_telegram_in_background, daemon=True)
        telegram_thread.start()

    app = MaudeApp()
    app.run()


if __name__ == "__main__":
    main()
