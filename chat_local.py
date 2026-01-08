#!/usr/bin/env python3
"""
MAUDE CODE - Terminal LLM Chat powered by Nemotron-3-Nano-30B
"""

import sys
import time
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.align import Align
import pyfiglet

console = Console()

# Local llama.cpp server
LOCAL_URL = "http://localhost:30000/v1"
MODEL = "nemotron"

# Toggle for showing reasoning
show_thinking = False

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
    """Generate banner text with mech positioned to the right."""
    banner = pyfiglet.figlet_format("MAUDE", font="banner3")
    code_text = pyfiglet.figlet_format("CODE", font="banner3")

    banner_lines = banner.rstrip('\n').split('\n')
    code_lines = code_text.rstrip('\n').split('\n')

    return banner_lines, code_lines


def animate_banner():
    """Display animated MAUDE banner with mech."""
    banner_lines, code_lines = get_banner_with_mech()

    # Animate the banner with fire colors
    with Live(console=console, refresh_per_second=12, transient=True) as live:
        for frame in range(25):
            content = Text()

            # Add animated MAUDE banner
            for line in banner_lines:
                content.append_text(fire_text(line, frame))
                content.append('\n')

            # Add CODE subtitle
            for line in code_lines:
                content.append_text(fire_text(line, frame + 2))
                content.append('\n')

            live.update(Align.center(content))
            time.sleep(0.06)

    # Final static banner with fire gradient
    final_content = Text()
    for line in banner_lines:
        final_content.append_text(fire_text(line, 24))
        final_content.append('\n')
    for line in code_lines:
        final_content.append_text(fire_text(line, 26))
        final_content.append('\n')

    console.print(Align.center(final_content))
    console.print()


def print_separator():
    """Print a styled separator."""
    console.print("─" * console.width, style="dim cyan")


def chat(client, messages: list, stream: bool = True):
    """Send chat request to local Nemotron."""
    global show_thinking
    try:
        if stream:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=4096,
                stream=True
            )

            full_response = ""
            reasoning = ""
            in_reasoning = False
            console.print("[bold magenta]MAUDE:[/bold magenta] ", end="")

            for chunk in response:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning += delta.reasoning_content
                    if show_thinking:
                        if not in_reasoning:
                            console.print("\n[dim cyan]<thinking>[/dim cyan]", end="")
                            in_reasoning = True
                        print(delta.reasoning_content, end="", flush=True)
                if delta.content:
                    if in_reasoning and show_thinking:
                        console.print("[dim cyan]</thinking>[/dim cyan]\n", end="")
                        in_reasoning = False
                    print(delta.content, end="", flush=True)
                    full_response += delta.content

            if in_reasoning and show_thinking:
                console.print("[dim cyan]</thinking>[/dim cyan]", end="")
            print()
            return full_response
        else:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=4096
            )
            msg = response.choices[0].message
            if show_thinking and hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                console.print(f"[dim cyan]<thinking>{msg.reasoning_content}</thinking>[/dim cyan]")
            return msg.content

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Is the server running? Start with: ./start_server.sh[/dim]")
        return None


def main():
    """Main chat loop."""
    console.clear()
    animate_banner()

    # Info panel
    console.print(Panel(
        "[dim]Powered by Nemotron-3-Nano-30B | llama.cpp on GB10 | 8K Context[/dim]\n"
        "[dim cyan]/quit[/dim cyan] exit  [dim cyan]/clear[/dim cyan] reset  [dim cyan]/think[/dim cyan] toggle reasoning",
        border_style="cyan",
        title="[bold cyan]MAUDE CODE[/bold cyan]",
        title_align="center"
    ))
    print_separator()
    console.print()

    try:
        client = create_client()
    except Exception as e:
        console.print(f"[red]Failed to connect: {e}[/red]")
        console.print("[dim]Make sure the server is running: ./start_server.sh[/dim]")
        sys.exit(1)

    messages = []
    system_prompt = {
        "role": "system",
        "content": """You are MAUDE, an advanced AI assistant powered by NVIDIA Nemotron.
You excel at coding, reasoning, and complex problem-solving.
Be concise but thorough. Show your reasoning when helpful.
You have a calm, helpful personality with subtle wit."""
    }
    messages.append(system_prompt)

    while True:
        try:
            console.print()
            user_input = console.input("[bold green]YOU >[/bold green] ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/quit":
                console.print("\n[dim magenta]MAUDE signing off. Until next time.[/dim magenta]\n")
                break

            if user_input.lower() == "/clear":
                messages = [system_prompt]
                console.clear()
                animate_banner()
                console.print(Panel(
                    "[dim]Powered by Nemotron-3-Nano-30B | llama.cpp on GB10 | 8K Context[/dim]\n"
                    "[dim cyan]/quit[/dim cyan] exit  [dim cyan]/clear[/dim cyan] reset  [dim cyan]/think[/dim cyan] toggle reasoning",
                    border_style="cyan",
                    title="[bold cyan]MAUDE CODE[/bold cyan]",
                    title_align="center"
                ))
                print_separator()
                console.print("[dim]Memory cleared.[/dim]")
                continue

            if user_input.lower() == "/think":
                global show_thinking
                show_thinking = not show_thinking
                status = "[green]ON[/green]" if show_thinking else "[red]OFF[/red]"
                console.print(f"[dim]Reasoning display: {status}[/dim]")
                continue

            messages.append({"role": "user", "content": user_input})
            console.print()
            response = chat(client, messages)

            if response:
                messages.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            console.print("\n[dim]Press /quit to exit[/dim]")
        except EOFError:
            break


if __name__ == "__main__":
    main()
