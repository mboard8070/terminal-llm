#!/usr/bin/env python3
"""
MAUDE VISION - Terminal LLM Chat with image understanding via Ollama + LLaVA
"""

import sys
import os
import time
import json
import base64
from pathlib import Path
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.align import Align
import pyfiglet
from PIL import Image

console = Console()

# Working directory for file operations
working_dir = Path.home()

# Ollama server with LLaVA
OLLAMA_URL = "http://localhost:11434/v1"
MODEL = "llava:13b"

# Color palette for animation - purple/blue gradient for vision
COLORS = ["purple", "blue", "cyan", "bright_cyan", "bright_blue", "magenta"]


def encode_image(image_path: str) -> tuple[str, str]:
    """
    Encode an image file to base64.
    Returns (base64_data, media_type) tuple.
    """
    path = Path(image_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }

    if suffix not in media_types:
        raise ValueError(f"Unsupported image format: {suffix}. Use jpg, png, gif, or webp.")

    media_type = media_types[suffix]

    # Validate it's actually an image
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")

    with open(path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    return image_data, media_type


def resolve_path(path_str: str) -> Path:
    """Resolve a path relative to working directory."""
    global working_dir
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = working_dir / path
    return path.resolve()


def create_client():
    """Create Ollama API client."""
    return OpenAI(
        base_url=OLLAMA_URL,
        api_key="ollama"
    )


def vision_text(text: str, offset: int = 0) -> Text:
    """Create vision-colored text (purple -> cyan gradient)."""
    result = Text()
    for i, char in enumerate(text):
        if char.strip():
            color = COLORS[(i + offset) % len(COLORS)]
            result.append(char, style=f"bold {color}")
        else:
            result.append(char)
    return result


def animate_banner():
    """Display animated MAUDE VISION banner."""
    banner = pyfiglet.figlet_format("MAUDE", font="banner3")
    vision_label = pyfiglet.figlet_format("VISION", font="banner3")

    banner_lines = banner.rstrip('\n').split('\n')
    vision_lines = vision_label.rstrip('\n').split('\n')

    with Live(console=console, refresh_per_second=12, transient=True) as live:
        for frame in range(25):
            content = Text()
            for line in banner_lines:
                content.append_text(vision_text(line, frame))
                content.append('\n')
            for line in vision_lines:
                content.append_text(vision_text(line, frame + 2))
                content.append('\n')
            live.update(Align.center(content))
            time.sleep(0.06)

    final_content = Text()
    for line in banner_lines:
        final_content.append_text(vision_text(line, 24))
        final_content.append('\n')
    for line in vision_lines:
        final_content.append_text(vision_text(line, 26))
        final_content.append('\n')

    console.print(Align.center(final_content))
    console.print()


def print_separator():
    """Print a styled separator."""
    console.print("─" * console.width, style="dim magenta")


def chat(client, messages: list):
    """Send chat request to Ollama LLaVA with streaming."""
    try:
        console.print("[bold magenta]thinking...[/bold magenta]", end="\r")
        start_time = time.time()

        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=4096,
            stream=True
        )

        full_content = ""
        token_count = 0
        first_token = True

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                if first_token:
                    console.print(" " * 40, end="\r")
                    console.print(f"[bold magenta]MAUDE:[/bold magenta] ", end="")
                    first_token = False
                print(delta.content, end="", flush=True)
                full_content += delta.content
                token_count += 1
                console.set_window_title(f"MAUDE VISION - {token_count} tokens")

        elapsed_time = time.time() - start_time
        if full_content:
            print()
            tokens_per_sec = token_count / elapsed_time if elapsed_time > 0 else 0
            console.print(f"[dim]{token_count} tokens in {elapsed_time:.1f}s ({tokens_per_sec:.1f} tok/s)[/dim]")
        elif first_token:
            console.print(" " * 40, end="\r")

        return full_content

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Is Ollama running? Start with: ollama serve[/dim]")
        return None


def main():
    """Main chat loop with vision support."""
    console.clear()
    animate_banner()

    console.print(Panel(
        "[dim]LLaVA 13B via Ollama | Vision-Language Model[/dim]\n"
        "[magenta]/image <path>[/magenta] [dim]to load an image |[/dim] [green]quit[/green] [dim]to leave[/dim]",
        border_style="magenta",
        title="[bold magenta]MAUDE VISION[/bold magenta]",
        title_align="center"
    ))
    print_separator()
    console.print()

    try:
        client = create_client()
    except Exception as e:
        console.print(f"[red]Failed to connect: {e}[/red]")
        console.print("[dim]Make sure Ollama is running: ollama serve[/dim]")
        sys.exit(1)

    messages = []
    pending_image = None

    system_prompt = {
        "role": "system",
        "content": """You are MAUDE VISION, an AI assistant with the ability to see and understand images.
You can analyze photos, screenshots, diagrams, charts, documents, and any visual content.
When shown an image, describe what you see accurately and answer questions about it.
Be concise but thorough. If you're uncertain about something in an image, say so."""
    }
    messages.append(system_prompt)

    while True:
        try:
            console.print()

            if pending_image:
                user_input = console.input("[bold green]YOU[/bold green] [magenta][+img][/magenta][bold green] >[/bold green] ").strip()
            else:
                user_input = console.input("[bold green]YOU >[/bold green] ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "bye", "goodbye"):
                console.print("\n[dim magenta]MAUDE VISION signing off. Until next time.[/dim magenta]\n")
                break

            if user_input.lower() == "clear":
                messages = [system_prompt]
                pending_image = None
                console.print("[dim]Conversation cleared.[/dim]")
                continue

            # Handle /image command
            if user_input.lower().startswith("/image"):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    console.print("[dim]Usage: /image <path_to_image>[/dim]")
                    continue

                image_path = parts[1].strip()
                try:
                    pending_image = encode_image(image_path)
                    console.print(f"[dim magenta]Image loaded: {image_path}[/dim magenta]")
                    console.print("[dim]Now type your question about the image.[/dim]")
                except (FileNotFoundError, ValueError) as e:
                    console.print(f"[red]Error: {e}[/red]")
                    pending_image = None
                continue

            # Build message with or without image
            if pending_image:
                base64_data, media_type = pending_image
                user_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{base64_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": user_input
                        }
                    ]
                }
                pending_image = None
            else:
                user_message = {"role": "user", "content": user_input}

            messages.append(user_message)
            console.print()
            response = chat(client, messages)

            if response:
                messages.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            console.print("\n[dim]Say \"quit\" to exit[/dim]")
        except EOFError:
            break


if __name__ == "__main__":
    main()
