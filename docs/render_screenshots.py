#!/usr/bin/env python3
"""Render MAUDE screenshot mockups using Rich → SVG → PNG."""
import os
os.environ["TERM"] = "xterm-256color"

from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
import cairosvg
from pathlib import Path

OUT = Path(__file__).parent / "screenshots"
OUT.mkdir(exist_ok=True)

COLORS = ["red", "bright_red", "orange1", "orange3", "yellow", "bright_yellow"]

def fire_text(text, offset=12):
    result = Text()
    for i, ch in enumerate(text):
        if ch.strip():
            color = COLORS[(i + offset) % len(COLORS)]
            result.append(ch, style=f"bold {color}")
        else:
            result.append(ch)
    return result

def save(console, name):
    svg = console.export_svg(title="MAUDE")
    svg_path = OUT / f"{name}.svg"
    png_path = OUT / f"{name}.png"
    svg_path.write_text(svg)
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png_path), scale=2.0)
    svg_path.unlink()
    print(f"  saved {png_path}")


# ── 1. Server TUI ──────────────────────────────────────────────

print("Rendering server TUI...")
c = Console(record=True, width=100, force_terminal=True, color_system="truecolor")

banner = [
    "  __  __    _    _   _ ____  _____  ",
    " |  \\/  |  / \\  | | | |  _ \\| ____| ",
    " | |\\/| | / _ \\ | | | | | | |  _|   ",
    " | |  | |/ ___ \\| |_| | |_| | |___  ",
    " |_|  |_/_/   \\_\\\\___/|____/|_____| ",
]
for line in banner:
    c.print(fire_text(line))
c.print()
c.print("[dim]nemotron | Files | Shell | Web | Vision[/dim]")
c.print("[dim]Type /help for commands, 'quit' to exit[/dim]")
c.print()

# User message
c.print("[bold green]YOU >[/bold green] what's the weather like in austin today?")
c.print()

# Tool trace
c.print("[dim cyan]  route: weather_query → nemotron (92%)[/dim cyan]")
c.print("[dim yellow]  [web_search] {\"query\": \"weather austin texas today\"}[/dim yellow]")
c.print("[dim]    → Austin, TX: 78°F, partly cloudy. High of 82°F... (1.3s)[/dim]")
c.print("[dim yellow]  [web_browse] {\"url\": \"https://weather.gov/austin\"}[/dim yellow]")
c.print("[dim]    → Current conditions: 78°F, humidity 45%, wind S 12mph... (2.1s)[/dim]")
c.print()

# MAUDE response
c.print("[bold magenta]MAUDE:[/bold magenta] It's a nice day in Austin! Currently 78°F and partly cloudy "
        "with humidity around 45%. The high today should reach about 82°F with "
        "south winds around 12 mph. Great weather to be outside!")
c.print("[dim]1847+312 tokens in 4.2s[/dim]")
c.print()

# Second message with model switch
c.print("[bold green]YOU >[/bold green] /model switch mistral")
c.print()
c.print("Switched to mistral (mistral-large-latest) via gateway")
c.print()

c.print("[bold green]YOU >[/bold green] explain how neural attention works")
c.print()
c.print("[bold magenta]MAUDE:[/bold magenta] Neural attention is a mechanism that allows a model to "
        "dynamically focus on different parts of its input when producing each "
        "element of its output. Think of it as a spotlight that the model can "
        "direct at the most relevant information for each step...")
c.print("[dim]2450+523 tokens in 3.1s[/dim]")

save(c, "server-tui")


# ── 2. Client CLI ──────────────────────────────────────────────

print("Rendering client CLI...")
c = Console(record=True, width=90, force_terminal=True, color_system="truecolor")

c.print("[bold cyan]MAUDE client v1.2.0[/bold cyan]")
c.print(f"[dim]Server: spark-e26c:30000 | Model: mistral-large-latest[/dim]")
c.print(f"[dim]Shared: ~/.maude/shared/ (syncing every 30s)[/dim]")
c.print()
c.print("Type 'quit' to exit, '/help' for commands.\n")

# First message with trace
c.print("You: search the web for DGX Spark specs")
c.print()
c.print("[dim]  [1200+180 tokens, 1.8s][/dim]")
c.print("[dim]  [web_search] {\"query\": \"NVIDIA DGX Spark specifications\"}[/dim]")
c.print("[dim]    → NVIDIA DGX Spark: Grace Blackwell, 128GB unified memory... (2.4s)[/dim]")
c.print("[dim]  [web_browse] {\"url\": \"https://nvidia.com/dgx-spark\"}[/dim]")
c.print("[dim]    → DGX Spark is a personal AI supercomputer powered by... (1.7s)[/dim]")
c.print("[dim]  [3200+490 tokens, 2.9s][/dim]")
c.print()
c.print("MAUDE: The NVIDIA DGX Spark is a compact personal AI computer featuring:")
c.print("  - Grace Blackwell architecture (GB10 Superchip)")
c.print("  - 128GB unified LPDDR5x memory")
c.print("  - Up to 1 petaflop of AI performance")
c.print("  - NVLink-C2C interconnect")
c.print("  - Compact desktop form factor")
c.print()

# Second message
c.print("You: list my shared files")
c.print()
c.print("MAUDE: ")
c.print("  [list_shared]")
c.print("  Shared folder (/home/mboard76/nvidia-workbench/terminal-llm/shared/):")
c.print("  [FILE] notes.txt (1,234 bytes)")
c.print("  [FILE] diagram.png (45,678 bytes)")
c.print()

# Third message with cached result
c.print("You: search the web for DGX Spark specs again")
c.print()
c.print("[dim]  [web_search] {\"query\": \"NVIDIA DGX Spark specifications\"}[/dim]")
c.print("[dim]    → [cached result] (0.0s)[/dim]")

save(c, "client-cli")


# ── 3. Android App ─────────────────────────────────────────────

print("Rendering Android app...")
c = Console(record=True, width=50, force_terminal=True, color_system="truecolor")

# Phone-style header
c.print(Panel(
    "[bold white]MAUDE[/bold white]  [dim]mistral-large-latest[/dim]",
    style="bold cyan",
    width=48
))

c.print()

# Chat bubbles - user
c.print("[bold green]You[/bold green]")
c.print(Panel(
    "Can you analyze the photo I just took?",
    style="green",
    width=40,
    padding=(0, 1)
))

c.print()

# Trace
c.print("[dim]  [view_image] analyzing photo...[/dim]")
c.print("[dim]    → The image shows a circuit board with... (3.2s)[/dim]")
c.print()

# Chat bubbles - MAUDE
c.print("[bold magenta]MAUDE[/bold magenta]")
c.print(Panel(
    "I can see a Raspberry Pi 5 board in your photo! "
    "It has the Broadcom BCM2712 SoC, dual micro-HDMI "
    "ports, USB-C power, and the new PCIe connector. "
    "The GPIO header is along the top edge.",
    style="magenta",
    width=44,
    padding=(0, 1)
))

c.print()

c.print("[bold green]You[/bold green]")
c.print(Panel(
    "What's the weather like today?",
    style="green",
    width=34,
    padding=(0, 1)
))

c.print()
c.print("[dim]  [web_search] weather today (1.1s)[/dim]")
c.print()

c.print("[bold magenta]MAUDE[/bold magenta]")
c.print(Panel(
    "It's currently 72°F and sunny. "
    "Perfect day to tinker with that Pi!",
    style="magenta",
    width=40,
    padding=(0, 1)
))

c.print()
# Input bar
c.print(Panel(
    "[dim]Type a message...[/dim]                📷",
    style="dim",
    width=48,
    padding=(0, 1)
))

save(c, "android-app")

print("\nDone! Screenshots saved to docs/screenshots/")
