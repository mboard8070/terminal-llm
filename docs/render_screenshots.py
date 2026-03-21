#!/usr/bin/env python3
"""Render realistic MAUDE screenshots."""

import asyncio
import os

os.environ["TERM"] = "xterm-256color"

from pathlib import Path

import cairosvg

OUT = Path(__file__).parent / "screenshots"
OUT.mkdir(exist_ok=True)


def svg_to_png(svg_str, path, scale=2.0):
    cairosvg.svg2png(bytestring=svg_str.encode("utf-8"), write_to=str(path), scale=scale)
    print(f"  saved {path}  ({path.stat().st_size // 1024}KB)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1.  SERVER TUI — real Textual app via async pilot
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("Rendering server TUI...")

import pyfiglet
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Input, RichLog, Static

COLORS = ["red", "bright_red", "orange1", "orange3", "yellow", "bright_yellow"]


def fire_text(text, offset=24):
    result = Text()
    for i, ch in enumerate(text):
        if ch.strip():
            result.append(ch, style=f"bold {COLORS[(i + offset) % len(COLORS)]}")
        else:
            result.append(ch)
    return result


class MockTUI(App):
    TITLE = "MAUDE"
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

    def compose(self) -> ComposeResult:
        yield RichLog(id="output", wrap=True, highlight=True, markup=True)
        yield Static("", id="status")
        with Container(id="input-container"):
            yield Input(placeholder="Type message... (quit to exit)", id="user-input")

    def fill_content(self):
        log = self.query_one("#output", RichLog)

        # Fire banner
        banner = pyfiglet.figlet_format("MAUDE", font="banner3")
        for line in banner.rstrip("\n").split("\n"):
            log.write(fire_text(line))
        log.write("")
        log.write("[dim]nemotron | Files | Shell | Web | Vision[/dim]")
        log.write("[dim]Type /help for commands, 'quit' to exit[/dim]\n")

        # Conversation
        log.write("[bold green]YOU >[/bold green] what's the weather like in austin today?\n")
        log.write("[dim cyan]  route: weather_query → nemotron (92%)[/dim cyan]")
        log.write('[dim yellow]  \\[web_search] {"query": "weather austin texas today"}[/dim yellow]')
        log.write("[dim]    → Austin, TX: 78°F, partly cloudy. High of 82°F... (1.3s)[/dim]")
        log.write('[dim yellow]  \\[web_browse] {"url": "https://weather.gov/austin"}[/dim yellow]')
        log.write("[dim]    → Current conditions: 78°F, humidity 45%, wind S 12mph... (2.1s)[/dim]\n")

        resp = Text()
        resp.append("MAUDE: ", style="bold magenta")
        resp.append(
            "It's a nice day in Austin! Currently 78°F and partly cloudy "
            "with humidity around 45%. The high today should reach about 82°F "
            "with south winds around 12 mph. Great weather to be outside!"
        )
        log.write(resp)
        log.write("[dim]1847+312 tokens in 4.2s[/dim]\n")

        log.write("[bold green]YOU >[/bold green] /model switch mistral\n")
        log.write("Switched to mistral (mistral-large-latest) via [bold cyan]gateway[/bold cyan]\n")

        log.write("[bold green]YOU >[/bold green] explain how neural attention works\n")

        resp2 = Text()
        resp2.append("MAUDE: ", style="bold magenta")
        resp2.append(
            "Neural attention is a mechanism that allows a model to dynamically "
            "focus on different parts of its input when producing each element of "
            "its output. Think of it as a learned spotlight — at every generation "
            "step the model scores how relevant each input token is, then reads a "
            "weighted mixture of their representations..."
        )
        log.write(resp2)
        log.write("[dim]2450+523 tokens in 3.1s[/dim]")


async def take_tui_screenshot():
    app = MockTUI()
    async with app.run_test(size=(110, 42)) as pilot:
        app.fill_content()
        await pilot.pause()
        svg = app.export_screenshot()
        svg_to_png(svg, OUT / "server-tui.png", scale=2.0)


asyncio.run(take_tui_screenshot())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2.  CLIENT CLI — Rich console export
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("Rendering client CLI...")

from rich.console import Console

c = Console(record=True, width=96, force_terminal=True, color_system="truecolor")

c.print("[bold cyan]MAUDE client v1.2.0[/bold cyan]")
c.print("[dim]Server: spark-e26c:30000 | Model: mistral-large-latest[/dim]")
c.print("[dim]Shared: ~/.maude/shared/ (syncing every 30s)[/dim]")
c.print()
c.print("Type 'quit' to exit, '/help' for commands.\n")

c.print("[bold]You:[/bold] search the web for DGX Spark specs\n")
c.print("[dim]  \\[1200+180 tokens, 1.8s][/dim]")
c.print('[dim]  \\[web_search] {"query": "NVIDIA DGX Spark specifications"}[/dim]')
c.print("[dim]    → NVIDIA DGX Spark: Grace Blackwell, 128GB unified memory... (2.4s)[/dim]")
c.print('[dim]  \\[web_browse] {"url": "https://nvidia.com/dgx-spark"}[/dim]')
c.print("[dim]    → DGX Spark is a personal AI supercomputer powered by... (1.7s)[/dim]")
c.print("[dim]  \\[3200+490 tokens, 2.9s][/dim]\n")

c.print(
    "MAUDE: The NVIDIA DGX Spark is a compact personal AI computer featuring:\n"
    "  • Grace Blackwell architecture (GB10 Superchip)\n"
    "  • 128GB unified LPDDR5x memory\n"
    "  • Up to 1 petaflop of AI performance\n"
    "  • NVLink-C2C interconnect\n"
    "  • Compact desktop form factor\n"
)

c.print("[bold]You:[/bold] list my shared files\n")
c.print(
    "MAUDE:\n"
    "  \\[list_shared]\n"
    "  Shared folder (/home/mboard76/nvidia-workbench/terminal-llm/shared/):\n"
    "  \\[FILE] notes.txt (1,234 bytes)\n"
    "  \\[FILE] diagram.png (45,678 bytes)\n"
)

c.print("[bold]You:[/bold] search the web for DGX Spark specs again\n")
c.print('[dim]  \\[web_search] {"query": "NVIDIA DGX Spark specifications"}[/dim]')
c.print("[dim]    → [bold]\\[cached result][/bold] (0.0s)[/dim]")

svg = c.export_svg(title="maude — bash")
svg_to_png(svg, OUT / "client-cli.png", scale=2.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3.  ANDROID APP — SVG matching the real phone UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("Rendering Android app...")

W, H = 390, 844

phone_svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <defs>
    <linearGradient id="fire" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#ff4500"/>
      <stop offset="100%" stop-color="#ff8c00"/>
    </linearGradient>
    <style>
      text {{ font-family: -apple-system, "SF Pro", Inter, Helvetica, Arial, sans-serif; }}
      .mono {{ font-family: "SF Mono", "Fira Code", monospace; }}
    </style>
  </defs>

  <!-- Phone body (rounded rect, dark) -->
  <rect width="{W}" height="{H}" rx="40" fill="#0d1117"/>

  <!-- Notch / dynamic island -->
  <rect x="145" y="6" width="100" height="24" rx="12" fill="#000"/>

  <!-- Status bar -->
  <text x="32" y="50" font-size="14" font-weight="600" fill="#e6edf3">9:41</text>
  <text x="{W - 80}" y="50" font-size="12" fill="#8b949e">LTE ▐▐▐ 🔋</text>

  <!-- ─── Header ─── -->
  <rect x="0" y="58" width="{W}" height="52" fill="#161b22"/>
  <line x1="0" y1="110" x2="{W}" y2="110" stroke="#30363d" stroke-width="0.5"/>
  <text x="20" y="90" font-size="20" fill="#8b949e">☰</text>
  <text x="54" y="91" font-size="20" font-weight="800" fill="url(#fire)">MAUDE</text>
  <text x="140" y="91" font-size="12" fill="#484f58">+ New</text>
  <text x="198" y="91" font-size="12" fill="#ff4500">🎙 Voice</text>
  <!-- Model chip -->
  <rect x="270" y="73" width="106" height="28" rx="8" fill="#0d1117" stroke="#30363d" stroke-width="0.5"/>
  <circle cx="284" cy="87" r="3.5" fill="#4ade80"/>
  <text x="294" y="91" font-size="11" fill="#8b949e">mistral-large</text>

  <!-- ─── Messages ─── -->

  <!-- User bubble 1 -->
  <rect x="96" y="126" width="278" height="42" rx="18" fill="url(#fire)"/>
  <text x="114" y="152" font-size="14.5" fill="white">What's the weather in Austin?</text>

  <!-- Trace line -->
  <text x="24" y="190" font-size="10" fill="#484f58" class="mono">[web_search] "weather austin texas today"  (1.3s)</text>

  <!-- AI bubble 1 -->
  <rect x="16" y="204" width="316" height="110" rx="18" fill="#161b22"/>
  <text x="28" y="222" font-size="9" fill="#484f58" letter-spacing="0.8" font-weight="600">MISTRAL-LARGE-LATEST</text>
  <text x="28" y="244" font-size="14" fill="#e6edf3">It's 78°F and partly cloudy in Austin</text>
  <text x="28" y="264" font-size="14" fill="#e6edf3">today! Humidity is around 45% with</text>
  <text x="28" y="284" font-size="14" fill="#e6edf3">south winds at 12 mph. The high should</text>
  <text x="28" y="304" font-size="14" fill="#e6edf3">reach about 82°F this afternoon.</text>

  <!-- User bubble 2 — photo request -->
  <rect x="66" y="332" width="308" height="42" rx="18" fill="url(#fire)"/>
  <text x="84" y="358" font-size="14.5" fill="white">Can you analyze this photo?  📷</text>

  <!-- Trace line -->
  <text x="24" y="396" font-size="10" fill="#484f58" class="mono">[view_image] /shared/IMG_2847.jpg  (3.2s)</text>

  <!-- AI bubble 2 — vision response -->
  <rect x="16" y="410" width="340" height="130" rx="18" fill="#161b22"/>
  <text x="28" y="428" font-size="9" fill="#484f58" letter-spacing="0.8" font-weight="600">MISTRAL-LARGE-LATEST</text>
  <text x="28" y="450" font-size="14" fill="#e6edf3">I can see a Raspberry Pi 5 board! It</text>
  <text x="28" y="470" font-size="14" fill="#e6edf3">has the Broadcom BCM2712 SoC, dual</text>
  <text x="28" y="490" font-size="14" fill="#e6edf3">micro-HDMI ports, USB-C power input,</text>
  <text x="28" y="510" font-size="14" fill="#e6edf3">and the new PCIe connector. The GPIO</text>
  <text x="28" y="530" font-size="14" fill="#e6edf3">header runs along the top edge.</text>

  <!-- User bubble 3 -->
  <rect x="56" y="558" width="318" height="42" rx="18" fill="url(#fire)"/>
  <text x="74" y="584" font-size="14.5" fill="white">Move it to ~/projects/hardware/</text>

  <!-- Trace line -->
  <text x="24" y="622" font-size="10" fill="#484f58" class="mono">[run_command] mv shared/IMG_2847.jpg …  (0.1s)</text>

  <!-- AI bubble 3 -->
  <rect x="16" y="636" width="290" height="58" rx="18" fill="#161b22"/>
  <text x="28" y="654" font-size="9" fill="#484f58" letter-spacing="0.8" font-weight="600">MISTRAL-LARGE-LATEST</text>
  <text x="28" y="676" font-size="14" fill="#e6edf3">Done! Moved to ~/projects/hardware/</text>

  <!-- ─── Input bar ─── -->
  <rect x="0" y="712" width="{W}" height="72" fill="#161b22"/>
  <line x1="0" y1="712" x2="{W}" y2="712" stroke="#30363d" stroke-width="0.5"/>
  <!-- Camera -->
  <rect x="12" y="726" width="44" height="44" rx="12" fill="#0d1117"/>
  <text x="22" y="756" font-size="22">📷</text>
  <!-- Attach -->
  <rect x="64" y="726" width="44" height="44" rx="12" fill="#0d1117"/>
  <text x="74" y="756" font-size="22">📎</text>
  <!-- Text input -->
  <rect x="116" y="726" width="216" height="44" rx="14" fill="#0d1117" stroke="#30363d" stroke-width="0.5"/>
  <text x="132" y="754" font-size="14" fill="#484f58">Message MAUDE...</text>
  <!-- Send -->
  <rect x="340" y="726" width="44" height="44" rx="12" fill="url(#fire)"/>
  <text x="354" y="755" font-size="20" fill="white" font-weight="bold">↑</text>

  <!-- ─── Tab bar ─── -->
  <rect x="0" y="784" width="{W}" height="60" fill="#161b22"/>
  <line x1="0" y1="784" x2="{W}" y2="784" stroke="#30363d" stroke-width="0.5"/>
  <!-- Tabs -->
  <text x="26" y="808" font-size="16" fill="#8b949e" text-anchor="middle">⊞</text>
  <text x="26" y="822" font-size="9" fill="#8b949e" text-anchor="middle">Home</text>

  <text x="104" y="808" font-size="16" fill="#ff4500" text-anchor="middle">◇</text>
  <text x="104" y="822" font-size="9" fill="#ff4500" text-anchor="middle">Chat</text>

  <text x="182" y="808" font-size="16" fill="#8b949e" text-anchor="middle">🎙</text>
  <text x="182" y="822" font-size="9" fill="#8b949e" text-anchor="middle">Voice</text>

  <text x="260" y="808" font-size="14" fill="#8b949e" text-anchor="middle">&gt;_</text>
  <text x="260" y="822" font-size="9" fill="#8b949e" text-anchor="middle">Term</text>

  <text x="338" y="808" font-size="16" fill="#8b949e" text-anchor="middle">⚙</text>
  <text x="338" y="822" font-size="9" fill="#8b949e" text-anchor="middle">Settings</text>

  <!-- Home indicator bar -->
  <rect x="130" y="836" width="130" height="4" rx="2" fill="#30363d"/>
</svg>'''

svg_to_png(phone_svg, OUT / "android-app.png", scale=2.0)

print("\nDone!")
