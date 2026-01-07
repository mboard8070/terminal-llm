#!/usr/bin/env python3
"""
Terminal LLM Chat using local Nemotron-3-Nano-30B via llama.cpp server.
This is a reasoning model - it thinks before responding.
"""

import sys
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

console = Console()

# Local llama.cpp server
LOCAL_URL = "http://localhost:30000/v1"
MODEL = "nemotron"

# Toggle for showing reasoning
show_thinking = False


def create_client():
    """Create local API client."""
    return OpenAI(
        base_url=LOCAL_URL,
        api_key="not-needed"  # Local server doesn't need auth
    )


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
            console.print("[cyan]Nemotron:[/cyan] ", end="")

            for chunk in response:
                delta = chunk.choices[0].delta
                # Handle reasoning content if present
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning += delta.reasoning_content
                    if show_thinking:
                        if not in_reasoning:
                            console.print("\n[dim]<thinking>[/dim]", end="")
                            in_reasoning = True
                        print(delta.reasoning_content, end="", flush=True)
                # Handle main content
                if delta.content:
                    if in_reasoning and show_thinking:
                        console.print("[dim]</thinking>[/dim]\n", end="")
                        in_reasoning = False
                    print(delta.content, end="", flush=True)
                    full_response += delta.content

            if in_reasoning and show_thinking:
                console.print("[dim]</thinking>[/dim]", end="")
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
                console.print(f"[dim]<thinking>{msg.reasoning_content}</thinking>[/dim]")
            return msg.content

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Is the server running? Start with: ./start_server.sh[/dim]")
        return None


def main():
    """Main chat loop."""
    console.print(Panel.fit(
        "[bold cyan]Nemotron-3-Nano-30B Local[/bold cyan]\n"
        "[dim]Running via llama.cpp on GB10[/dim]\n"
        "[dim]Context: 8192 tokens | Full GPU offload[/dim]",
        border_style="cyan"
    ))
    console.print("[dim]Commands: /quit, /clear, /think (toggle reasoning)[/dim]\n")

    try:
        client = create_client()
    except Exception as e:
        console.print(f"[red]Failed to connect: {e}[/red]")
        console.print("[dim]Make sure the server is running: ./start_server.sh[/dim]")
        sys.exit(1)

    messages = []

    system_prompt = {
        "role": "system",
        "content": """You are Nemotron, NVIDIA's advanced AI assistant.
You excel at coding, reasoning, and complex problem-solving.
Be concise but thorough. Show your reasoning when helpful."""
    }
    messages.append(system_prompt)

    while True:
        try:
            user_input = console.input("[green]You:[/green] ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/quit":
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.lower() == "/clear":
                messages = [system_prompt]
                console.print("[dim]Conversation cleared.[/dim]")
                continue

            if user_input.lower() == "/think":
                global show_thinking
                show_thinking = not show_thinking
                status = "ON" if show_thinking else "OFF"
                console.print(f"[dim]Show reasoning: {status}[/dim]")
                continue

            messages.append({"role": "user", "content": user_input})

            response = chat(client, messages)

            if response:
                messages.append({"role": "assistant", "content": response})

            print()

        except KeyboardInterrupt:
            console.print("\n[dim]Use /quit to exit[/dim]")
        except EOFError:
            break


if __name__ == "__main__":
    main()
