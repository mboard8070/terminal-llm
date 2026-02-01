#!/usr/bin/env python3
"""
Terminal LLM Chat using NVIDIA Nemotron via NGC API.
Best model for coding, reasoning, and complex tasks.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Load environment variables
env_path = Path(__file__).parent / "variables.env"
load_dotenv(env_path)

# Initialize console for rich output
console = Console()

# NVIDIA API configuration
NVIDIA_API_KEY = os.getenv("NGC_PERSONAL_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Available models (best for coding/reasoning)
MODELS = {
    "llama-3.3": "meta/llama-3.3-70b-instruct",                # Excellent for coding/reasoning
    "nemotron-nano": "nvidia/llama-3.1-nemotron-nano-8b-v1",   # Fast Nemotron variant
    "nemotron-51b": "nvidia/llama-3.1-nemotron-51b-instruct",  # Mid-size Nemotron
    "llama-405b": "meta/llama-3.1-405b-instruct",              # Largest available
    "codellama": "meta/codellama-70b",                          # Code-specific
}

# Default model
DEFAULT_MODEL = "nemotron-nano"


def create_client():
    """Create NVIDIA API client."""
    if not NVIDIA_API_KEY:
        console.print("[red]Error: NGC_PERSONAL_KEY not found in variables.env[/red]")
        sys.exit(1)

    return OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=NVIDIA_API_KEY
    )


def chat(client, messages: list, model: str = DEFAULT_MODEL, stream: bool = True):
    """Send chat request to Nemotron."""
    model_id = MODELS.get(model, MODELS[DEFAULT_MODEL])

    try:
        if stream:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.2,
                max_tokens=4096,
                stream=True
            )

            full_response = ""
            console.print("[cyan]Nemotron:[/cyan] ", end="")
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            print()  # newline after response
            return full_response
        else:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.2,
                max_tokens=4096
            )
            return response.choices[0].message.content

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return None


def main():
    """Main chat loop."""
    console.print(Panel.fit(
        "[bold magenta]NVIDIA Terminal LLM[/bold magenta]\n"
        "[dim]Default: Nemotron-Nano-8B (local) / Llama-3.3-70B (remote)[/dim]\n"
        "[dim]Best for: Coding, Reasoning, Complex Tasks[/dim]",
        border_style="magenta"
    ))
    console.print(f"[dim]Commands: /quit, /clear, /model [{' | '.join(MODELS.keys())}][/dim]\n")

    client = create_client()
    messages = []
    current_model = DEFAULT_MODEL

    # System prompt for coding/reasoning
    system_prompt = {
        "role": "system",
        "content": """You are a highly capable AI assistant powered by NVIDIA Nemotron.
You excel at:
- Writing clean, efficient code in any language
- Debugging and explaining complex code
- Reasoning through difficult problems step by step
- Explaining technical concepts clearly

Be concise but thorough. When writing code, include comments for complex logic."""
    }
    messages.append(system_prompt)

    while True:
        try:
            user_input = console.input("[green]You:[/green] ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "/quit":
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.lower() == "/clear":
                messages = [system_prompt]
                console.print("[dim]Conversation cleared.[/dim]")
                continue

            if user_input.lower().startswith("/model"):
                parts = user_input.split()
                if len(parts) > 1 and parts[1] in MODELS:
                    current_model = parts[1]
                    console.print(f"[dim]Switched to {current_model}[/dim]")
                else:
                    console.print(f"[dim]Available models: {', '.join(MODELS.keys())}[/dim]")
                continue

            # Add user message
            messages.append({"role": "user", "content": user_input})

            # Get response
            response = chat(client, messages, model=current_model)

            if response:
                messages.append({"role": "assistant", "content": response})

            print()  # Extra spacing

        except KeyboardInterrupt:
            console.print("\n[dim]Use /quit to exit[/dim]")
        except EOFError:
            break


if __name__ == "__main__":
    main()