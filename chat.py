#!/usr/bin/env python3
"""
Terminal LLM Chat — multi-model client.
Supports local (NVIDIA), Mistral, Codestral, and Claude models.
"""

import os
import sys
from pathlib import Path

import uuid
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
import threading
import time
import conversation_sync
from collab import get_hub as get_collab_hub

# Load environment variables
env_path = Path(__file__).parent / "variables.env"
load_dotenv(env_path)

# Initialize console for rich output
console = Console()

# NVIDIA API configuration
NVIDIA_API_KEY = os.getenv("NGC_PERSONAL_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
GATEWAY_URL = "http://localhost:30080/v1"

# Available models — local go direct to NVIDIA, cloud go through gateway
MODELS = {
    # Local / NVIDIA
    "llama-3.3": "meta/llama-3.3-70b-instruct",
    "nemotron-nano": "nvidia/llama-3.1-nemotron-nano-8b-v1",
    "nemotron-51b": "nvidia/llama-3.1-nemotron-51b-instruct",
    "llama-405b": "meta/llama-3.1-405b-instruct",
    "codellama": "meta/codellama-70b",
    # Cloud (via gateway)
    "mistral": "mistral-large-latest",
    "codestral": "codestral-latest",
    "devstral": "devstral-2512",
    "devstral-small": "devstral-small-latest",
    "devstral-medium": "devstral-medium-latest",
    "claude": "claude-opus-4-20250514",
    "sonnet": "claude-sonnet-4-20250514",
    # Local (via gateway)
    "nemotron": "nemotron",
    "llava": "llava",
}

# Models that route through the gateway
_CLOUD_MODELS = {"mistral", "codestral", "devstral", "devstral-small", "devstral-medium", "claude", "sonnet", "nemotron", "llava"}

# Default model
DEFAULT_MODEL = "mistral"


def create_nvidia_client():
    """Create NVIDIA API client."""
    if not NVIDIA_API_KEY:
        console.print("[red]Error: NGC_PERSONAL_KEY not found in variables.env[/red]")
        sys.exit(1)
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)


def create_gateway_client():
    """Create gateway client for cloud models."""
    return OpenAI(base_url=GATEWAY_URL, api_key="not-needed")


def chat(client, messages: list, model_id: str, stream: bool = True):
    """Send chat request and stream response."""
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
            console.print("[bold magenta]MAUDE:[/bold magenta] ", end="")
            for chunk in response:
                choice = chunk.choices[0] if chunk.choices else None
                if choice and choice.delta and choice.delta.content:
                    print(choice.delta.content, end="", flush=True)
                    full_response += choice.delta.content
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
        "[bold magenta]MAUDE Terminal Chat[/bold magenta]\n"
        "[dim]NVIDIA: nemotron-nano, llama-3.3, nemotron-51b, llama-405b, codellama[/dim]\n"
        "[dim]Local:  nemotron, llava[/dim]\n"
        "[dim]Cloud:  mistral, codestral, devstral, devstral-small, devstral-medium, claude, sonnet[/dim]",
        border_style="magenta"
    ))
    console.print(f"[dim]Commands: /quit, /clear, /model <name>[/dim]\n")

    nvidia_client = create_nvidia_client()
    gateway_client = create_gateway_client()
    messages = []
    current_model = DEFAULT_MODEL
    conv_id = str(uuid.uuid4())
    conv_title = ""
    collab_activity = ["idle"]

    # Start presence heartbeat
    def _heartbeat_loop():
        hub = get_collab_hub()
        while True:
            try:
                hub.heartbeat("cli-client", "cli", collab_activity[0], conv_id)
            except Exception:
                pass
            time.sleep(30)
    threading.Thread(target=_heartbeat_loop, daemon=True).start()

    # System prompt
    system_prompt = {
        "role": "system",
        "content": """You are MAUDE, a highly capable AI assistant.
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
                conv_id = str(uuid.uuid4())
                conv_title = ""
                console.print("[dim]Conversation cleared.[/dim]")
                continue

            if user_input.lower().startswith("/model"):
                parts = user_input.split()
                if len(parts) > 1 and parts[1] in MODELS:
                    current_model = parts[1]
                    via = "gateway" if current_model in _CLOUD_MODELS else "NVIDIA API"
                    console.print(f"[dim]Switched to {current_model} ({MODELS[current_model]}) via {via}[/dim]")
                else:
                    console.print(f"[dim]Available models: {', '.join(MODELS.keys())}[/dim]")
                continue

            # Add user message
            messages.append({"role": "user", "content": user_input})

            # Pick the right client and model ID
            if current_model in _CLOUD_MODELS:
                client = gateway_client
            else:
                client = nvidia_client
            model_id = MODELS[current_model]

            # Get response
            response = chat(client, messages, model_id=model_id)

            if response:
                messages.append({"role": "assistant", "content": response})
                # Sync to gateway for cross-device history
                if not conv_title:
                    conv_title = conversation_sync.generate_title(user_input)
                conversation_sync.save_conversation(
                    conv_id, conv_title, current_model, messages
                )
                # Emit activity event
                collab_activity[0] = f"chatting: {user_input[:40]}"
                try:
                    get_collab_hub().emit(
                        "chat", f"Asked about: {user_input[:50]}",
                        data={"model": current_model},
                        client_id="cli-client",
                        conversation_id=conv_id,
                    )
                except Exception:
                    pass

            print()  # Extra spacing

        except KeyboardInterrupt:
            console.print("\n[dim]Use /quit to exit[/dim]")
        except EOFError:
            break


if __name__ == "__main__":
    main()
