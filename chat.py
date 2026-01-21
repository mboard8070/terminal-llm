#!/usr/bin/env python3
"""
Terminal LLM Chat using NVIDIA Nemotron via NGC API.
Best model for coding, reasoning, and complex tasks.
Supports vision with Nemotron Nano V2 VL model.
"""

import os
import sys
import base64
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from PIL import Image

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
    "nemotron-vision": "nvidia/nemotron-nano-12b-v2-vl",       # Vision-capable model
}

# Models that support vision/images
VISION_MODELS = {"nemotron-vision"}

# Default model
DEFAULT_MODEL = "llama-3.3"


def create_client():
    """Create NVIDIA API client."""
    if not NVIDIA_API_KEY:
        console.print("[red]Error: NGC_PERSONAL_KEY not found in variables.env[/red]")
        sys.exit(1)

    return OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=NVIDIA_API_KEY
    )


def encode_image(image_path: str) -> tuple[str, str]:
    """
    Encode an image file to base64.
    Returns (base64_data, media_type) tuple.
    """
    path = Path(image_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # Determine media type from extension
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

    # Read and encode
    with open(path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    return image_data, media_type


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
        "[bold cyan]NVIDIA Terminal LLM[/bold cyan]\n"
        "[dim]Default: Llama-3.3-70B-Instruct[/dim]\n"
        "[dim]Vision: Use /model nemotron-vision + /image <path>[/dim]",
        border_style="cyan"
    ))
    console.print(f"[dim]Commands: /quit, /clear, /image <path>, /model [{' | '.join(MODELS.keys())}][/dim]\n")

    client = create_client()
    messages = []
    current_model = DEFAULT_MODEL
    pending_image = None  # Holds (base64_data, media_type) for next message

    # System prompt for coding/reasoning
    system_prompt = {
        "role": "system",
        "content": """You are a highly capable AI assistant powered by NVIDIA Nemotron.
You excel at:
- Writing clean, efficient code in any language
- Debugging and explaining complex code
- Reasoning through difficult problems step by step
- Explaining technical concepts clearly
- Analyzing images when provided (with vision model)

Be concise but thorough. When writing code, include comments for complex logic."""
    }
    messages.append(system_prompt)

    while True:
        try:
            # Show image indicator if one is pending
            if pending_image:
                user_input = console.input("[green]You[/green] [magenta][+img][/magenta][green]:[/green] ").strip()
            else:
                user_input = console.input("[green]You:[/green] ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "/quit":
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.lower() == "/clear":
                messages = [system_prompt]
                pending_image = None
                console.print("[dim]Conversation cleared.[/dim]")
                continue

            if user_input.lower().startswith("/image"):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    console.print("[dim]Usage: /image <path_to_image>[/dim]")
                    continue

                image_path = parts[1].strip()
                try:
                    pending_image = encode_image(image_path)
                    console.print(f"[dim]Image loaded: {image_path}[/dim]")
                    if current_model not in VISION_MODELS:
                        console.print("[yellow]Warning: Current model doesn't support vision. Use /model nemotron-vision[/yellow]")
                except (FileNotFoundError, ValueError) as e:
                    console.print(f"[red]Error: {e}[/red]")
                    pending_image = None
                continue

            if user_input.lower().startswith("/model"):
                parts = user_input.split()
                if len(parts) > 1 and parts[1] in MODELS:
                    current_model = parts[1]
                    vision_note = " (supports images)" if current_model in VISION_MODELS else ""
                    console.print(f"[dim]Switched to {current_model}{vision_note}[/dim]")
                else:
                    console.print(f"[dim]Available models: {', '.join(MODELS.keys())}[/dim]")
                    console.print(f"[dim]Vision models: {', '.join(VISION_MODELS)}[/dim]")
                continue

            # Build user message (text-only or multimodal)
            if pending_image and current_model in VISION_MODELS:
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
                pending_image = None  # Clear after use
            else:
                if pending_image:
                    console.print("[yellow]Image ignored - current model doesn't support vision[/yellow]")
                    pending_image = None
                user_message = {"role": "user", "content": user_input}

            messages.append(user_message)

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
