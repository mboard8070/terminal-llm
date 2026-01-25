"""Mochi video generation skill via ComfyUI on remote desktop."""

import os
import json
import time
import requests
from skills import skill

# Default ComfyUI host - can be overridden by COMFYUI_HOST env var
COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "mattwell")  # Tailscale hostname
COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))


def get_comfyui_url():
    """Get the ComfyUI API URL."""
    return f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"


def check_comfyui_status() -> dict:
    """Check if ComfyUI is running and get system info."""
    try:
        url = f"{get_comfyui_url()}/system_stats"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return {"online": True, "stats": response.json()}
    except:
        pass
    return {"online": False}


def queue_prompt(workflow: dict, client_id: str = "maude") -> dict:
    """Queue a prompt/workflow for execution."""
    url = f"{get_comfyui_url()}/prompt"
    payload = {
        "prompt": workflow,
        "client_id": client_id
    }
    response = requests.post(url, json=payload, timeout=30)
    return response.json()


def get_history(prompt_id: str) -> dict:
    """Get the history/results for a prompt."""
    url = f"{get_comfyui_url()}/history/{prompt_id}"
    response = requests.get(url, timeout=10)
    return response.json()


def wait_for_completion(prompt_id: str, timeout: int = 300) -> dict:
    """Wait for a prompt to complete."""
    start = time.time()
    while time.time() - start < timeout:
        history = get_history(prompt_id)
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(2)
    return {"error": "Timeout waiting for generation"}


# Basic Mochi workflow template - will need to be customized based on actual ComfyUI setup
MOCHI_WORKFLOW = {
    "3": {
        "class_type": "MochiTextEncode",
        "inputs": {
            "prompt": "",  # Will be filled in
            "clip": ["4", 0]
        }
    },
    "4": {
        "class_type": "MochiLoader",
        "inputs": {
            "model_name": "mochi_preview_bf16.safetensors"
        }
    },
    "5": {
        "class_type": "MochiSampler",
        "inputs": {
            "seed": 0,
            "steps": 50,
            "cfg": 4.5,
            "conditioning": ["3", 0],
            "model": ["4", 0]
        }
    },
    "6": {
        "class_type": "MochiDecode",
        "inputs": {
            "samples": ["5", 0],
            "vae": ["4", 1]
        }
    },
    "7": {
        "class_type": "SaveVideo",
        "inputs": {
            "filename_prefix": "mochi_",
            "video": ["6", 0]
        }
    }
}


@skill(
    name="mochi",
    description="Generate videos using Mochi AI via ComfyUI on remote desktop",
    version="1.0.0",
    author="MAUDE",
    triggers=["mochi", "video", "generate video", "create video"],
    parameters={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Text description of the video to generate"
            },
            "action": {
                "type": "string",
                "enum": ["generate", "status", "check"],
                "description": "Action to perform (default: generate)",
                "default": "generate"
            },
            "steps": {
                "type": "integer",
                "description": "Number of sampling steps (default: 50)",
                "default": 50
            },
            "seed": {
                "type": "integer",
                "description": "Random seed for reproducibility (default: random)",
                "default": -1
            }
        },
        "required": ["prompt"]
    }
)
def mochi(prompt: str, action: str = "generate", steps: int = 50, seed: int = -1) -> str:
    """Generate video using Mochi via ComfyUI."""

    if action == "status" or action == "check":
        status = check_comfyui_status()
        if status["online"]:
            stats = status.get("stats", {})
            device = stats.get("system", {}).get("device_name", "Unknown")
            vram = stats.get("system", {}).get("vram_total", 0) / (1024**3)
            return f"ComfyUI Status: Online\nHost: {COMFYUI_HOST}:{COMFYUI_PORT}\nDevice: {device}\nVRAM: {vram:.1f} GB"
        else:
            return f"ComfyUI Status: Offline\nHost: {COMFYUI_HOST}:{COMFYUI_PORT}\n\nMake sure ComfyUI is running on the remote machine."

    # Check if ComfyUI is available
    status = check_comfyui_status()
    if not status["online"]:
        return (
            f"Cannot connect to ComfyUI at {COMFYUI_HOST}:{COMFYUI_PORT}\n\n"
            "Please ensure:\n"
            "1. ComfyUI is running on the remote machine\n"
            "2. The machine is accessible via Tailscale\n"
            "3. ComfyUI is listening on all interfaces (--listen 0.0.0.0)"
        )

    try:
        # Prepare workflow
        import copy
        import random

        workflow = copy.deepcopy(MOCHI_WORKFLOW)

        # Set prompt
        workflow["3"]["inputs"]["prompt"] = prompt

        # Set steps
        workflow["5"]["inputs"]["steps"] = steps

        # Set seed
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        workflow["5"]["inputs"]["seed"] = seed

        # Queue the prompt
        result = queue_prompt(workflow)

        if "error" in result:
            return f"Error queuing prompt: {result['error']}"

        prompt_id = result.get("prompt_id")
        if not prompt_id:
            return f"Failed to queue prompt: {result}"

        return (
            f"Video generation started!\n"
            f"Prompt ID: {prompt_id}\n"
            f"Host: {COMFYUI_HOST}\n"
            f"Steps: {steps}\n"
            f"Seed: {seed}\n\n"
            f"Generation is running in the background on the remote machine.\n"
            f"Check ComfyUI on {COMFYUI_HOST} for progress and results."
        )

    except requests.exceptions.ConnectionError:
        return f"Connection failed to {COMFYUI_HOST}:{COMFYUI_PORT}. Is ComfyUI running?"
    except Exception as e:
        return f"Error: {e}"


@skill(
    name="comfyui",
    description="Check ComfyUI status and manage remote generation",
    version="1.0.0",
    author="MAUDE",
    triggers=["comfyui", "comfy"],
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "queue", "history"],
                "description": "Action to perform",
                "default": "status"
            },
            "prompt_id": {
                "type": "string",
                "description": "Prompt ID for history lookup"
            }
        }
    }
)
def comfyui(action: str = "status", prompt_id: str = None) -> str:
    """Manage ComfyUI connection."""

    if action == "status":
        status = check_comfyui_status()
        if status["online"]:
            stats = status.get("stats", {})
            system = stats.get("system", {})

            output = [
                f"ComfyUI: Online",
                f"Host: {COMFYUI_HOST}:{COMFYUI_PORT}",
                f"Device: {system.get('device_name', 'Unknown')}",
            ]

            vram_total = system.get("vram_total", 0)
            vram_free = system.get("vram_free", 0)
            if vram_total:
                output.append(f"VRAM: {vram_free/(1024**3):.1f} / {vram_total/(1024**3):.1f} GB free")

            return "\n".join(output)
        else:
            return f"ComfyUI: Offline ({COMFYUI_HOST}:{COMFYUI_PORT})"

    elif action == "queue":
        try:
            url = f"{get_comfyui_url()}/queue"
            response = requests.get(url, timeout=5)
            data = response.json()
            running = len(data.get("queue_running", []))
            pending = len(data.get("queue_pending", []))
            return f"Queue: {running} running, {pending} pending"
        except Exception as e:
            return f"Error checking queue: {e}"

    elif action == "history" and prompt_id:
        try:
            history = get_history(prompt_id)
            if prompt_id in history:
                result = history[prompt_id]
                status = result.get("status", {})
                if status.get("completed"):
                    outputs = result.get("outputs", {})
                    return f"Prompt {prompt_id}: Completed\nOutputs: {json.dumps(outputs, indent=2)}"
                else:
                    return f"Prompt {prompt_id}: In progress"
            else:
                return f"Prompt {prompt_id} not found in history"
        except Exception as e:
            return f"Error: {e}"

    return "Usage: comfyui status | queue | history <prompt_id>"
