"""
Mesh-aware video generation skill.

Routes video generation requests to the best available endpoint via CapabilityRouter.
Supports ComfyUI (Mochi) and other video generation backends.
"""

import copy
import json
import random
import time
import requests
from typing import Optional

from skills import skill


def _get_router():
    """Lazy import router to avoid circular dependencies."""
    try:
        from routing import get_router
        return get_router()
    except ImportError:
        return None


def _get_comfyui_url_from_router(provider: str = None) -> Optional[str]:
    """Get ComfyUI URL from the capability router."""
    router = _get_router()
    if not router:
        return None

    from capabilities import CapabilityType

    # Try to find video generation capability
    result = router.find_video_gen(provider if provider != "auto" else None)
    if result:
        node, cap = result
        return cap.endpoint_url

    # Fall back to ComfyUI capability
    result = router.find_comfyui()
    if result:
        node, cap = result
        return cap.endpoint_url

    return None


def _check_comfyui_status(base_url: str) -> dict:
    """Check if ComfyUI is running and get system info."""
    try:
        url = f"{base_url}/system_stats"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return {"online": True, "stats": response.json(), "url": base_url}
    except:
        pass
    return {"online": False, "url": base_url}


def _queue_prompt(base_url: str, workflow: dict, client_id: str = "maude") -> dict:
    """Queue a prompt/workflow for execution."""
    url = f"{base_url}/prompt"
    payload = {
        "prompt": workflow,
        "client_id": client_id
    }
    response = requests.post(url, json=payload, timeout=30)
    return response.json()


def _get_history(base_url: str, prompt_id: str) -> dict:
    """Get the history/results for a prompt."""
    url = f"{base_url}/history/{prompt_id}"
    response = requests.get(url, timeout=10)
    return response.json()


# Basic Mochi workflow template
MOCHI_WORKFLOW = {
    "3": {
        "class_type": "MochiTextEncode",
        "inputs": {
            "prompt": "",
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
    name="generate_video",
    description="Generate videos using AI. Automatically routes to the best available video generation endpoint on the mesh.",
    version="1.0.0",
    author="MAUDE",
    triggers=["generate video", "create video", "make video", "video generation"],
    parameters={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Text description of the video to generate"
            },
            "provider": {
                "type": "string",
                "description": "Specific provider to use (auto, mochi, etc.)",
                "default": "auto"
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
            },
            "action": {
                "type": "string",
                "enum": ["generate", "status", "list"],
                "description": "Action to perform",
                "default": "generate"
            }
        },
        "required": ["prompt"]
    }
)
def generate_video(
    prompt: str,
    provider: str = "auto",
    steps: int = 50,
    seed: int = -1,
    action: str = "generate"
) -> str:
    """Generate video using the mesh capability router."""

    # Get router
    router = _get_router()

    if action == "list":
        # List available video generation capabilities
        if not router:
            return "Capability router not available. Cannot list video generators."

        caps = router.list_capabilities()
        video_caps = caps.get("VIDEO_GEN", [])
        comfyui_caps = caps.get("COMFYUI", [])

        if not video_caps and not comfyui_caps:
            return (
                "No video generation capabilities found on the mesh.\n\n"
                "To add video generation:\n"
                "1. Start ComfyUI with Mochi on a mesh node\n"
                "2. Or use: /mesh register video_gen mochi http://host:8188"
            )

        lines = ["Available Video Generation:\n"]
        if video_caps:
            lines.append("  [VIDEO_GEN]")
            for cap in video_caps:
                lines.append(f"    - {cap}")
        if comfyui_caps:
            lines.append("  [COMFYUI]")
            for cap in comfyui_caps:
                lines.append(f"    - {cap}")

        return "\n".join(lines)

    if action == "status":
        # Check status of video generation endpoint
        base_url = _get_comfyui_url_from_router(provider)
        if not base_url:
            return "No video generation endpoint found on mesh."

        status = _check_comfyui_status(base_url)
        if status["online"]:
            stats = status.get("stats", {})
            device = stats.get("system", {}).get("device_name", "Unknown")
            vram = stats.get("system", {}).get("vram_total", 0) / (1024**3)
            return (
                f"Video Generation Status: Online\n"
                f"Endpoint: {base_url}\n"
                f"Device: {device}\n"
                f"VRAM: {vram:.1f} GB"
            )
        else:
            return f"Video Generation Status: Offline\nEndpoint: {base_url}"

    # Generate action
    if not prompt:
        return "Error: prompt is required for video generation"

    # Find video generation endpoint via router
    base_url = _get_comfyui_url_from_router(provider)
    if not base_url:
        # Fall back to environment variable
        import os
        comfyui_host = os.environ.get("COMFYUI_HOST", "mattwell")
        comfyui_port = os.environ.get("COMFYUI_PORT", "8188")
        base_url = f"http://{comfyui_host}:{comfyui_port}"

    # Check if endpoint is available
    status = _check_comfyui_status(base_url)
    if not status["online"]:
        return (
            f"Cannot connect to video generation at {base_url}\n\n"
            "Please ensure:\n"
            "1. ComfyUI is running on the target machine\n"
            "2. The machine is accessible via Tailscale/network\n"
            "3. ComfyUI is listening on all interfaces (--listen 0.0.0.0)"
        )

    try:
        # Prepare workflow
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
        result = _queue_prompt(base_url, workflow)

        if "error" in result:
            return f"Error queuing video generation: {result['error']}"

        prompt_id = result.get("prompt_id")
        if not prompt_id:
            return f"Failed to queue video generation: {result}"

        # Extract hostname from URL for display
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        host = parsed.hostname or base_url

        return (
            f"Video generation started!\n"
            f"Prompt ID: {prompt_id}\n"
            f"Endpoint: {host}\n"
            f"Provider: {provider if provider != 'auto' else 'mochi (auto)'}\n"
            f"Steps: {steps}\n"
            f"Seed: {seed}\n\n"
            f"Generation is running in the background.\n"
            f"Check ComfyUI at {base_url} for progress and results."
        )

    except requests.exceptions.ConnectionError:
        return f"Connection failed to {base_url}. Is the video generation service running?"
    except Exception as e:
        return f"Error: {e}"
