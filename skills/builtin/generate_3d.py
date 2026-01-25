"""
Mesh-aware 3D model generation skill.

Routes 3D generation requests to the best available endpoint via CapabilityRouter.
Supports various 3D generation backends (Meshy API, local services, etc.).
"""

import json
import os
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


def _get_3d_endpoint_from_router(provider: str = None) -> Optional[tuple]:
    """Get 3D generation endpoint info from the capability router.

    Returns:
        Tuple of (endpoint_url, endpoint_type, name) or None
    """
    router = _get_router()
    if not router:
        return None

    from capabilities import CapabilityType

    # Try to find 3D generation capability
    result = router.find_3d_gen(provider if provider != "auto" else None)
    if result:
        node, cap = result
        return (cap.endpoint_url, cap.endpoint_type.value, cap.name)

    return None


def _check_endpoint_health(url: str) -> bool:
    """Check if a 3D generation endpoint is healthy."""
    try:
        # Try common health check endpoints
        for path in ["/health", "/status", "/api/health", ""]:
            try:
                response = requests.get(f"{url}{path}", timeout=5)
                if response.status_code in [200, 204]:
                    return True
            except:
                continue
    except:
        pass
    return False


@skill(
    name="generate_3d",
    description="Generate 3D models from text or images. Routes to the best available 3D generation endpoint on the mesh.",
    version="1.0.0",
    author="MAUDE",
    triggers=["generate 3d", "create 3d", "make 3d model", "3d generation", "3d model"],
    parameters={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Text description of the 3D model to generate"
            },
            "image_path": {
                "type": "string",
                "description": "Optional path to an image to use as reference"
            },
            "output_format": {
                "type": "string",
                "enum": ["glb", "obj", "fbx", "stl"],
                "description": "Output format for the 3D model",
                "default": "glb"
            },
            "provider": {
                "type": "string",
                "description": "Specific provider to use (auto, meshy, local, etc.)",
                "default": "auto"
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
def generate_3d(
    prompt: str,
    image_path: str = None,
    output_format: str = "glb",
    provider: str = "auto",
    action: str = "generate"
) -> str:
    """Generate 3D model using the mesh capability router."""

    router = _get_router()

    if action == "list":
        # List available 3D generation capabilities
        if not router:
            return "Capability router not available. Cannot list 3D generators."

        caps = router.list_capabilities()
        model_3d_caps = caps.get("MODEL_3D", [])
        gaussian_caps = caps.get("GAUSSIAN_SPLAT", [])

        if not model_3d_caps and not gaussian_caps:
            return (
                "No 3D generation capabilities found on the mesh.\n\n"
                "To add 3D generation:\n"
                "1. Register a Meshy API endpoint: /mesh register model_3d meshy https://api.meshy.ai\n"
                "2. Or start a local 3D generation service and register it\n"
                "3. Set MESHY_API_KEY environment variable for Meshy API"
            )

        lines = ["Available 3D Generation:\n"]
        if model_3d_caps:
            lines.append("  [MODEL_3D]")
            for cap in model_3d_caps:
                lines.append(f"    - {cap}")
        if gaussian_caps:
            lines.append("  [GAUSSIAN_SPLAT]")
            for cap in gaussian_caps:
                lines.append(f"    - {cap}")

        return "\n".join(lines)

    if action == "status":
        # Check status of 3D generation endpoint
        endpoint_info = _get_3d_endpoint_from_router(provider)
        if not endpoint_info:
            # Check for Meshy API key as fallback
            meshy_key = os.environ.get("MESHY_API_KEY")
            if meshy_key:
                return (
                    "3D Generation Status: Meshy API (Cloud)\n"
                    "Endpoint: https://api.meshy.ai\n"
                    "API Key: Configured"
                )
            return "No 3D generation endpoint found on mesh and no MESHY_API_KEY configured."

        url, endpoint_type, name = endpoint_info
        is_healthy = _check_endpoint_health(url)

        return (
            f"3D Generation Status: {'Online' if is_healthy else 'Offline'}\n"
            f"Provider: {name}\n"
            f"Endpoint: {url}\n"
            f"Type: {endpoint_type}"
        )

    # Generate action
    if not prompt:
        return "Error: prompt is required for 3D generation"

    # Find 3D generation endpoint via router
    endpoint_info = _get_3d_endpoint_from_router(provider)

    # Check for Meshy API as fallback
    meshy_key = os.environ.get("MESHY_API_KEY")

    if endpoint_info:
        url, endpoint_type, name = endpoint_info

        # Check endpoint health
        if not _check_endpoint_health(url) and not meshy_key:
            return (
                f"3D generation endpoint at {url} is not responding.\n"
                "Please check that the service is running."
            )

        # For custom endpoints, we'd need to know their API format
        # For now, return info about the endpoint
        return (
            f"3D Generation Request Prepared:\n"
            f"Provider: {name}\n"
            f"Endpoint: {url}\n"
            f"Prompt: {prompt}\n"
            f"Format: {output_format}\n"
            f"Image: {image_path or 'None'}\n\n"
            f"Note: This endpoint uses a custom API. "
            f"The actual generation would need endpoint-specific implementation."
        )

    elif meshy_key:
        # Use Meshy API
        return _generate_with_meshy(prompt, image_path, output_format, meshy_key)

    else:
        return (
            "No 3D generation capability available.\n\n"
            "Options:\n"
            "1. Set MESHY_API_KEY for cloud 3D generation via Meshy\n"
            "2. Register a local 3D endpoint: /mesh register model_3d <name> <url>\n"
            "3. Start a 3D generation service on a mesh node"
        )


def _generate_with_meshy(prompt: str, image_path: str, output_format: str, api_key: str) -> str:
    """Generate 3D model using Meshy API."""
    import base64

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Determine if this is text-to-3D or image-to-3D
    if image_path and os.path.exists(image_path):
        # Image-to-3D
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        # Meshy image-to-3D endpoint
        url = "https://api.meshy.ai/v2/image-to-3d"
        payload = {
            "image_url": f"data:image/png;base64,{image_data}",
            "enable_pbr": True
        }
    else:
        # Text-to-3D
        url = "https://api.meshy.ai/v2/text-to-3d"
        payload = {
            "mode": "preview",
            "prompt": prompt,
            "art_style": "realistic",
            "negative_prompt": "low quality, blurry, distorted"
        }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code == 202:
            # Request accepted, task created
            data = response.json()
            task_id = data.get("result")
            return (
                f"3D Generation Started!\n"
                f"Task ID: {task_id}\n"
                f"Provider: Meshy API\n"
                f"Prompt: {prompt}\n"
                f"Format: {output_format}\n\n"
                f"Generation is processing in the cloud.\n"
                f"Check status at: https://app.meshy.ai/\n"
                f"Or use: /skills run generate_3d action=status"
            )
        elif response.status_code == 401:
            return "Error: Invalid Meshy API key. Check MESHY_API_KEY."
        elif response.status_code == 429:
            return "Error: Meshy API rate limit exceeded. Please wait and try again."
        else:
            return f"Error from Meshy API: {response.status_code} - {response.text}"

    except requests.exceptions.Timeout:
        return "Error: Meshy API request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Meshy API. Check your internet connection."
    except Exception as e:
        return f"Error: {e}"
