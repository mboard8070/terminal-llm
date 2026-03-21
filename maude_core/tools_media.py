"""
Image generation tool — Flux via ComfyUI.
"""

import http.client
import json
import os
from pathlib import Path
from urllib.parse import urlparse

from tool_registry import register_tool

from .log import log


def tool_generate_image(
    prompt: str, width: int = 1024, height: int = 1024, seed: int = -1, steps: int = 28, lora: str = None
) -> str:
    """Generate an image using Flux via ComfyUI API."""
    import random
    import shutil
    import time

    COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "localhost")
    COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
    comfyui_url = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"

    # Try mesh router first
    try:
        from routing import get_router

        router = get_router()
        result = router.find_comfyui()
        if result:
            _, cap = result
            comfyui_url = cap.endpoint_url
    except Exception:
        pass

    # Check if ComfyUI is reachable
    parsed = urlparse(comfyui_url)
    try:
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 8188, timeout=5)
        conn.request("GET", "/system_stats")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        if resp.status != 200:
            return f"Error: ComfyUI not responding at {comfyui_url}. Start it with: cd ~/nvidia-workbench/ComfyUI && ./start.sh"
    except Exception:
        return f"Error: Cannot connect to ComfyUI at {comfyui_url}. Start it with: cd ~/nvidia-workbench/ComfyUI && ./start.sh"

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    # Build Flux workflow
    filename_prefix = f"maude/gen_{seed}"
    workflow = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["10", 0],
                "positive": ["55", 0],
                "negative": ["19", 0],
                "latent_image": ["6", 0],
                "seed": seed,
                "control_after_generate": "fixed",
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "5": {
            "class_type": "CLIPTextEncodeFlux",
            "inputs": {"clip": ["4", 0], "clip_l": prompt, "t5xxl": prompt, "guidance": 4},
        },
        "19": {
            "class_type": "CLIPTextEncodeFlux",
            "inputs": {"clip": ["4", 0], "clip_l": "", "t5xxl": "", "guidance": 4},
        },
        "55": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["5", 0], "guidance": 3.5}},
        "4": {
            "class_type": "DualCLIPLoader",
            "inputs": {"clip_name1": "t5xxl_fp16.safetensors", "clip_name2": "clip_l.safetensors", "type": "flux"},
        },
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["8", 0]}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "10": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-dev.safetensors", "weight_dtype": "default"}},
        "38": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": filename_prefix}},
    }

    # Add LoRA if requested
    if lora:
        lora_map = {
            "stillion-style": "stillion-style.safetensors",
            "marker-mech-style": "marker-mech-style.safetensors",
        }
        lora_file = lora_map.get(lora, f"{lora}.safetensors")
        workflow["50"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["10", 0],
                "clip": ["4", 0],
                "lora_name": lora_file,
                "strength_model": 1.0,
                "strength_clip": 1.0,
            },
        }
        # Rewire: KSampler uses LoRA output instead of raw model
        workflow["3"]["inputs"]["model"] = ["50", 0]
        # CLIP encoder uses LoRA clip output
        workflow["5"]["inputs"]["clip"] = ["50", 1]
        workflow["19"]["inputs"]["clip"] = ["50", 1]

    # Queue the prompt
    try:
        body = json.dumps({"prompt": workflow, "client_id": "maude"}).encode()
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 8188, timeout=30)
        conn.request("POST", "/prompt", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        result = json.loads(resp.read())
        conn.close()

        if "error" in result:
            return f"Error queuing prompt: {result['error']}"
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            return f"Failed to queue prompt: {result}"
    except Exception as e:
        return f"Error queuing prompt: {e}"

    log(f"Flux generation queued: {prompt_id} (seed={seed}, steps={steps})")

    # Poll for completion (up to 5 minutes)
    for _ in range(150):
        time.sleep(2)
        try:
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 8188, timeout=10)
            conn.request("GET", f"/history/{prompt_id}")
            resp = conn.getresponse()
            history = json.loads(resp.read())
            conn.close()

            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                if "38" in outputs:
                    images = outputs["38"].get("images", [])
                    if images:
                        img_info = images[0]
                        # Build path to the generated image in ComfyUI output dir
                        comfyui_output = Path.home() / "nvidia-workbench" / "ComfyUI" / "app" / "output"
                        subfolder = img_info.get("subfolder", "")
                        filename = img_info["filename"]
                        src = comfyui_output / subfolder / filename if subfolder else comfyui_output / filename

                        # Copy to shared folder for serving via gateway
                        shared_dir = Path.home() / "nvidia-workbench" / "terminal-llm" / "shared"
                        shared_dir.mkdir(parents=True, exist_ok=True)
                        dest_name = f"flux_{seed}.png"
                        dest = shared_dir / dest_name
                        shutil.copy2(str(src), str(dest))

                        log(f"Image generated: {dest}")
                        return (
                            f"Image generated successfully!\n"
                            f"Seed: {seed}\n"
                            f"File: {dest}\n"
                            f"Display with: ![{prompt[:50]}](/download/{dest_name})"
                        )
        except Exception:
            continue

    return f"Timeout waiting for image generation (prompt_id: {prompt_id}). Check ComfyUI at {comfyui_url}"


# ── Registry wrapper ──────────────────────────────────────────


@register_tool("generate_image")
def _dispatch_generate_image(args):
    return tool_generate_image(
        args.get("prompt", ""),
        args.get("width", 1024),
        args.get("height", 1024),
        args.get("seed", -1),
        args.get("steps", 28),
        args.get("lora"),
    )
