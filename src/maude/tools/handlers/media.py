"""
Image generation tool — Flux via ComfyUI.
"""

import http.client
import json
import os
import ssl
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

from maude.config import runtime_paths

from maude_core.log import log
from tool_registry import register_tool

REPLICATE_BASE = "api.replicate.com"
FLUX2_MODEL_MAP = {
    "pro": "black-forest-labs/flux-2-pro",
    "dev": "black-forest-labs/flux-2-dev",
    "klein": "black-forest-labs/flux-2-klein",
}

LOCAL_COMFYUI_MODEL_ALIASES = {
    "flux1": "flux1-dev",
    "flux1-dev": "flux1-dev",
    "flux": "flux1-dev",
    "flux2-klein": "flux2-klein-4b",
    "flux2_klein": "flux2-klein-4b",
    "flux2-klein-4b": "flux2-klein-4b",
    "flux2_klein_4b": "flux2-klein-4b",
    "klein": "flux2-klein-4b",
    "klein-4b": "flux2-klein-4b",
}

LOCAL_COMFYUI_WORKFLOW_PRESETS = {
    "flux2-klein-4b": (
        "MAUDE_COMFYUI_FLUX2_KLEIN_WORKFLOW",
        runtime_paths().project_root / "data" / "comfyui_workflows" / "flux2_klein_4b.json",
    ),
}


def _default_local_image_model() -> str:
    configured = os.environ.get("MAUDE_DEFAULT_IMAGE_MODEL", "flux2-klein-4b")
    return LOCAL_COMFYUI_MODEL_ALIASES.get(configured.lower(), configured.lower())


def _default_steps_for_model(model: str) -> int:
    return 4 if model == "flux2-klein-4b" else 28


def tool_generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
    steps: int | None = None,
    lora: str = None,
    model: str | None = None,
    workflow_path: str | None = None,
) -> str:
    """Generate an image using a local ComfyUI workflow."""
    import random
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
    if not _comfyui_ready(parsed):
        _start_comfyui_service()
        for _ in range(30):
            time.sleep(1)
            if _comfyui_ready(parsed):
                break
        else:
            return (
                f"Error: Cannot connect to ComfyUI at {comfyui_url}. "
                "Start/check it with: systemctl --user status maude-comfyui"
            )

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    requested_model = str(model or _default_local_image_model()).lower()
    local_model = LOCAL_COMFYUI_MODEL_ALIASES.get(requested_model, requested_model)
    if steps is None:
        steps = _default_steps_for_model(local_model)
    filename_prefix = f"maude/gen_{local_model.replace('/', '_')}_{seed}"

    if local_model != "flux1-dev":
        try:
            workflow = _load_local_comfyui_workflow(local_model, workflow_path)
            _patch_comfyui_workflow(workflow, prompt, width, height, seed, steps, filename_prefix)
        except Exception as exc:
            return str(exc)
    else:
        # Build Flux 1 Dev workflow
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
    if local_model == "flux1-dev" and lora:
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

    timeout_seconds = int(os.environ.get("MAUDE_COMFYUI_IMAGE_TIMEOUT_SECONDS", "1200"))
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        time.sleep(2)
        try:
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 8188, timeout=10)
            conn.request("GET", f"/history/{prompt_id}")
            resp = conn.getresponse()
            history = json.loads(resp.read())
            conn.close()

            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                saved = _copy_first_comfyui_output_image(outputs, prompt, seed, local_model)
                if saved:
                    dest, dest_name = saved
                    log(f"Image generated: {dest}")
                    return (
                        f"Image generated successfully!\n"
                        f"Model: {local_model}\n"
                        f"Seed: {seed}\n"
                        f"File: {dest}\n"
                        f"Display with: ![{prompt[:50]}](/download/{dest_name})"
                    )
        except Exception:
            continue

    return (
        f"Timeout waiting for image generation after {timeout_seconds}s "
        f"(prompt_id: {prompt_id}). Check ComfyUI at {comfyui_url}"
    )



def _load_local_comfyui_workflow(model: str, workflow_path: str | None = None) -> dict:
    preset = LOCAL_COMFYUI_WORKFLOW_PRESETS.get(model)
    if not preset and not workflow_path:
        supported = ", ".join(sorted(LOCAL_COMFYUI_WORKFLOW_PRESETS))
        raise ValueError(f"Error: Unknown local ComfyUI image model '{model}'. Supported custom models: {supported}")

    env_name, default_path = preset if preset else ("MAUDE_COMFYUI_WORKFLOW", None)
    raw_path = workflow_path or os.environ.get(env_name, "") or str(default_path or "")
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Error: ComfyUI workflow for {model} not found at {path}. "
            f"Export the workflow in API format and save it there, or set {env_name}."
        )

    data = json.loads(path.read_text())
    workflow = data.get("prompt", data) if isinstance(data, dict) else data
    if not isinstance(workflow, dict):
        raise ValueError(f"Error: ComfyUI workflow at {path} is not a JSON object.")
    if "nodes" in workflow and not any(isinstance(v, dict) and "class_type" in v for v in workflow.values()):
        raise ValueError(
            f"Error: {path} looks like a ComfyUI UI workflow export. "
            "Save/export it in API prompt format before using it with MAUDE."
        )
    return json.loads(json.dumps(workflow))


def _replace_prompt_placeholders(value, prompt: str):
    if isinstance(value, str):
        for token in ("{{prompt}}", "{{PROMPT}}", "__PROMPT__", "$PROMPT"):
            value = value.replace(token, prompt)
    return value


def _patch_comfyui_workflow(
    workflow: dict,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    filename_prefix: str,
) -> None:
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        class_type = str(node.get("class_type", "")).lower()
        is_negative = "negative" in class_type
        for key, value in list(inputs.items()):
            key_l = str(key).lower()
            if isinstance(value, str):
                inputs[key] = _replace_prompt_placeholders(value, prompt)
                value = inputs[key]
            if key_l in {"seed", "noise_seed"}:
                inputs[key] = seed
            elif key_l == "steps":
                inputs[key] = steps
            elif key_l == "width":
                inputs[key] = width
            elif key_l == "height":
                inputs[key] = height
            elif key_l == "filename_prefix":
                inputs[key] = filename_prefix
            elif key_l in {"prompt", "positive", "caption", "text"} and not is_negative:
                inputs[key] = prompt
            elif key_l in {"clip_l", "t5xxl"} and not is_negative and str(value).strip():
                inputs[key] = prompt


def _copy_first_comfyui_output_image(outputs: dict, prompt: str, seed: int, model: str):
    import shutil

    for output in outputs.values():
        if not isinstance(output, dict):
            continue
        images = output.get("images", [])
        if not images:
            continue
        img_info = images[0]
        comfyui_output = runtime_paths().comfyui_output_dir
        subfolder = img_info.get("subfolder", "")
        filename = img_info["filename"]
        src = comfyui_output / subfolder / filename if subfolder else comfyui_output / filename

        shared_dir = runtime_paths().shared_dir
        shared_dir.mkdir(parents=True, exist_ok=True)
        safe_model = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in model)
        dest_name = f"{safe_model}_{seed}.png"
        dest = shared_dir / dest_name
        shutil.copy2(str(src), str(dest))
        return dest, dest_name
    return None

def _comfyui_ready(parsed) -> bool:
    try:
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 8188, timeout=5)
        conn.request("GET", "/system_stats")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        return resp.status == 200
    except Exception:
        return False


def _start_comfyui_service() -> None:
    try:
        subprocess.run(
            ["systemctl", "--user", "start", "maude-comfyui.service"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        pass


def tool_generate_image_flux2_klein(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
    steps: int | None = None,
    workflow_path: str | None = None,
) -> str:
    """Generate an image with the local Flux2 Klein 4B ComfyUI workflow."""
    return tool_generate_image(
        prompt=prompt,
        width=width,
        height=height,
        seed=seed,
        steps=steps,
        model="flux2-klein-4b",
        workflow_path=workflow_path,
    )


def _replicate_request(method: str, path: str, body: dict | None = None, token: str = "") -> dict:
    ctx = ssl.create_default_context()
    conn = http.client.HTTPSConnection(REPLICATE_BASE, 443, timeout=300, context=ctx)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = json.dumps(body).encode() if body else None
    conn.request(method, path, body=payload, headers=headers)
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return data


def tool_generate_image_flux2(
    prompt: str,
    model: str = "pro",
    aspect_ratio: str = "1:1",
    seed: int = -1,
) -> str:
    """Generate an image via Replicate Flux 2 (text-to-image).

    Args:
        prompt: Text description of the image
        model: "pro" (highest quality), "dev" (open weights), or "klein" (cheapest)
        aspect_ratio: "1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"
        seed: -1 for random, otherwise explicit seed
    """
    import random
    import urllib.request

    if os.environ.get("MAUDE_ENABLE_FLUX2", "").lower() not in {"1", "true", "yes"}:
        return "Error: Flux 2 is disabled by default. Use generate_image for local Flux 1 via ComfyUI, or set MAUDE_ENABLE_FLUX2=true to allow cloud Flux 2."

    token = os.environ.get("REPLICATE_API_TOKEN", "")
    if not token:
        return "Error: REPLICATE_API_TOKEN not set in environment."

    model_id = FLUX2_MODEL_MAP.get(model.lower(), FLUX2_MODEL_MAP["pro"])
    if seed == -1:
        seed = random.randint(0, 2**31 - 1)

    input_params = {"prompt": prompt, "aspect_ratio": aspect_ratio, "seed": seed}

    log(f"Flux 2 generate: model={model_id} seed={seed} ar={aspect_ratio}")

    try:
        prediction = _replicate_request("POST", f"/v1/models/{model_id}/predictions", {"input": input_params}, token)
    except Exception as e:
        return f"Error starting Replicate prediction: {e}"

    if prediction.get("error"):
        return f"Replicate error: {prediction['error']}"

    poll_url = prediction.get("urls", {}).get("get", "")
    if not poll_url:
        return f"No poll URL in prediction response: {prediction}"
    poll_path = poll_url.replace(f"https://{REPLICATE_BASE}", "")

    for _ in range(300):
        time.sleep(1)
        try:
            result = _replicate_request("GET", poll_path, None, token)
        except Exception:
            continue
        status = result.get("status", "")
        if status == "succeeded":
            output = result.get("output")
            url = output[0] if isinstance(output, list) and output else (output or "")
            if not url:
                return f"Prediction succeeded but no output URL: {result}"
            shared_dir = runtime_paths().shared_dir
            shared_dir.mkdir(parents=True, exist_ok=True)
            dest_name = f"flux2_{model}_{seed}.png"
            dest = shared_dir / dest_name
            try:
                urllib.request.urlretrieve(url, str(dest))
            except Exception as e:
                return f"Generated {url} but failed to download locally: {e}"
            log(f"Flux 2 image saved: {dest}")
            return (
                f"Image generated successfully!\n"
                f"Model: {model_id}\n"
                f"Seed: {seed}\n"
                f"File: {dest}\n"
                f"Display with: ![{prompt[:50]}](/download/{dest_name})"
            )
        if status in ("failed", "canceled"):
            return f"Prediction {status}: {result.get('error', 'unknown')}"

    return "Timeout waiting for Flux 2 generation (>5 min)."


# ── Registry wrapper ──────────────────────────────────────────


@register_tool("generate_image")
def _dispatch_generate_image(args):
    return tool_generate_image(
        args.get("prompt", ""),
        args.get("width", 1024),
        args.get("height", 1024),
        args.get("seed", -1),
        args.get("steps"),
        args.get("lora"),
        args.get("model"),
        args.get("workflow_path"),
    )


@register_tool("generate_image_flux2")
def _dispatch_generate_image_flux2(args):
    return tool_generate_image_flux2(
        args.get("prompt", ""),
        args.get("model", "pro"),
        args.get("aspect_ratio", "1:1"),
        args.get("seed", -1),
    )


@register_tool("generate_image_flux2_klein")
def _dispatch_generate_image_flux2_klein(args):
    return tool_generate_image_flux2_klein(
        args.get("prompt", ""),
        args.get("width", 1024),
        args.get("height", 1024),
        args.get("seed", -1),
        args.get("steps", 28),
        args.get("workflow_path"),
    )
