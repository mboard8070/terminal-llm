"""
Replicate image generation for the MAUDE Gateway.

Provides img2img stylization via Flux 2 Klein and text-to-image generation.
Used by the /api/image/stylize endpoint and the generate_image_replicate tool.
"""

import http.client
import json
import os
import ssl
import time

from .state import logger

REPLICATE_BASE = "api.replicate.com"
DEFAULT_MODEL = "black-forest-labs/flux-2-klein"


def _get_token() -> str:
    return os.environ.get("REPLICATE_API_TOKEN", "")


def _api_request(method: str, path: str, body: dict | None = None) -> dict:
    """Make an HTTPS request to the Replicate API."""
    token = _get_token()
    if not token:
        raise RuntimeError("REPLICATE_API_TOKEN not set")

    ctx = ssl.create_default_context()
    conn = http.client.HTTPSConnection(REPLICATE_BASE, 443, timeout=300, context=ctx)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = json.dumps(body).encode() if body else None
    conn.request(method, path, body=payload, headers=headers)
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return data


def _poll_prediction(prediction: dict, timeout: int = 300) -> dict:
    """Poll a Replicate prediction until completion."""
    poll_url = prediction.get("urls", {}).get("get", "")
    if not poll_url:
        return prediction

    # Extract path from full URL
    path = poll_url.replace(f"https://{REPLICATE_BASE}", "")

    for _ in range(timeout):
        time.sleep(1)
        result = _api_request("GET", path)
        status = result.get("status", "")
        if status == "succeeded":
            return result
        if status in ("failed", "canceled"):
            error = result.get("error", "Unknown error")
            raise RuntimeError(f"Prediction {status}: {error}")
    raise RuntimeError("Prediction timed out")


def stylize_image(
    prompt: str,
    image_base64: str,
    strength: float = 0.55,
    model: str = DEFAULT_MODEL,
    width: int = 1024,
    height: int = 1024,
) -> dict:
    """Run img2img stylization via Replicate.

    Args:
        prompt: Style description (e.g. "oil painting, impressionist")
        image_base64: Base64-encoded input image (no data URI prefix)
        strength: How much to deviate from input (0.0=keep, 1.0=ignore)
        model: Replicate model identifier
        width: Output width
        height: Output height

    Returns:
        dict with "url" (output image URL) and "prediction_id"
    """
    # Ensure data URI format
    if not image_base64.startswith("data:"):
        image_base64 = f"data:image/png;base64,{image_base64}"

    input_params = {
        "prompt": prompt,
        "input_images": [image_base64],
        "go_fast": True,
        "width": width,
        "height": height,
    }

    # Map strength to guidance (same formula as Pixelus pipeline)
    input_params["guidance"] = 3.0 + strength * 27.0

    logger.info("Replicate stylize: model=%s prompt=%.60s strength=%.2f", model, prompt, strength)

    prediction = _api_request(
        "POST",
        f"/v1/models/{model}/predictions",
        {"input": input_params},
    )

    if prediction.get("error"):
        raise RuntimeError(f"Replicate error: {prediction['error']}")

    result = _poll_prediction(prediction)
    output = result.get("output")

    # Output can be a string URL or a list of URLs
    if isinstance(output, list):
        url = output[0] if output else ""
    else:
        url = output or ""

    return {
        "url": url,
        "prediction_id": result.get("id", ""),
        "model": model,
    }


def generate_image(
    prompt: str,
    model: str = DEFAULT_MODEL,
    width: int = 1024,
    height: int = 1024,
) -> dict:
    """Text-to-image generation via Replicate.

    Args:
        prompt: Image description
        model: Replicate model identifier
        width: Output width
        height: Output height

    Returns:
        dict with "url" (output image URL) and "prediction_id"
    """
    input_params = {
        "prompt": prompt,
        "go_fast": True,
        "width": width,
        "height": height,
    }

    logger.info("Replicate generate: model=%s prompt=%.60s", model, prompt)

    prediction = _api_request(
        "POST",
        f"/v1/models/{model}/predictions",
        {"input": input_params},
    )

    if prediction.get("error"):
        raise RuntimeError(f"Replicate error: {prediction['error']}")

    result = _poll_prediction(prediction)
    output = result.get("output")

    if isinstance(output, list):
        url = output[0] if output else ""
    else:
        url = output or ""

    return {
        "url": url,
        "prediction_id": result.get("id", ""),
        "model": model,
    }
