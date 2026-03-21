"""
Image generation skill — Flux via ComfyUI.

Thin wrapper around maude_core.tools_media.tool_generate_image().
Routes to the best available ComfyUI endpoint via the mesh capability router.
"""

from skills import skill


@skill(
    name="generate_image",
    description="Generate images using Flux AI via ComfyUI. Automatically routes to the best available endpoint on the mesh.",
    version="1.0.0",
    author="MAUDE",
    triggers=["generate image", "create image", "draw", "flux", "picture of", "illustration"],
    parameters={
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Text description of the image to generate"},
            "width": {"type": "integer", "description": "Image width in pixels (default: 1024)", "default": 1024},
            "height": {"type": "integer", "description": "Image height in pixels (default: 1024)", "default": 1024},
            "seed": {
                "type": "integer",
                "description": "Random seed for reproducibility (-1 for random)",
                "default": -1,
            },
            "steps": {"type": "integer", "description": "Number of sampling steps (default: 28)", "default": 28},
            "lora": {
                "type": "string",
                "description": "Optional LoRA name (e.g. 'stillion-style', 'marker-mech-style')",
            },
        },
        "required": ["prompt"],
    },
)
def generate_image(
    prompt: str, width: int = 1024, height: int = 1024, seed: int = -1, steps: int = 28, lora: str = None
) -> str:
    """Generate image using Flux via ComfyUI."""
    from maude_core.tools_media import tool_generate_image

    return tool_generate_image(prompt, width, height, seed, steps, lora)
