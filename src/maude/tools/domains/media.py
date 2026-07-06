"""Domain-owned tool schemas."""

TOOL_NAMES = {
    "generate_3d",
    "generate_image",
    "generate_image_flux2",
    "generate_image_flux2_klein",
    "hyperframes_browser_ensure",
    "hyperframes_doctor",
    "hyperframes_init",
    "hyperframes_lint",
    "hyperframes_render",
    "video_pre_publish_checklist",
}

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image using local ComfyUI. Defaults to Flux2 Klein 4B for faster local mobile generation. Use "
            "model=flux1-dev only when Flux 1 Dev is explicitly requested. Saves to shared folder and returns a download URL. Use markdown "
            "![desc](/download/filename.png) to display.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Text description of the image to generate"},
                    "width": {"type": "integer", "description": "Image width (default 1024)"},
                    "height": {"type": "integer", "description": "Image height (default 1024)"},
                    "seed": {"type": "integer", "description": "Seed for reproducibility (-1 for random)"},
                    "steps": {"type": "integer", "description": "Sampling steps. Default: 4 for Flux2 Klein 4B, 28 for Flux 1 Dev"},
                    "lora": {"type": "string", "description": "Optional LoRA name for Flux 1 Dev: stillion-style, marker-mech-style"},
                    "model": {
                        "type": "string",
                        "enum": ["flux1-dev", "flux2-klein-4b", "flux2-klein", "klein", "klein-4b"],
                        "description": "Local ComfyUI model/workflow. Default: flux2-klein-4b. Use flux1-dev only when Flux 1 Dev is explicitly requested.",
                    },
                    "workflow_path": {
                        "type": "string",
                        "description": "Optional path to an API-format ComfyUI workflow JSON. Defaults to data/comfyui_workflows/flux2_klein_4b.json for flux2-klein-4b.",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image_flux2_klein",
            "description": "Generate an image using the local Flux2 Klein 4B ComfyUI workflow. Use this when the user asks for Flux2 Klein, Flux 2 Klein, or Klein 4B. Requires the local API-format workflow JSON at data/comfyui_workflows/flux2_klein_4b.json or MAUDE_COMFYUI_FLUX2_KLEIN_WORKFLOW.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Text description of the image to generate"},
                    "width": {"type": "integer", "description": "Image width (default 1024)"},
                    "height": {"type": "integer", "description": "Image height (default 1024)"},
                    "seed": {"type": "integer", "description": "Seed for reproducibility (-1 for random)"},
                    "steps": {"type": "integer", "description": "Sampling steps (default 4)"},
                    "workflow_path": {
                        "type": "string",
                        "description": "Optional path to the API-format ComfyUI workflow JSON. Defaults to data/comfyui_workflows/flux2_klein_4b.json.",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image_flux2",
            "description": "Generate an image using Flux 2 via Replicate (cloud, paid). Only use when the user "
            "explicitly asks for Flux 2 / flux2. Do not use as a fallback for local ComfyUI "
            "failures. Saves to shared folder and returns a download URL. Use markdown "
            "![desc](/download/filename.png) to display.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Text description of the image to generate"},
                    "model": {
                        "type": "string",
                        "enum": ["pro", "dev", "klein"],
                        "description": "Flux 2 variant: pro (best quality), dev (open "
                        "weights), klein (cheapest). Default: pro",
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"],
                        "description": "Output aspect ratio. Default: 1:1",
                    },
                    "seed": {"type": "integer", "description": "Seed for reproducibility (-1 for random)"},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_3d",
            "description": "Generate a 3D GLB model from an image using TRELLIS 2 (local, free) or Tripo (cloud, "
            "better topology). Returns file path to the GLB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to the input image file"},
                    "engine": {
                        "type": "string",
                        "enum": ["trellis", "tripo", "both"],
                        "description": "Which engine to use: trellis (free, local), "
                        "tripo (cloud, costs credits), or both in "
                        "parallel",
                    },
                    "resolution": {
                        "type": "string",
                        "enum": ["512", "1024", "1536"],
                        "description": "TRELLIS resolution (default 1024)",
                    },
                    "seed": {"type": "integer", "description": "Random seed for TRELLIS generation"},
                    "quad": {"type": "boolean", "description": "Enable quad remeshing for Tripo (cleaner topology)"},
                    "face_limit": {
                        "type": "integer",
                        "description": "Target face count for Tripo (e.g. 50000, 100000)",
                    },
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hyperframes_doctor",
            "description": "Run HyperFrames diagnostics for Node.js, FFmpeg, Chrome, Docker, and render readiness. "
            "Use before first HyperFrames render or when video rendering fails.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hyperframes_browser_ensure",
            "description": "Install or verify the managed Chrome browser required by HyperFrames local rendering. "
            "Use when hyperframes_doctor reports Chrome is missing.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hyperframes_init",
            "description": "Create a new HyperFrames HTML-to-video project under the configured HyperFrames runtime workspace using the "
            "HyperFrames CLI. Use when starting a new programmatic video composition.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Project name, e.g. product-intro"},
                    "example": {
                        "type": "string",
                        "description": "Built-in HyperFrames example/template name. Default: blank",
                    },
                    "tailwind": {
                        "type": "boolean",
                        "description": "Create a Tailwind-enabled project if supported by the CLI.",
                    },
                    "install_skills": {
                        "type": "boolean",
                        "description": "Allow HyperFrames to install its AI coding skills during init. Default: false",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hyperframes_lint",
            "description": "Lint a HyperFrames project for missing timing attributes, adapter libraries, media "
            "issues, and other structural problems before rendering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_path": {"type": "string", "description": "Path to the HyperFrames project directory"},
                    "verbose": {"type": "boolean", "description": "Include informational lint findings."},
                },
                "required": ["project_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hyperframes_render",
            "description": "Render a HyperFrames project to MP4, MOV, or WebM via the HyperFrames CLI. Can share "
            "the finished video through Maude's shared download folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_path": {"type": "string", "description": "Path to the HyperFrames project directory"},
                    "output": {
                        "type": "string",
                        "description": "Optional output path. Defaults to project/renders/<project>-timestamp.<format>",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["mp4", "mov", "webm"],
                        "description": "Output format. Default: mp4",
                    },
                    "fps": {"type": "integer", "enum": [24, 30, 60], "description": "Frame rate. Default: 30"},
                    "quality": {
                        "type": "string",
                        "enum": ["draft", "standard", "high"],
                        "description": "Encoding quality. Use draft for quick previews, high for final delivery.",
                    },
                    "docker": {"type": "boolean", "description": "Use Docker mode for deterministic rendering."},
                    "gpu": {"type": "boolean", "description": "Use hardware video encoding when available."},
                    "workers": {"type": "string", "description": "Worker count, e.g. auto, 1, 2, 4. Default: auto"},
                    "share": {
                        "type": "boolean",
                        "description": "Copy the rendered video to Maude's shared download folder. Default: true",
                    },
                    "timeout": {"type": "integer", "description": "Render timeout in seconds. Default: 900"},
                },
                "required": ["project_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "video_pre_publish_checklist",
            "description": "Create a required markdown pre-publish checklist artifact for any video before "
            "publishing or posting. Use after rendering and before youtube_upload or social_post "
            "with video. Blocks if raw URLs are visible, link placement is missing, transitions are "
            "clipped, text is unreadable, placeholders remain, media is missing, or the MP4 has not "
            "been reviewed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Optional HyperFrames or video project "
                        "directory; checklist is saved under "
                        "project/reviews/ when provided.",
                    },
                    "video_path": {"type": "string", "description": "Path to the rendered video file."},
                    "platform": {
                        "type": "string",
                        "description": "Destination platform, e.g. YouTube Shorts, X, LinkedIn, TikTok, Instagram.",
                    },
                    "title": {"type": "string", "description": "Final video/post title."},
                    "description": {
                        "type": "string",
                        "description": "Final post/video description or caption. Put actual clickable links here.",
                    },
                    "link_url": {
                        "type": "string",
                        "description": "Clean URL to include in the description/caption, if needed.",
                    },
                    "link_placement": {
                        "type": "string",
                        "description": "Where the clickable link is placed, e.g. description, caption, profile link.",
                    },
                    "cta_text": {
                        "type": "string",
                        "description": "Designed on-screen CTA text. Do not use raw URLs unless explicitly requested.",
                    },
                    "timing_notes": {
                        "type": "string",
                        "description": "Notes confirming screen "
                        "wipes/reveals/transitions had enough "
                        "duration and hold time.",
                    },
                    "render_review": {
                        "type": "string",
                        "description": "Notes from watching/scrubbing the rendered MP4.",
                    },
                    "tags": {"type": "string", "description": "Comma-separated tags or hashtags."},
                    "privacy": {"type": "string", "description": "Privacy or visibility setting."},
                    "media_attached": {
                        "type": "boolean",
                        "description": "True only when the rendered media file exists and is the file to publish.",
                    },
                    "transition_timing_ok": {
                        "type": "boolean",
                        "description": "True only after confirming screen wipes/reveals/transitions complete and hold.",
                    },
                    "text_readable": {
                        "type": "boolean",
                        "description": "True only after confirming on-screen text is readable in the render.",
                    },
                    "no_placeholders": {
                        "type": "boolean",
                        "description": "True only after confirming no "
                        "placeholders, broken assets, or "
                        "missing media are visible.",
                    },
                    "render_reviewed": {
                        "type": "boolean",
                        "description": "True only after reviewing the actual "
                        "rendered MP4, not just source HTML or "
                        "a still frame.",
                    },
                    "raw_url_in_video": {
                        "type": "boolean",
                        "description": "Must be false unless the user explicitly requested raw URLs on screen.",
                    },
                    "link_in_description": {
                        "type": "boolean",
                        "description": "True when the real clickable link is in the description/caption/profile field.",
                    },
                },
                "required": ["video_path", "title", "description", "cta_text", "render_review"],
            },
        },
    },
]


def schemas() -> list[dict]:
    return list(SCHEMAS)
