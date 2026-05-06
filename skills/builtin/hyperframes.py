"""
HyperFrames skill wrapper.

Exposes Maude's native HyperFrames tools in the built-in skills list so
Maude can recognize requests that ask for the "HyperFrames skill".
"""

from skills import skill


@skill(
    name="hyperframes",
    description="Create, lint, diagnose, and render HyperFrames HTML/CSS/JS video compositions.",
    version="1.0.0",
    author="MAUDE",
    triggers=[
        "hyperframes",
        "hyperframe",
        "html to video",
        "programmatic video",
        "render video",
        "motion graphics",
    ],
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["doctor", "browser_ensure", "init", "lint", "render"],
                "description": "HyperFrames action to run. Default: doctor",
                "default": "doctor",
            },
            "project_path": {"type": "string", "description": "Path to a HyperFrames project for lint/render"},
            "name": {"type": "string", "description": "Project name for init"},
            "example": {"type": "string", "description": "HyperFrames init example/template. Default: blank"},
            "tailwind": {"type": "boolean", "description": "Enable Tailwind support during init"},
            "install_skills": {"type": "boolean", "description": "Allow HyperFrames init to install its own agent skills"},
            "output": {"type": "string", "description": "Optional render output path"},
            "format": {"type": "string", "enum": ["mp4", "mov", "webm"], "description": "Render format"},
            "fps": {"type": "integer", "enum": [24, 30, 60], "description": "Render frame rate"},
            "quality": {"type": "string", "enum": ["draft", "standard", "high"], "description": "Render quality"},
            "docker": {"type": "boolean", "description": "Use Docker mode for deterministic rendering"},
            "gpu": {"type": "boolean", "description": "Use hardware video encoding when available"},
            "workers": {"type": "string", "description": "Render worker count, e.g. auto, 1, 2, 4"},
            "share": {"type": "boolean", "description": "Copy rendered video to Maude's shared download folder"},
            "timeout": {"type": "integer", "description": "Render timeout in seconds"},
        },
        "required": [],
    },
)
def hyperframes(
    action: str = "doctor",
    project_path: str = None,
    name: str = None,
    example: str = "blank",
    tailwind: bool = False,
    install_skills: bool = False,
    output: str = None,
    format: str = "mp4",
    fps: int = 30,
    quality: str = "standard",
    docker: bool = False,
    gpu: bool = False,
    workers: str = "auto",
    share: bool = True,
    timeout: int = 900,
) -> str:
    """Run HyperFrames through Maude's native HyperFrames tool implementations."""
    from maude_core.tools_hyperframes import (
        tool_hyperframes_browser_ensure,
        tool_hyperframes_doctor,
        tool_hyperframes_init,
        tool_hyperframes_lint,
        tool_hyperframes_render,
    )

    action = (action or "doctor").lower()
    if action == "doctor":
        return tool_hyperframes_doctor()
    if action in ("browser_ensure", "browser", "ensure"):
        return tool_hyperframes_browser_ensure()
    if action == "init":
        if not name:
            return "Error: name is required for HyperFrames init."
        return tool_hyperframes_init(name, example, tailwind, install_skills)
    if action == "lint":
        if not project_path:
            return "Error: project_path is required for HyperFrames lint."
        return tool_hyperframes_lint(project_path)
    if action == "render":
        if not project_path:
            return "Error: project_path is required for HyperFrames render."
        return tool_hyperframes_render(
            project_path,
            output,
            format,
            fps,
            quality,
            docker,
            gpu,
            workers,
            share,
            timeout,
        )
    return "Error: unknown HyperFrames action. Use doctor, browser_ensure, init, lint, or render."
