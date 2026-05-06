"""
HyperFrames tools - HTML/CSS/JS video composition rendering.
"""

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from tool_registry import register_tool

from .log import log
from .paths import resolve_path
from .tools_shared import SHARED_DIR

HYPERFRAMES_WORKSPACE = Path.home() / "nvidia-workbench" / "terminal-llm" / "data" / "hyperframes"
_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_.-]+")
_FORMATS = {"mp4", "mov", "webm"}
_QUALITIES = {"draft", "standard", "high"}


def _slug(value: str, fallback: str = "maude-video") -> str:
    slug = _SAFE_NAME_RE.sub("-", (value or "").strip()).strip(".-").lower()
    return slug or fallback


def _npx() -> str | None:
    return shutil.which("npx")


def _run_hyperframes(args: list[str], cwd: Path | None = None, timeout: int = 120) -> tuple[int, str]:
    npx = _npx()
    if not npx:
        return 127, "Error: npx not found. Install Node.js 22+ and npm before using HyperFrames."

    env = {
        "CI": "1",
        "HYPERFRAMES_NO_UPDATE_CHECK": "1",
        "HYPERFRAMES_NO_TELEMETRY": "1",
    }
    cmd = [npx, "--yes", "hyperframes", *args]
    log(f"$ {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env={**os.environ, **env},
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, f"Error: hyperframes {' '.join(args)} timed out after {timeout}s."
    except Exception as e:
        return 1, f"Error running HyperFrames: {e}"

    output = "\n".join(part for part in (proc.stdout.strip(), proc.stderr.strip()) if part)
    return proc.returncode, output[-8000:] if len(output) > 8000 else output


def _project_path(path: str | None) -> Path:
    if not path:
        return HYPERFRAMES_WORKSPACE
    return resolve_path(path)


def tool_hyperframes_doctor() -> str:
    """Run HyperFrames dependency diagnostics."""
    code, output = _run_hyperframes(["doctor"], timeout=180)
    has_failed_check = "Some checks failed" in output
    prefix = "HyperFrames doctor passed." if code == 0 and not has_failed_check else "HyperFrames doctor failed."
    return f"{prefix}\n\n{output}".strip()


def tool_hyperframes_browser_ensure() -> str:
    """Install or verify the managed Chrome browser HyperFrames needs."""
    code, output = _run_hyperframes(["browser", "ensure"], timeout=600)
    prefix = "HyperFrames browser is ready." if code == 0 else "HyperFrames browser setup failed."
    return f"{prefix}\n\n{output}".strip()


def tool_hyperframes_init(
    name: str, example: str = "blank", tailwind: bool = False, install_skills: bool = False
) -> str:
    """Scaffold a HyperFrames project under data/hyperframes."""
    project_name = _slug(name)
    example_name = _slug(example or "blank", "blank")
    HYPERFRAMES_WORKSPACE.mkdir(parents=True, exist_ok=True)
    project_dir = HYPERFRAMES_WORKSPACE / project_name
    if project_dir.exists():
        return f"Error: HyperFrames project already exists: {project_dir}"

    args = ["init", project_name, "--non-interactive", "--example", example_name]
    if tailwind:
        args.append("--tailwind")
    if not install_skills:
        args.append("--skip-skills")

    code, output = _run_hyperframes(args, cwd=HYPERFRAMES_WORKSPACE, timeout=300)
    if code != 0:
        return f"Error creating HyperFrames project '{project_name}'.\n\n{output}".strip()

    return (
        f"Created HyperFrames project: {project_dir}\n"
        f"Example: {example_name}\n"
        f"Next: edit {project_dir / 'index.html'}, then run hyperframes_lint and hyperframes_render.\n\n"
        f"{output}"
    ).strip()


def tool_hyperframes_lint(project_path: str, verbose: bool = False) -> str:
    """Lint a HyperFrames project for composition issues."""
    project_dir = _project_path(project_path)
    if not project_dir.exists():
        return f"Error: project path not found: {project_dir}"
    args = ["lint", str(project_dir)]
    if verbose:
        args.append("--verbose")
    code, output = _run_hyperframes(args, cwd=project_dir, timeout=180)
    prefix = "HyperFrames lint passed." if code == 0 else "HyperFrames lint found issues."
    return f"{prefix}\nProject: {project_dir}\n\n{output}".strip()


def tool_hyperframes_render(
    project_path: str,
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
    """Render a HyperFrames project to video and optionally share it."""
    project_dir = _project_path(project_path)
    if not project_dir.exists():
        return f"Error: project path not found: {project_dir}"
    if not (project_dir / "index.html").exists():
        return f"Error: {project_dir} does not look like a HyperFrames project; missing index.html."

    fmt = (format or "mp4").lower()
    if fmt not in _FORMATS:
        return f"Error: unsupported format '{format}'. Use one of: {', '.join(sorted(_FORMATS))}."
    if quality not in _QUALITIES:
        return f"Error: unsupported quality '{quality}'. Use one of: {', '.join(sorted(_QUALITIES))}."

    fps = int(fps or 30)
    if fps not in (24, 30, 60):
        return "Error: fps must be one of 24, 30, or 60."

    render_dir = project_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    if output:
        raw_output = Path(output)
        output_path = raw_output if raw_output.is_absolute() else project_dir / raw_output
    else:
        output_path = render_dir / f"{project_dir.name}-{int(time.time())}.{fmt}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "render",
        "--output",
        str(output_path),
        "--format",
        fmt,
        "--fps",
        str(fps),
        "--quality",
        quality,
        "--workers",
        str(workers or "auto"),
    ]
    if docker:
        args.append("--docker")
    if gpu:
        args.append("--gpu")

    code, output_text = _run_hyperframes(args, cwd=project_dir, timeout=int(timeout or 900))
    if code != 0:
        return f"Error rendering HyperFrames project: {project_dir}\n\n{output_text}".strip()
    if not output_path.exists():
        return f"HyperFrames reported success, but output was not found: {output_path}\n\n{output_text}".strip()

    result = f"Rendered HyperFrames video: {output_path}\n\n{output_text}".strip()
    if share:
        SHARED_DIR.mkdir(parents=True, exist_ok=True)
        shared_name = f"hyperframes_{project_dir.name}_{int(time.time())}.{fmt}"
        shared_path = SHARED_DIR / shared_name
        shutil.copy2(output_path, shared_path)
        result += f"\n\nShared copy: {shared_path}\nDownload: /download/{shared_name}"
    return result


# Registry wrappers


@register_tool("hyperframes_doctor")
def _dispatch_hyperframes_doctor(args):
    return tool_hyperframes_doctor()


@register_tool("hyperframes_browser_ensure")
def _dispatch_hyperframes_browser_ensure(args):
    return tool_hyperframes_browser_ensure()


@register_tool("hyperframes_init")
def _dispatch_hyperframes_init(args):
    return tool_hyperframes_init(
        args.get("name", ""),
        args.get("example", "blank"),
        args.get("tailwind", False),
        args.get("install_skills", False),
    )


@register_tool("hyperframes_lint")
def _dispatch_hyperframes_lint(args):
    return tool_hyperframes_lint(args.get("project_path", ""), args.get("verbose", False))


@register_tool("hyperframes_render")
def _dispatch_hyperframes_render(args):
    return tool_hyperframes_render(
        args.get("project_path", ""),
        args.get("output"),
        args.get("format", "mp4"),
        args.get("fps", 30),
        args.get("quality", "standard"),
        args.get("docker", False),
        args.get("gpu", False),
        args.get("workers", "auto"),
        args.get("share", True),
        args.get("timeout", 900),
    )
