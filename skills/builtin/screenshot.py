"""Cross-machine screenshot skill - capture screen from remote devices."""

import base64
from datetime import datetime
from pathlib import Path
from skills import skill


SCREENSHOTS_DIR = Path(__file__).resolve().parent.parent.parent / "shared" / "screenshots"


def _dispatch_shell(command: str, target: str) -> str:
    """Dispatch a shell command to a remote machine via collab."""
    from collab_tools import execute_collab_tool
    args = {
        "prompt": command,
        "capability": "SHELL",
    }
    if target:
        args["target"] = target
    return execute_collab_tool("dispatch_task", args)


def _is_dispatch_error(result: str) -> bool:
    """Check if a dispatch result indicates failure."""
    if not result:
        return True
    r = result.strip()
    if r.startswith("ERROR"):
        return True
    if "not found or offline" in r:
        return True
    if "no result yet" in r:
        return True
    if r.startswith("Task failed"):
        return True
    return False


def _extract_output(result: str) -> str:
    """Extract actual output from dispatch_task result string."""
    if not result:
        return ""
    for prefix in ("Task completed on ", "Task failed on "):
        if result.startswith(prefix):
            newline = result.find("\n")
            if newline != -1:
                return result[newline + 1:].strip()
    return result.strip()


def _platform_hint(target: str) -> str:
    """Guess target platform from target name for command construction."""
    t = target.lower()
    if t in ("windows", "win"):
        return "windows"
    if t in ("macos", "mac", "darwin"):
        return "macos"
    if t in ("linux",):
        return "linux"
    return ""


def _build_screenshot_command(platform: str) -> str:
    """Build a platform-aware screenshot + base64 command."""
    if platform == "windows":
        return (
            'Add-Type -AssemblyName System.Windows.Forms,System.Drawing; '
            '$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; '
            '$bmp = New-Object System.Drawing.Bitmap($bounds.Width, $bounds.Height); '
            '$gfx = [System.Drawing.Graphics]::FromImage($bmp); '
            '$gfx.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size); '
            '$ms = New-Object System.IO.MemoryStream; '
            '$bmp.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png); '
            '$gfx.Dispose(); $bmp.Dispose(); '
            '[Convert]::ToBase64String($ms.ToArray())'
        )
    if platform == "macos":
        return (
            'screencapture -x /tmp/maude_ss.png && '
            'base64 /tmp/maude_ss.png && '
            'rm -f /tmp/maude_ss.png'
        )
    # Linux: try multiple tools
    return (
        'SS=/tmp/maude_ss.png && '
        '{ scrot "$SS" 2>/dev/null || '
        'import -window root "$SS" 2>/dev/null || '
        'gnome-screenshot -f "$SS" 2>/dev/null || '
        '{ echo "ERROR: no screenshot tool found (install scrot, imagemagick, or gnome-screenshot)"; exit 1; }; } && '
        'base64 "$SS" && rm -f "$SS"'
    )


@skill(
    name="screenshot",
    description="Take a screenshot of a remote device's screen. Captures the screen, transfers the image back, and saves it locally.",
    version="1.0.0",
    author="MAUDE",
    triggers=["screenshot", "screen capture", "capture screen", "print screen"],
    parameters={
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Target device — hostname, client_id, or platform (e.g. 'windows', 'macos'). Leave empty for default.",
                "default": ""
            },
            "save_path": {
                "type": "string",
                "description": "Custom save path for the screenshot. Leave empty for auto-generated path in shared/screenshots/.",
                "default": ""
            }
        }
    }
)
def screenshot(target: str = "", save_path: str = "") -> str:
    """Take a screenshot of a remote device."""
    platform = _platform_hint(target)
    command = _build_screenshot_command(platform)
    result = _dispatch_shell(command, target)

    if _is_dispatch_error(result):
        return f"Screenshot failed:\n{result}"

    # Extract base64 data from result
    b64_data = _extract_output(result)
    if not b64_data:
        return f"Screenshot failed: no image data received.\nRaw result: {result[:200]}"

    # Clean up base64 — remove any whitespace/newlines that shell may have added
    b64_data = b64_data.replace("\n", "").replace("\r", "").replace(" ", "")

    try:
        image_bytes = base64.b64decode(b64_data)
    except Exception as e:
        return f"Screenshot failed: invalid base64 data — {e}\nFirst 100 chars: {b64_data[:100]}"

    if len(image_bytes) < 100:
        return f"Screenshot failed: image too small ({len(image_bytes)} bytes), likely an error."

    # Determine save path
    if save_path:
        out_path = Path(save_path)
    else:
        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        device = target or "device"
        device = device.replace(" ", "_").replace("/", "_")[:20]
        out_path = SCREENSHOTS_DIR / f"screenshot_{device}_{timestamp}.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(image_bytes)

    size_kb = len(image_bytes) / 1024
    return (
        f"Screenshot saved: {out_path}\n"
        f"Size: {size_kb:.1f} KB ({len(image_bytes)} bytes)\n"
        f"Source: {target or 'target device'}"
    )
