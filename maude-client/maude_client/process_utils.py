"""Process helpers for MAUDE client commands.

subprocess.run(timeout=...) kills only the direct child. On macOS/Linux that can
leave shell grandchildren running; on Windows it can leave child processes under
PowerShell. These helpers start commands in their own process group and tear down
that whole group/tree on timeout or cancellation.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import platform
import signal
import subprocess
import time
from collections.abc import Sequence


_IS_WINDOWS = platform.system().lower() == "windows"


@dataclass
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def shell_command(command: str) -> list[str]:
    """Return the platform shell argv for a user command."""
    if _IS_WINDOWS:
        return ["powershell", "-NoProfile", "-Command", command]
    return ["bash", "-lc", command]


def _popen_kwargs(cwd: str | None = None) -> dict:
    kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "cwd": cwd,
    }
    if _IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return kwargs


def terminate_process_tree(proc: subprocess.Popen, grace_seconds: float = 2.0) -> None:
    """Terminate a subprocess and its children as best as the platform allows."""
    if proc.poll() is not None:
        return

    if _IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        proc.terminate()

    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.05)

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception:
        proc.kill()


def run_process(
    command: Sequence[str],
    *,
    timeout: int,
    cwd: str | None = None,
) -> ProcessResult:
    """Run a command with whole-process-tree cleanup on timeout."""
    proc = subprocess.Popen(list(command), **_popen_kwargs(cwd=cwd))
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return ProcessResult(proc.returncode, stdout or "", stderr or "")
    except subprocess.TimeoutExpired:
        terminate_process_tree(proc)
        stdout, stderr = proc.communicate()
        return ProcessResult(proc.returncode if proc.returncode is not None else -9, stdout or "", stderr or "", True)
