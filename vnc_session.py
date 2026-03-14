"""
VNC Session Manager — enables remote interaction with visible browser windows.

When browser_login is triggered from a remote client, this module:
  1. Starts Xvfb (virtual display) on a free display number
  2. Starts x11vnc to share that display
  3. Starts noVNC/websockify for browser-based access
  4. Returns a URL the user can open on any device

The browser_login tool then launches Chromium on the virtual display,
and the user logs in via the noVNC web interface.

Requirements: apt install xvfb x11vnc novnc python3-websockify
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

try:
    from maude_core import log
except ImportError:
    def log(msg: str):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

VNC_DISPLAY = ":42"           # Virtual display number (high to avoid conflicts)
VNC_PORT = 5942               # x11vnc port (5900 + display number)
NOVNC_PORT = 6080             # noVNC web port
NOVNC_WEB_DIR = "/usr/share/novnc"  # Ubuntu default noVNC install path
SCREEN_SIZE = "1280x900x24"   # Virtual screen resolution
PID_DIR = Path.home() / ".config" / "maude" / "vnc"  # PID file storage


# ─────────────────────────────────────────────────────────────────────────────
# VNC Session
# ─────────────────────────────────────────────────────────────────────────────

class VncSession:
    """Manages Xvfb + x11vnc + noVNC lifecycle with PID file tracking."""

    def __init__(self):
        self._display: str = VNC_DISPLAY
        PID_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def is_active(self) -> bool:
        """Check if VNC processes are running (via PID files, not in-memory state)."""
        return self._pid_alive("xvfb")

    @property
    def display(self) -> str:
        return self._display

    def start(self) -> str:
        """Start the VNC session. Returns the noVNC URL."""
        if self.is_active:
            return self._get_url()

        # Kill any stale processes first
        self._cleanup()

        # Check dependencies
        for cmd in ("Xvfb", "x11vnc", "websockify"):
            if not _which(cmd):
                return (
                    f"Error: '{cmd}' not found. Install with:\n"
                    f"  sudo apt-get install -y xvfb x11vnc novnc python3-websockify"
                )

        try:
            # 1. Start Xvfb
            log(f"Starting Xvfb on {self._display}")
            xvfb = subprocess.Popen(
                ["Xvfb", self._display, "-screen", "0", SCREEN_SIZE, "-ac"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.5)

            if xvfb.poll() is not None:
                return f"Error: Xvfb failed to start on {self._display}. Display may be in use."
            self._write_pid("xvfb", xvfb.pid)

            # 2. Start x11vnc
            log(f"Starting x11vnc on port {VNC_PORT}")
            x11vnc = subprocess.Popen(
                [
                    "x11vnc",
                    "-display", self._display,
                    "-rfbport", str(VNC_PORT),
                    "-nopw",              # No password (local network only)
                    "-forever",           # Don't exit after first client disconnects
                    "-shared",            # Allow multiple viewers
                    "-noxdamage",
                    "-quiet",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.5)

            if x11vnc.poll() is not None:
                self._cleanup()
                return "Error: x11vnc failed to start."
            self._write_pid("x11vnc", x11vnc.pid)

            # 3. Start noVNC via websockify
            log(f"Starting noVNC on port {NOVNC_PORT}")
            novnc = subprocess.Popen(
                [
                    "websockify",
                    "--web", NOVNC_WEB_DIR,
                    str(NOVNC_PORT),
                    f"localhost:{VNC_PORT}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(1.0)

            if novnc.poll() is not None:
                self._cleanup()
                return "Error: noVNC/websockify failed to start."
            self._write_pid("novnc", novnc.pid)

            url = self._get_url()
            log(f"VNC session ready: {url}")
            return url

        except Exception as e:
            self._cleanup()
            return f"Error starting VNC session: {e}"

    def stop(self) -> str:
        """Stop the VNC session and clean up all processes."""
        if not self.is_active:
            return "No VNC session running."

        self._cleanup()
        return "VNC session stopped."

    def _get_url(self) -> str:
        """Build the noVNC access URL."""
        ts_ip = _get_tailscale_ip()
        lan_ip = _get_lan_ip()

        urls = []
        if ts_ip:
            urls.append(f"http://{ts_ip}:{NOVNC_PORT}/vnc.html?autoconnect=true")
        if lan_ip:
            urls.append(f"http://{lan_ip}:{NOVNC_PORT}/vnc.html?autoconnect=true")

        return urls[0] if urls else f"http://localhost:{NOVNC_PORT}/vnc.html?autoconnect=true"

    def get_all_urls(self) -> list:
        """Return all access URLs (Tailscale + LAN + localhost)."""
        urls = []
        ts_ip = _get_tailscale_ip()
        lan_ip = _get_lan_ip()

        if ts_ip:
            urls.append(f"http://{ts_ip}:{NOVNC_PORT}/vnc.html?autoconnect=true")
        if lan_ip:
            urls.append(f"http://{lan_ip}:{NOVNC_PORT}/vnc.html?autoconnect=true")
        urls.append(f"http://localhost:{NOVNC_PORT}/vnc.html?autoconnect=true")
        return urls

    # ── PID file management ──────────────────────────────────────────────

    def _write_pid(self, name: str, pid: int):
        """Write a PID file for a VNC process."""
        (PID_DIR / f"{name}.pid").write_text(str(pid))

    def _read_pid(self, name: str) -> Optional[int]:
        """Read a PID from file, return None if missing or stale."""
        path = PID_DIR / f"{name}.pid"
        if not path.exists():
            return None
        try:
            pid = int(path.read_text().strip())
            # Check if process is actually running
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            path.unlink(missing_ok=True)
            return None

    def _pid_alive(self, name: str) -> bool:
        """Check if a tracked process is still running."""
        return self._read_pid(name) is not None

    def _kill_pid(self, name: str):
        """Kill a process by its PID file."""
        pid = self._read_pid(name)
        if pid is not None:
            try:
                os.kill(pid, signal.SIGTERM)
                # Wait briefly for clean shutdown
                for _ in range(10):
                    time.sleep(0.3)
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                else:
                    # Still alive, force kill
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                log(f"Stopped {name} (pid {pid})")
            except ProcessLookupError:
                pass  # Already dead
            except Exception as e:
                log(f"Error stopping {name}: {e}")

        # Clean up PID file
        (PID_DIR / f"{name}.pid").unlink(missing_ok=True)

    def _cleanup(self):
        """Kill all VNC processes using PID files."""
        for name in ("novnc", "x11vnc", "xvfb"):
            self._kill_pid(name)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _which(cmd: str) -> bool:
    """Check if a command is available."""
    try:
        subprocess.run(["which", cmd], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _get_tailscale_ip() -> Optional[str]:
    """Get Tailscale IPv4 address if available."""
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _get_lan_ip() -> Optional[str]:
    """Get the primary LAN IP address."""
    try:
        result = subprocess.run(
            ["hostname", "-I"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            ips = result.stdout.strip().split()
            # Return first non-tailscale IP
            ts_ip = _get_tailscale_ip()
            for ip in ips:
                if ip != ts_ip and not ip.startswith("172."):
                    return ip
            return ips[0] if ips else None
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Module Singleton
# ─────────────────────────────────────────────────────────────────────────────

_vnc: Optional[VncSession] = None


def get_vnc_session() -> VncSession:
    global _vnc
    if _vnc is None:
        _vnc = VncSession()
    return _vnc
