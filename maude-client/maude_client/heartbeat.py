"""
MAUDE Client Heartbeat - Reports client status to the server.
"""

import os
import sys
import time
import socket
import platform
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional

from maude_client import __version__
from maude_client.config import SERVER_HOST, SERVER_LLM_PORT

# Configuration
HEARTBEAT_INTERVAL = 30  # seconds
HEARTBEAT_ENDPOINT = f"https://{SERVER_HOST}:{SERVER_LLM_PORT}/api/collab/presence"

# Client identification
def get_client_id() -> str:
    """Generate a unique client ID based on hostname and user."""
    hostname = socket.gethostname()
    user = os.environ.get("USER", os.environ.get("USERNAME", "unknown"))
    return f"{hostname}-{user}"

def get_hostname() -> str:
    """Get the machine's hostname."""
    return socket.gethostname()

def get_platform() -> str:
    """Get the platform/OS name."""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    return system

class HeartbeatClient:
    """Background heartbeat sender with retries and backoff."""

    def __init__(self, endpoint: str = HEARTBEAT_ENDPOINT, interval: int = HEARTBEAT_INTERVAL):
        self.endpoint = endpoint
        self.interval = interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.client_id = get_client_id()
        self.hostname = get_hostname()
        self.platform = get_platform()
        self.version = __version__
        self._fail_count = 0
        self._last_ok = 0.0
        self._activity = "running"
        self._lock = threading.Lock()
        self._session = self._build_session()


    @staticmethod
    def _build_session() -> requests.Session:
        """HTTP session resilient to brief gateway restarts / peer-closed blips."""
        session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=(502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        # Gateway may use self-signed certs on the mesh.
        session.verify = False
        return session

    def set_activity(self, activity: str):
        """Update activity string shown in presence (e.g. working/idle)."""
        with self._lock:
            self._activity = activity or "running"

    def _current_activity(self) -> str:
        with self._lock:
            return self._activity

    def _send_heartbeat(self, status: str = "running") -> bool:
        """Send a single heartbeat to the server with short retries."""
        payload = {
            "client_id": self.client_id,
            "client_type": self.platform,
            "hostname": self.hostname,
            "platform": self.platform,
            "activity": status,
        }
        # Retry a few times so a single blip does not drop presence.
        attempts = 3 if status != "stopping" else 1
        for attempt in range(attempts):
            try:
                response = self._session.post(
                    self.endpoint,
                    json=payload,
                    timeout=5,
                )
                if response.status_code == 200:
                    self._fail_count = 0
                    self._last_ok = time.time()
                    return True
            except Exception as exc:
                # Includes ConnectionError / ProtocolError ("peer closed").
                if attempt + 1 >= attempts and (self._fail_count + 1) % 5 == 0:
                    print(f"[heartbeat] error: {type(exc).__name__}: {exc}", flush=True)

            # Brief backoff between retries
            if attempt + 1 < attempts and self.running:
                time.sleep(1.0 * (attempt + 1))
        self._fail_count += 1
        # Log occasionally so silent death is visible in client console.
        if self._fail_count == 1 or self._fail_count % 5 == 0:
            print(
                f"[heartbeat] failed x{self._fail_count} "
                f"(last_ok={int(time.time() - self._last_ok) if self._last_ok else 'never'}s ago)",
                flush=True,
            )
        return False

    def _heartbeat_loop(self):
        """Background loop that sends heartbeats.

        On consecutive failures, retry more aggressively (every 5s) so the
        client re-registers quickly after a transient network/server blip.
        """
        while self.running:
            ok = self._send_heartbeat(self._current_activity())
            # Adaptive interval: faster recovery when failing.
            sleep_for = self.interval if ok else min(5, self.interval)
            for _ in range(int(sleep_for)):
                if not self.running:
                    break
                time.sleep(1)

    def start(self):
        """Start the heartbeat background thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.thread.start()

        # Send initial heartbeat immediately
        self._send_heartbeat("running")

    def stop(self):
        """Stop the heartbeat and send final 'stopping' status."""
        self.running = False
        self._send_heartbeat("stopping")

        if self.thread:
            self.thread.join(timeout=2)

# Global instance
_heartbeat_client: Optional[HeartbeatClient] = None

def start_heartbeat(endpoint: str = HEARTBEAT_ENDPOINT, interval: int = HEARTBEAT_INTERVAL):
    """Start the global heartbeat client."""
    global _heartbeat_client
    if _heartbeat_client is None:
        _heartbeat_client = HeartbeatClient(endpoint, interval)
    _heartbeat_client.start()

def stop_heartbeat():
    """Stop the global heartbeat client."""
    global _heartbeat_client
    if _heartbeat_client:
        _heartbeat_client.stop()
        _heartbeat_client = None

def set_heartbeat_activity(activity: str):
    """Update presence activity for the running heartbeat client."""
    if _heartbeat_client:
        _heartbeat_client.set_activity(activity)

# For testing
if __name__ == "__main__":
    print(f"Client ID: {get_client_id()}")
    print(f"Hostname: {get_hostname()}")
    print(f"Platform: {get_platform()}")
    print()
    print("Starting heartbeat (Ctrl+C to stop)...")

    start_heartbeat()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping heartbeat...")
        stop_heartbeat()
        print("Done.")
