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
from typing import Optional

# Configuration
HEARTBEAT_INTERVAL = 30  # seconds
HEARTBEAT_ENDPOINT = "http://100.107.132.16:3003/api/heartbeat"  # Spark Tailscale IP

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
    """Background heartbeat sender."""

    def __init__(self, endpoint: str = HEARTBEAT_ENDPOINT, interval: int = HEARTBEAT_INTERVAL):
        self.endpoint = endpoint
        self.interval = interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.client_id = get_client_id()
        self.hostname = get_hostname()
        self.platform = get_platform()
        self.version = "1.0.0"

    def _send_heartbeat(self, status: str = "running") -> bool:
        """Send a single heartbeat to the server."""
        try:
            response = requests.post(
                self.endpoint,
                json={
                    "clientId": self.client_id,
                    "hostname": self.hostname,
                    "platform": self.platform,
                    "version": self.version,
                    "status": status,
                },
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            # Silently fail - server might be unreachable
            return False

    def _heartbeat_loop(self):
        """Background loop that sends heartbeats."""
        while self.running:
            self._send_heartbeat("running")
            # Sleep in small increments to allow quick shutdown
            for _ in range(self.interval):
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
