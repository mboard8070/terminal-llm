"""
MAUDE Shared Folder Sync - Background bidirectional rsync over SSH.

Syncs ~/.maude/shared/ <-> server:~/nvidia-workbench/terminal-llm/shared/
every SYNC_INTERVAL seconds using rsync --update (newer files win).
"""

import os
import time
import subprocess
import threading
import logging
from pathlib import Path
from typing import Optional

from maude_client.config import (
    SERVER_SSH_HOST, LOCAL_SHARED_DIR, SERVER_SHARED_DIR, SYNC_INTERVAL
)

# Set up logging
LOG_DIR = Path("~/.maude").expanduser()
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "sync.log"

logger = logging.getLogger("shared_sync")
logger.setLevel(logging.INFO)

_file_handler = logging.FileHandler(LOG_FILE)
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_file_handler)


def _ensure_dirs():
    """Ensure local shared directory exists."""
    local = Path(LOCAL_SHARED_DIR).expanduser()
    local.mkdir(parents=True, exist_ok=True)
    return local


def _run_rsync(src: str, dst: str) -> subprocess.CompletedProcess:
    """Run a single rsync command (add/update only, never delete)."""
    cmd = [
        "rsync", "-avz", "--update",
        "-e", "ssh",
        src, dst
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


def sync_now() -> str:
    """Run a single bidirectional sync. Returns status message."""
    local_dir = _ensure_dirs()
    # Trailing slashes are important for rsync (sync contents, not the dir itself)
    local_path = str(local_dir) + "/"
    remote_path = f"{SERVER_SSH_HOST}:{SERVER_SHARED_DIR}/"

    results = []

    # Ensure remote directory exists
    try:
        subprocess.run(
            ["ssh", SERVER_SSH_HOST, f"mkdir -p {SERVER_SHARED_DIR}"],
            capture_output=True, timeout=10
        )
    except Exception as e:
        msg = f"Failed to create remote shared dir: {e}"
        logger.error(msg)
        return msg

    # Pull first: server -> local (add/update only — never delete)
    try:
        r = _run_rsync(remote_path, local_path)
        if r.returncode == 0:
            pulled = [l for l in r.stdout.splitlines() if not l.startswith("receiving") and not l.startswith("sent") and not l.startswith("total") and l.strip()]
            if pulled:
                logger.info(f"Pulled: {pulled}")
                results.append(f"Pulled {len(pulled)} item(s) from server")
            else:
                results.append("Pull: nothing new")
        else:
            logger.warning(f"Pull failed: {r.stderr}")
            results.append(f"Pull error: {r.stderr.strip()}")
    except subprocess.TimeoutExpired:
        results.append("Pull: timed out")
        logger.error("Pull rsync timed out")
    except Exception as e:
        results.append(f"Pull error: {e}")
        logger.error(f"Pull error: {e}")

    # Process deletions manifest — remove local files the server deleted
    deletions_file = local_dir / ".maude_deletions"
    if deletions_file.exists():
        try:
            deleted_names = [
                line.strip() for line in deletions_file.read_text().splitlines() if line.strip()
            ]
            for name in deleted_names:
                local_file = local_dir / name
                if local_file.exists():
                    local_file.unlink()
                    logger.info(f"Deleted local shared file (server removed): {name}")
            # Clear the manifest locally so push doesn't re-upload stale entries
            deletions_file.unlink()
            # Clear the manifest on the server too
            try:
                subprocess.run(
                    ["ssh", SERVER_SSH_HOST, f"rm -f {SERVER_SHARED_DIR}/.maude_deletions"],
                    capture_output=True, timeout=10
                )
            except Exception as e:
                logger.warning(f"Failed to clear remote deletions manifest: {e}")
            if deleted_names:
                results.append(f"Removed {len(deleted_names)} deleted file(s) locally")
        except Exception as e:
            logger.error(f"Error processing deletions manifest: {e}")

    # Push second: local -> server (no --delete — server decides what to keep)
    try:
        r = _run_rsync(local_path, remote_path)
        if r.returncode == 0:
            pushed = [l for l in r.stdout.splitlines() if not l.startswith("sending") and not l.startswith("sent") and not l.startswith("total") and l.strip()]
            if pushed:
                logger.info(f"Pushed: {pushed}")
                results.append(f"Pushed {len(pushed)} item(s) to server")
            else:
                results.append("Push: nothing new")
        else:
            logger.warning(f"Push failed: {r.stderr}")
            results.append(f"Push error: {r.stderr.strip()}")
    except subprocess.TimeoutExpired:
        results.append("Push: timed out")
        logger.error("Push rsync timed out")
    except Exception as e:
        results.append(f"Push error: {e}")
        logger.error(f"Push error: {e}")

    return " | ".join(results)


class SharedSyncDaemon:
    """Background thread that syncs the shared folder periodically."""

    def __init__(self, interval: int = SYNC_INTERVAL):
        self.interval = interval
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def _sync_loop(self):
        """Background loop that runs sync periodically."""
        while self.running:
            try:
                sync_now()
            except Exception as e:
                logger.error(f"Sync loop error: {e}")

            # Sleep in small increments to allow quick shutdown
            for _ in range(self.interval):
                if not self.running:
                    break
                time.sleep(1)

    def start(self):
        """Start the background sync thread."""
        if self.running:
            return

        _ensure_dirs()
        self.running = True
        self.thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.thread.start()
        logger.info("Shared sync started")

    def stop(self):
        """Stop the background sync thread."""
        self.running = False
        logger.info("Shared sync stopped")
        if self.thread:
            self.thread.join(timeout=5)


# Global instance
_sync_daemon: Optional[SharedSyncDaemon] = None


def start_sync(interval: int = SYNC_INTERVAL):
    """Start the global sync daemon."""
    global _sync_daemon
    if _sync_daemon is None:
        _sync_daemon = SharedSyncDaemon(interval)
    _sync_daemon.start()


def stop_sync():
    """Stop the global sync daemon."""
    global _sync_daemon
    if _sync_daemon:
        _sync_daemon.stop()
        _sync_daemon = None


# Standalone testing
if __name__ == "__main__":
    print(f"Local shared dir:  {Path(LOCAL_SHARED_DIR).expanduser()}")
    print(f"Server shared dir: {SERVER_SSH_HOST}:{SERVER_SHARED_DIR}")
    print(f"Sync interval:     {SYNC_INTERVAL}s")
    print()

    print("Running initial sync...")
    result = sync_now()
    print(f"Result: {result}")
    print()

    print("Starting background sync (Ctrl+C to stop)...")
    start_sync()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping sync...")
        stop_sync()
        print("Done.")
