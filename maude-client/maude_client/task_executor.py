"""
MAUDE Client Task Executor - Polls for and executes tasks dispatched to this client.
"""

import os
import sys
import time
import platform
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, Callable

from maude_client.config import SERVER_HOST, SERVER_LLM_PORT
from maude_client.heartbeat import get_client_id, set_heartbeat_activity
from maude_client.process_utils import run_process, shell_command

# Configuration
POLL_INTERVAL = 10  # seconds
COMMAND_TIMEOUT = 120  # seconds
POLL_ENDPOINT = f"https://{SERVER_HOST}:{SERVER_LLM_PORT}/api/collab/tasks/poll"
TASKS_ENDPOINT = f"https://{SERVER_HOST}:{SERVER_LLM_PORT}/api/collab/tasks"


class TaskExecutor:
    """Background daemon that polls for queued tasks, executes them, and reports results."""

    def __init__(self, on_task_start: Callable = None, on_task_complete: Callable = None):
        self.client_id = get_client_id()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.on_task_start = on_task_start
        self.on_task_complete = on_task_complete
        self._platform = platform.system().lower()
        self._session = self._build_session()
        self._poll_fail_count = 0

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
        session.verify = False
        return session

    def _poll_tasks(self) -> list:
        """Poll the server for queued tasks targeting this client."""
        try:
            resp = self._session.get(
                POLL_ENDPOINT,
                params={"client_id": self.client_id},
                timeout=10,
            )
            if resp.status_code == 200:
                self._poll_fail_count = 0
                return resp.json()
        except Exception as exc:
            self._poll_fail_count += 1
            if self._poll_fail_count == 1 or self._poll_fail_count % 5 == 0:
                print(
                    f"[task_executor] poll failed x{self._poll_fail_count}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
        return []

    def _claim_task(self, task_id: str) -> bool:
        """Claim a task (set status to running). Returns False if already claimed."""
        try:
            resp = self._session.post(
                f"{TASKS_ENDPOINT}/{task_id}/claim",
                json={},
                timeout=10,
            )
            return resp.status_code == 200
        except Exception as exc:
            print(f"[task_executor] claim failed: {type(exc).__name__}: {exc}", flush=True)
            return False

    def _report_result(self, task_id: str, status: str, result: str):
        """Report task result back to the server."""
        # Retry result report — losing completion is worse than a delayed report.
        last_exc = None
        for attempt in range(3):
            try:
                self._session.post(
                    f"{TASKS_ENDPOINT}/{task_id}/result",
                    json={"status": status, "result": result},
                    timeout=15,
                )
                return
            except Exception as exc:
                last_exc = exc
                time.sleep(1.0 * (attempt + 1))
        print(f"[task_executor] result report failed: {last_exc}", flush=True)

    def _execute_command(self, command: str) -> tuple:
        """Execute a shell command. Returns (status, output).

        Commands are launched in their own process group so a timeout cleans up
        shell grandchildren (important on Windows where taskkill /T is used).
        This also prevents Mac helper processes from being
        orphaned after a dispatched task times out.
        """
        try:
            result = run_process(shell_command(command), timeout=COMMAND_TIMEOUT)
            output = result.stdout
            if result.stderr:
                output += ("\n" if output else "") + result.stderr

            if result.timed_out:
                return ("failed", f"Command timed out after {COMMAND_TIMEOUT}s and its process tree was terminated")
            if result.returncode == 0:
                return ("completed", output or "(no output)")
            return ("completed", f"(exit code {result.returncode})\n{output}" if output else f"(exit code {result.returncode})")
        except Exception as e:
            return ("failed", f"Execution error: {e}")

    def _execute_task(self, task: dict):
        """Execute a single task and report the result."""
        task_id = task.get("id", "")
        prompt = task.get("prompt", "")
        capability = task.get("capability", "shell")

        if not self._claim_task(task_id):
            return  # Already claimed by someone else

        set_heartbeat_activity(f"working:{task_id[:16]}")
        if self.on_task_start:
            self.on_task_start(task)

        try:
            cap = (capability or "").strip().upper()
            if cap in ("SHELL", "COMMAND", ""):
                status, result = self._execute_command(prompt)
            else:
                status, result = ("failed", f"Unsupported capability on client: {capability}")
            self._report_result(task_id, status, result)
            if self.on_task_complete:
                self.on_task_complete(task, status, result)
        finally:
            set_heartbeat_activity("running")

    def _poll_loop(self):
        """Background loop that polls for and executes tasks."""
        while self.running:
            tasks = self._poll_tasks()
            for task in tasks:
                if not self.running:
                    break
                self._execute_task(task)
            # Faster recovery after poll failures (gateway restart / peer closed).
            sleep_for = POLL_INTERVAL if self._poll_fail_count == 0 else min(3, POLL_INTERVAL)
            for _ in range(int(sleep_for)):
                if not self.running:
                    break
                time.sleep(1)

    def start(self):
        """Start the task executor background thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the task executor."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)


# Global instance
_executor: Optional[TaskExecutor] = None


def start_task_executor(on_task_start: Callable = None, on_task_complete: Callable = None):
    """Start the global task executor."""
    global _executor
    if _executor is None:
        _executor = TaskExecutor(on_task_start, on_task_complete)
    _executor.start()


def stop_task_executor():
    """Stop the global task executor."""
    global _executor
    if _executor:
        _executor.stop()
        _executor = None
