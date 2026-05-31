"""
MAUDE Client Task Executor - Polls for and executes tasks dispatched to this client.
"""

import os
import sys
import time
import platform
import threading
import requests
from typing import Optional, Callable

from maude_client.config import SERVER_HOST, SERVER_LLM_PORT
from maude_client.heartbeat import get_client_id
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

    def _poll_tasks(self) -> list:
        """Poll the server for queued tasks targeting this client."""
        try:
            resp = requests.get(
                POLL_ENDPOINT,
                params={"client_id": self.client_id},
                timeout=10,
                verify=False,
            )
            if resp.status_code == 200:
                return resp.json()
        except requests.exceptions.RequestException:
            pass
        return []

    def _claim_task(self, task_id: str) -> bool:
        """Claim a task (set status to running). Returns False if already claimed."""
        try:
            resp = requests.post(
                f"{TASKS_ENDPOINT}/{task_id}/claim",
                json={},
                timeout=10,
                verify=False,
            )
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _report_result(self, task_id: str, status: str, result: str):
        """Report task result back to the server."""
        try:
            requests.post(
                f"{TASKS_ENDPOINT}/{task_id}/result",
                json={"status": status, "result": result},
                timeout=10,
                verify=False,
            )
        except requests.exceptions.RequestException:
            pass

    def _execute_command(self, command: str) -> tuple:
        """Execute a shell command. Returns (status, output).

        Commands are launched in their own process group so a timeout cleans up
        shell grandchildren too. This prevents Mac helper processes from being
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
            return ("failed", str(e))

    def _handle_task(self, task: dict):
        """Claim, execute, and report a single task."""
        task_id = task.get("id", "")
        prompt = task.get("prompt", "")
        capability = task.get("capability", "SHELL")

        if self.on_task_start:
            self.on_task_start(task)

        # Claim the task
        if not self._claim_task(task_id):
            return  # Already claimed by someone else

        if capability == "SHELL":
            status, result = self._execute_command(prompt)
        else:
            status, result = "failed", f"Client can only execute SHELL tasks (got {capability})"

        self._report_result(task_id, status, result)

        if self.on_task_complete:
            self.on_task_complete(task, status, result)

    def _poll_loop(self):
        """Background loop that polls for and executes tasks."""
        while self.running:
            tasks = self._poll_tasks()
            for task in tasks:
                if not self.running:
                    break
                self._handle_task(task)

            # Sleep in small increments for quick shutdown
            for _ in range(POLL_INTERVAL):
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
