#!/usr/bin/env python3
"""
Claude Permission Watcher - Monitors Claude Code for permission prompts
and forwards them to MAUDE for approval.
"""

import subprocess
import time
import re
import json
from pathlib import Path
from datetime import datetime

# Configuration
CLAUDE_SESSION = "claude"
MAUDE_SESSION = "maude"
POLL_INTERVAL = 0.5  # seconds
ACTIVITY_LOG = Path.home() / ".config" / "maude" / "activity.log"
KILL_SWITCH = Path.home() / ".config" / "maude" / "stop"

# Patterns that indicate Claude is waiting for permission
PERMISSION_PATTERNS = [
    r"Do you want to proceed\?",
    r"Allow .*\?",
    r"Do you want to .*\?",
    r"Press Enter to allow",
    r"yes.*no.*always",
    r"\[y/n\]",
    r"\(y\)es.*\(n\)o",
    r"Allow this",
    r"Approve\?",
    r"1\. Yes",
    r"❯.*Yes",
]

# Pattern to detect the permission prompt line (usually at bottom)
PROMPT_INDICATORS = [
    "Do you want to proceed",
    "Allow",
    "approve",
    "y/n",
    "yes/no",
    "(y)",
    "1. Yes",
]


def log_activity(message: str):
    """Log activity to shared log file."""
    try:
        ACTIVITY_LOG.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(ACTIVITY_LOG, "a") as f:
            f.write(f"[{timestamp}] [WATCHER] {message}\n")
    except Exception:
        pass


def check_kill_switch() -> bool:
    """Check if kill switch is active."""
    return KILL_SWITCH.exists()


def get_tmux_content(session: str, lines: int = 50) -> str:
    """Capture recent content from a tmux session."""
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", session, "-p", "-S", f"-{lines}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout if result.returncode == 0 else ""
    except Exception:
        return ""


def send_to_tmux(session: str, text: str, send_enter: bool = True):
    """Send text to a tmux session."""
    try:
        subprocess.run(
            ["tmux", "send-keys", "-t", session, "-l", text],
            capture_output=True,
            timeout=5
        )
        if send_enter:
            time.sleep(0.1)
            subprocess.run(
                ["tmux", "send-keys", "-t", session, "Enter"],
                capture_output=True,
                timeout=5
            )
    except Exception as e:
        log_activity(f"Error sending to {session}: {e}")


def detect_permission_prompt(content: str) -> str | None:
    """
    Detect if Claude is showing a permission prompt.
    Returns the prompt text if found, None otherwise.
    """
    lines = content.strip().split("\n")

    # Look at the last 20 lines for permission patterns
    recent_lines = lines[-20:] if len(lines) > 20 else lines
    recent_text = "\n".join(recent_lines)

    for pattern in PERMISSION_PATTERNS:
        match = re.search(pattern, recent_text, re.IGNORECASE)
        if match:
            # Extract context around the match
            for i, line in enumerate(recent_lines):
                if re.search(pattern, line, re.IGNORECASE):
                    # Get a few lines of context
                    start = max(0, i - 3)
                    end = min(len(recent_lines), i + 2)
                    context = "\n".join(recent_lines[start:end])
                    return context.strip()

    return None


def format_for_maude(prompt: str) -> str:
    """Format the permission prompt for MAUDE to display."""
    return f"[CLAUDE NEEDS APPROVAL]\n{prompt}\n\nReply 'approve' or 'deny'"


def watch_for_maude_response(timeout: int = 300) -> str | None:
    """
    Watch MAUDE's tmux session for an approval response.
    Returns '1' + Enter for approve, 'Escape' for deny, None for timeout.
    """
    start_time = time.time()
    last_content = ""

    while time.time() - start_time < timeout:
        if check_kill_switch():
            log_activity("Kill switch activated during approval wait")
            return None

        content = get_tmux_content(MAUDE_SESSION, 30)

        # Look for new content with approval/deny
        if content != last_content:
            recent = content.lower()

            # Check for approval keywords in recent output
            if "approve" in recent[-500:] or "yes" in recent[-200:] or "allow" in recent[-200:]:
                return "1"  # Claude expects "1" for Yes
            elif "deny" in recent[-500:] or "reject" in recent[-200:] or "cancel" in recent[-200:]:
                return "Escape"  # Escape to cancel

        last_content = content
        time.sleep(POLL_INTERVAL)

    return None


def main():
    """Main watcher loop."""
    print(f"Claude Permission Watcher starting...")
    print(f"Monitoring session: {CLAUDE_SESSION}")
    print(f"Forwarding to: {MAUDE_SESSION}")
    print(f"Activity log: {ACTIVITY_LOG}")
    print(f"Kill switch: {KILL_SWITCH}")
    print(f"Press Ctrl+C or create {KILL_SWITCH} to stop")
    print()

    log_activity("Watcher started")

    last_prompt = None
    last_prompt_time = 0

    try:
        while True:
            # Check kill switch
            if check_kill_switch():
                print("Kill switch detected. Stopping.")
                log_activity("Kill switch activated - stopping watcher")
                break

            # Check if Claude session exists
            result = subprocess.run(
                ["tmux", "has-session", "-t", CLAUDE_SESSION],
                capture_output=True
            )
            if result.returncode != 0:
                time.sleep(2)
                continue

            # Get Claude's current output
            content = get_tmux_content(CLAUDE_SESSION)

            # Check for permission prompt
            prompt = detect_permission_prompt(content)

            if prompt and (prompt != last_prompt or time.time() - last_prompt_time > 30):
                log_activity(f"Permission prompt detected: {prompt[:100]}...")
                print(f"\n[PERMISSION PROMPT DETECTED]\n{prompt}\n")

                # Send to MAUDE
                maude_message = format_for_maude(prompt)
                send_to_tmux(MAUDE_SESSION, maude_message)
                log_activity("Forwarded prompt to MAUDE")

                # Wait for response from MAUDE
                print("Waiting for approval from MAUDE...")
                response = watch_for_maude_response()

                if response:
                    # Send response to Claude
                    if response == "Escape":
                        # Send actual Escape key
                        subprocess.run(
                            ["tmux", "send-keys", "-t", CLAUDE_SESSION, "Escape"],
                            capture_output=True,
                            timeout=5
                        )
                    else:
                        send_to_tmux(CLAUDE_SESSION, response)
                    log_activity(f"Sent response to Claude: {response}")
                    print(f"Sent '{response}' to Claude")
                else:
                    log_activity("Approval timeout - no response from MAUDE")
                    print("Timeout waiting for approval")

                last_prompt = prompt
                last_prompt_time = time.time()

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopping watcher...")
        log_activity("Watcher stopped by user")


if __name__ == "__main__":
    main()
