#!/usr/bin/env python3
"""
Eddie Bridge for MAUDE - Calls Claude through Clawdbot's gateway.

MAUDE can delegate complex reasoning tasks to Eddie (Claude) via this bridge.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional


def call_eddie(
    task: str,
    context: str = None,
    timeout: int = 120
) -> Optional[str]:
    """
    Send a task to Eddie via Clawdbot sessions.
    
    Args:
        task: The task/prompt for Eddie
        context: Additional context
        timeout: Timeout in seconds
    
    Returns:
        Eddie's response or None on failure
    """
    # Build the message
    message = task
    if context:
        message = f"{task}\n\nContext:\n{context}"
    
    # Use clawdbot agent to run a Claude turn
    # Use the main agent with a MAUDE-specific session
    try:
        result = subprocess.run(
            [
                "clawdbot", "agent",
                "--agent", "main",
                "--session-id", "maude-bridge",
                "--message", message,
                "--timeout", str(timeout)
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 10,
            env={**os.environ, "CLAWDBOT_AGENT_QUIET": "1"}
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"Eddie bridge error: {result.stderr}", file=sys.stderr)
            return None
            
    except subprocess.TimeoutExpired:
        print("Eddie bridge timeout", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("clawdbot CLI not found", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Eddie bridge error: {e}", file=sys.stderr)
        return None


def quick_ask(prompt: str) -> Optional[str]:
    """Quick one-shot question to Eddie."""
    return call_eddie(prompt)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        result = call_eddie(prompt)
        if result:
            print(result)
        else:
            sys.exit(1)
    else:
        print("Usage: eddie_bridge.py <prompt>")
        print("Sends a prompt to Eddie (Claude) via Clawdbot")
        sys.exit(1)
