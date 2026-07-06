"""
AI delegation tool implementations — ask_frontier, send_to_claude.
"""

import subprocess

from maude_core.log import log
from tool_registry import register_tool


def tool_ask_frontier(question: str, context: str = None, provider: str = None) -> str:
    """Escalate a question to a frontier cloud model."""
    try:
        from frontier import RateLimitError, ask_frontier, list_available_providers

        available = list_available_providers()
        if not available:
            return "Error: No frontier providers configured. Set API keys with /keys set <provider> <key>"

        provider_name = provider if provider in available else None
        log("Escalating to frontier model...")

        response = ask_frontier(
            query=question,
            context=context,
            provider_name=provider_name,
            system_prompt="You are an expert assistant. Be thorough but concise.",
        )

        log(f"{response.provider}: {response.input_tokens}+{response.output_tokens} tokens, ${response.cost_usd:.4f}")

        return f"[Expert response from {response.provider}]\n\n{response.content}"

    except RateLimitError as e:
        wait_msg = f" Try again in {e.retry_after}s." if e.retry_after else " Wait a minute and try again."
        return f"[Rate limit] {e.provider} free tier limit reached.{wait_msg} Local models are still available."

    except Exception as e:
        return f"Error calling frontier model: {e}"


def tool_send_to_claude(message: str, session: str = "claude") -> str:
    """Send a message to Claude Code running in tmux and capture the response."""
    import shutil
    import time

    # Check if tmux is available
    if not shutil.which("tmux"):
        return "Error: tmux is not installed"

    # Check if the session exists
    result = subprocess.run(["tmux", "has-session", "-t", session], capture_output=True)
    if result.returncode != 0:
        return f"Error: tmux session '{session}' not found. Start Claude with: ./start_claude.sh"

    # Capture pane content before sending (to know what's new)
    _before = subprocess.run(["tmux", "capture-pane", "-t", session, "-p"], capture_output=True, text=True).stdout

    # Send the message using -l (literal) flag to avoid interpretation issues
    log(f"Sending to Claude: {message[:50]}...")

    # First send the message text literally
    result = subprocess.run(["tmux", "send-keys", "-t", session, "-l", message], capture_output=True, text=True)
    if result.returncode != 0:
        return f"Error sending to Claude: {result.stderr}"

    # Small delay then send Enter key separately
    time.sleep(0.1)
    result = subprocess.run(["tmux", "send-keys", "-t", session, "Enter"], capture_output=True, text=True)

    if result.returncode != 0:
        return f"Error sending Enter: {result.stderr}"

    # Wait for Claude to respond - poll until we see the prompt again
    log("Waiting for Claude's response...")
    max_wait = 120  # seconds
    poll_interval = 1
    waited = 0
    last_content = ""

    while waited < max_wait:
        time.sleep(poll_interval)
        waited += poll_interval

        # Capture current pane content
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", session, "-p", "-S", "-500"], capture_output=True, text=True
        )
        current = result.stdout

        # Check if Claude is done (prompt visible and content stopped changing)
        if ("\u276f" in current.split("\n")[-5:] or ">" in current.split("\n")[-3:]) and current == last_content:
            break
        last_content = current

        # Also break if we've been waiting and content hasn't changed
        if waited > 5 and current == last_content:
            time.sleep(2)  # One more pause to be sure
            final_check = subprocess.run(
                ["tmux", "capture-pane", "-t", session, "-p", "-S", "-500"], capture_output=True, text=True
            ).stdout
            if final_check == current:
                break

    # Extract new content (what appeared after our message)
    after = subprocess.run(
        ["tmux", "capture-pane", "-t", session, "-p", "-S", "-500"], capture_output=True, text=True
    ).stdout

    # Find the response - everything after our message
    lines = after.strip().split("\n")

    # Look for our message and get everything after until the prompt
    response_lines = []
    found_message = False
    for line in lines:
        if message[:40] in line:  # Found our message
            found_message = True
            continue
        if found_message:
            # Stop at the prompt
            if line.strip().startswith("\u276f") or line.strip() == ">":
                break
            if "bypass permissions" in line:
                continue
            response_lines.append(line)

    response = "\n".join(response_lines).strip()

    # Truncate long responses to avoid filling MAUDE's context
    MAX_RESPONSE = 2000
    if len(response) > MAX_RESPONSE:
        response = response[:MAX_RESPONSE] + "\n\n[Response truncated - see Claude's tmux session for full output]"

    # Detect if Claude is asking a follow-up question
    follow_up_indicators = [
        "would you like",
        "do you want",
        "should i",
        "shall i",
        "let me know",
        "what would you",
        "which option",
        "prefer",
    ]
    is_follow_up = any(ind in response.lower() for ind in follow_up_indicators)

    if response:
        log(f"Got response from Claude ({len(response)} chars)")
        if is_follow_up:
            return f"Claude's response:\n\n{response}\n\n[Claude is asking a follow-up question - relay this to the user and wait for their answer before proceeding]"
        return f"Claude completed the task:\n\n{response}"
    else:
        return "Claude received the message but no response was captured. Check tmux session."


# ── Registry wrappers ──────────────────────────────────────────


@register_tool("ask_frontier")
def _dispatch_ask_frontier(args):
    return tool_ask_frontier(args.get("question", ""), args.get("context"), args.get("provider"))


@register_tool("send_to_claude")
def _dispatch_send_to_claude(args):
    return tool_send_to_claude(args.get("message", ""), args.get("session", "claude"))
