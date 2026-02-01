#!/bin/bash
# Start Claude Code in a tmux session for MAUDE integration

SESSION_NAME="claude"
# Skip permissions since MAUDE delegates tasks autonomously
CLAUDE_FLAGS="--dangerously-skip-permissions"

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Claude session already running. Attaching..."
    tmux attach -t "$SESSION_NAME"
else
    echo "Starting Claude Code in tmux session '$SESSION_NAME'..."
    echo "Note: Running with --dangerously-skip-permissions for MAUDE integration"
    tmux new-session -d -s "$SESSION_NAME"
    tmux send-keys -t "$SESSION_NAME" "cd /home/mboard76/nvidia-workbench/terminal-llm && claude $CLAUDE_FLAGS" Enter
    sleep 1
    tmux attach -t "$SESSION_NAME"
fi
