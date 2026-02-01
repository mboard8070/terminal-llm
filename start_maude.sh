#!/bin/bash
# Start MAUDE in a tmux session

SESSION_NAME="maude"

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "MAUDE session already running. Attaching..."
    tmux attach -t "$SESSION_NAME"
else
    echo "Starting MAUDE in tmux session '$SESSION_NAME'..."
    tmux new-session -d -s "$SESSION_NAME"
    tmux send-keys -t "$SESSION_NAME" "cd /home/mboard76/nvidia-workbench/terminal-llm && ./maude" Enter
    sleep 1
    tmux attach -t "$SESSION_NAME"
fi
