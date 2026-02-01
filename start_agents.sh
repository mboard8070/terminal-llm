#!/bin/bash
# Start MAUDE + Claude Code agent system
#
# This launches:
# - Claude Code in tmux session 'claude'
# - MAUDE in tmux session 'maude'
# - Permission watcher to forward Claude's approval requests to MAUDE
#
# Usage: ./start_agents.sh [--attach maude|claude|watcher]

cd /home/mboard76/nvidia-workbench/terminal-llm

# Clear kill switch if it exists
rm -f ~/.config/maude/stop

# Initialize activity log
mkdir -p ~/.config/maude
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SYSTEM] Agent system starting" >> ~/.config/maude/activity.log

# Start Claude if not running
if ! tmux has-session -t claude 2>/dev/null; then
    echo "Starting Claude Code..."
    tmux new-session -d -s claude
    tmux send-keys -t claude "cd /home/mboard76/nvidia-workbench/terminal-llm && claude" Enter
    sleep 2
fi

# Start MAUDE if not running
if ! tmux has-session -t maude 2>/dev/null; then
    echo "Starting MAUDE..."
    tmux new-session -d -s maude
    tmux send-keys -t maude "cd /home/mboard76/nvidia-workbench/terminal-llm && ./maude" Enter
    sleep 2
fi

# Start watcher if not running
if ! tmux has-session -t watcher 2>/dev/null; then
    echo "Starting permission watcher..."
    tmux new-session -d -s watcher
    tmux send-keys -t watcher "cd /home/mboard76/nvidia-workbench/terminal-llm && python claude_watcher.py" Enter
fi

echo ""
echo "Agent system running!"
echo ""
echo "Sessions:"
echo "  tmux attach -t maude    # Your main interface"
echo "  tmux attach -t claude   # Watch Claude work"
echo "  tmux attach -t watcher  # Permission watcher logs"
echo ""
echo "Activity log: tail -f ~/.config/maude/activity.log"
echo "Kill switch:  touch ~/.config/maude/stop"
echo ""

# Attach to requested session or maude by default
ATTACH_TO="${1:-maude}"
if [[ "$1" == "--attach" ]]; then
    ATTACH_TO="${2:-maude}"
fi

tmux attach -t "$ATTACH_TO"
