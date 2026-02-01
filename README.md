# MAUDE

**Multi-Agent Unified Dispatch Engine** — A local AI coding assistant that collaborates with Claude Code.

MAUDE runs on-device using llama.cpp, providing file operations, shell access, web browsing, and vision capabilities. When complex tasks arise, MAUDE delegates to Claude Code running in a companion tmux session, with MAUDE acting as the permission gatekeeper.

## Features

- **Multi-Agent Architecture**: MAUDE + Claude Code working together
- **Textual TUI** with animated banner and async processing
- **32K context window** for long conversations
- **File operations**: read, write, edit, search (single file or entire directory)
- **Shell access**: run commands, manage git, pip, etc.
- **Web browsing**: fetch and parse web pages, search DuckDuckGo
- **Vision**: screenshot webpages and analyze images using LLaVA
- **Claude Code delegation**: complex tasks automatically routed to Claude
- **Permission proxy**: MAUDE approves/denies Claude's permission requests
- **Scheduled tasks**: automate recurring tasks with natural language

## Models

**Primary Model (MAUDE):**
- **Nemotron-3-Nano-30B** (Q8_K_XL quantization) via llama.cpp
- 32K context window, runs on GPU with 99 layers offloaded

**Vision Model:**
- **LLaVA 7B/13B** via Ollama for image and screenshot analysis

**Available Alternative Models:**
- Codestral 22B - optimized for code
- CodeLlama 7B - lightweight code model
- Gemma 3 12B - general purpose
- Mistral Nemo Instruct - general purpose

**Delegation Target:**
- **Claude Code** (Anthropic) - frontier AI for complex reasoning

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (tested on DGX Spark)
- Ollama (for LLaVA vision model)
- Claude Code CLI (`claude`)
- tmux

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Build llama.cpp and download model
./setup_local.sh

# Install Playwright for web screenshots
playwright install chromium

# Install LLaVA for vision (via Ollama)
ollama pull llava:7b

# Install Claude Code (if not already installed)
# See: https://docs.anthropic.com/en/docs/claude-code
```

## Quick Start

**Terminal 1** - Start the agent system:
```bash
ssh -L 3000:localhost:3001 mboard76@spark-e26c
cd ~/nvidia-workbench/terminal-llm
./maude
```

The `./maude` command automatically starts 4 tmux sessions:
- `nemo` - Nemotron inference server (llama.cpp)
- `maude` - Your main interface (attaches here)
- `claude` - Claude Code for complex tasks
- `watcher` - Permission bridge

**Terminal 2** - Watch Claude work (optional):
```bash
ssh -L 3000:localhost:3001 mboard76@spark-e26c
cd ~/nvidia-workbench/terminal-llm && tmux attach -t claude
```

**Terminal 3** - Watch permission flow (optional):
```bash
ssh -L 3000:localhost:3001 mboard76@spark-e26c
cd ~/nvidia-workbench/terminal-llm && tmux attach -t watcher
```

Each session is independent - open as many terminals as you want.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         YOU                                     │
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      MAUDE                               │   │
│  │              (Nemotron-3-Nano-30B)                       │   │
│  │                                                          │   │
│  │  • Handles simple tasks directly                         │   │
│  │  • Web browsing & search (DuckDuckGo)                    │   │
│  │  • Vision analysis (LLaVA via Ollama)                    │   │
│  │  • Delegates complex tasks to Claude                     │   │
│  │  • Approves/denies Claude's permission requests          │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                       │
│            send_to_claude()                                     │
│                         │                                       │
│                         ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   CLAUDE CODE                            │   │
│  │              (Anthropic Claude)                          │   │
│  │                                                          │   │
│  │  • Complex reasoning & multi-file refactoring            │   │
│  │  • Git operations, PR creation                           │   │
│  │  • Deep code analysis                                    │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                       │
│              Permission needed?                                 │
│                         │                                       │
│                         ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                PERMISSION WATCHER                        │   │
│  │                                                          │   │
│  │  • Detects Claude's permission prompts                   │   │
│  │  • Forwards to MAUDE for approval                        │   │
│  │  • Sends response back to Claude                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Tools

### File Operations
| Tool | Description |
|------|-------------|
| `read_file` | Read file with line numbers (supports ranges) |
| `write_file` | Create or overwrite files |
| `edit_file` | Replace specific line ranges |
| `search_file` | Search within a single file |
| `search_directory` | Search across all files in a directory |
| `list_directory` | Browse directory contents |
| `change_directory` | Navigate filesystem |
| `run_command` | Execute shell commands |

### Web & Vision
| Tool | Description |
|------|-------------|
| `web_browse` | Fetch and parse web pages (text extraction) |
| `web_search` | Search via DuckDuckGo |
| `web_view` | Screenshot webpage + LLaVA visual analysis |
| `view_image` | Analyze local images with LLaVA |

### Delegation & Automation
| Tool | Description |
|------|-------------|
| `send_to_claude` | Delegate tasks to Claude Code |
| `schedule_task` | Create and manage scheduled automated tasks |

## Vision Capabilities

MAUDE can analyze images and webpages using LLaVA (Large Language and Vision Assistant):

**Screenshot & Analyze Webpages:**
```
"What does the Apple homepage look like?"
"Describe the layout of artstation.com/artwork/k4eYxl"
```

**Analyze Local Images:**
```
"What's in the image at ~/screenshots/diagram.png?"
"Describe this architecture diagram"
```

Vision requires Ollama running with LLaVA:
```bash
ollama serve  # Start Ollama
ollama pull llava:7b  # Download model
```

## Claude Code Delegation

MAUDE automatically delegates to Claude when you say things like:
- "ask Claude to review this code"
- "have Claude handle the git rebase"
- "let Claude refactor the authentication module"
- "delegate to Claude for the architecture decision"

MAUDE also delegates automatically for:
- Complex multi-file refactoring
- Git operations beyond simple commits
- Deep code analysis across large codebases
- Tasks where MAUDE is uncertain

When Claude needs permission (to run commands, edit files, etc.), the watcher forwards the request to MAUDE. MAUDE decides whether to approve, and the response is sent back to Claude automatically.

## Session Management

After starting with `./maude`, you'll be attached to the MAUDE session.

**From the same terminal:**
| Command | Action |
|---------|--------|
| `Ctrl+B d` | Detach from current session |
| `tmux attach -t maude` | Reattach to MAUDE |
| `tmux attach -t claude` | Attach to Claude |
| `tmux attach -t watcher` | Attach to Watcher |
| `tmux attach -t nemo` | Attach to Nemotron server |

**From a new SSH session (view multiple at once):**
```bash
ssh -L 3000:localhost:3001 mboard76@spark-e26c
cd ~/nvidia-workbench/terminal-llm && tmux attach -t claude
```

## Monitoring & Control

```bash
# Watch activity log
tail -f ~/.config/maude/activity.log

# Emergency stop (halts the watcher)
touch ~/.config/maude/stop

# Kill all sessions
tmux kill-session -t maude
tmux kill-session -t claude
tmux kill-session -t watcher
tmux kill-session -t nemo
```

## Scheduled Tasks

MAUDE can schedule automated tasks using natural language:

```
"Remind me every morning at 8am to check the weather"
"Schedule a daily summary of my tasks at 6pm"
"Every weekday at 9am, check my calendar"
```

MAUDE converts these to cron expressions automatically. Manage tasks with:
- "Show my scheduled tasks"
- "Disable the weather reminder"
- "Remove task abc123"

Supported shortcuts: `@hourly`, `@daily`, `@morning`, `@evening`, `@weekly`, `@workdays`

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `LLM_SERVER_URL` | `http://localhost:30000/v1` | llama.cpp server endpoint |
| `MAUDE_MODEL` | `nemotron` | Model name for requests |
| `MAUDE_NUM_CTX` | `32768` | Context window size |
| `VISION_SERVER_URL` | `http://localhost:11434/v1` | Ollama endpoint for LLaVA |
| `MAUDE_VISION_MODEL` | `llava:7b` | Vision model name |

### Using a Different Model

```bash
# Use Codestral (better for code)
./start_codestral.sh  # Instead of start_nemo.sh
export MAUDE_MODEL="codestral"
./maude

# Use a model via Ollama
export LLM_SERVER_URL="http://localhost:11434/v1"
export MAUDE_MODEL="gemma2:9b"
./maude
```

## Project Structure

```
terminal-llm/
├── maude               # Main launcher (starts all agents)
├── maude_core.py       # Core tools and functionality
├── chat_local.py       # MAUDE TUI application
├── claude_watcher.py   # Permission bridge between Claude and MAUDE
├── start_nemo.sh       # Nemotron server launcher
├── start_codestral.sh  # Codestral server launcher
├── start_claude.sh     # Claude Code standalone launcher
├── start_maude.sh      # MAUDE standalone launcher
├── llama.cpp/          # Inference engine
└── models/             # Downloaded GGUF models
    ├── Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf
    ├── codestral-22b-v0.1-Q6_K.gguf
    ├── codellama-7b-instruct.Q6_K.gguf
    └── ...
```

## License

MIT License
