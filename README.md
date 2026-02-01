# MAUDE

**Multi-Agent Unified Dispatch Engine** — A local AI coding assistant with optional frontier model escalation.

MAUDE runs on-device using llama.cpp or Ollama, providing file operations, shell access, web browsing, and vision capabilities. Use any local model (Nemotron, Codestral, Llama, Mistral, etc.) with optional escalation to frontier cloud models when needed.

## Features

- **Textual TUI** with animated banner and async processing
- **32K context window** for long conversations
- **File operations**: read, write, edit, search (single file or entire directory)
- **Shell access**: run commands, manage git, pip, etc.
- **Web tools**: browse pages, search DuckDuckGo, visual page analysis
- **Vision**: analyze local images and webpage screenshots via LLaVA
- **Frontier escalation** (optional): delegate complex tasks to Claude, GPT, Gemini, or Grok

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support
- Ollama (for LLaVA vision model)

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
```

## Usage

```bash
# Terminal 1: Start the inference server
./start_server.sh

# Terminal 2: Launch MAUDE
./maude
```

Or use environment variables for remote servers:
```bash
export LLM_SERVER_URL="http://remote-host:30000/v1"
export VISION_SERVER_URL="http://remote-host:11434/v1"
./maude
```

## Tools

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
| `web_browse` | Fetch and parse web pages |
| `web_search` | Search via DuckDuckGo |
| `web_view` | Screenshot + visual analysis of webpages |
| `view_image` | Analyze local images |
| `ask_frontier` | Escalate to cloud AI (optional, requires API keys) |
| `schedule_task` | Create and manage scheduled automated tasks |
| `send_to_claude` | Delegate tasks to Claude Code running in tmux |

## Frontier Model Escalation (Optional)

MAUDE works fully offline with just the local Nemotron model. Optionally, configure API keys in `.env` to enable escalation to frontier models for complex tasks:

```bash
# Priority order: Claude > GPT > Gemini > Grok > Mistral
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
XAI_API_KEY=...
MISTRAL_API_KEY=...
```

MAUDE escalates when:
- Complex multi-step reasoning is needed
- Architecture or design decisions required
- Deep domain expertise (security, algorithms, etc.)
- Uncertainty about the correct approach

Cost is displayed after each frontier call.

## Scheduled Tasks

MAUDE can schedule automated tasks using natural language. Just tell MAUDE when and what to do:

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

Tasks are stored in `~/.config/maude/schedules.json` and viewable in the Command Center.

## Claude Code Integration

MAUDE can delegate complex tasks to Claude Code running in a tmux session. This enables two-way communication where MAUDE sends tasks and receives Claude's responses.

### Setup

```bash
# Terminal 1: Start Claude Code in tmux (then detach with Ctrl+B, D)
./start_claude.sh

# Terminal 2: Run MAUDE
./maude
```

### Usage

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

The response from Claude is captured and returned to MAUDE, completing the loop.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `LLM_SERVER_URL` | `http://localhost:30000/v1` | LLM server endpoint |
| `MAUDE_MODEL` | `nemotron` | Model name (nemotron, codestral, llama3, etc.) |
| `MAUDE_NUM_CTX` | `32768` | Context window size |
| `VISION_SERVER_URL` | `http://localhost:11434/v1` | Ollama endpoint for vision |
| `MAUDE_VISION_MODEL` | `llava:7b` | Vision model name |

### Using a Different Model

Nemotron-3-Nano-30B is recommended for coding tasks, but you can use any model:

```bash
# Use Codestral
export MAUDE_MODEL="codestral"
./maude

# Use Llama 3 via Ollama
export LLM_SERVER_URL="http://localhost:11434/v1"
export MAUDE_MODEL="llama3:70b"
./maude
```

## Project Structure

```
terminal-llm/
├── maude             # Launcher script
├── maude_core.py     # Core tools and functionality
├── chat_local.py     # MAUDE TUI application
├── frontier.py       # Frontier model client
├── providers.py      # Provider configurations
├── start_server.sh   # llama.cpp server launcher
├── start_claude.sh   # Claude Code tmux launcher
├── setup_local.sh    # Automated setup
├── llama.cpp/        # Inference engine
└── models/           # Downloaded models
```

## License

MIT License
