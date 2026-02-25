# MAUDE

**Multi-Agent Unified Dispatch Engine** — An on-device AI assistant running on DGX Spark with cloud model routing, multi-client access, and tool execution.

MAUDE runs locally using Nemotron via llama.cpp, with automatic routing to cloud models (Mistral, Codestral) via a unified gateway. Accessible from the server TUI, a Mac/PC CLI client, a phone PWA, and Telegram.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Clients                                                             │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Server   │  │ Mac/PC   │  │ Phone    │  │ Telegram │            │
│  │ TUI      │  │ CLI      │  │ PWA      │  │ Bot      │            │
│  │chat_local│  │maude-    │  │maude-    │  │run_      │            │
│  │  .py     │  │ client   │  │ phone    │  │telegram  │            │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
│       │              │              │              │                  │
│       │  local       │   Tailscale / HTTPS         │  local          │
│       │              │              │              │                  │
│       ▼              ▼              ▼              ▼                  │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │                     GATEWAY (port 30000)                      │   │
│  │  • Model routing (local ↔ cloud)                              │   │
│  │  • Server-side tool execution for cloud models                │   │
│  │  • SSE streaming with pipeline trace events                   │   │
│  │  • HTTPS with self-signed certs                               │   │
│  │  • Shared folder / file transfer serving                      │   │
│  └──────────┬────────────────────────┬───────────────────────────┘   │
│             │                        │                               │
│      local models              cloud models                          │
│             │                        │                               │
│             ▼                        ▼                                │
│  ┌──────────────────┐  ┌─────────────────────────────────────┐      │
│  │ llama.cpp        │  │ Mistral API / Codestral API         │      │
│  │ (port 30080)     │  │ (tool loop + result caching)        │      │
│  │ Nemotron 30B     │  └─────────────────────────────────────┘      │
│  └──────────────────┘                                                │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                     maude_core.py                             │    │
│  │  Shared tool implementations: files, shell, web, vision,     │    │
│  │  image gen, Google, scheduling, shared folder, caching        │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

## Features

- **Multi-client access**: Server TUI, Mac/PC CLI, phone PWA, Telegram bot
- **Model routing**: Auto-routes between local Nemotron and cloud Mistral/Codestral via gateway
- **Tool execution**: Files, shell, web browse/search, vision, image generation
- **Tool result caching**: TTL-based cache (30min web, 5min vision) to reduce redundant calls
- **Pipeline trace**: Inline display of tool calls, args, result previews, and timing
- **Typewriter effects**: Word-by-word response reveal on server TUI and client CLI
- **Shared folder**: Bidirectional rsync between server and clients (`~/.maude/shared/`)
- **Auto-routing**: Classifies messages and routes to specialized subagents
- **Frontier delegation**: Escalates to cloud AI (Mistral, Codestral) for complex tasks
- **Scheduled tasks**: Cron-based automation with natural language scheduling
- **Google integration**: Gmail, Drive, Sheets, Calendar, Slides, Contacts, YouTube
- **Voice mode**: Speech input/output via Whisper transcription
- **Conversation memory**: Persistent context across sessions

## Models

| Model | Type | Use Case |
|-------|------|----------|
| **Nemotron-3-Nano-30B** | Local (llama.cpp) | Default — general tasks, tool use |
| **Mistral Large** | Cloud (API) | Complex reasoning, longer context |
| **Codestral** | Cloud (API) | Code generation and analysis |
| **LLaVA 13B** | Local (Ollama) | Vision — image and screenshot analysis |

Switch models at runtime with `/model switch mistral` or `/model switch nemotron`.

## Clients

### Server TUI (`chat_local.py`)
Textual-based terminal UI running directly on the DGX Spark. Full tool access, animated banner, voice mode, typewriter response display.

```bash
./maude          # Starts all services + TUI
```

### Mac/PC Client (`maude-client`)
Pip-installable CLI that connects to the Spark server via Tailscale.

```bash
pip install "https://github.com/mboard8070/terminal-llm/archive/main.tar.gz#subdirectory=maude-client"
maude            # Connect and chat
```

Commands: `/help`, `/model`, `/voice`, `/sync`, `/update`, `/version`, `clear`, `quit`

### Phone PWA (`maude-phone`)
Capacitor-based progressive web app served by the gateway. Camera integration for photo analysis.

### Telegram Bot (`run_telegram.py`)
Accessible from anywhere via Telegram. Runs as a systemd service on the server.

## Gateway

The gateway (`gateway.py`) runs on port 30000 and is the single entry point for all remote clients.

- **Model routing**: Routes requests to local llama.cpp or cloud APIs based on model name
- **Tool execution**: Runs server-side tool loops for cloud models (Mistral/Codestral)
- **SSE trace events**: Streams `tool_call`, `tool_result`, and `llm_call` trace events to clients
- **File serving**: Serves shared folder files, PWA assets, and generated images
- **HTTPS**: Self-signed certs for Tailscale connections

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
| `run_command` | Execute shell commands (rm, mv, cp, pip, git, etc.) |

### Web & Vision
| Tool | Description |
|------|-------------|
| `web_browse` | Fetch and parse web pages (text extraction) |
| `web_search` | Search via DuckDuckGo |
| `web_view` | Screenshot webpage + LLaVA visual analysis |
| `view_image` | Analyze local images with LLaVA |
| `generate_image` | Generate images via Flux/ComfyUI |

### Shared Folder & Transfers
| Tool | Description |
|------|-------------|
| `list_shared` | List files in the shared folder |
| `share_file` | Copy a file into the shared folder for clients |
| `list_transfers` | List files uploaded by clients |
| `get_transfer` | Copy a client upload to the working directory |

### Delegation & Automation
| Tool | Description |
|------|-------------|
| `ask_frontier` | Escalate to cloud AI for complex reasoning |
| `send_to_claude` | Delegate tasks to Claude Code |
| `schedule_task` | Create/manage cron-based automated tasks |

### Google Integration
| Tool | Description |
|------|-------------|
| `gmail_list` / `gmail_read` / `gmail_send` | Email operations |
| `drive_list` / `drive_search` / `drive_read` / `drive_upload` / `drive_delete` | Drive operations |
| `sheets_read` / `sheets_write` / `sheets_create` | Spreadsheet operations |
| `calendar_list` / `calendar_create` / `calendar_delete_event` | Calendar operations |
| `slides_create` / `slides_add_slide` | Presentation operations |
| `contacts_list` / `contacts_search` | Contact operations |
| `youtube_search` | YouTube search |

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Build llama.cpp and download model
./setup_local.sh

# Install Playwright for web screenshots
playwright install chromium

# Install LLaVA for vision (via Ollama)
ollama pull llava:13b

# Set up API keys for cloud models (optional)
export MISTRAL_API_KEY="your-key"
export CODESTRAL_API_KEY="your-key"
```

## Quick Start

```bash
cd ~/nvidia-workbench/terminal-llm
./maude
```

This starts the inference server, gateway, and TUI. Use `/model switch mistral` to switch to cloud models.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `LLM_SERVER_URL` | `http://localhost:30080/v1` | llama.cpp server endpoint |
| `MAUDE_MODEL` | `nemotron` | Default model name |
| `MAUDE_NUM_CTX` | `32768` | Context window size (local models) |
| `VISION_SERVER_URL` | `http://localhost:11434/v1` | Ollama endpoint for LLaVA |
| `MAUDE_VISION_MODEL` | `llava:13b` | Vision model name |
| `MISTRAL_API_KEY` | — | Mistral API key (for cloud routing) |
| `CODESTRAL_API_KEY` | — | Codestral API key (for cloud routing) |

## Project Structure

```
terminal-llm/
├── maude                  # Main launcher
├── maude_core.py          # Shared tools, caching, rate limiting
├── chat_local.py          # Server TUI (Textual)
├── gateway.py             # Gateway — model routing, tool loops, SSE, file serving
├── auto_router.py         # Message classification and subagent routing
├── execution.py           # Subagent execution
├── frontier.py            # Cloud AI escalation
├── memory.py              # Persistent conversation memory
├── scheduler.py           # Cron-based task scheduling
├── google_tools.py        # Google Workspace integration
├── voice.py               # Voice mode (Whisper)
├── run_telegram.py        # Telegram bot
├── maude-client/          # Mac/PC CLI client (pip-installable)
│   ├── maude_client/
│   │   ├── cli.py         # Client main loop, spinner, typewriter, SSE trace
│   │   ├── client_tools.py# Client-side tool implementations
│   │   ├── shared_sync.py # Bidirectional rsync daemon
│   │   └── config.py      # Server connection config
│   └── pyproject.toml
├── maude-phone/           # Phone PWA (Capacitor + Vite)
│   ├── src/               # TypeScript/HTML frontend
│   └── capacitor.config.ts
├── shared/                # Shared folder (synced to clients)
├── transfers/             # Client uploads
└── certs/                 # HTTPS certificates for gateway
```

## License

MIT License
