# Terminal-LLM

A terminal-based AI chat application that provides coding assistance and system interaction capabilities using NVIDIA's LLM models. Supports both cloud-based and fully offline local inference.

## Features

### Remote Mode (`chat.py`)
- Multi-model support with 5 pre-configured NVIDIA models:
  - Llama-3.3-70B-Instruct (default)
  - NVIDIA Nemotron-51B-Instruct
  - Llama-3.1-405B-Instruct
  - CodeLlama-70B
  - Nemotron-Nano-8B-V1
- Streaming responses for real-time output
- Rich terminal UI with markdown rendering
- Interactive commands: `/quit`, `/clear`, `/model [name]`

### Local Mode (`chat_local.py` / "MAUDE CODE")
- Fully offline operation using Nemotron-3-Nano-30B
- Built-in file system tools:
  - `read_file` - Read files up to 1MB
  - `write_file` - Create or modify files
  - `list_directory` - Browse directories
  - `get_working_directory` / `change_directory` - Navigate the filesystem
  - `run_command` - Execute shell commands (pip, python, git, npm, etc.)
- Token counting with performance metrics
- Animated ASCII art banner

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (for local mode)
- NVIDIA NGC API key (for remote mode)

## Installation

### Remote Mode

```bash
# Install dependencies
pip install -r requirements.txt

# Set up NVIDIA API key
echo 'NGC_PERSONAL_KEY="your_key_here"' > variables.env
```

### Local Mode

```bash
# Run the automated setup script
./setup_local.sh
```

This script will:
1. Create a Python virtual environment
2. Clone and build llama.cpp with CUDA support
3. Download the Nemotron-3-Nano-30B model (~38GB)

## Usage

### Remote Mode

```bash
python chat.py
```

### Local Mode

**Option 1: Separate terminals**
```bash
# Terminal 1: Start the inference server
./start_server.sh

# Terminal 2: Launch the chat interface
./maude
```

**Option 2: Combined launcher**
```bash
./run.sh
```

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/quit` | Exit the application |
| `/clear` | Clear conversation history |
| `/model <name>` | Switch to a different model (remote mode) |

## Configuration

### Environment Variables

| Variable | Description | Required For |
|----------|-------------|--------------|
| `NGC_PERSONAL_KEY` | NVIDIA NGC API authentication token | Remote mode |

### Local Server Settings

The llama.cpp server is configured in `start_server.sh`:

| Setting | Value |
|---------|-------|
| Port | 30000 |
| GPU layers | 99 (full offload) |
| Context size | 8192 tokens |
| Threads | 8 |

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Temperature | 0.2 |
| Max tokens | 4096 |
| Streaming | Enabled |

## Project Structure

```
terminal-llm/
├── chat.py              # Remote mode (NVIDIA NGC API)
├── chat_local.py        # Local mode (llama.cpp)
├── maude                # Shortcut script for local mode
├── run.sh               # Combined launcher
├── setup_local.sh       # Automated local setup
├── start_server.sh      # llama.cpp server launcher
├── requirements.txt     # Python dependencies
├── llama.cpp/           # llama.cpp repository (after setup)
├── models/              # Downloaded model files
└── venv/                # Python virtual environment
```

## NVIDIA AI Workbench

This project is compatible with NVIDIA AI Workbench. The `.project/spec.yaml` file defines GPU resources, storage layout, and environment specifications for seamless integration.

## License

MIT License
