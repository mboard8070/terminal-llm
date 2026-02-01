# MAUDE Client

Local client for MAUDE that connects to the Spark server for LLM inference.

```
┌─────────────────────────┐         ┌─────────────────────────────┐
│   Your Mac/PC           │         │      Spark Server           │
│                         │  SSH    │                             │
│  MAUDE Client ──────────┼─tunnel──┼──► Nemotron (LLM)           │
│  • Local file ops       │         │                             │
│  • Server file transfer │───scp───┼──► Server filesystem        │
│  • Server commands      │         │    Server MAUDE (optional)  │
└─────────────────────────┘         └─────────────────────────────┘
```

## Quick Start

### 1. Download the client

On your Mac/PC:

```bash
mkdir -p ~/maude-client
scp -r mboard76@spark-e26c:nvidia-workbench/terminal-llm/maude-client/* ~/maude-client/
```

### 2. Run setup

```bash
cd ~/maude-client
chmod +x setup.sh
./setup.sh
```

### 3. Start the client

```bash
maude
```

Or manually:
```bash
~/.maude/start_client.sh
```

## Configuration

Edit `config.py` to customize:

```python
# Server connection
SERVER_HOST = "localhost"
SERVER_LLM_PORT = 30000
SERVER_SSH_HOST = "mboard76@spark-e26c"

# Your work directory on the server
SERVER_WORK_DIR = "~/nvidia-workbench/terminal-llm"

# Client name (shown in logs)
CLIENT_NAME = "macbook"  # Change this!
```

## Tools

### Local Tools (operate on your Mac/PC)

| Tool | Description |
|------|-------------|
| `read_file` | Read local files |
| `write_file` | Write local files |
| `edit_file` | Edit local files |
| `list_directory` | Browse local filesystem |
| `search_files` | Search local files |
| `run_command` | Run local shell commands |

### Server Tools (operate on Spark)

| Tool | Description |
|------|-------------|
| `upload_to_server` | Send files to Spark |
| `download_from_server` | Get files from Spark |
| `list_server_files` | Browse Spark filesystem |
| `run_server_command` | Run commands on Spark |
| `send_to_server_maude` | Message server MAUDE |

## Usage Examples

**Work with local files:**
```
"Read my ~/Documents/notes.txt file"
"List what's in my Downloads folder"
"Search for TODO in my project"
```

**Transfer files:**
```
"Upload report.pdf to the server"
"Download the latest results from the server"
"List files on the server"
```

**Server commands:**
```
"Check if Nemotron is running on the server"
"Run git status on the server"
"Send a message to server MAUDE"
```

## SSH Setup

The client needs passwordless SSH access to the server. Set this up once:

```bash
# Generate SSH key if you don't have one
ssh-keygen -t ed25519

# Copy to server
ssh-copy-id mboard76@spark-e26c
```

## Troubleshooting

**Cannot connect to server:**
- Make sure the Spark server is running Nemotron (`./start_nemo.sh`)
- Check if SSH tunnel is running: `pgrep -f "ssh.*30000"`
- Manually start tunnel: `ssh -L 30000:localhost:30000 mboard76@spark-e26c -N &`

**File transfer fails:**
- Check SSH key is set up: `ssh mboard76@spark-e26c echo OK`
- Verify paths are correct in `config.py`

**Slow responses:**
- Normal - inference runs on server, may take a few seconds
- Check server load: `ssh mboard76@spark-e26c htop`
