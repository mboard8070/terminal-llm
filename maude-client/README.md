# MAUDE Client

Local client for MAUDE that connects to the Spark server for LLM inference via Tailscale.

```
┌─────────────────────────┐              ┌─────────────────────────────┐
│   Your Mac/PC           │              │      Spark Server           │
│                         │  Tailscale   │                             │
│  MAUDE Client ──────────┼──────────────┼──► Gateway (:30000)         │
│  • Local file ops       │              │    ├─► Nemotron (LLM)       │
│  • Server file transfer │              │    ├─► /upload, /download   │
│  • Server commands      │              │    └─► /list, /shared       │
└─────────────────────────┘              └─────────────────────────────┘
```

## Quick Start

### 1. Install Tailscale

Make sure Tailscale is installed and connected on your Mac/PC:
- https://tailscale.com/download

Verify connectivity:
```bash
ping desktop-aveak19
```

### 2. Install the client

```bash
pip install --upgrade "git+ssh://git@github.com/mboard8070/terminal-llm.git#subdirectory=maude-client"
```

### 3. Run

```bash
maude
```

### 4. Update (from inside the client)

```
/update
```

Or from the command line:
```bash
pip install --upgrade "git+ssh://git@github.com/mboard8070/terminal-llm.git#subdirectory=maude-client"
```

## Configuration

Edit `config.py` to customize:

```python
# Server connection (via Tailscale)
SERVER_HOST = "desktop-aveak19"
SERVER_LLM_PORT = 30000
SERVER_FILE_PORT = 30000  # Same port — gateway handles both

# SSH for server commands (optional)
SERVER_SSH_HOST = "Matt@desktop-aveak19"

# Client name (shown in logs)
CLIENT_NAME = "maude-client"
```

## Commands

| Command | Description |
|---------|-------------|
| `quit` | Exit MAUDE |
| `clear` | Clear conversation history |
| `/help` | Show all commands |
| `/version` | Show client version |
| `/update` | Update client from GitHub and restart |
| `/voice deps` | Check voice dependencies |
| `/voice start` | Single voice interaction |
| `/voice talk` | Continuous voice mode |
| `/sync` | Sync shared folder now |

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
| `upload_to_server` | Push files to server transfers folder |
| `download_from_server` | Pull files from server shared folder |
| `list_server_files` | List server shared folder |
| `list_shared` | List shared folder contents |
| `pull_shared` | Pull a specific file from shared |
| `sync_shared` | Sync all files from shared |
| `run_server_command` | Run commands on Spark via SSH |
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
"List shared files"
"Pull pixleus_draft.md from the server"
"Upload report.pdf to the server"
"Sync the shared folder"
```

**Server commands:**
```
"Check if Nemotron is running on the server"
"Run git status on the server"
```

## Troubleshooting

**Cannot connect to server:**
- Make sure Tailscale is connected: `ping desktop-aveak19`
- Make sure the gateway is running on desktop-aveak19
- Verify the gateway is up: `curl -k https://desktop-aveak19:30000/health`

**File transfer fails:**
- Check gateway health: `curl -k https://desktop-aveak19:30000/health`
- List files to verify: `curl -k https://desktop-aveak19:30000/list`
- Verify paths in `config.py`

**Slow responses:**
- Normal — inference runs on the gateway, may take a few seconds
