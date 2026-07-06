# MAUDE Portable Local Install Plan

## Goal

MAUDE should install and run on Windows, macOS, and Linux without requiring a DGX Spark. A user chooses where the gateway lives, then uses the terminal client, desktop browser, mobile browser, or installed PWA against that gateway.

The Spark becomes one supported gateway host, not the default assumption.

## Repo Direction

Use this repo (`maude`) as the full product repo because it already contains the gateway, phone app, client, tool registry, shared folder routes, and model routing. Borrow the simpler packaging and command shape from `maude-cloud`, especially:

- `maude` as the primary console command
- `maude --serve` or `maude gateway` for gateway-only mode
- user data under platform user config/data directories
- setup wizard before first run

## Target Modes

| Mode | Command | Purpose |
| --- | --- | --- |
| Desktop TUI + local gateway | `maude` | Normal single-machine install. Starts or connects to the local gateway, then opens the TUI. |
| Gateway only | `maude gateway` | Runs the API/web/mobile gateway on this machine. |
| Client only | `maude client --gateway URL` | Connects this terminal to any reachable MAUDE gateway. |
| Web app | `maude web` | Opens the local browser to the gateway-served web app. |
| Setup | `maude setup` | Configures model providers, gateway exposure, certs, and optional tools. |
| Doctor | `maude doctor` | Checks OS, Python, browser, API keys, gateway reachability, and optional services. |

## Gateway Placement

During setup, the user chooses one gateway exposure level:

| Choice | Bind | Access |
| --- | --- | --- |
| This computer only | `127.0.0.1` | Terminal and browser on the same machine. |
| LAN | `0.0.0.0` | Other devices on the same network can connect. |
| Tailscale/private mesh | `0.0.0.0` plus detected Tailscale URL | Mobile/remote access through a private network. |
| Public HTTPS | explicit host/cert | Advanced deployment only. |

Default should be local-only. The setup wizard can print a QR code for mobile access when LAN or Tailscale is enabled.

## Ports

Use portable defaults and keep old Spark ports as compatibility aliases:

- Gateway API/web: `8080` by default for new installs
- Legacy gateway: accept `30000`/`30080` when configured or detected
- Local model servers: optional and only checked when local model routes are enabled

## Data Layout

Runtime files should not live in the repo checkout. Use platform data directories:

| Data | Path |
| --- | --- |
| config | `~/.config/maude/config.json` on Linux/macOS, `%APPDATA%\Maude\config.json` on Windows |
| shared files | user data dir `shared/` |
| uploads/transfers | user data dir `transfers/` |
| conversations | user data dir `conversations/` |
| logs | user cache/log dir |
| certs | user config dir `certs/` |

## Tool Strategy

Tools need capability checks instead of Spark assumptions.

Core cross-platform tools:

- file read/write/search/list within configured allowed roots
- shell command execution with OS-aware subprocess behavior
- web search/browse
- image viewing through configured cloud/local vision provider
- shared folder upload/download
- GitHub tools when `gh` is installed and authenticated
- Google tools when credentials are configured
- scheduler with OS-specific backend

Optional tools:

- local llama/ollama routes
- Playwright browser screenshots and visible browser login
- ComfyUI/Flux image generation
- voice input/output
- VNC/noVNC
- Telegram bot
- remote mesh/subagents

Each optional tool should report `unavailable` with a concrete reason instead of failing inside a task.

## Mobile And Web

The gateway should serve the web/PWA app directly:

- `/app/` serves the bundled app
- `/health` returns gateway URL, OS, version, bind mode, and available capabilities
- setup prints local, LAN, and Tailscale URLs when available
- setup can generate a QR code for `/app/`
- HTTPS is optional for local/LAN, required only for features that browsers restrict

The phone UI should say "Gateway" or "MAUDE host", not "Spark". Spark can appear only as a detected hostname or optional node type.

## Packaging

Short-term:

- Keep source install working with `pip install -e .`
- Add console scripts in `pyproject.toml`
- Add OS-specific setup instructions
- Build the phone app into static assets that the gateway can serve

Long-term:

- Publish Python package when stable
- Add signed Windows/macOS installers if needed
- Add Docker image for Linux/server users
- Add system service helpers:
  - Windows Task Scheduler or service wrapper
  - macOS LaunchAgent
  - Linux systemd user service

## Migration Work

1. Add a real CLI entrypoint for this repo.
2. Move gateway defaults to configurable host/port/bind values.
3. Replace Spark-specific client config with `MAUDE_GATEWAY_URL`.
4. Update client and phone copy from "Spark" to "Gateway" or "host".
5. Add `maude setup` and `maude doctor`.
6. Serve bundled web/PWA assets from the gateway in portable mode.
7. Add capability detection for tools.
8. Add tests for Windows/macOS/Linux path and subprocess behavior.
9. Keep Spark profile as `maude spark` or an equivalent named legacy profile.

## Compatibility Requirements

- Existing Spark/Tailscale installs must keep working.
- Existing client `/update`, `/version`, `/sync`, and shared-folder flows should continue to work.
- Existing gateway routes should remain stable:
  - `/v1/chat/completions`
  - `/v1/models`
  - `/health`
  - `/list`
  - `/download/*`
  - `/upload/*`
  - `/api/tools`
  - `/api/tools/execute`
  - `/app/*`

## First Implementation Slice

The safest first slice is:

1. Add CLI commands: `maude gateway`, `maude client`, `maude setup`, `maude doctor`.
2. Introduce `MAUDE_GATEWAY_URL`, `MAUDE_GATEWAY_HOST`, and `MAUDE_GATEWAY_PORT`.
3. Make local gateway mode default to `127.0.0.1:8080`.
4. Keep `spark-e26c:30000` only as a named legacy profile.
5. Update docs and UI labels to remove Spark as the default identity.

