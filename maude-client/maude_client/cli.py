#!/usr/bin/env python3
"""
MAUDE Client - Local interface connecting to Spark server for inference.

Connects via Tailscale to spark-e26c:30000.

Run:
  maude
  python -m maude_client
"""

import os
import sys
import json
import time
import asyncio
import tempfile
import subprocess
import threading
_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:
    # Force-remove readline/pyreadline to prevent it from hooking input()
    sys.modules.pop("readline", None)
    # Enable ANSI/VT processing on Windows console
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass
else:
    try:
        import readline
        readline.set_completer(None)
        _rl_doc = getattr(readline, "__doc__", None) or ""
        if "libedit" in _rl_doc:
            readline.parse_and_bind("bind ^I rl_insert")  # macOS libedit
        else:
            readline.parse_and_bind("tab: self-insert")   # GNU readline
    except Exception:
        pass
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from typing import Optional, Generator, Callable

from maude_client import __version__
from maude_client.config import (
    SERVER_HOST, SERVER_LLM_PORT, MODEL_NAME,
    CONTEXT_SIZE, TEMPERATURE, CLIENT_NAME
)
from maude_client.tool_router import ToolRouter
from maude_client.heartbeat import start_heartbeat, stop_heartbeat, get_hostname, get_platform
from maude_client.task_executor import start_task_executor, stop_task_executor


# ─────────────────────────────────────────────────────────────────
# Spinner & Typewriter
# ─────────────────────────────────────────────────────────────────

_BRAILLE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_ASCII_SPIN = "|/-\\"

class Spinner:
    """Spinner shown while waiting for first response chunk."""

    def __init__(self, label: str = "thinking"):
        self._label = label
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frames = _ASCII_SPIN if _IS_WINDOWS else _BRAILLE

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        i = 0
        while self._running:
            frame = self._frames[i % len(self._frames)]
            print(f"\r{frame} {self._label}...", end="", flush=True)
            time.sleep(0.1)
            i += 1

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        # Clear spinner line
        label_len = len(self._label) + 6  # frame + space + label + "..."
        sys.stdout.write("\r" + " " * label_len + "\r")
        sys.stdout.flush()


def typewriter_print(chunk: str):
    """Print a chunk with typewriter effect for large pieces, instant for small."""
    if not chunk:
        return
    # Skip typewriter for tool output and errors
    if chunk.startswith("[Tool:") or chunk.startswith("[Error"):
        print(chunk, end="", flush=True)
        return
    # Small chunks (natural streaming) → instant
    if len(chunk) <= 40:
        print(chunk, end="", flush=True)
        return
    # Large chunks → char-by-char
    for ch in chunk:
        print(ch, end="", flush=True)
        if ch not in ("\n", " "):
            time.sleep(0.006)


# ─────────────────────────────────────────────────────────────────
# Voice Mode Support
# ─────────────────────────────────────────────────────────────────

class VoiceMode:
    """Voice mode for Mac client with server-side transcription."""

    # Transcription server port (via SSH tunnel)
    TRANSCRIPTION_PORT = 30001

    def __init__(self):
        self.whisper_model = None
        self._active = False
        self._use_server = True  # Default to server-side transcription

    def check_dependencies(self) -> dict:
        """Check which voice dependencies are available."""
        deps = {}

        # Check sounddevice
        try:
            import sounddevice
            deps["sounddevice"] = True
        except ImportError:
            deps["sounddevice"] = False

        # Check server transcription
        try:
            resp = requests.get(
                f"https://{SERVER_HOST}:{self.TRANSCRIPTION_PORT}/health",
                timeout=2, verify=False
            )
            deps["server_transcription"] = resp.status_code == 200
        except:
            deps["server_transcription"] = False

        # Check local whisper (fallback)
        try:
            from faster_whisper import WhisperModel
            deps["local_whisper"] = "faster-whisper"
        except ImportError:
            try:
                import whisper
                deps["local_whisper"] = "whisper"
            except ImportError:
                deps["local_whisper"] = False

        # Check TTS (macOS 'say' command)
        deps["tts"] = subprocess.run(
            ["which", "say"], capture_output=True
        ).returncode == 0

        return deps

    def check_server_available(self) -> bool:
        """Check if transcription server is available."""
        try:
            resp = requests.get(
                f"https://{SERVER_HOST}:{self.TRANSCRIPTION_PORT}/health",
                timeout=2, verify=False
            )
            return resp.status_code == 200
        except:
            return False

    def load_whisper(self):
        """Load local Whisper model (fallback)."""
        if self.whisper_model is not None:
            return

        try:
            from faster_whisper import WhisperModel
            print("Loading local Whisper (faster-whisper)...", end=" ", flush=True)
            self.whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
            self._whisper_type = "faster"
            print("OK")
        except ImportError:
            try:
                import whisper
                print("Loading local Whisper...", end=" ", flush=True)
                self.whisper_model = whisper.load_model("tiny")
                self._whisper_type = "original"
                print("OK")
            except ImportError:
                raise RuntimeError("Neither faster-whisper nor whisper installed")

    def record_audio(self, silence_threshold=0.02, silence_duration=1.5) -> bytes:
        """Record audio until silence detected."""
        import sounddevice as sd
        import numpy as np

        sample_rate = 16000
        chunk_duration = 0.5
        chunk_samples = int(chunk_duration * sample_rate)
        max_silence_chunks = int(silence_duration / chunk_duration)

        print("🎤 Listening... (speak now)")

        chunks = []
        silence_chunks = 0

        while True:
            chunk = sd.rec(chunk_samples, samplerate=sample_rate, channels=1, dtype=np.int16)
            sd.wait()

            amplitude = np.abs(chunk).mean() / 32768.0

            if amplitude > silence_threshold:
                chunks.append(chunk)
                silence_chunks = 0
            elif chunks:
                chunks.append(chunk)
                silence_chunks += 1
                if silence_chunks >= max_silence_chunks:
                    break

        if not chunks:
            return b""

        # Convert to WAV bytes
        recording = np.concatenate(chunks)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            import scipy.io.wavfile as wavfile
            wavfile.write(temp_path, sample_rate, recording)
            with open(temp_path, "rb") as f:
                return f.read()
        finally:
            os.unlink(temp_path)

    def transcribe_server(self, audio_bytes: bytes) -> Optional[str]:
        """Transcribe audio using server GPU."""
        try:
            files = {"audio": ("audio.wav", audio_bytes, "audio/wav")}
            resp = requests.post(
                f"https://{SERVER_HOST}:{self.TRANSCRIPTION_PORT}/transcribe",
                files=files,
                timeout=30, verify=False
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("text", "")
            else:
                print(f"[Server error: {resp.status_code}]")
                return None
        except Exception as e:
            print(f"[Server transcription failed: {e}]")
            return None

    def transcribe_local(self, audio_bytes: bytes) -> str:
        """Transcribe audio locally (fallback)."""
        self.load_whisper()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            if self._whisper_type == "faster":
                segments, _ = self.whisper_model.transcribe(temp_path)
                return " ".join(seg.text for seg in segments).strip()
            else:
                result = self.whisper_model.transcribe(temp_path)
                return result["text"].strip()
        finally:
            os.unlink(temp_path)

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio - try server first, fall back to local."""
        if self._use_server and self.check_server_available():
            result = self.transcribe_server(audio_bytes)
            if result is not None:
                return result
            print("[Falling back to local transcription]")

        return self.transcribe_local(audio_bytes)

    def speak(self, text: str):
        """Speak text using macOS 'say' command."""
        try:
            # Clean text for speech (remove markdown, etc.)
            clean_text = text.replace("**", "").replace("*", "").replace("`", "")
            subprocess.run(["say", clean_text], check=True)
        except FileNotFoundError:
            print(f"[TTS unavailable] {text}")

    def listen_and_transcribe(self) -> Optional[str]:
        """Record audio and return transcribed text."""
        audio = self.record_audio()
        if not audio:
            return None

        if self._use_server:
            print("🔄 Transcribing (server GPU)...")
        else:
            print("🔄 Transcribing (local)...")
        text = self.transcribe(audio)
        return text


# Global voice mode instance
_voice_mode = None


def get_voice_mode() -> VoiceMode:
    """Get or create voice mode instance."""
    global _voice_mode
    if _voice_mode is None:
        _voice_mode = VoiceMode()
    return _voice_mode


def handle_voice_command(args: list, chat_func: Callable) -> bool:
    """
    Handle /voice commands.
    Returns True if command was handled, False otherwise.
    """
    if not args:
        print("""
Voice Commands:
  /voice deps   - Check voice dependencies
  /voice start  - Single voice interaction
  /voice talk   - Continuous voice conversation (say 'stop' to end)
""")
        return True

    action = args[0].lower()
    vm = get_voice_mode()

    if action == "deps":
        deps = vm.check_dependencies()
        print("\nVoice Dependencies:")
        for dep, status in deps.items():
            if status is True:
                print(f"  {dep}: OK")
            elif status is False:
                print(f"  {dep}: MISSING")
            else:
                print(f"  {dep}: {status}")
        return True

    elif action == "start":
        try:
            text = vm.listen_and_transcribe()
            if text:
                print(f"\n📝 You said: {text}")
                print("\nMAUDE: ", end="", flush=True)
                response_text = ""
                for chunk in chat_func(text):
                    print(chunk, end="", flush=True)
                    response_text += chunk
                print()
                # Speak the response
                vm.speak(response_text)
        except Exception as e:
            print(f"\n[Voice error: {e}]")
        return True

    elif action == "talk":
        print("\n🎙️ Talk mode started. Say 'stop', 'exit', or 'quit' to end.\n")
        try:
            while True:
                text = vm.listen_and_transcribe()
                if not text:
                    continue

                print(f"\n📝 You said: {text}")

                # Check for exit commands
                if text.lower().strip() in ["stop", "exit", "quit", "goodbye"]:
                    vm.speak("Goodbye!")
                    print("\n👋 Talk mode ended.")
                    break

                print("\nMAUDE: ", end="", flush=True)
                response_text = ""
                for chunk in chat_func(text):
                    print(chunk, end="", flush=True)
                    response_text += chunk
                print()

                # Speak the response
                vm.speak(response_text)

        except KeyboardInterrupt:
            print("\n\n👋 Talk mode interrupted.")
        return True

    return False

# Tool router (initialized in main())
router: ToolRouter = None

# API endpoint
API_URL = f"https://{SERVER_HOST}:{SERVER_LLM_PORT}/v1/chat/completions"

# Conversation history
messages = []

# Current model (mutable at runtime via /model command)
current_model = MODEL_NAME

# System prompt
_MY_HOSTNAME = get_hostname()
_MY_PLATFORM = get_platform()

SYSTEM_PROMPT = f"""You are MAUDE (Multi-Agent Unified Dispatch Engine), a helpful AI assistant.

You are running as a CLIENT on the user's {_MY_PLATFORM} machine ({_MY_HOSTNAME}), connected to a Spark server for inference.

LOCAL TOOLS (operate on THIS machine):
- read_file, write_file, edit_file: Local file operations
- list_directory, search_files: Browse and search local files
- run_command: Run local shell commands

FILE TRANSFER TOOLS:
- pull_shared: Pull/download a file from the server's shared folder. Use this when the user says "pull", "grab", "fetch", or "download" a file. Just needs the filename.
- upload_to_server: Push/upload a local file to the server.
- download_from_server: Download a file from any path on the server.
- list_shared: List files available in the shared folder (shows both local and server).
- list_server_files: Browse server filesystem.
- sync_shared: Trigger immediate sync of shared folder.

IMPORTANT: When the user says "pull <filename>" or "download <filename>", use the pull_shared tool with that filename. Do NOT use run_command for file transfers.

SHARED FOLDER:
- Server shared folder: ~/nvidia-workbench/terminal-llm/shared/
- Local shared folder: ~/.maude/shared/
- Use pull_shared to grab specific files, or sync_shared to sync everything.

SERVER TOOLS (operate on Spark):
- run_server_command: Run commands on server
- send_to_server_maude: Message the server MAUDE instance (for tasks that need the server-side MAUDE specifically)

WEB TOOLS:
- web_search: Search the web using DuckDuckGo. Use for current events, news, docs, prices, reviews, or any question needing up-to-date info.
- web_browse: Fetch and read content from a web URL.
You CAN and SHOULD search the web when the user asks about anything requiring current information.

COLLABORATION TOOLS (cross-machine task dispatch):
- mesh_status: Show who's online across all devices in the MAUDE mesh. Shows client_id and platform for each device.
- dispatch_task: Send a shell command to ANY device on the mesh by targeting it.
  To target a device, use the "target" parameter with the device's hostname, client_id, or platform name (e.g. "windows", "macos").
  Example: dispatch_task(prompt="dir Desktop", target="windows", capability="SHELL")
  Example: dispatch_task(prompt="ls ~/Desktop", target="Mattwell", capability="SHELL")
  The target client will execute the command and report the result within ~10 seconds.
  Use mesh_status first to see available devices and their names/platforms.
- create_project, list_projects: Manage shared projects.
- list_tasks: Check dispatched task status and results.

CROSS-MACHINE: You CAN run commands on other devices! Use mesh_status to find devices, then dispatch_task with target= to send shell commands. This works across Mac, Windows, and Linux clients.

BROWSER LOGIN (server-side):
- browser_login: Opens a visible browser on Spark via VNC for manual login to any site.
  Accepts shorthand names: "x", "linkedin", "instagram", "facebook", "github", "reddit", "tiktok", "bluesky" or any URL.
  Returns a noVNC URL you open on any device to interact with the login page.
  IMPORTANT: When the user asks to "log in", "login", or "sign in" to any website, USE browser_login — do NOT give text instructions.
- browser_check_session: Check if a saved login is still valid for a site.

BROWSER WORKFLOWS (server-side):
- workflow_create, workflow_run, workflow_list, workflow_get, workflow_delete, workflow_history, workflow_schedule, workflow_unschedule
  Create repeatable browser automations with change detection and email notifications.

Note: Google Workspace tools (Gmail, Drive, Sheets, Calendar, etc.) and browser tools are handled server-side by the gateway's tool loop. Just ask naturally and the gateway will call the right tools.

Current client: {CLIENT_NAME} ({_MY_PLATFORM})
Be concise and helpful."""


def _is_mistralish_model(model: str) -> bool:
    """Return True for models routed through Mistral-compatible APIs."""
    name = (model or "").lower()
    return any(part in name for part in ("mistral", "codestral", "devstral"))


def _format_http_error(response: requests.Response) -> str:
    """Include provider error details instead of only the HTTP status line."""
    detail = response.text.strip()
    if detail:
        try:
            parsed = response.json()
            detail = json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass
        return f"{response.status_code} {response.reason}: {detail}"
    return f"{response.status_code} {response.reason} for url: {response.url}"


def _clean_description(value, limit: int = 900) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text


def _sanitize_json_schema_for_mistral(schema) -> dict:
    """Reduce JSON Schema to the conservative subset Mistral accepts for tools."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    raw_type = schema.get("type", "object")
    if isinstance(raw_type, list):
        raw_type = next((t for t in raw_type if t != "null"), raw_type[0] if raw_type else "string")
    if raw_type not in {"object", "array", "string", "integer", "number", "boolean"}:
        raw_type = "string"

    clean = {"type": raw_type}
    desc = _clean_description(schema.get("description"), 300)
    if desc:
        clean["description"] = desc

    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        clean["enum"] = [v for v in enum if isinstance(v, (str, int, float, bool)) and v is not None][:100]
        if not clean["enum"]:
            clean.pop("enum", None)

    if raw_type == "object":
        props = schema.get("properties")
        if not isinstance(props, dict):
            props = {}
        clean_props = {}
        for key, value in props.items():
            key = str(key)
            if not key:
                continue
            clean_props[key] = _sanitize_json_schema_for_mistral(value)
        clean["properties"] = clean_props

        required = schema.get("required")
        if isinstance(required, list):
            clean_required = [str(k) for k in required if str(k) in clean_props]
            if clean_required:
                clean["required"] = clean_required

    elif raw_type == "array":
        clean["items"] = _sanitize_json_schema_for_mistral(schema.get("items") or {"type": "string"})

    return clean


def _sanitize_tools_for_mistral(tools: list) -> list:
    """Normalize tool definitions before sending them to Mistral-compatible APIs."""
    clean_tools = []
    seen = set()
    for tool in tools or []:
        fn = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = str(fn.get("name") or "")
        name = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in name)[:64]
        if not name or name in seen:
            continue
        seen.add(name)
        clean_tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": _clean_description(fn.get("description"), 900) or name,
                    "parameters": _sanitize_json_schema_for_mistral(fn.get("parameters") or {}),
                },
            }
        )
    return clean_tools


def _prepare_payload_for_provider(payload: dict) -> dict:
    """Apply provider-specific request normalization without mutating callers."""
    prepared = dict(payload)
    if prepared.get("tools") and _is_mistralish_model(prepared.get("model", current_model)):
        prepared["tools"] = _sanitize_tools_for_mistral(prepared.get("tools"))
        if not prepared["tools"]:
            prepared.pop("tools", None)
            prepared.pop("tool_choice", None)
    return prepared


def _post_chat_payload(payload: dict, *, retry_without_tools: bool = True) -> requests.Response:
    """POST a chat payload, falling back only if sanitized Mistral tools still fail."""
    payload = _prepare_payload_for_provider(payload)
    response = requests.post(API_URL, json=payload, stream=True, timeout=300, verify=False)
    if (
        response.status_code == 422
        and retry_without_tools
        and payload.get("tools")
        and _is_mistralish_model(payload.get("model", current_model))
    ):
        response.close()
        fallback_payload = dict(payload)
        fallback_payload.pop("tools", None)
        fallback_payload.pop("tool_choice", None)
        response = requests.post(API_URL, json=fallback_payload, stream=True, timeout=300, verify=False)

    if response.status_code >= 400:
        raise RuntimeError(_format_http_error(response))
    return response


def check_server_connection() -> bool:
    """Check if the LLM server is reachable."""
    try:
        response = requests.get(
            f"https://{SERVER_HOST}:{SERVER_LLM_PORT}/v1/models",
            timeout=5, verify=False
        )
        return response.status_code == 200
    except:
        return False


def _format_trace(data: dict) -> str:
    """Format a trace event for terminal display (dim/muted)."""
    _dim = "" if _IS_WINDOWS else "\033[2m"
    _reset = "" if _IS_WINDOWS else "\033[0m"
    t = data.get("type", "")
    if t == "tool_call":
        return f"{_dim}  [{data.get('name', '')}] {data.get('args', '')}{_reset}\n"
    elif t == "tool_result":
        elapsed = data.get("elapsed", 0)
        return f"{_dim}    -> {data.get('preview', '')} ({elapsed}s){_reset}\n"
    elif t == "llm_call":
        pt = data.get("prompt_tokens", 0)
        ct = data.get("completion_tokens", 0)
        elapsed = data.get("elapsed", 0)
        return f"{_dim}  [{pt}+{ct} tokens, {elapsed}s]{_reset}\n"
    return ""


def stream_chat(user_message: str) -> Generator[str, None, None]:
    """Send message and stream the response."""
    messages.append({"role": "user", "content": user_message})

    # Fast dispatch — try direct tool call first
    result = router.fast_dispatch(user_message)
    if result:
        tool_name, args, tool_result = result
        yield f"\n[{tool_name}]\n{tool_result}\n"
        messages.append({"role": "assistant", "content": f"[Used {tool_name}]\n{tool_result}"})
        return

    # Build request
    payload = {
        "model": current_model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "tools": router.get_tools_for_message(user_message),
        "tool_choice": "auto",
        "temperature": TEMPERATURE,
        "max_tokens": 4096,
        "stream": True
    }

    try:
        response = _post_chat_payload(payload)

        full_content = ""
        tool_calls = []
        current_tool_call = None
        current_event_type = None

        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode('utf-8')

            # Track SSE event type
            if line.startswith('event: '):
                current_event_type = line[7:].strip()
                continue

            if not line.startswith('data: '):
                continue

            data = line[6:]
            if data == '[DONE]':
                break

            # Handle trace events
            if current_event_type == "trace":
                current_event_type = None
                try:
                    trace_data = json.loads(data)
                    trace_line = _format_trace(trace_data)
                    if trace_line:
                        yield trace_line
                except json.JSONDecodeError:
                    pass
                continue

            current_event_type = None

            try:
                chunk = json.loads(data)
                choices = chunk.get('choices', [])
                if not choices:
                    continue
                delta = choices[0].get('delta', {})

                # Handle content
                if 'content' in delta and delta['content']:
                    full_content += delta['content']
                    yield delta['content']

                # Handle tool calls
                if 'tool_calls' in delta:
                    for tc in delta['tool_calls']:
                        idx = tc.get('index', 0)
                        while len(tool_calls) <= idx:
                            tool_calls.append({
                                'id': '',
                                'type': 'function',
                                'function': {'name': '', 'arguments': ''}
                            })

                        if 'id' in tc:
                            tool_calls[idx]['id'] = tc['id']
                        if 'function' in tc:
                            if 'name' in tc['function']:
                                tool_calls[idx]['function']['name'] = tc['function']['name']
                            if 'arguments' in tc['function']:
                                tool_calls[idx]['function']['arguments'] += tc['function']['arguments']

            except (json.JSONDecodeError, IndexError, KeyError):
                continue

        # Process tool calls if any
        if tool_calls and tool_calls[0]['function']['name']:
            yield "\n"

            # Save assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": full_content or None,
                "tool_calls": tool_calls
            })

            # Execute each tool
            for tc in tool_calls:
                func_name = tc['function']['name']
                try:
                    args = json.loads(tc['function']['arguments'])
                except:
                    args = {}

                yield f"\n[Tool: {func_name}]\n"

                result = router.execute(func_name, args)

                # Truncate long results
                if len(result) > 3000:
                    result = result[:3000] + "\n... (truncated)"

                yield f"{result}\n"

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc['id'],
                    "content": result
                })

            # Get follow-up response
            yield "\n"
            for chunk in stream_chat_continuation():
                yield chunk

        else:
            # No tool calls, save assistant message
            if full_content:
                messages.append({"role": "assistant", "content": full_content})

    except requests.exceptions.ConnectionError:
        yield "\n[Error: Cannot connect to server. Is Tailscale connected and the server running?]\n"
    except Exception as e:
        yield f"\n[Error: {e}]\n"


def stream_chat_continuation() -> Generator[str, None, None]:
    """Continue chat after tool execution."""
    payload = {
        "model": current_model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "tools": router.get_tools_for_message(""),
        "tool_choice": "auto",
        "temperature": TEMPERATURE,
        "max_tokens": 4096,
        "stream": True
    }

    try:
        response = _post_chat_payload(payload)

        full_content = ""
        tool_calls = []
        current_event_type = None

        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode('utf-8')

            # Track SSE event type
            if line.startswith('event: '):
                current_event_type = line[7:].strip()
                continue

            if not line.startswith('data: '):
                continue

            data = line[6:]
            if data == '[DONE]':
                break

            # Handle trace events
            if current_event_type == "trace":
                current_event_type = None
                try:
                    trace_data = json.loads(data)
                    trace_line = _format_trace(trace_data)
                    if trace_line:
                        yield trace_line
                except json.JSONDecodeError:
                    pass
                continue

            current_event_type = None

            try:
                chunk = json.loads(data)
                choices = chunk.get('choices', [])
                if not choices:
                    continue
                delta = choices[0].get('delta', {})

                if 'content' in delta and delta['content']:
                    full_content += delta['content']
                    yield delta['content']

                if 'tool_calls' in delta:
                    for tc in delta['tool_calls']:
                        idx = tc.get('index', 0)
                        while len(tool_calls) <= idx:
                            tool_calls.append({
                                'id': '',
                                'type': 'function',
                                'function': {'name': '', 'arguments': ''}
                            })
                        if 'id' in tc:
                            tool_calls[idx]['id'] = tc['id']
                        if 'function' in tc:
                            if 'name' in tc['function']:
                                tool_calls[idx]['function']['name'] = tc['function']['name']
                            if 'arguments' in tc['function']:
                                tool_calls[idx]['function']['arguments'] += tc['function']['arguments']

            except (json.JSONDecodeError, IndexError, KeyError):
                continue

        # Handle recursive tool calls (limit depth)
        if tool_calls and tool_calls[0]['function']['name']:
            messages.append({
                "role": "assistant",
                "content": full_content or None,
                "tool_calls": tool_calls
            })

            for tc in tool_calls:
                func_name = tc['function']['name']
                try:
                    args = json.loads(tc['function']['arguments'])
                except:
                    args = {}

                yield f"\n[Tool: {func_name}]\n"
                result = router.execute(func_name, args)

                if len(result) > 3000:
                    result = result[:3000] + "\n... (truncated)"

                yield f"{result}\n"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc['id'],
                    "content": result
                })

            # One more continuation (prevent infinite loops)
            yield "\n"
            for chunk in final_response():
                yield chunk
        else:
            if full_content:
                messages.append({"role": "assistant", "content": full_content})

    except Exception as e:
        yield f"\n[Error: {e}]\n"


def final_response() -> Generator[str, None, None]:
    """Get final response without tool calls."""
    payload = {
        "model": current_model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "temperature": TEMPERATURE,
        "max_tokens": 2048,
        "stream": True
    }

    try:
        response = requests.post(API_URL, json=payload, stream=True, timeout=120, verify=False)
        full_content = ""

        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode('utf-8')
            if not line.startswith('data: '):
                continue
            data = line[6:]
            if data == '[DONE]':
                break
            try:
                chunk = json.loads(data)
                choices = chunk.get('choices', [])
                if not choices:
                    continue
                delta = choices[0].get('delta', {})
                if 'content' in delta and delta['content']:
                    full_content += delta['content']
                    yield delta['content']
            except:
                continue

        if full_content:
            messages.append({"role": "assistant", "content": full_content})

    except Exception as e:
        yield f"\n[Error: {e}]\n"


def print_banner():
    """Print startup banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                     MAUDE CLIENT                               ║
    ║         Connected to Spark Server for Inference                ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"    Server:  {SERVER_HOST}:{SERVER_LLM_PORT}")
    print(f"    Model:   {current_model}")
    print(f"    Client:  {CLIENT_NAME}")
    print(f"    Version: {__version__}")
    print()


def main():
    """Main chat loop."""
    print_banner()

    # Check connection
    print("Checking server connection...", end=" ", flush=True)
    if check_server_connection():
        print("OK")
    else:
        print("FAILED")
        print("\nCannot connect to server. Make sure:")
        print("  1. Tailscale is connected")
        print(f"  2. Server is running on spark-e26c:{SERVER_LLM_PORT}")
        print("\nThen restart this client.")
        sys.exit(1)

    # Initialize tool router
    global router
    router = ToolRouter()
    print("Fetching tool catalog...", end=" ", flush=True)
    router.fetch_catalog()
    if router.is_online:
        tool_count = len(router._all_tools)
        print(f"OK ({tool_count} tools)")
    else:
        print("OFFLINE (local tools only)")
    for warning in router.health_warnings():
        print(f"  WARNING: {warning}")

    # Start heartbeat
    print("Starting heartbeat...", end=" ", flush=True)
    try:
        start_heartbeat()
        print("OK")
    except Exception as e:
        print(f"Warning: {e}")

    # Start task executor
    print("Starting task executor...", end=" ", flush=True)
    try:
        start_task_executor()
        print("OK")
    except Exception as e:
        print(f"Warning: {e}")

    print("\nType 'quit' to exit, '/help' for commands.\n")

    try:
        while True:
            try:
                if _IS_WINDOWS:
                    sys.stdout.write("\nYou: ")
                    sys.stdout.flush()
                    user_input = sys.stdin.readline()
                    if not user_input:
                        break
                    user_input = user_input.strip()
                else:
                    user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break

                if user_input.lower() == 'clear':
                    messages.clear()
                    print("Conversation cleared.")
                    continue

                # Handle /voice commands
                if user_input.startswith("/voice"):
                    parts = user_input.split()[1:] if len(user_input.split()) > 1 else []
                    handle_voice_command(parts, stream_chat)
                    continue

                # Handle /sync command — pull all files from the server's shared folder
                if user_input == "/sync":
                    print("Pulling shared folder from server...", end=" ", flush=True)
                    from maude_client.client_tools import sync_shared
                    print(sync_shared())
                    continue

                # Handle /model command
                if user_input.startswith("/model"):
                    global current_model
                    parts = user_input.split(maxsplit=1)
                    if len(parts) == 1:
                        # Show current model and list available
                        print(f"\nCurrent model: {current_model}")
                        try:
                            resp = requests.get(
                                f"https://{SERVER_HOST}:{SERVER_LLM_PORT}/v1/models",
                                timeout=5, verify=False
                            )
                            if resp.status_code == 200:
                                models = [m["id"] for m in resp.json().get("data", [])]
                                print(f"Available:      {', '.join(models)}")
                            else:
                                print("[Could not fetch model list]")
                        except Exception:
                            print("[Could not fetch model list]")
                    else:
                        # Handle "/model switch <name>" or "/model <name>"
                        model_arg = parts[1].strip()
                        tokens = model_arg.split()
                        if tokens[0].lower() in ("switch", "use", "set") and len(tokens) > 1:
                            model_arg = "-".join(tokens[1:])
                        else:
                            model_arg = "-".join(tokens)
                        current_model = model_arg
                        print(f"Switched to model: {current_model}")
                    continue

                # Handle /update command
                if user_input == "/update":
                    print("Updating MAUDE client...")
                    # Use tarball URL to avoid git clone entirely (bypasses hook issues)
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir",
                         "https://github.com/mboard8070/terminal-llm/archive/Astra.tar.gz#subdirectory=maude-client"],
                        capture_output=False
                    )
                    if result.returncode == 0:
                        print("\nUpdate complete. Restarting...")
                        os.execv(sys.executable, [sys.executable, "-m", "maude_client"])
                    else:
                        print("\nUpdate failed. Check your SSH key and internet connection.")
                    continue

                # Handle /version command
                if user_input == "/version":
                    print(f"MAUDE client v{__version__}")
                    continue

                # Handle /help
                if user_input == "/help":
                    print(f"""
Commands:
  quit          - Exit MAUDE
  clear         - Clear conversation history
  /help         - Show this help
  /version      - Show client version (v{__version__})
  /update       - Update client from GitHub and restart
  /voice deps   - Check voice dependencies
  /voice start  - Single voice interaction
  /voice talk   - Continuous voice mode
  /model        - Show current model and list available
  /model <name> - Switch to a different model
  /sync         - Sync shared folder now

Features:
  - Shared folder: ~/.maude/shared/ auto-syncs with server every 30s
  - Dynamic tool filtering: Only relevant tools sent per message
  - Fast dispatch: Common queries (list files, etc.) skip the LLM
  - Server delegation: Ask MAUDE to use Gmail, Drive, Calendar, etc.
    via the server MAUDE instance (mention 'server' in your message)
""")
                    continue

                spinner = Spinner("thinking")
                spinner.start()
                first_chunk = True
                for chunk in stream_chat(user_input):
                    if first_chunk:
                        spinner.stop()
                        print("MAUDE: ", end="", flush=True)
                        first_chunk = False
                    typewriter_print(chunk)
                if first_chunk:
                    # No chunks received at all
                    spinner.stop()
                print()

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                break
    finally:
        # Stop heartbeat and task executor on exit
        stop_task_executor()
        stop_heartbeat()


if __name__ == "__main__":
    main()
