#!/usr/bin/env python3
"""
MAUDE Client - Local interface connecting to Spark server for inference.

Connects via Tailscale to the MAUDE gateway (default: server:30000).

Run:
  maude
  python -m maude_client
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time

_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:
    # Force-remove readline/pyreadline. Keep input private/raw below, but enable
    # ANSI on stdout so MAUDE output remains readable in modern terminals.
    sys.modules.pop("readline", None)
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass
else:
    # macOS ships libedit-backed readline. Keep it simple and non-blocking so
    # keystrokes echo immediately; avoid completion hooks that can swallow input.
    try:
        import readline

        readline.set_completer(None)
        try:
            readline.set_completion_display_matches_hook(None)
        except Exception:
            pass
        _rl_doc = getattr(readline, "__doc__", None) or ""
        if "libedit" in _rl_doc:
            # libedit: insert tab literally, no completion menu.
            readline.parse_and_bind("bind ^I rl_insert")
            try:
                readline.parse_and_bind("bind -e")  # emacs mode, predictable editing
            except Exception:
                pass
        else:
            readline.parse_and_bind("set disable-completion on")
            readline.parse_and_bind("tab: self-insert")
    except Exception:
        pass
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from collections.abc import Callable, Generator

from maude_client import __version__
from maude_client.config import CLIENT_NAME, MODEL_NAME, SERVER_HOST, SERVER_LLM_PORT, TEMPERATURE
from maude_client.heartbeat import get_hostname, get_platform, start_heartbeat, stop_heartbeat
from maude_client.task_executor import start_task_executor, stop_task_executor
from maude_client.tool_router import ToolRouter
from maude_client.writing_rules import application_writing_block

# ─────────────────────────────────────────────────────────────────
# Spinner & Typewriter
# ─────────────────────────────────────────────────────────────────

_BRAILLE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_ASCII_SPIN = "|/-\\"

_COLOR_ENABLED = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
_RESET = "\033[0m" if _COLOR_ENABLED else ""
_USER = "\033[92m" if _COLOR_ENABLED else ""
_ASSISTANT = "\033[95m" if _COLOR_ENABLED else ""
_RESPONSE = "\033[97m" if _COLOR_ENABLED else ""
_TOOL = "\033[96m" if _COLOR_ENABLED else ""
_DIM = "\033[2m" if _COLOR_ENABLED else ""
_WARN = "\033[93m" if _COLOR_ENABLED else ""


def color(text: str, style: str) -> str:
    return f"{style}{text}{_RESET}" if style else text


def _windows_input_restore():
    """Put the Windows console in a private, non-line-editing input mode."""
    if not _IS_WINDOWS:
        return lambda: None
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-10)
        original = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(original)):
            return lambda: None

        ENABLE_PROCESSED_INPUT = 0x0001
        ENABLE_WINDOW_INPUT = 0x0008
        ENABLE_EXTENDED_FLAGS = 0x0080
        raw_mode = ENABLE_PROCESSED_INPUT | ENABLE_WINDOW_INPUT | ENABLE_EXTENDED_FLAGS
        kernel32.SetConsoleMode(handle, raw_mode)

        def restore() -> None:
            try:
                kernel32.SetConsoleMode(handle, original.value)
            except Exception:
                pass

        return restore
    except Exception:
        return lambda: None


def _windows_input(label: str) -> str:
    import msvcrt

    restore_console = _windows_input_restore()
    try:
        sys.stdout.write(f"\r\n{color(label + ':', _USER)} ")
        sys.stdout.flush()
        chars: list[str] = []
        while True:
            ch = msvcrt.getwch()
            if ch in ("\r", "\n"):
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return "".join(chars).strip()
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch == "\x1a":
                raise EOFError
            if ch in ("\x00", "\xe0"):
                # Extended key prefix: consume the key code and ignore arrows/F-keys.
                msvcrt.getwch()
                continue
            if ch == "\t":
                # Never move the cursor on tab while MAUDE owns the prompt.
                continue
            if ch in ("\b", "\x7f"):
                if chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if ch >= " ":
                chars.append(ch)
                sys.stdout.write(ch)
                sys.stdout.flush()
    finally:
        restore_console()


def prompt_input(label: str) -> str:
    if _IS_WINDOWS:
        return _windows_input(label)
    # Paint the prompt ourselves and flush before blocking. input() still uses
    # readline/libedit for history and editing, but an empty prompt avoids a
    # second write that can race background thread output on macOS.
    if _COLOR_ENABLED:
        sys.stdout.write(f"\n{color(label + ':', _USER)} ")
    else:
        sys.stdout.write(f"\n{label}: ")
    sys.stdout.flush()
    return input().strip()


def _windows_console_mode(handle_id: int) -> str:
    if not _IS_WINDOWS:
        return "n/a"
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(handle_id)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return f"0x{mode.value:08x}"
        return "not-a-console"
    except Exception as exc:
        return f"error: {exc}"


def debug_windows_input() -> None:
    print(f"MAUDE client v{__version__}")
    print("Windows input path: raw msvcrt + private console mode")
    print(f"Python: {sys.executable}")
    print(f"Platform: {sys.platform}")
    print(f"Args: {sys.argv}")
    print(f"stdin mode: {_windows_console_mode(-10)}")
    print(f"stdout mode: {_windows_console_mode(-11)}")
    print(f"WT_SESSION: {os.environ.get('WT_SESSION', '')}")
    print(f"TERM_PROGRAM: {os.environ.get('TERM_PROGRAM', '')}")
    print(f"MAUDE_CLIENT_TASKS: {os.environ.get('MAUDE_CLIENT_TASKS', '')}")
    if not _IS_WINDOWS:
        print("Input diagnostic is only meaningful on Windows.")
        return

    import msvcrt

    print("Press keys. Ctrl-C exits. Enter records CR. Esc exits.")
    idx = 0
    while True:
        ch = msvcrt.getwch()
        code = ord(ch)
        name = repr(ch)
        if ch in ("\x00", "\xe0"):
            nxt = msvcrt.getwch()
            print(f"key {idx}: prefix={name} ord={code} next={nxt!r} next_ord={ord(nxt)}")
        else:
            print(f"key {idx}: char={name} ord={code}")
        sys.stdout.flush()
        idx += 1
        if ch in ("\x03", "\x1b"):
            break


class Spinner:
    """Spinner shown while waiting for first response chunk."""

    def __init__(self, label: str = "thinking"):
        self._label = label
        self._running = False
        self._thread: threading.Thread | None = None
        self._frames = _ASCII_SPIN if _IS_WINDOWS else _BRAILLE

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        i = 0
        while self._running:
            frame = self._frames[i % len(self._frames)]
            print(f"\r{_DIM}{frame} {self._label}...{_RESET}", end="", flush=True)
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
    """Print a response chunk immediately.

    Older builds slept ~6ms per character on large chunks. That blocked the
    main thread, filled the macOS TTY input buffer unevenly, and made typing
    feel like keys needed 3-4 presses before a letter appeared.
    """
    if not chunk:
        return
    if chunk.startswith("[status]"):
        print(color(chunk, _DIM), end="", flush=True)
        return
    if chunk.startswith("[Tool:") or chunk.startswith("[tool:"):
        print(color(chunk, _TOOL), end="", flush=True)
        return
    if chunk.startswith("[Error"):
        print(color(chunk, _WARN), end="", flush=True)
        return
    print(f"{_RESPONSE}{chunk}{_RESET}" if _COLOR_ENABLED else chunk, end="", flush=True)


# Quiet-by-default UI: behave more like the phone/web chat surface.
# Full tool payloads still go to the model; only compact status hits the TTY.
_VERBOSE_UI = os.environ.get("MAUDE_CLIENT_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}
_MODEL_TOOL_RESULT_CHARS = int(os.environ.get("MAUDE_CTX_MAX_TOOL_CHARS", "4000"))
_VERBOSE_DISPLAY_CHARS = 1200


def _short_arg_summary(args, limit: int = 72) -> str:
    """One-line arg summary for status lines."""
    if not args:
        return ""
    preferred = ("path", "pattern", "query", "command", "filename", "name", "url", "repo")
    parts = []
    for key in preferred:
        if key in args and args[key] not in (None, ""):
            val = str(args[key]).replace("\n", " ").strip()
            if len(val) > 48:
                val = val[:45] + "..."
            parts.append(val)
            break
    if not parts:
        try:
            raw = json.dumps(args, ensure_ascii=False)
        except Exception:
            raw = str(args)
        raw = raw.replace("\n", " ").strip()
        parts.append(raw[:48] + ("..." if len(raw) > 48 else ""))
    summary = " ".join(parts)
    if len(summary) > limit:
        summary = summary[: limit - 3] + "..."
    return summary


def _summarize_tool_result(func_name: str, result: str) -> str:
    """Compact human-facing summary; never dump bulk file/search output."""
    text = (result or "").strip()
    if not text:
        return "done"
    lower = text.lower()
    if lower.startswith("error"):
        first = text.splitlines()[0]
        return first if len(first) <= 160 else first[:157] + "..."

    lines = [ln for ln in text.splitlines() if ln.strip()]
    line_count = len(lines)

    if func_name in {"search_files", "search_directory", "search_file"}:
        if "no matches" in lower:
            return "no matches"
        extra = 0
        if "more matches" in lower:
            try:
                extra = int(text.rsplit("and", 1)[-1].split("more", 1)[0].strip())
            except Exception:
                extra = 0
        total = line_count
        if extra:
            total = max(line_count - 1, 0) + extra
        return f"{total} matches"

    if func_name in {"list_directory", "list_server_files", "list_transfers"}:
        return f"{line_count} items" if line_count else "empty"

    if func_name == "read_file":
        return f"{line_count} lines" if line_count else "empty file"

    if func_name == "run_command":
        return f"{line_count} output lines" if line_count else "ok"

    if func_name in {"write_file", "edit_file", "upload_to_server", "download_from_server"}:
        return lines[0][:120] if lines else "done"

    if line_count <= 2 and len(text) <= 160:
        return text.replace("\n", " ")
    return f"ok · {line_count} lines"


def _prepare_tool_result_for_model(result: str, tool_name: str = "tool") -> str:
    """Keep enough context for the model without unbounded payloads."""
    text = result if isinstance(result, str) else str(result)
    try:
        # maude_client.context_hygiene uses maude_core when present, else local port
        from maude_client.context_hygiene import compact_tool_result

        return compact_tool_result(tool_name, text)
    except Exception:
        if len(text) > _MODEL_TOOL_RESULT_CHARS:
            return text[:_MODEL_TOOL_RESULT_CHARS] + "\n... (truncated)"
        return text


def _hygiene_client_messages(*, full: bool = True) -> None:
    """Bound client-side conversation history.

    full=True  — sliding window + drop old tools (start of user turn)
    full=False — only stub old tool payloads (between tool rounds)
    """
    global messages
    try:
        if full:
            from maude_client.context_hygiene import apply_hygiene_in_place

            apply_hygiene_in_place(messages)
        else:
            from maude_client.context_hygiene import drop_old_tool_payloads

            drop_old_tool_payloads(messages, in_place=True)
    except Exception:
        # Lightweight fallback: stub tool results older than last 2
        tool_idxs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        if len(tool_idxs) > 2:
            for i in tool_idxs[:-2]:
                content = messages[i].get("content") or ""
                if isinstance(content, str) and len(content) > 200 and not content.startswith("[prior"):
                    messages[i]["content"] = f"[prior tool result summarized] {content[:160]}..."
        if full:
            keep = int(os.environ.get("MAUDE_CTX_KEEP_RECENT_TURNS", "12"))
            keep = max(4, keep * 2)
            if len(messages) > keep + 4:
                messages[:] = messages[-keep:]


def _format_tool_status(func_name: str, args=None, result: str = None) -> str:
    """Chat-like tool status line for the terminal."""
    arg_bit = _short_arg_summary(args or {})
    if result is None:
        if arg_bit:
            return f"\n[status] {func_name} {arg_bit}\n"
        return f"\n[status] {func_name}\n"
    summary = _summarize_tool_result(func_name, result)
    if _VERBOSE_UI:
        preview = (result or "").strip()
        if len(preview) > _VERBOSE_DISPLAY_CHARS:
            preview = preview[:_VERBOSE_DISPLAY_CHARS] + "\n... (truncated)"
        header = f"\n[Tool: {func_name}] {summary}\n"
        return f"{header}{preview}\n"
    return f"\n[status] {func_name} · {summary}\n"


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
            resp = requests.get(f"https://{SERVER_HOST}:{self.TRANSCRIPTION_PORT}/health", timeout=2, verify=False)
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
        deps["tts"] = subprocess.run(["which", "say"], capture_output=True).returncode == 0

        return deps

    def check_server_available(self) -> bool:
        """Check if transcription server is available."""
        try:
            resp = requests.get(f"https://{SERVER_HOST}:{self.TRANSCRIPTION_PORT}/health", timeout=2, verify=False)
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
            print(color("OK", _TOOL))
        except ImportError:
            try:
                import whisper

                print("Loading local Whisper...", end=" ", flush=True)
                self.whisper_model = whisper.load_model("tiny")
                self._whisper_type = "original"
                print(color("OK", _TOOL))
            except ImportError:
                raise RuntimeError("Neither faster-whisper nor whisper installed")

    def record_audio(self, silence_threshold=0.02, silence_duration=1.5) -> bytes:
        """Record audio until silence detected."""
        import numpy as np
        import sounddevice as sd

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

    def transcribe_server(self, audio_bytes: bytes) -> str | None:
        """Transcribe audio using server GPU."""
        try:
            files = {"audio": ("audio.wav", audio_bytes, "audio/wav")}
            resp = requests.post(
                f"https://{SERVER_HOST}:{self.TRANSCRIPTION_PORT}/transcribe", files=files, timeout=30, verify=False
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

    def listen_and_transcribe(self) -> str | None:
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

_active_task_message = ""

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
Be concise and helpful.

{application_writing_block()}"""


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


def _chat_timeout_for_model(model: str | None = None) -> int:
    """HTTP timeout for /v1/chat/completions.

    Grok CLI runs can take several minutes (server default MAUDE_GROK_TIMEOUT=600),
    so the client budget must be higher than the generic 300s used for other models.
    """
    name = (model or current_model or "").lower()
    if "grok" in name:
        return int(os.environ.get("MAUDE_CLIENT_GROK_TIMEOUT", "900"))
    return int(os.environ.get("MAUDE_CLIENT_CHAT_TIMEOUT", "300"))


def _post_chat_payload(payload: dict, *, retry_without_tools: bool = True) -> requests.Response:
    """POST a chat payload, falling back only if sanitized Mistral tools still fail."""
    payload = _prepare_payload_for_provider(payload)
    timeout = _chat_timeout_for_model(payload.get("model", current_model))
    response = requests.post(API_URL, json=payload, stream=True, timeout=timeout, verify=False)
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
        response = requests.post(API_URL, json=fallback_payload, stream=True, timeout=timeout, verify=False)

    if response.status_code >= 400:
        raise RuntimeError(_format_http_error(response))
    return response


def check_server_connection() -> bool:
    """Check if the LLM server is reachable."""
    try:
        response = requests.get(f"https://{SERVER_HOST}:{SERVER_LLM_PORT}/v1/models", timeout=5, verify=False)
        return response.status_code == 200
    except:
        return False


def _format_trace(data: dict) -> str:
    """Format a trace event for terminal display (dim/muted, quiet by default)."""
    t = data.get("type", "")
    # Default UI matches chat: hide noisy token/result dumps.
    if not _VERBOSE_UI:
        if t == "tool_call":
            # Prefer friendly task labels (e.g. "Running Grok agent") over raw names.
            label = (data.get("task") or data.get("name") or "tool").strip()
            return f"{_DIM}  · {label}{_RESET}\n"
        if t == "keepalive":
            name = data.get("name") or "working"
            elapsed = data.get("elapsed", 0)
            return f"{_DIM}  ⠿ {name} ({elapsed}s){_RESET}\n"
        if t == "context_trim":
            removed = data.get("removed", 0)
            if not removed:
                return ""
            return f"{_DIM}  · context trimmed ({removed} msgs){_RESET}\n"
        if t in {"tool_result", "llm_call"}:
            return ""
        return ""

    if t == "tool_call":
        args = data.get("args", "")
        args_s = str(args)
        if len(args_s) > 100:
            args_s = args_s[:97] + "..."
        label = data.get("task") or data.get("name", "")
        return f"{_DIM}  [{label}] {args_s}{_RESET}\n"
    if t == "tool_result":
        elapsed = data.get("elapsed", 0)
        preview = str(data.get("preview", "") or "")
        if len(preview) > 100:
            preview = preview[:97] + "..."
        return f"{_DIM}    -> {preview} ({elapsed}s){_RESET}\n"
    if t == "keepalive":
        name = data.get("name") or "working"
        elapsed = data.get("elapsed", 0)
        return f"{_DIM}  ⠿ still working: {name} ({elapsed}s){_RESET}\n"
    if t == "context_trim":
        removed = data.get("removed", 0)
        return f"{_DIM}  [context] trimmed {removed} messages{_RESET}\n"
    if t == "llm_call":
        pt = data.get("prompt_tokens", 0)
        ct = data.get("completion_tokens", 0)
        elapsed = data.get("elapsed", 0)
        return f"{_DIM}  [{pt}+{ct} tokens, {elapsed}s]{_RESET}\n"
    return ""


def _parse_tool_args(raw) -> dict:
    """Best-effort JSON parse for streamed tool arguments."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _run_tool_calls(tool_calls: list) -> Generator[str, None, None]:
    """Execute tool calls and append ordered tool results to messages."""
    for tc in tool_calls:
        func_name = (tc.get("function") or {}).get("name") or "tool"
        args = _parse_tool_args((tc.get("function") or {}).get("arguments"))
        try:
            result = router.execute(func_name, args)
        except Exception as exc:
            result = f"Error executing {func_name}: {exc}"
        model_result = _prepare_tool_result_for_model(result, tool_name=func_name)
        yield _format_tool_status(func_name, args, result)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.get("id") or "",
                "name": func_name,
                "content": model_result,
            }
        )


def _stream_model_turn(payload: dict) -> Generator[str, None, None]:
    """Stream one model turn.

    Yields:
      ("content", text)
      ("trace", text)
      ("done", full_content, tool_calls)
      ("error", text)
    """
    try:
        response = _post_chat_payload(payload)
    except requests.exceptions.ConnectionError:
        yield (
            "error",
            "\n[Error: Cannot connect to server. Is Tailscale connected and the server running?]\n",
        )
        return
    except Exception as e:
        yield ("error", f"\n[Error: {e}]\n")
        return

    full_content = ""
    tool_calls = []
    current_event_type = None

    try:
        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode("utf-8")

            # Default gateway transport: SSE comments (`: trace {...}`).
            # Without this, Grok keepalives / tool progress never reach the TUI.
            if line.startswith(": trace "):
                try:
                    trace_data = json.loads(line[len(": trace ") :])
                    trace_line = _format_trace(trace_data)
                    if trace_line:
                        yield ("trace", trace_line)
                except json.JSONDecodeError:
                    pass
                continue

            if line.startswith("event: "):
                current_event_type = line[7:].strip()
                continue

            if not line.startswith("data: "):
                continue

            data = line[6:]
            if data == "[DONE]":
                break

            if current_event_type == "trace":
                current_event_type = None
                try:
                    trace_data = json.loads(data)
                    trace_line = _format_trace(trace_data)
                    if trace_line:
                        yield ("trace", trace_line)
                except json.JSONDecodeError:
                    pass
                continue

            current_event_type = None

            try:
                chunk = json.loads(data)
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {}) or {}

                if delta.get("content"):
                    full_content += delta["content"]
                    yield ("content", delta["content"])

                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        while len(tool_calls) <= idx:
                            tool_calls.append(
                                {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            )
                        if tc.get("id"):
                            tool_calls[idx]["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            tool_calls[idx]["function"]["name"] = fn["name"]
                        if fn.get("arguments"):
                            tool_calls[idx]["function"]["arguments"] += fn["arguments"]
            except (json.JSONDecodeError, IndexError, KeyError, TypeError):
                continue
    finally:
        try:
            response.close()
        except Exception:
            pass

    # Drop incomplete/empty tool call slots
    tool_calls = [tc for tc in tool_calls if (tc.get("function") or {}).get("name")]
    yield ("done", full_content, tool_calls)


def stream_chat(user_message: str) -> Generator[str, None, None]:
    """Send message and stream the response with a multi-round tool loop.

    Unlike the old client path, this keeps going across many tool rounds
    (like the phone/web UI) instead of stopping after 1-2 tool calls.
    """
    global _active_task_message
    messages.append({"role": "user", "content": user_message})
    _active_task_message = user_message
    _hygiene_client_messages(full=True)

    # Fast dispatch only for simple single-shot intents. Multi-step tasks
    # ("check X then do Y", "find and fix", etc.) go through the model loop.
    if _looks_like_single_shot(user_message):
        result = router.fast_dispatch(user_message)
        if result:
            tool_name, args, tool_result = result
            model_result = _prepare_tool_result_for_model(tool_result, tool_name=tool_name)
            yield _format_tool_status(tool_name, args, tool_result)
            # Ask the model to turn the tool result into a normal answer
            # instead of ending the turn with raw tool output.
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "fast_dispatch_1",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(args or {}),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": "fast_dispatch_1",
                    "content": model_result,
                }
            )
            # Fall through into the normal continuation loop below.
        # if fast_dispatch misses, fall through to model

    # Keep the original user text available for tool-group selection on later rounds.
    # Re-select each round so sticky domain activation expands schemas mid-loop.
    max_rounds = int(os.environ.get("MAUDE_CLIENT_MAX_TOOL_ROUNDS", "12"))

    try:
        for round_idx in range(max_rounds):
            # Between tool rounds: stub old tool bodies only (keep the user ask)
            if round_idx > 0:
                _hygiene_client_messages(full=False)
            tools = router.get_tools_for_message(user_message, messages=messages)
            payload = {
                "model": current_model,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                "temperature": TEMPERATURE,
                "max_tokens": 4096,
                "stream": True,
            }
            # Always offer tools until the final forced text round.
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            full_content = ""
            tool_calls = []
            errored = False

            for item in _stream_model_turn(payload):
                kind = item[0]
                if kind == "content" or kind == "trace":
                    yield item[1]
                elif kind == "error":
                    yield item[1]
                    errored = True
                    break
                elif kind == "done":
                    full_content = item[1] or ""
                    tool_calls = item[2] or []

            if errored:
                return

            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": full_content or None,
                        "tool_calls": tool_calls,
                    }
                )
                yield "\n"
                for status in _run_tool_calls(tool_calls):
                    yield status
                yield "\n"
                continue

            # Final text answer
            if full_content:
                messages.append({"role": "assistant", "content": full_content})
            return

        # Hit round limit — force one text-only wrap-up so the user isn't left hanging.
        yield (f"\n{_DIM}[status] reached tool-round limit ({max_rounds}); summarizing progress{_RESET}\n")
        payload = {
            "model": current_model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}]
            + messages
            + [
                {
                    "role": "user",
                    "content": (
                        "Stop calling tools. Briefly summarize what you already did, "
                        "what you learned, and the next concrete step."
                    ),
                }
            ],
            "temperature": TEMPERATURE,
            "max_tokens": 1024,
            "stream": True,
        }
        full_content = ""
        for item in _stream_model_turn(payload):
            if item[0] == "content":
                full_content += item[1]
                yield item[1]
            elif item[0] == "trace":
                yield item[1]
            elif item[0] == "error":
                yield item[1]
                return
            elif item[0] == "done":
                full_content = item[1] or full_content
        if full_content:
            messages.append({"role": "assistant", "content": full_content})
    finally:
        _active_task_message = ""


def _looks_like_single_shot(message: str) -> bool:
    """Allow fast_dispatch only for simple one-action requests."""
    msg = (message or "").strip().lower()
    if not msg:
        return False
    # Multi-step cues should always use the full agent loop.
    multi_cues = [
        " and ",
        " then ",
        " after ",
        " before ",
        " also ",
        "fix",
        "debug",
        "implement",
        "create",
        "build",
        "update",
        "refactor",
        "deploy",
        "commit",
        "push",
        "investigate",
        "why ",
        "how ",
        "make sure",
        "until ",
        "all ",
    ]
    if any(c in msg for c in multi_cues):
        return False
    # Keep fast path for short direct ops: list/read/search/run one thing.
    if len(msg) > 160:
        return False
    simple_prefixes = (
        "list ",
        "ls ",
        "read ",
        "cat ",
        "show ",
        "open ",
        "search ",
        "find ",
        "grep ",
        "run ",
        "pwd",
        "status",
    )
    return msg.startswith(simple_prefixes) or msg in {"ls", "pwd", "status"}


def print_banner():
    """Print startup banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                     MAUDE CLIENT                               ║
    ║              Multi-Unit Dispatch Engine                        ║
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
    global current_model

    if "--diag-input" in sys.argv or os.environ.get("MAUDE_DIAG_INPUT") == "1":
        debug_windows_input()
        return

    print_banner()

    # Check connection
    print("Checking server connection...", end=" ", flush=True)
    if check_server_connection():
        print(color("OK", _TOOL))
    else:
        print(color("FAILED", _WARN))
        print("\nCannot connect to server. Make sure:")
        print("  1. Tailscale is connected")
        print(f"  2. Server is running on {SERVER_HOST}:{SERVER_LLM_PORT}")
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
        print(color("OFFLINE (local tools only)", _WARN))
    for warning in router.health_warnings():
        print(color(f"  WARNING: {warning}", _WARN))

    # Start heartbeat
    print("Starting heartbeat...", end=" ", flush=True)
    try:
        start_heartbeat()
        print(color("OK", _TOOL))
    except Exception as e:
        print(f"Warning: {e}")

    # Start task executor. Windows shell tasks are launched detached from this
    # console in process_utils, so they can run without corrupting chat input.
    if os.environ.get("MAUDE_CLIENT_TASKS") == "0":
        print("Task executor disabled (MAUDE_CLIENT_TASKS=0).")
    else:
        print("Starting task executor...", end=" ", flush=True)
        try:
            start_task_executor()
            print(color("OK", _TOOL))
        except Exception as e:
            print(f"Warning: {e}")

    if _IS_WINDOWS:
        print("Windows input path: raw msvcrt + private console mode")
    print("\nType 'quit' to exit, '/help' for commands.\n")

    try:
        while True:
            try:
                user_input = prompt_input("You")

                if not user_input:
                    continue

                if user_input.lower() == "quit":
                    print(color("Goodbye!", _DIM))
                    break

                if user_input.lower() == "clear":
                    messages.clear()
                    try:
                        from maude_client.context_hygiene import clear_mission_scratch

                        clear_mission_scratch()
                    except Exception:
                        pass
                    print(color("Conversation cleared.", _DIM))
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

                # Handle /current model command
                if user_input.startswith("/current"):
                    current_parts = user_input.split()
                    if len(current_parts) > 1 and current_parts[1].lower() != "model":
                        print("Usage: /current model")
                    else:
                        print(f"\nCurrent model: {current_model}")
                    continue

                # Handle /model command
                if user_input.startswith("/model"):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) == 1:
                        print(f"\nCurrent model: {current_model}")
                        try:
                            resp = requests.get(
                                f"https://{SERVER_HOST}:{SERVER_LLM_PORT}/v1/models", timeout=8, verify=False
                            )
                            if resp.status_code == 200:
                                models = [m["id"] for m in resp.json().get("data", [])]
                                print(f"Available:      {', '.join(models)}")
                            else:
                                print(f"[Gateway returned HTTP {resp.status_code} for /v1/models]")
                        except Exception as exc:
                            print(f"[Could not fetch model list from {SERVER_HOST}:{SERVER_LLM_PORT}: {exc}]")
                    else:
                        model_arg = parts[1].strip()
                        tokens = model_arg.split()
                        if tokens[0].lower() in ("switch", "use", "set") and len(tokens) > 1:
                            model_arg = "-".join(tokens[1:])
                        else:
                            model_arg = "-".join(tokens)
                        if not model_arg:
                            print("Usage: /model <name>   or   /model switch <name>")
                        else:
                            current_model = model_arg
                            print(f"Switched to model: {current_model}")
                    continue

                # Handle /update command
                if user_input == "/update":
                    print("Updating MAUDE client...")
                    print(f"Using Python: {sys.executable}")
                    package_url = (
                        "https://github.com/mboard8070/terminal-llm/archive/main.tar.gz#subdirectory=maude-client"
                    )
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "--upgrade",
                            "--force-reinstall",
                            "--no-cache-dir",
                            package_url,
                        ],
                        capture_output=False,
                    )
                    if result.returncode == 0:
                        if _IS_WINDOWS:
                            print("\nUpdate complete. Close this window and start MAUDE again.")
                            break
                        print("\nUpdate complete. Restarting...")
                        os.execv(sys.executable, [sys.executable, "-m", "maude_client"])
                    else:
                        print("\nUpdate failed. Check the pip output above for details.")
                    continue

                # Handle /version command
                if user_input == "/version":
                    print(color(f"MAUDE client v{__version__}", _DIM))
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
  /current model - Show model in use
  /sync         - Sync shared folder now

Features:
  - Quiet chat UI: tools show compact status, not full file dumps
  - Set MAUDE_CLIENT_VERBOSE=1 for old detailed tool output
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
                try:
                    for chunk in stream_chat(user_input):
                        if first_chunk:
                            spinner.stop()
                            print(color("MAUDE: ", _ASSISTANT), end="", flush=True)
                            first_chunk = False
                        typewriter_print(chunk)
                finally:
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
