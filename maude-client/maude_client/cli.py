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
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from typing import Optional, Generator, Callable

from maude_client import __version__
from maude_client.config import (
    SERVER_HOST, SERVER_LLM_PORT, MODEL_NAME,
    CONTEXT_SIZE, TEMPERATURE, CLIENT_NAME
)
from maude_client.client_tools import TOOLS, execute_tool, get_tools_for_message, fast_dispatch
from maude_client.heartbeat import start_heartbeat, stop_heartbeat
from maude_client.shared_sync import start_sync, stop_sync, sync_now


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

# API endpoint
API_URL = f"https://{SERVER_HOST}:{SERVER_LLM_PORT}/v1/chat/completions"

# Conversation history
messages = []

# System prompt
SYSTEM_PROMPT = f"""You are MAUDE (Multi-Agent Unified Dispatch Engine), a helpful AI assistant.

You are running as a CLIENT on the user's Mac, connected to a Spark server for inference.

LOCAL TOOLS (operate on Mac):
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
- send_to_server_maude: Message the server MAUDE instance

The server MAUDE also has: Gmail, Google Drive, Sheets, Calendar, Slides, Contacts, YouTube, Substack, browser automation, social media posting, web search, and more. Use send_to_server_maude to delegate those tasks.

Current client: {CLIENT_NAME}
Be concise and helpful."""


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


def stream_chat(user_message: str) -> Generator[str, None, None]:
    """Send message and stream the response."""
    messages.append({"role": "user", "content": user_message})

    # Fast dispatch — try direct tool call first
    result = fast_dispatch(user_message)
    if result:
        tool_name, args, tool_result = result
        yield f"\n[{tool_name}]\n{tool_result}\n"
        messages.append({"role": "assistant", "content": f"[Used {tool_name}]\n{tool_result}"})
        return

    # Build request
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "tools": get_tools_for_message(user_message),
        "tool_choice": "auto",
        "temperature": TEMPERATURE,
        "max_tokens": 4096,
        "stream": True
    }

    try:
        response = requests.post(API_URL, json=payload, stream=True, timeout=300, verify=False)
        response.raise_for_status()

        full_content = ""
        tool_calls = []
        current_tool_call = None

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
                delta = chunk.get('choices', [{}])[0].get('delta', {})

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

            except json.JSONDecodeError:
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

                result = execute_tool(func_name, args)

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
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "tools": get_tools_for_message(""),
        "tool_choice": "auto",
        "temperature": TEMPERATURE,
        "max_tokens": 4096,
        "stream": True
    }

    try:
        response = requests.post(API_URL, json=payload, stream=True, timeout=300, verify=False)
        response.raise_for_status()

        full_content = ""
        tool_calls = []

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
                delta = chunk.get('choices', [{}])[0].get('delta', {})

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

            except json.JSONDecodeError:
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
                result = execute_tool(func_name, args)

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
        "model": MODEL_NAME,
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
                delta = chunk.get('choices', [{}])[0].get('delta', {})
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
    print(f"    Model:   {MODEL_NAME}")
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

    # Start heartbeat
    print("Starting heartbeat...", end=" ", flush=True)
    try:
        start_heartbeat()
        print("OK")
    except Exception as e:
        print(f"Warning: {e}")

    # Start shared folder sync
    print("Starting shared folder sync...", end=" ", flush=True)
    try:
        start_sync()
        print("OK")
    except Exception as e:
        print(f"Warning: {e}")

    print("\nType 'quit' to exit, '/help' for commands.\n")

    try:
        while True:
            try:
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

                # Handle /sync command
                if user_input == "/sync":
                    print("Syncing shared folder...", end=" ", flush=True)
                    result = sync_now()
                    print(result)
                    continue

                # Handle /update command
                if user_input == "/update":
                    print("Updating MAUDE client...")
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "--upgrade",
                         "git+ssh://git@github.com/mboard8070/terminal-llm.git#subdirectory=maude-client"],
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
  /sync         - Sync shared folder now

Features:
  - Shared folder: ~/.maude/shared/ auto-syncs with server every 30s
  - Dynamic tool filtering: Only relevant tools sent per message
  - Fast dispatch: Common queries (list files, etc.) skip the LLM
  - Server delegation: Ask MAUDE to use Gmail, Drive, Calendar, etc.
    via the server MAUDE instance (mention 'server' in your message)
""")
                    continue

                print("\nMAUDE: ", end="", flush=True)
                for chunk in stream_chat(user_input):
                    print(chunk, end="", flush=True)
                print()

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                break
    finally:
        # Stop heartbeat and sync on exit
        stop_sync()
        stop_heartbeat()


if __name__ == "__main__":
    main()
