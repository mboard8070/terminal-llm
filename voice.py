"""
Voice Mode for MAUDE.

Supports multiple backends:
1. PersonaPlex - Full-duplex speech-to-speech (primary, requires GB10/A100)
2. Whisper + TTS - Traditional STT/TTS pipeline (fallback)
3. Cloud APIs - OpenAI Whisper/TTS, ElevenLabs (optional)
"""

import os
import sys
import asyncio
import subprocess
import tempfile
import base64
import json
import wave
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any
from enum import Enum
from rich.console import Console

console = Console()


class VoiceBackend(Enum):
    """Available voice backends."""
    PERSONAPLEX = "personaplex"  # Full-duplex speech-to-speech
    WHISPER_LOCAL = "whisper_local"  # Local Whisper + TTS
    WHISPER_CLOUD = "whisper_cloud"  # OpenAI Whisper + TTS
    ELEVENLABS = "elevenlabs"  # ElevenLabs TTS


@dataclass
class VoiceConfig:
    """Voice mode configuration."""
    # Backend selection
    backend: VoiceBackend = VoiceBackend.WHISPER_LOCAL

    # PersonaPlex settings
    personaplex_url: str = "wss://localhost:8998/ws"
    personaplex_voice: str = "NATF2"  # Natural female voice 2
    personaplex_role: str = "You are MAUDE, a helpful AI assistant."

    # Whisper settings (local)
    whisper_model: str = "base"  # tiny, base, small, medium, large
    whisper_device: str = "cuda"  # cuda, cpu

    # TTS settings
    tts_provider: str = "piper"  # piper, espeak, say (macOS), openai, elevenlabs
    tts_voice: str = "en_US-lessac-medium"  # Piper voice
    tts_rate: float = 1.0

    # ElevenLabs settings
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel

    # OpenAI TTS settings
    openai_tts_voice: str = "nova"
    openai_tts_model: str = "tts-1"

    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration: float = 0.5  # seconds per chunk for streaming

    # Behavior
    auto_listen: bool = False  # Continuous listening mode
    wake_word: Optional[str] = None  # e.g., "Hey MAUDE"
    silence_threshold: float = 0.03  # For voice activity detection
    silence_duration: float = 1.5  # Seconds of silence to end utterance


class PersonaPlexClient:
    """Client for PersonaPlex speech-to-speech server."""

    def __init__(self, config: VoiceConfig):
        self.config = config
        self.ws = None
        self._connected = False

    async def connect(self):
        """Connect to PersonaPlex WebSocket server."""
        try:
            import websockets
            import ssl

            # PersonaPlex uses self-signed SSL
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            self.ws = await websockets.connect(
                self.config.personaplex_url,
                ssl=ssl_context
            )
            self._connected = True
            console.print("[dim green]Connected to PersonaPlex[/dim green]")

            # Send initial configuration
            await self._configure()

        except Exception as e:
            console.print(f"[red]Failed to connect to PersonaPlex: {e}[/red]")
            self._connected = False
            raise

    async def _configure(self):
        """Send voice and role configuration to PersonaPlex."""
        config_msg = {
            "type": "config",
            "voice": self.config.personaplex_voice,
            "role": self.config.personaplex_role
        }
        await self.ws.send(json.dumps(config_msg))

    async def disconnect(self):
        """Disconnect from PersonaPlex."""
        if self.ws:
            await self.ws.close()
            self._connected = False

    async def send_audio(self, audio_data: bytes):
        """Send audio data to PersonaPlex."""
        if not self._connected:
            raise RuntimeError("Not connected to PersonaPlex")

        # Send as binary frame
        await self.ws.send(audio_data)

    async def receive_audio(self) -> Optional[bytes]:
        """Receive audio response from PersonaPlex."""
        if not self._connected:
            return None

        try:
            response = await asyncio.wait_for(self.ws.recv(), timeout=0.1)
            if isinstance(response, bytes):
                return response
            else:
                # JSON message (status, etc.)
                data = json.loads(response)
                if data.get("type") == "transcript":
                    console.print(f"[dim cyan]Transcript: {data.get('text')}[/dim cyan]")
                return None
        except asyncio.TimeoutError:
            return None

    @property
    def is_connected(self) -> bool:
        return self._connected


class WhisperSTT:
    """Local Whisper speech-to-text."""

    def __init__(self, config: VoiceConfig):
        self.config = config
        self.model = None

    def load_model(self):
        """Load Whisper model."""
        try:
            from faster_whisper import WhisperModel
            console.print(f"[dim]Loading Whisper {self.config.whisper_model}...[/dim]")
            self.model = WhisperModel(
                self.config.whisper_model,
                device=self.config.whisper_device,
                compute_type="float16" if self.config.whisper_device == "cuda" else "int8"
            )
            console.print("[dim green]Whisper loaded[/dim green]")
        except ImportError:
            console.print("[yellow]faster-whisper not installed, trying whisper...[/yellow]")
            try:
                import whisper
                self.model = whisper.load_model(self.config.whisper_model)
            except ImportError:
                raise RuntimeError("Neither faster-whisper nor whisper is installed")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text."""
        if self.model is None:
            self.load_model()

        try:
            # faster-whisper
            from faster_whisper import WhisperModel
            segments, _ = self.model.transcribe(audio_path)
            return " ".join(segment.text for segment in segments).strip()
        except:
            # Original whisper
            result = self.model.transcribe(audio_path)
            return result["text"].strip()

    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes to text."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            return self.transcribe(temp_path)
        finally:
            os.unlink(temp_path)


class TextToSpeech:
    """Text-to-speech with multiple backend support."""

    def __init__(self, config: VoiceConfig):
        self.config = config

    async def speak(self, text: str) -> Optional[bytes]:
        """Convert text to speech audio."""
        provider = self.config.tts_provider.lower()

        if provider == "piper":
            return await self._speak_piper(text)
        elif provider == "espeak":
            return await self._speak_espeak(text)
        elif provider == "say":
            return await self._speak_say(text)
        elif provider == "openai":
            return await self._speak_openai(text)
        elif provider == "elevenlabs":
            return await self._speak_elevenlabs(text)
        else:
            console.print(f"[red]Unknown TTS provider: {provider}[/red]")
            return None

    async def _speak_piper(self, text: str) -> Optional[bytes]:
        """Use Piper TTS (local, fast, good quality)."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            # Piper command
            process = await asyncio.create_subprocess_exec(
                "piper",
                "--model", self.config.tts_voice,
                "--output_file", temp_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate(input=text.encode())

            if process.returncode == 0:
                with open(temp_path, "rb") as f:
                    audio = f.read()
                os.unlink(temp_path)
                return audio
            else:
                console.print("[yellow]Piper failed, falling back to espeak[/yellow]")
                return await self._speak_espeak(text)

        except FileNotFoundError:
            console.print("[yellow]Piper not found, falling back to espeak[/yellow]")
            return await self._speak_espeak(text)

    async def _speak_espeak(self, text: str) -> Optional[bytes]:
        """Use espeak TTS (basic but universal)."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            process = await asyncio.create_subprocess_exec(
                "espeak-ng", "-w", temp_path, text,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            with open(temp_path, "rb") as f:
                audio = f.read()
            os.unlink(temp_path)
            return audio

        except FileNotFoundError:
            # Try espeak without -ng
            try:
                process = await asyncio.create_subprocess_exec(
                    "espeak", "-w", temp_path, text,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()

                with open(temp_path, "rb") as f:
                    audio = f.read()
                os.unlink(temp_path)
                return audio
            except:
                console.print("[red]espeak not found[/red]")
                return None

    async def _speak_say(self, text: str) -> Optional[bytes]:
        """Use macOS 'say' command."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
                temp_path = f.name

            process = await asyncio.create_subprocess_exec(
                "say", "-o", temp_path, text,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            with open(temp_path, "rb") as f:
                audio = f.read()
            os.unlink(temp_path)
            return audio

        except FileNotFoundError:
            console.print("[red]'say' command not found (macOS only)[/red]")
            return None

    async def _speak_openai(self, text: str) -> Optional[bytes]:
        """Use OpenAI TTS API."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            console.print("[red]OPENAI_API_KEY not set[/red]")
            return None

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            response = client.audio.speech.create(
                model=self.config.openai_tts_model,
                voice=self.config.openai_tts_voice,
                input=text
            )
            return response.content

        except Exception as e:
            console.print(f"[red]OpenAI TTS error: {e}[/red]")
            return None

    async def _speak_elevenlabs(self, text: str) -> Optional[bytes]:
        """Use ElevenLabs TTS API."""
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            console.print("[red]ELEVENLABS_API_KEY not set[/red]")
            return None

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{self.config.elevenlabs_voice_id}",
                    headers={"xi-api-key": api_key},
                    json={
                        "text": text,
                        "model_id": "eleven_monolingual_v1",
                        "voice_settings": {
                            "stability": 0.5,
                            "similarity_boost": 0.75
                        }
                    }
                )

                if response.status_code == 200:
                    return response.content
                else:
                    console.print(f"[red]ElevenLabs error: {response.status_code}[/red]")
                    return None

        except Exception as e:
            console.print(f"[red]ElevenLabs error: {e}[/red]")
            return None


class AudioPlayer:
    """Cross-platform audio playback."""

    @staticmethod
    async def play(audio_data: bytes, format: str = "wav"):
        """Play audio data."""
        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            # Try different players
            players = ["ffplay", "aplay", "paplay", "afplay"]  # afplay is macOS

            for player in players:
                try:
                    if player == "ffplay":
                        cmd = [player, "-nodisp", "-autoexit", "-loglevel", "quiet", temp_path]
                    else:
                        cmd = [player, temp_path]

                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                    await process.wait()
                    return
                except FileNotFoundError:
                    continue

            console.print("[red]No audio player found (install ffmpeg)[/red]")

        finally:
            os.unlink(temp_path)


class AudioRecorder:
    """Cross-platform audio recording."""

    def __init__(self, config: VoiceConfig):
        self.config = config
        self._recording = False

    async def record(self, duration: float = None, until_silence: bool = True) -> bytes:
        """
        Record audio from microphone.

        Args:
            duration: Fixed duration in seconds (if set)
            until_silence: Stop recording after silence detected
        """
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            console.print("[red]sounddevice not installed. Run: pip install sounddevice[/red]")
            return b""

        sample_rate = self.config.sample_rate
        channels = self.config.channels

        if duration:
            # Fixed duration recording
            console.print(f"[dim cyan]Recording for {duration}s...[/dim cyan]")
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                dtype=np.int16
            )
            sd.wait()
        else:
            # Record until silence
            console.print("[dim cyan]Listening... (speak now)[/dim cyan]")

            chunks = []
            silence_chunks = 0
            max_silence_chunks = int(self.config.silence_duration / self.config.chunk_duration)
            chunk_samples = int(self.config.chunk_duration * sample_rate)

            self._recording = True

            while self._recording:
                chunk = sd.rec(chunk_samples, samplerate=sample_rate, channels=channels, dtype=np.int16)
                sd.wait()

                # Check for voice activity
                amplitude = np.abs(chunk).mean() / 32768.0

                if amplitude > self.config.silence_threshold:
                    chunks.append(chunk)
                    silence_chunks = 0
                elif chunks:  # Only count silence after speech started
                    chunks.append(chunk)
                    silence_chunks += 1

                    if silence_chunks >= max_silence_chunks:
                        break

            if not chunks:
                return b""

            recording = np.concatenate(chunks)

        # Convert to WAV bytes
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            import scipy.io.wavfile as wavfile
            wavfile.write(temp_path, sample_rate, recording)

            with open(temp_path, "rb") as f:
                return f.read()
        finally:
            os.unlink(temp_path)

    def stop(self):
        """Stop recording."""
        self._recording = False


class VoiceMode:
    """
    Main voice mode controller for MAUDE.

    Supports multiple backends:
    - PersonaPlex: Full-duplex speech-to-speech
    - Whisper + TTS: Traditional pipeline
    """

    def __init__(self, config: VoiceConfig = None):
        self.config = config or VoiceConfig()

        # Initialize components based on backend
        self.personaplex: Optional[PersonaPlexClient] = None
        self.whisper: Optional[WhisperSTT] = None
        self.tts: Optional[TextToSpeech] = None
        self.recorder: Optional[AudioRecorder] = None

        self._active = False
        self._talk_mode = False

    async def initialize(self):
        """Initialize voice components based on configured backend."""
        console.print(f"[dim]Initializing voice mode ({self.config.backend.value})...[/dim]")

        if self.config.backend == VoiceBackend.PERSONAPLEX:
            self.personaplex = PersonaPlexClient(self.config)
            try:
                await self.personaplex.connect()
            except Exception as e:
                console.print(f"[yellow]PersonaPlex unavailable, falling back to Whisper[/yellow]")
                self.config.backend = VoiceBackend.WHISPER_LOCAL
                await self._init_whisper_pipeline()
        else:
            await self._init_whisper_pipeline()

        self._active = True
        console.print("[green]Voice mode initialized[/green]")

    async def _init_whisper_pipeline(self):
        """Initialize Whisper + TTS pipeline."""
        self.whisper = WhisperSTT(self.config)
        self.tts = TextToSpeech(self.config)
        self.recorder = AudioRecorder(self.config)

    async def shutdown(self):
        """Shutdown voice components."""
        self._active = False
        self._talk_mode = False

        if self.personaplex:
            await self.personaplex.disconnect()

    async def listen(self) -> Optional[str]:
        """
        Listen for voice input and return transcribed text.
        """
        if self.config.backend == VoiceBackend.PERSONAPLEX:
            # PersonaPlex handles this internally
            console.print("[yellow]Use talk_mode() for PersonaPlex[/yellow]")
            return None

        # Whisper pipeline
        if not self.recorder:
            await self._init_whisper_pipeline()

        audio_data = await self.recorder.record()
        if not audio_data:
            return None

        text = self.whisper.transcribe_bytes(audio_data)
        console.print(f"[dim]Heard: {text}[/dim]")
        return text

    async def speak(self, text: str):
        """
        Speak text using TTS.
        """
        if self.config.backend == VoiceBackend.PERSONAPLEX:
            # PersonaPlex generates speech directly
            console.print("[yellow]PersonaPlex handles speech generation[/yellow]")
            return

        if not self.tts:
            await self._init_whisper_pipeline()

        audio_data = await self.tts.speak(text)
        if audio_data:
            await AudioPlayer.play(audio_data)

    async def talk_mode(self, maude_callback: Callable[[str], str]):
        """
        Continuous conversation mode.

        For PersonaPlex: Full-duplex streaming
        For Whisper: Listen -> Process -> Speak loop
        """
        console.print("[bold green]Talk mode enabled[/bold green]")
        console.print("[dim]Say 'stop', 'exit', or 'quit' to end. Press Ctrl+C to interrupt.[/dim]")

        self._talk_mode = True

        if self.config.backend == VoiceBackend.PERSONAPLEX:
            await self._personaplex_talk_mode(maude_callback)
        else:
            await self._whisper_talk_mode(maude_callback)

        self._talk_mode = False
        console.print("[dim]Talk mode ended[/dim]")

    async def _personaplex_talk_mode(self, maude_callback: Callable):
        """Full-duplex conversation with PersonaPlex."""
        if not self.personaplex or not self.personaplex.is_connected:
            console.print("[red]PersonaPlex not connected[/red]")
            return

        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            console.print("[red]sounddevice required for PersonaPlex[/red]")
            return

        sample_rate = self.config.sample_rate
        chunk_samples = int(self.config.chunk_duration * sample_rate)

        async def audio_sender():
            """Send microphone audio to PersonaPlex."""
            while self._talk_mode:
                chunk = sd.rec(chunk_samples, samplerate=sample_rate, channels=1, dtype=np.int16)
                sd.wait()
                await self.personaplex.send_audio(chunk.tobytes())

        async def audio_receiver():
            """Receive and play audio from PersonaPlex."""
            while self._talk_mode:
                audio = await self.personaplex.receive_audio()
                if audio:
                    # Play received audio
                    audio_array = np.frombuffer(audio, dtype=np.int16)
                    sd.play(audio_array, sample_rate)

        # Run sender and receiver concurrently
        try:
            await asyncio.gather(
                audio_sender(),
                audio_receiver()
            )
        except KeyboardInterrupt:
            self._talk_mode = False

    async def _whisper_talk_mode(self, maude_callback: Callable):
        """Traditional listen-process-speak loop."""
        while self._talk_mode:
            try:
                # Listen
                text = await self.listen()
                if not text:
                    continue

                # Check for exit commands
                if text.lower().strip() in ["stop", "exit", "quit", "goodbye"]:
                    await self.speak("Goodbye!")
                    break

                console.print(f"[green]You:[/green] {text}")

                # Process with MAUDE
                response = maude_callback(text)
                console.print(f"[magenta]MAUDE:[/magenta] {response}")

                # Speak response
                await self.speak(response)

            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    def stop_talk_mode(self):
        """Stop talk mode."""
        self._talk_mode = False
        if self.recorder:
            self.recorder.stop()


# ─────────────────────────────────────────────────────────────────
# Command Handlers
# ─────────────────────────────────────────────────────────────────

_voice_mode: Optional[VoiceMode] = None


async def get_voice_mode() -> VoiceMode:
    """Get or create the global voice mode instance."""
    global _voice_mode
    if _voice_mode is None:
        _voice_mode = VoiceMode()
        await _voice_mode.initialize()
    return _voice_mode


def handle_voice_command(args: list) -> str:
    """Handle /voice command."""
    if not args:
        return """Voice Mode Commands:

/voice start         - Start voice mode (listen once)
/voice talk          - Start continuous talk mode
/voice stop          - Stop voice mode
/voice config        - Show current configuration
/voice backend <b>   - Set backend (personaplex, whisper_local, whisper_cloud)
/voice tts <provider> - Set TTS (piper, espeak, openai, elevenlabs)

Backends:
  personaplex    - Full-duplex speech-to-speech (requires PersonaPlex server)
  whisper_local  - Local Whisper STT + TTS
  whisper_cloud  - OpenAI Whisper API + TTS

TTS Providers:
  piper          - Local, fast, good quality
  espeak         - Local, basic
  openai         - OpenAI TTS API
  elevenlabs     - ElevenLabs API"""

    action = args[0].lower()

    if action == "config":
        global _voice_mode
        if _voice_mode:
            cfg = _voice_mode.config
            return f"""Voice Configuration:
  Backend: {cfg.backend.value}
  TTS Provider: {cfg.tts_provider}
  Whisper Model: {cfg.whisper_model}
  PersonaPlex URL: {cfg.personaplex_url}
  PersonaPlex Voice: {cfg.personaplex_voice}"""
        return "Voice mode not initialized. Run /voice start first."

    elif action == "backend" and len(args) > 1:
        backend = args[1].lower()
        try:
            new_backend = VoiceBackend(backend)
            if _voice_mode:
                _voice_mode.config.backend = new_backend
            return f"Voice backend set to: {backend}"
        except ValueError:
            return f"Unknown backend: {backend}. Options: personaplex, whisper_local, whisper_cloud"

    elif action == "tts" and len(args) > 1:
        provider = args[1].lower()
        if provider in ["piper", "espeak", "say", "openai", "elevenlabs"]:
            if _voice_mode:
                _voice_mode.config.tts_provider = provider
            return f"TTS provider set to: {provider}"
        return f"Unknown TTS provider: {provider}"

    elif action in ["start", "talk", "stop"]:
        return f"Voice action '{action}' requires async execution. Use voice mode from the chat interface."

    return f"Unknown voice command: {action}"


def check_voice_dependencies() -> Dict[str, bool]:
    """Check which voice dependencies are available."""
    deps = {}

    # Check sounddevice
    try:
        import sounddevice
        deps["sounddevice"] = True
    except ImportError:
        deps["sounddevice"] = False

    # Check faster-whisper
    try:
        from faster_whisper import WhisperModel
        deps["faster_whisper"] = True
    except ImportError:
        deps["faster_whisper"] = False

    # Check websockets (for PersonaPlex)
    try:
        import websockets
        deps["websockets"] = True
    except ImportError:
        deps["websockets"] = False

    # Check piper
    deps["piper"] = subprocess.run(
        ["which", "piper"], capture_output=True
    ).returncode == 0

    # Check espeak
    deps["espeak"] = subprocess.run(
        ["which", "espeak-ng"], capture_output=True
    ).returncode == 0 or subprocess.run(
        ["which", "espeak"], capture_output=True
    ).returncode == 0

    # Check ffplay
    deps["ffplay"] = subprocess.run(
        ["which", "ffplay"], capture_output=True
    ).returncode == 0

    return deps
