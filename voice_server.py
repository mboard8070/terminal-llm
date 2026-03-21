#!/usr/bin/env python3
"""
MAUDE Voice Server — Nemotron ASR (0.6B) + Magpie TTS (357M).

Replaces PersonaPlex with a cascaded pipeline that routes through the
MAUDE gateway, giving voice access to all tools and any LLM model.

Binary protocol (unchanged from PersonaPlex — mobile apps work as-is):
  0x00  Server → Client  Handshake (models ready)
  0x01  Client → Server  Audio frame (Opus-encoded OGG pages @ 24kHz)
  0x02  Server → Client  Text (UTF-8)
  0x03  Server → Client  Audio frame (PCM float32 @ 24kHz, 1920 samples)

Usage:
    python voice_server.py [--port 8998] [--ssl ./certs]
"""

import argparse
import asyncio
import json
import logging
import os
import ssl as _ssl
import threading
from typing import ClassVar
from urllib.parse import parse_qs, urlparse

import aiohttp
import numpy as np
import sphn
import websockets

logger = logging.getLogger("voice_server")

# Audio constants (match moshi_bridge protocol)
SAMPLE_RATE = 24000
FRAME_SIZE = 1920  # 80ms @ 24kHz
FRAME_BYTES = FRAME_SIZE * 4  # float32
TTS_SAMPLE_RATE = 22050  # Magpie native output
ASR_SAMPLE_RATE = 16000  # Nemotron ASR input

# VAD parameters
VAD_ENERGY_THRESHOLD = 0.003
VAD_SILENCE_FRAMES = 12  # ~960ms of silence = end of speech
VAD_MIN_SPEECH_FRAMES = 5  # ~400ms minimum to bother transcribing

# Gateway
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:30080/v1")
TOOL_MODEL = os.environ.get("TOOL_MODEL", "mistral")
VOICE_PORT = int(os.environ.get("VOICE_PORT", "8998"))


# ---------------------------------------------------------------------------
# Nemotron ASR
# ---------------------------------------------------------------------------
class NemoASR:
    """Nemotron Speech Streaming ASR (0.6B) — batch transcription."""

    def __init__(self):
        self.model = None
        self._lock = threading.Lock()

    def load(self):
        import nemo.collections.asr as nemo_asr

        logger.info("Loading Nemotron ASR 0.6B...")
        self.model = nemo_asr.models.ASRModel.from_pretrained("nvidia/nemotron-speech-streaming-en-0.6b")
        self.model.eval()
        logger.info("Nemotron ASR loaded")

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        """Transcribe PCM float32 audio to text."""
        if self.model is None:
            return ""
        with self._lock:
            # Resample to 16kHz (ASR model input rate)
            if sample_rate != ASR_SAMPLE_RATE:
                audio_16k = sphn.resample(audio, sample_rate, ASR_SAMPLE_RATE)
            else:
                audio_16k = audio

            # NeMo transcribe returns list of str or Hypothesis objects
            result = self.model.transcribe([audio_16k])
            if isinstance(result, list) and len(result) > 0:
                item = result[0]
                if isinstance(item, str):
                    return item.strip()
                # Hypothesis object — extract .text
                if hasattr(item, "text"):
                    return item.text.strip()
                # Fallback: some models return nested lists
                if isinstance(item, list) and len(item) > 0:
                    inner = item[0]
                    if hasattr(inner, "text"):
                        return inner.text.strip()
                    return str(inner).strip()
            return ""


# ---------------------------------------------------------------------------
# Magpie TTS
# ---------------------------------------------------------------------------
class MagpieTTS:
    """Magpie TTS Multilingual (357M) — text to speech."""

    MAX_TEXT_CHARS = 500
    SPEAKERS: ClassVar[dict[str, int]] = {"John": 0, "Sofia": 1, "Aria": 2, "Jason": 3, "Leo": 4}

    def __init__(self, speaker: str = "Sofia"):
        self.model = None
        self.speaker_idx = self.SPEAKERS.get(speaker, 1)
        self._lock = threading.Lock()

    def load(self):
        from nemo.collections.tts.models import MagpieTTSModel

        logger.info("Loading Magpie TTS 357M...")
        self.model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
        self.model.eval()
        logger.info("Magpie TTS loaded (speaker: %s)", self._speaker_name())

    def _speaker_name(self) -> str:
        for name, idx in self.SPEAKERS.items():
            if idx == self.speaker_idx:
                return name
        return "unknown"

    @staticmethod
    def _clean_for_speech(text: str) -> str:
        """Strip markdown, code blocks, and special chars that TTS can't handle."""
        import re

        # Remove code blocks
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"`[^`]+`", "", text)
        # Remove markdown formatting
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"#{1,6}\s+", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove bullet points and numbered lists
        text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
        # Collapse whitespace
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def synthesize(self, text: str) -> np.ndarray | None:
        """Convert text to PCM float32 @ 24kHz (resampled from 22kHz)."""
        text = self._clean_for_speech(text)[: self.MAX_TEXT_CHARS]
        if not text:
            return None
        # Truncate at last sentence boundary for natural speech
        for sep in (". ", "! ", "? "):
            idx = text.rfind(sep)
            if 20 < idx < len(text) - 1:
                text = text[: idx + 1]
                break
        with self._lock:
            try:
                audio, _audio_len = self.model.do_tts(text, language="en", speaker_index=self.speaker_idx)
                pcm_22k = audio.squeeze().cpu().numpy().astype(np.float32)
                # Resample 22kHz → 24kHz so client's 24k→48k resampler works
                pcm_24k = sphn.resample(pcm_22k, TTS_SAMPLE_RATE, SAMPLE_RATE)
                return pcm_24k
            except Exception as e:
                logger.error("TTS synthesis failed: %s", e)
                return None


# ---------------------------------------------------------------------------
# Audio Buffer with VAD
# ---------------------------------------------------------------------------
class AudioBuffer:
    """VAD-aware audio accumulator. Yields complete utterances."""

    def __init__(self):
        self._buf = np.array([], dtype=np.float32)
        self._speech_buf = np.array([], dtype=np.float32)
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False
        self._utterance_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self._closed = False

    def feed(self, pcm: np.ndarray):
        """Feed decoded PCM samples. Detects speech boundaries."""
        if self._closed:
            return
        self._buf = np.concatenate((self._buf, pcm))

        while len(self._buf) >= FRAME_SIZE:
            frame = self._buf[:FRAME_SIZE]
            self._buf = self._buf[FRAME_SIZE:]
            rms = float(np.sqrt(np.mean(frame**2)))

            if rms > VAD_ENERGY_THRESHOLD:
                self._speech_frames += 1
                self._silence_frames = 0
                if not self._in_speech:
                    self._in_speech = True
                    self._speech_buf = frame.copy()
                else:
                    self._speech_buf = np.concatenate((self._speech_buf, frame))
            elif self._in_speech:
                self._speech_buf = np.concatenate((self._speech_buf, frame))
                self._silence_frames += 1
                if self._silence_frames >= VAD_SILENCE_FRAMES:
                    if self._speech_frames >= VAD_MIN_SPEECH_FRAMES:
                        self._utterance_queue.put_nowait(self._speech_buf.copy())
                    self._in_speech = False
                    self._speech_frames = 0
                    self._silence_frames = 0
                    self._speech_buf = np.array([], dtype=np.float32)

    async def get_utterance(self) -> np.ndarray | None:
        """Wait for next complete utterance. Returns None when closed."""
        return await self._utterance_queue.get()

    def close(self):
        self._closed = True
        if self._in_speech and self._speech_frames >= VAD_MIN_SPEECH_FRAMES:
            self._utterance_queue.put_nowait(self._speech_buf.copy())
        self._utterance_queue.put_nowait(None)


# ---------------------------------------------------------------------------
# Gateway Client
# ---------------------------------------------------------------------------
class GatewayClient:
    """Async HTTP client for the MAUDE gateway LLM + tools API."""

    def __init__(self):
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def chat(self, messages: list[dict]) -> str:
        """POST to gateway and return response text."""
        await self._ensure_session()
        payload = {
            "model": TOOL_MODEL,
            "messages": messages,
            "stream": False,
            "max_tokens": 512,
        }
        try:
            async with self._session.post(
                f"{GATEWAY_URL}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("Gateway error: %s", e)
            return "Sorry, I couldn't complete that request."

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# ---------------------------------------------------------------------------
# WebSocket Connection Handler
# ---------------------------------------------------------------------------
asr_model: NemoASR | None = None
tts_model: MagpieTTS | None = None


async def handle_connection(websocket):
    """Handle a single voice WebSocket connection."""
    peer = websocket.remote_address
    logger.info("[%s] voice connection opened", peer)

    # Parse query params (text_prompt comes from mobile app)
    params = parse_qs(urlparse(websocket.request.path).query)
    text_prompt = params.get("text_prompt", [""])[0]

    # Send handshake — models are pre-loaded, no wait needed
    await websocket.send(bytes([0x00]))
    logger.info("[%s] handshake sent", peer)

    # Per-connection state
    send_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    audio_buffer = AudioBuffer()
    conversation: list[dict] = []
    gateway = GatewayClient()

    if text_prompt:
        conversation.append({"role": "system", "content": text_prompt})

    opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)

    async def recv_loop():
        """Receive Opus audio from client, decode, feed to VAD."""
        try:
            async for message in websocket:
                if isinstance(message, bytes) and len(message) > 0 and message[0] == 0x01:
                    opus_reader.append_bytes(message[1:])
                    while True:
                        pcm = opus_reader.read_pcm()
                        if pcm.shape[-1] == 0:
                            break
                        audio_buffer.feed(pcm.flatten())
        except websockets.ConnectionClosed:
            pass
        finally:
            audio_buffer.close()

    async def process_loop():
        """On speech boundaries, run ASR → LLM → TTS pipeline."""
        loop = asyncio.get_event_loop()
        try:
            while True:
                utterance = await audio_buffer.get_utterance()
                if utterance is None:
                    break

                # ASR
                text = await loop.run_in_executor(None, asr_model.transcribe, utterance, SAMPLE_RATE)
                if not text or len(text.strip()) < 2:
                    continue

                logger.info("[%s] ASR: %s", peer, text[:120])
                send_queue.put_nowait(b"\x02" + f"[You]: {text}\n".encode())
                conversation.append({"role": "user", "content": text})

                # LLM (through gateway — gets tools, web search, etc.)
                response = await gateway.chat(conversation)
                logger.info("[%s] LLM: %s", peer, response[:120])
                conversation.append({"role": "assistant", "content": response})
                send_queue.put_nowait(b"\x02" + f"[MAUDE]: {response}\n".encode())

                # TTS
                pcm = await loop.run_in_executor(None, tts_model.synthesize, response)
                if pcm is not None and len(pcm) > 0:
                    for i in range(0, len(pcm), FRAME_SIZE):
                        chunk = pcm[i : i + FRAME_SIZE]
                        if len(chunk) < FRAME_SIZE:
                            chunk = np.pad(chunk, (0, FRAME_SIZE - len(chunk)))
                        send_queue.put_nowait(b"\x03" + chunk.astype(np.float32).tobytes())
                        # Pace at real-time to avoid client ring buffer overflow
                        await asyncio.sleep(FRAME_SIZE / SAMPLE_RATE)

        except asyncio.CancelledError:
            pass

    async def send_loop():
        """Drain send_queue to WebSocket."""
        try:
            while True:
                msg = await send_queue.get()
                if msg is None:
                    break
                await websocket.send(msg)
        except websockets.ConnectionClosed:
            pass

    tasks = [
        asyncio.create_task(recv_loop()),
        asyncio.create_task(process_loop()),
        asyncio.create_task(send_loop()),
    ]

    _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

    await gateway.close()
    logger.info("[%s] voice connection closed", peer)


# ---------------------------------------------------------------------------
# Request Router
# ---------------------------------------------------------------------------
async def route_request(connection, request):
    """Route HTTP requests. Accept /api/chat for voice, /api/status for health."""
    path = urlparse(request.path).path
    if path == "/api/status":
        body = json.dumps(
            {
                "backend": "nemotron-asr-magpie-tts",
                "asr_loaded": asr_model is not None and asr_model.model is not None,
                "tts_loaded": tts_model is not None and tts_model.model is not None,
            }
        ).encode()
        return websockets.http11.Response(200, "OK", websockets.Headers({"Content-Type": "application/json"}), body)
    if path == "/api/chat":
        return None  # Accept WebSocket upgrade
    return websockets.http11.Response(404, "Not Found", websockets.Headers(), b"Not found")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    global asr_model, tts_model, TOOL_MODEL

    parser = argparse.ArgumentParser(description="MAUDE Voice Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=VOICE_PORT, type=int)
    parser.add_argument("--ssl", type=str, default=None, help="Dir with cert.pem + key.pem")
    parser.add_argument("--speaker", default="Sofia", choices=list(MagpieTTS.SPEAKERS))
    parser.add_argument("--model", default=TOOL_MODEL, help="LLM model for responses")
    args = parser.parse_args()

    TOOL_MODEL = args.model

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Load models (one-time, ~20s)
    asr_model = NemoASR()
    asr_model.load()

    tts_model = MagpieTTS(speaker=args.speaker)
    tts_model.load()

    # SSL
    ssl_context = None
    if args.ssl:
        cert = os.path.join(args.ssl, "cert.pem")
        key = os.path.join(args.ssl, "key.pem")
        if os.path.exists(cert) and os.path.exists(key):
            ssl_context = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(certfile=cert, keyfile=key)
            logger.info("SSL enabled from %s", args.ssl)

    logger.info("Starting voice server on %s:%d", args.host, args.port)

    async with websockets.serve(
        handle_connection,
        args.host,
        args.port,
        ssl=ssl_context,
        max_size=2**20,
        process_request=route_request,
    ):
        logger.info("Voice server ready")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
