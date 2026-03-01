#!/usr/bin/env python3
"""
moshi_bridge.py — WebSocket-to-pipe bridge for moshi.cpp PersonaPlex inference.

Drop-in replacement for the Python PersonaPlex server. Spawns a personaplex_pipe
subprocess per connection, bridges WebSocket audio ↔ stdin/stdout pipes.

Uses threads for subprocess I/O to avoid asyncio pipe overhead.

Usage:
    python moshi_bridge.py [--port 8998] [--ssl ./certs] [--static ./dist]
"""

import argparse
import asyncio
import json
import logging
import os
import queue
import re
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from urllib.request import Request, urlopen

import aiohttp
from aiohttp import web
import numpy as np
import sphn

# ── Constants ─────────────────────────────────────────────────────

FRAME_SIZE = 1920             # samples per frame @ 24kHz
FRAME_BYTES = FRAME_SIZE * 4  # float32 bytes per frame
SAMPLE_RATE = 24000

PIPE_BIN = os.environ.get(
    "MOSHI_PIPE_BIN",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "moshi.cpp/build/bin/personaplex_pipe"),
)

logger = logging.getLogger("moshi_bridge")

# ── Tool-calling constants ────────────────────────────────────────

MOSHI_STT_BIN = os.environ.get(
    "MOSHI_STT_BIN",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "moshi.cpp/build/bin/moshi-stt"),
)
MOSHI_TTS_BIN = os.environ.get(
    "MOSHI_TTS_BIN",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "moshi.cpp/build/bin/moshi-tts"),
)
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:30080/v1")
TOOL_MODEL = os.environ.get("TOOL_MODEL", "mistral")

# VAD parameters (frame = 1920 samples @ 24kHz = 80ms)
VAD_ENERGY_THRESHOLD = 0.003   # RMS threshold for speech detection
VAD_SILENCE_FRAMES = 12        # ~960ms of silence triggers end-of-speech
VAD_MIN_SPEECH_FRAMES = 5      # ~400ms minimum speech to transcribe
STT_BUFFER_SAMPLES = 10 * SAMPLE_RATE  # rolling 10s buffer

# Keyword patterns that trigger a tool call through the gateway.
# Kept broad to tolerate STT transcription noise.
TOOL_TRIGGER_PATTERNS = [
    # Web / search / lookup
    r'\b(search\b|look\s?up\b|google\b|find out\b|browse\b|web\s*search)',
    # Weather (standalone — very common voice query)
    r'\bweather\b',
    # News / prices / scores
    r'\b(latest news|headline|stock price|bitcoin|score)\b',
    # Email
    r'\b(email|inbox|gmail|send .{0,10}message|check .{0,6}mail)\b',
    # Calendar
    r'\b(schedule|calendar|meeting|appointment)\b',
    # Files / system
    r'\b(list files|read file|system status|disk space)\b',
]


# ── Helpers ───────────────────────────────────────────────────────

def get_lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


# ── SidecarSTT ────────────────────────────────────────────────────

class SidecarSTT:
    """Background STT via the moshi-stt 1B model.

    Receives copies of user PCM audio, detects speech boundaries with a
    simple energy-based VAD, and transcribes completed utterances.
    """

    def __init__(self, stt_bin: str, peer: str):
        self._stt_bin = stt_bin
        self._peer = peer
        self._audio_q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._transcript = ""
        self._transcript_ts = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # -- public API (called from other threads) --

    def feed_audio(self, frame: np.ndarray):
        """Enqueue a PCM frame (float32, 24kHz). Non-blocking."""
        if self._running:
            self._audio_q.put(frame.copy())

    def get_transcript(self):
        """Return (text, monotonic-timestamp) of the latest transcription."""
        with self._lock:
            return self._transcript, self._transcript_ts

    def stop(self):
        self._running = False
        self._audio_q.put(None)

    # -- background thread --

    def _run(self):
        buf = np.array([], dtype=np.float32)
        speech_frames = 0
        silence_frames = 0
        in_speech = False

        while self._running:
            try:
                pcm = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if pcm is None:
                break

            flat = pcm.flatten()
            buf = np.concatenate((buf, flat))
            if buf.shape[0] > STT_BUFFER_SAMPLES:
                buf = buf[-STT_BUFFER_SAMPLES:]

            rms = np.sqrt(np.mean(flat ** 2)) if flat.shape[0] > 0 else 0.0
            if rms > VAD_ENERGY_THRESHOLD:
                speech_frames += 1
                silence_frames = 0
                if not in_speech:
                    logger.info(f"[{self._peer}][stt] speech start (rms={rms:.5f})")
                in_speech = True
            else:
                silence_frames += 1
                if in_speech and silence_frames >= VAD_SILENCE_FRAMES:
                    dur = buf.shape[0] / SAMPLE_RATE
                    logger.info(f"[{self._peer}][stt] speech end: "
                                f"{speech_frames} frames, {dur:.1f}s audio")
                    if speech_frames >= VAD_MIN_SPEECH_FRAMES:
                        self._transcribe(buf)
                    in_speech = False
                    speech_frames = 0
                    silence_frames = 0
                    buf = np.array([], dtype=np.float32)

    def _transcribe(self, audio: np.ndarray):
        if audio.shape[0] < SAMPLE_RATE:
            logger.info(f"[{self._peer}][stt] skip: too short ({audio.shape[0]} samples)")
            return
        logger.info(f"[{self._peer}][stt] transcribing {audio.shape[0]} samples "
                     f"({audio.shape[0]/SAMPLE_RATE:.1f}s), "
                     f"rms={np.sqrt(np.mean(audio**2)):.5f}")
        tmp_wav = tmp_txt = None
        try:
            fd_w, tmp_wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd_w)
            fd_t, tmp_txt = tempfile.mkstemp(suffix=".txt")
            os.close(fd_t)

            audio_i16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
            with wave.open(tmp_wav, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_i16.tobytes())

            result = subprocess.run(
                [self._stt_bin, "-i", tmp_wav, "-o", tmp_txt],
                capture_output=True, timeout=15,
            )
            if result.returncode == 0 and os.path.exists(tmp_txt):
                text = open(tmp_txt).read().strip()
                if text:
                    with self._lock:
                        self._transcript = text
                        self._transcript_ts = time.monotonic()
                    logger.info(f"[{self._peer}][stt] {text[:120]}")
        except Exception as e:
            logger.warning(f"[{self._peer}][stt] failed: {e}")
        finally:
            for p in (tmp_wav, tmp_txt):
                if p:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass


# ── ToolResultSpeaker ─────────────────────────────────────────────

class ToolResultSpeaker:
    """Synthesize text to 24kHz float32 PCM via moshi-tts."""

    MAX_TEXT_CHARS = 300

    def __init__(self, tts_bin: str = MOSHI_TTS_BIN):
        self._tts_bin = tts_bin

    def synthesize(self, text: str) -> "np.ndarray | None":
        """Run moshi-tts, return float32 PCM @ 24kHz or None on failure."""
        text = text.strip()[:self.MAX_TEXT_CHARS]
        if not text:
            return None
        # Truncate at last sentence boundary for natural speech
        for sep in (". ", "! ", "? "):
            idx = text.rfind(sep)
            if idx > 100:
                text = text[:idx + 1]
                break
        tmp_wav = None
        try:
            fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            cmd = [self._tts_bin, "-d", "CUDA0", "-o", tmp_wav, text]
            logger.info(f"[tts] synthesizing {len(text)} chars")
            result = subprocess.run(cmd, capture_output=True, timeout=45)
            if result.returncode != 0:
                logger.warning(f"[tts] moshi-tts failed (rc={result.returncode}): "
                               f"{result.stderr[:200]}")
                return None
            with wave.open(tmp_wav, "r") as wf:
                n_frames = wf.getnframes()
                if n_frames == 0:
                    return None
                raw = wf.readframes(n_frames)
                sample_width = wf.getsampwidth()
                sr = wf.getframerate()
            if sample_width == 4:
                pcm = np.frombuffer(raw, dtype=np.float32)
            elif sample_width == 2:
                pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                logger.warning(f"[tts] unexpected sample width: {sample_width}")
                return None
            # Resample if not 24kHz (simple linear interpolation)
            if sr != SAMPLE_RATE and sr > 0:
                ratio = SAMPLE_RATE / sr
                new_len = int(len(pcm) * ratio)
                indices = np.linspace(0, len(pcm) - 1, new_len)
                pcm = np.interp(indices, np.arange(len(pcm)), pcm).astype(np.float32)
            return pcm
        except subprocess.TimeoutExpired:
            logger.warning("[tts] moshi-tts timed out")
            return None
        except Exception as e:
            logger.warning(f"[tts] synthesis failed: {e}")
            return None
        finally:
            if tmp_wav:
                try:
                    os.unlink(tmp_wav)
                except OSError:
                    pass


# ── ToolBridge ────────────────────────────────────────────────────

class ToolBridge:
    """Detects tool-worthy queries and routes them through the MAUDE gateway."""

    def __init__(self, ws_send_queue: asyncio.Queue,
                 loop: asyncio.AbstractEventLoop, peer: str):
        self._ws_send_queue = ws_send_queue
        self._loop = loop
        self._peer = peer

    def detect_trigger(self, text: str) -> bool:
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in TOOL_TRIGGER_PATTERNS)

    def call_gateway(self, query: str, history: list) -> str:
        """Blocking POST to the gateway's server-side tool loop."""
        messages = history + [{"role": "user", "content": query}]
        payload = json.dumps({
            "model": TOOL_MODEL,
            "messages": messages,
            "stream": False,
            "max_tokens": 512,
        }).encode()
        req = Request(
            f"{GATEWAY_URL}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"[{self._peer}][tool] gateway error: {e}")
            return f"Sorry, I couldn't complete that request."

    def send_text_to_client(self, text: str):
        """Inject a 0x02 text message into the WS send queue."""
        self._loop.call_soon_threadsafe(
            self._ws_send_queue.put_nowait,
            b"\x02" + text.encode("utf-8"),
        )


# ── WebSocket Handler ─────────────────────────────────────────────

async def handle_chat(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    # Extract query params
    text_prompt = request.query.get("text_prompt", "")
    voice_prompt = request.query.get("voice_prompt", "NATF2")

    # Strip extension to get bare voice name
    voice_name = voice_prompt
    for ext in (".pt", ".gguf", ".safetensors"):
        if voice_name.endswith(ext):
            voice_name = voice_name[:-len(ext)]
            break

    peer = request.remote
    logger.info(f"[{peer}] incoming connection, voice={voice_name}")

    # Build subprocess command
    cmd = [PIPE_BIN]
    if voice_name:
        cmd += ["-v", voice_name]
    if text_prompt:
        cmd += ["-p", wrap_with_system_tags(text_prompt)]

    logger.info(f"[{peer}] spawning personaplex_pipe...")

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,  # unbuffered
        )
    except FileNotFoundError:
        logger.error(f"personaplex_pipe not found at: {PIPE_BIN}")
        await ws.close(message=b"inference engine not found")
        return ws
    except Exception as e:
        logger.error(f"failed to spawn subprocess: {e}")
        await ws.close(message=b"spawn failed")
        return ws

    # ── stderr reader thread (logging) ────────────────────────────

    def stderr_thread():
        for line in proc.stderr:
            logger.info(f"[{peer}][cpp] {line.decode(errors='replace').rstrip()}")

    t_stderr = threading.Thread(target=stderr_thread, daemon=True)
    t_stderr.start()

    # ── Wait for ready signal (blocking, in executor) ─────────────

    loop = asyncio.get_event_loop()

    def wait_ready():
        b = proc.stdout.read(1)
        if not b or b[0] != 0x00:
            return False
        return True

    try:
        ready = await asyncio.wait_for(loop.run_in_executor(None, wait_ready), timeout=120)
    except asyncio.TimeoutError:
        ready = False

    if not ready:
        logger.error(f"[{peer}] subprocess failed to become ready")
        proc.kill()
        await ws.close()
        return ws

    logger.info(f"[{peer}] model ready, sending handshake")
    await ws.send_bytes(b"\x00")

    # ── Shared state ──────────────────────────────────────────────

    close_event = threading.Event()
    # Queue for WS → stdin: raw PCM numpy arrays
    pcm_queue = queue.Queue()
    # Queue for stdout → WS: raw bytes to send
    ws_send_queue = asyncio.Queue()

    # ── Tool-calling state ─────────────────────────────────────────
    tool_mode_active = threading.Event()
    conversation_history: list = []

    stt = None
    tool_bridge = None
    tts_speaker = None
    if os.path.exists(MOSHI_STT_BIN):
        stt = SidecarSTT(MOSHI_STT_BIN, peer)
        tool_bridge = ToolBridge(ws_send_queue, loop, peer)
        logger.info(f"[{peer}] tool support enabled (stt: {MOSHI_STT_BIN})")
        if os.path.exists(MOSHI_TTS_BIN):
            tts_speaker = ToolResultSpeaker(MOSHI_TTS_BIN)
            logger.info(f"[{peer}] TTS enabled (tts: {MOSHI_TTS_BIN})")
        else:
            logger.warning(f"[{peer}] TTS disabled (moshi-tts not at {MOSHI_TTS_BIN})")
    else:
        logger.warning(f"[{peer}] tool support disabled "
                       f"(moshi-stt not at {MOSHI_STT_BIN})")

    # ── stdin writer thread ───────────────────────────────────────

    def stdin_writer():
        """Accumulate PCM from queue, write frame-sized chunks to stdin.

        Forks audio to SidecarSTT for transcription.  During tool mode,
        feeds silence to moshi (keeps the model alive) while STT still
        receives the real audio.
        """
        silence_frame = np.zeros(FRAME_SIZE, dtype=np.float32).tobytes()
        buf = np.array([], dtype=np.float32)
        frames_written = 0
        samples_received = 0
        import time as _time
        t0 = _time.monotonic()
        try:
            while not close_event.is_set():
                try:
                    pcm = pcm_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if pcm is None:
                    break
                samples_received += pcm.shape[-1]
                buf = np.concatenate((buf, pcm)) if buf.shape[0] > 0 else pcm
                while buf.shape[-1] >= FRAME_SIZE:
                    frame = buf[:FRAME_SIZE]
                    buf = buf[FRAME_SIZE:]

                    # Always feed real audio to STT
                    if stt is not None:
                        stt.feed_audio(frame)

                    # During tool mode, feed silence to moshi
                    if tool_mode_active.is_set():
                        proc.stdin.write(silence_frame)
                    else:
                        proc.stdin.write(frame.astype(np.float32).tobytes())
                    proc.stdin.flush()
                    frames_written += 1
                    if frames_written % 25 == 0:
                        dt = _time.monotonic() - t0
                        rate = samples_received / dt if dt > 0 else 0
                        logger.info(f"[{peer}][stdin] frame {frames_written}: "
                                    f"{samples_received} samples in {dt:.1f}s = "
                                    f"{rate:.0f} samp/s ({rate/24000:.2f}x realtime)")
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    # ── stdout reader thread ──────────────────────────────────────

    def stdout_reader():
        """Read framed protocol from stdout, push to WS send queue.

        During tool mode, audio frames (0x03) are read but discarded
        so moshi's stdout doesn't block.  Text tokens (0x02) always
        pass through.
        """
        try:
            while not close_event.is_set():
                type_byte = proc.stdout.read(1)
                if not type_byte:
                    break
                kind = type_byte[0]

                if kind == 0x03:  # audio frame
                    data = proc.stdout.read(FRAME_BYTES)
                    if len(data) == FRAME_BYTES and not tool_mode_active.is_set():
                        loop.call_soon_threadsafe(
                            ws_send_queue.put_nowait,
                            b"\x03" + data
                        )

                elif kind == 0x02:  # text token
                    len_bytes = proc.stdout.read(2)
                    if len(len_bytes) == 2:
                        text_len = struct.unpack("<H", len_bytes)[0]
                        text_data = proc.stdout.read(text_len)
                        if len(text_data) == text_len:
                            loop.call_soon_threadsafe(
                                ws_send_queue.put_nowait,
                                b"\x02" + text_data
                            )

                elif kind == 0x00:
                    pass  # extra ready signal, ignore

                else:
                    logger.warning(f"[{peer}] unknown stdout byte: {kind:#x}")
        except (OSError, ValueError):
            pass
        finally:
            close_event.set()
            # Wake up ws_sender
            loop.call_soon_threadsafe(ws_send_queue.put_nowait, None)

    # Start threads
    t_stdin = threading.Thread(target=stdin_writer, daemon=True)
    t_stdout = threading.Thread(target=stdout_reader, daemon=True)
    t_stdin.start()
    t_stdout.start()

    # ── async: WS recv → Opus decode → pcm_queue ─────────────────

    async def recv_loop():
        import time as _time
        # Use 24kHz to match model rate — sphn resamples internally
        opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)
        ws_msgs = 0
        pcm_total = 0
        t0 = _time.monotonic()
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    data = msg.data
                    if len(data) > 0 and data[0] == 1:  # audio
                        ws_msgs += 1
                        opus_reader.append_bytes(data[1:])
                        # Drain all decoded PCM — each Ogg page may
                        # contain multiple Opus frames
                        while True:
                            pcm = opus_reader.read_pcm()
                            if pcm.shape[-1] == 0:
                                break
                            pcm_total += pcm.shape[-1]
                            pcm_queue.put_nowait(pcm)
                        if ws_msgs % 100 == 0:
                            dt = _time.monotonic() - t0
                            rate = pcm_total / dt if dt > 0 else 0
                            logger.info(
                                f"[{peer}][recv] {ws_msgs} msgs, "
                                f"{pcm_total} pcm@24k in {dt:.1f}s "
                                f"({rate:.0f} samp/s = {rate/24000:.2f}x)"
                            )
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                ):
                    break
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            close_event.set()
            pcm_queue.put(None)
            logger.info(f"[{peer}] recv_loop done")

    # ── async: ws_send_queue → WS ────────────────────────────────

    async def ws_sender():
        try:
            while True:
                msg = await ws_send_queue.get()
                if msg is None:
                    break
                if not ws.closed:
                    await ws.send_bytes(msg)
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            close_event.set()
            logger.info(f"[{peer}] ws_sender done")

    # ── async: tool monitor (polls STT, triggers gateway) ───────

    async def tool_monitor_loop():
        """Poll STT transcripts, fire tool calls through the gateway."""
        if stt is None or tool_bridge is None:
            return
        last_ts = 0.0
        try:
            while not close_event.is_set():
                await asyncio.sleep(1.0)
                text, ts = stt.get_transcript()
                if ts <= last_ts or not text:
                    continue
                last_ts = ts

                conversation_history.append({"role": "user", "content": text})

                if not tool_bridge.detect_trigger(text):
                    continue

                logger.info(f"[{peer}][tool] trigger: {text[:80]}")
                tool_mode_active.set()
                tool_bridge.send_text_to_client("\n[Searching...]\n")

                try:
                    result = await loop.run_in_executor(
                        None,
                        tool_bridge.call_gateway,
                        text,
                        conversation_history[:-1],
                    )
                    tool_bridge.send_text_to_client(
                        f"\n[Tool result:]\n{result}\n"
                    )
                    conversation_history.append(
                        {"role": "assistant", "content": result}
                    )
                    logger.info(f"[{peer}][tool] done ({len(result)} chars)")

                    # Synthesize and stream TTS audio to client
                    if tts_speaker is not None:
                        logger.info(f"[{peer}][tool] tts starting...")
                        pcm = await loop.run_in_executor(
                            None, tts_speaker.synthesize, result
                        )
                        if pcm is not None and len(pcm) > 0:
                            n_frames = 0
                            for i in range(0, len(pcm), FRAME_SIZE):
                                chunk = pcm[i:i + FRAME_SIZE]
                                if len(chunk) < FRAME_SIZE:
                                    chunk = np.pad(chunk, (0, FRAME_SIZE - len(chunk)))
                                frame_msg = b"\x03" + chunk.astype(np.float32).tobytes()
                                await ws_send_queue.put(frame_msg)
                                n_frames += 1
                                # Pace at real-time to avoid ring buffer overflow
                                await asyncio.sleep(FRAME_SIZE / SAMPLE_RATE)
                            logger.info(f"[{peer}][tool] tts done, "
                                        f"sent {n_frames} frames "
                                        f"({len(pcm) / SAMPLE_RATE:.1f}s)")
                        else:
                            logger.warning(f"[{peer}][tool] tts returned no audio")
                except Exception as e:
                    logger.error(f"[{peer}][tool] error: {e}")
                    tool_bridge.send_text_to_client(f"\n[Error: {e}]\n")
                finally:
                    tool_mode_active.clear()
        except asyncio.CancelledError:
            pass

    # Run async loops
    tasks = [
        asyncio.create_task(recv_loop()),
        asyncio.create_task(ws_sender()),
        asyncio.create_task(tool_monitor_loop()),
    ]

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    close_event.set()
    pcm_queue.put(None)

    for t in pending:
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

    # Stop STT background thread
    if stt is not None:
        stt.stop()

    # Kill subprocess
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass
    proc.wait()

    # Wait for threads to finish
    t_stdin.join(timeout=2)
    t_stdout.join(timeout=2)

    if not ws.closed:
        await ws.close()

    logger.info(f"[{peer}] connection closed")
    return ws


async def handle_status(request):
    return web.json_response({
        "backend": "moshi.cpp",
        "pipe_bin": PIPE_BIN,
        "pipe_exists": os.path.exists(PIPE_BIN),
    })


# ── SSL ───────────────────────────────────────────────────────────

def create_ssl_context(cert_dir: str):
    cert_file = os.path.join(cert_dir, "cert.pem")
    key_file = os.path.join(cert_dir, "key.pem")

    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        try:
            from personaplex.moshi.moshi.utils.connection import create_cert_if_needed
            cert_file_p, key_file_p = create_cert_if_needed(cert_dir)
            if cert_file_p:
                cert_file = str(cert_file_p)
                key_file = str(key_file_p)
        except ImportError:
            pass

    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        logger.warning(f"SSL certs not found in {cert_dir}, falling back to HTTP")
        return None, "http"

    import ssl
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=cert_file, keyfile=key_file)
    return ctx, "https"


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WebSocket bridge for moshi.cpp PersonaPlex")
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--ssl", type=str, default=None,
                        help="Directory containing cert.pem and key.pem")
    parser.add_argument("--static", type=str, default=None,
                        help="Directory to serve static files from")
    parser.add_argument("--pipe-bin", type=str, default=None,
                        help="Path to personaplex_pipe binary")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    global PIPE_BIN
    if args.pipe_bin:
        PIPE_BIN = args.pipe_bin

    if not os.path.exists(PIPE_BIN):
        logger.error(f"personaplex_pipe not found at: {PIPE_BIN}")
        logger.error("Set MOSHI_PIPE_BIN env var or use --pipe-bin")
        sys.exit(1)

    logger.info(f"using pipe binary: {PIPE_BIN}")

    app = web.Application()
    app.router.add_get("/api/chat", handle_chat)
    app.router.add_get("/api/status", handle_status)

    if args.static:
        static_path = args.static
        if os.path.exists(static_path):
            async def handle_root(_):
                return web.FileResponse(os.path.join(static_path, "index.html"))

            logger.info(f"serving static content from {static_path}")
            app.router.add_get("/", handle_root)
            app.router.add_static("/", path=static_path, follow_symlinks=True, name="static")
        else:
            logger.warning(f"static path does not exist: {static_path}")

    ssl_context = None
    protocol = "http"
    if args.ssl:
        ssl_context, protocol = create_ssl_context(args.ssl)

    host_ip = get_lan_ip() if args.host in ("0.0.0.0", "::", "localhost") else args.host
    logger.info(f"Access the Web UI at {protocol}://{host_ip}:{args.port}")

    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


if __name__ == "__main__":
    main()
