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
import logging
import os
import queue
import signal
import socket
import struct
import subprocess
import sys
import threading
from pathlib import Path

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

    # ── stdin writer thread ───────────────────────────────────────

    def stdin_writer():
        """Accumulate PCM from queue, write frame-sized chunks to stdin."""
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
        """Read framed protocol from stdout, push to WS send queue."""
        try:
            while not close_event.is_set():
                type_byte = proc.stdout.read(1)
                if not type_byte:
                    break
                kind = type_byte[0]

                if kind == 0x03:  # audio frame
                    data = proc.stdout.read(FRAME_BYTES)
                    if len(data) == FRAME_BYTES:
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

    # Run async loops
    tasks = [
        asyncio.create_task(recv_loop()),
        asyncio.create_task(ws_sender()),
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
