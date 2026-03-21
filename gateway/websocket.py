"""
WebSocket helpers (RFC 6455), SSH terminal proxy, and voice server proxy.

All standalone functions — no class methods here.
"""

import base64
import fcntl
import hashlib
import json
import os
import pty
import select
import signal
import struct
import sys
import termios
import threading
import time
import uuid
from urllib.parse import urlparse

from .state import VOICE_PORT, http_terminal_sessions, logger

# ─────────────────────────────────────────────────────────────────
# WebSocket helpers (RFC 6455)
# ─────────────────────────────────────────────────────────────────

WS_MAGIC = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def ws_accept_key(key):
    """Compute Sec-WebSocket-Accept from client key."""
    return base64.b64encode(hashlib.sha1(key.encode() + WS_MAGIC).digest()).decode()


def ws_encode_frame(data, opcode=0x01):
    """Encode a WebSocket frame (text=0x01, binary=0x02, close=0x08, ping=0x09, pong=0x0A)."""
    frame = bytearray()
    frame.append(0x80 | opcode)  # FIN + opcode
    if isinstance(data, str):
        data = data.encode("utf-8")
    length = len(data)
    if length < 126:
        frame.append(length)
    elif length < 65536:
        frame.append(126)
        frame.extend(struct.pack(">H", length))
    else:
        frame.append(127)
        frame.extend(struct.pack(">Q", length))
    frame.extend(data)
    return bytes(frame)


def ws_decode_frame(raw):
    """Decode a WebSocket frame. Returns (opcode, payload, bytes_consumed)."""
    if len(raw) < 2:
        return None, None, 0
    b0, b1 = raw[0], raw[1]
    opcode = b0 & 0x0F
    masked = b1 & 0x80
    length = b1 & 0x7F
    offset = 2
    if length == 126:
        if len(raw) < 4:
            return None, None, 0
        length = struct.unpack(">H", raw[2:4])[0]
        offset = 4
    elif length == 127:
        if len(raw) < 10:
            return None, None, 0
        length = struct.unpack(">Q", raw[2:10])[0]
        offset = 10
    if masked:
        if len(raw) < offset + 4 + length:
            return None, None, 0
        mask = raw[offset : offset + 4]
        offset += 4
        payload = bytearray(raw[offset : offset + length])
        for i in range(length):
            payload[i] ^= mask[i % 4]
        payload = bytes(payload)
    else:
        if len(raw) < offset + length:
            return None, None, 0
        payload = raw[offset : offset + length]
    return opcode, payload, offset + length


# ─────────────────────────────────────────────────────────────────
# SSH WebSocket Terminal
# ─────────────────────────────────────────────────────────────────


def handle_terminal_websocket(handler):
    """Handle a WebSocket connection for terminal access."""
    # Complete WebSocket handshake
    key = handler.headers.get("Sec-WebSocket-Key", "")
    accept = ws_accept_key(key)
    handler.send_response(101)
    handler.send_header("Upgrade", "websocket")
    handler.send_header("Connection", "Upgrade")
    handler.send_header("Sec-WebSocket-Accept", accept)
    handler.end_headers()

    sock = handler.request

    # Open a PTY with bash
    master_fd, slave_fd = pty.openpty()
    pid = os.fork()
    if pid == 0:
        # Child process
        os.close(master_fd)
        os.setsid()
        fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        os.dup2(slave_fd, 2)
        os.close(slave_fd)
        os.execvp("/bin/bash", ["/bin/bash", "--login"])
        sys.exit(0)

    os.close(slave_fd)

    # Set master_fd non-blocking
    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    ws_buffer = bytearray()
    running = True

    last_ping = [time.time()]

    def read_from_pty():
        """Read PTY output and send as WebSocket frames. Also sends keepalive pings."""
        nonlocal running
        while running:
            try:
                r, _, _ = select.select([master_fd], [], [], 0.1)
                if master_fd in r:
                    data = os.read(master_fd, 4096)
                    if not data:
                        running = False
                        break
                    frame = ws_encode_frame(data, opcode=0x02)  # binary
                    try:
                        sock.sendall(frame)
                    except Exception:
                        running = False
                        break
                # Send keepalive ping every 25 seconds
                now = time.time()
                if now - last_ping[0] > 25:
                    try:
                        sock.sendall(ws_encode_frame(b"keepalive", opcode=0x09))
                        last_ping[0] = now
                    except Exception:
                        running = False
                        break
            except OSError:
                running = False
                break

    pty_thread = threading.Thread(target=read_from_pty, daemon=True)
    pty_thread.start()

    try:
        while running:
            r, _, _ = select.select([sock], [], [], 0.1)
            if sock not in r:
                continue
            try:
                chunk = sock.recv(8192)
            except Exception:
                break
            if not chunk:
                break
            ws_buffer.extend(chunk)

            while True:
                opcode, payload, consumed = ws_decode_frame(ws_buffer)
                if opcode is None:
                    break
                ws_buffer = ws_buffer[consumed:]

                if opcode == 0x01 or opcode == 0x02:
                    # Text or binary — write to PTY
                    # Check for resize message
                    if opcode == 0x01:
                        try:
                            msg = json.loads(payload)
                            if isinstance(msg, dict) and msg.get("type") == "resize":
                                cols = msg.get("cols", 80)
                                rows = msg.get("rows", 24)
                                winsize = struct.pack("HHHH", rows, cols, 0, 0)
                                fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
                                continue
                            # Regular text input (including valid JSON that isn't a resize command)
                            os.write(master_fd, payload)
                        except (json.JSONDecodeError, ValueError):
                            os.write(master_fd, payload)
                    else:
                        os.write(master_fd, payload)
                elif opcode == 0x08:
                    # Close
                    running = False
                    break
                elif opcode == 0x09:
                    # Ping -> Pong
                    sock.sendall(ws_encode_frame(payload, opcode=0x0A))
    except Exception:
        pass
    finally:
        running = False
        try:
            os.close(master_fd)
        except Exception:
            pass
        try:
            os.kill(pid, signal.SIGTERM)
            os.waitpid(pid, 0)
        except Exception:
            pass
        pty_thread.join(timeout=2)


# ─────────────────────────────────────────────────────────────────
# HTTP Terminal Transport (iOS fallback — SSE output + POST input)
# ─────────────────────────────────────────────────────────────────


def create_terminal_session():
    """Create a PTY session for HTTP-based terminal access. Returns session_id."""
    session_id = uuid.uuid4().hex[:12]
    master_fd, slave_fd = pty.openpty()
    pid = os.fork()
    if pid == 0:
        os.close(master_fd)
        os.setsid()
        fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        os.dup2(slave_fd, 2)
        os.close(slave_fd)
        os.execvp("/bin/bash", ["/bin/bash", "--login"])
        sys.exit(0)
    os.close(slave_fd)
    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    http_terminal_sessions[session_id] = {
        "master_fd": master_fd,
        "pid": pid,
        "created": time.time(),
    }
    logger.info("HTTP terminal session created: %s (pid=%d)", session_id, pid)
    return session_id


def cleanup_terminal_session(session_id):
    """Clean up a terminal session."""
    session = http_terminal_sessions.pop(session_id, None)
    if session:
        try:
            os.close(session["master_fd"])
        except Exception:
            pass
        try:
            os.kill(session["pid"], signal.SIGTERM)
            os.waitpid(session["pid"], 0)
        except Exception:
            pass
        logger.info("HTTP terminal session cleaned up: %s", session_id)


def handle_terminal_stream(handler, session_id):
    """Stream PTY output as SSE events with base64-encoded data."""
    session = http_terminal_sessions.get(session_id)
    if not session:
        handler.send_response(404)
        handler.end_headers()
        handler.wfile.write(b'{"error":"session not found"}')
        return

    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()

    master_fd = session["master_fd"]
    try:
        while session_id in http_terminal_sessions:
            r, _, _ = select.select([master_fd], [], [], 0.1)
            if master_fd in r:
                try:
                    data = os.read(master_fd, 4096)
                except OSError:
                    break
                if not data:
                    break
                b64 = base64.b64encode(data).decode()
                handler.wfile.write(f"data: {b64}\n\n".encode())
                handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
    finally:
        cleanup_terminal_session(session_id)


# ─────────────────────────────────────────────────────────────────
# Voice Server WebSocket Proxy
# ─────────────────────────────────────────────────────────────────


def handle_voice_proxy(handler):
    """Proxy WebSocket from client to voice server on port 8998."""
    import socket as _socket

    # Complete WebSocket handshake with the client
    client_key = handler.headers.get("Sec-WebSocket-Key", "")
    accept = ws_accept_key(client_key)
    handler.send_response(101)
    handler.send_header("Upgrade", "websocket")
    handler.send_header("Connection", "Upgrade")
    handler.send_header("Sec-WebSocket-Accept", accept)
    handler.end_headers()

    client_sock = handler.request

    # Build the upstream WebSocket URL with query params
    parsed = urlparse(handler.path)
    upstream_path = parsed.path
    if parsed.query:
        upstream_path += "?" + parsed.query

    # Connect to voice server upstream (plain TCP on localhost)
    upstream_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    upstream_sock.settimeout(10)
    try:
        upstream_sock.connect(("localhost", VOICE_PORT))
    except Exception as e:
        try:
            client_sock.sendall(ws_encode_frame(f"voice server connection failed: {e}", opcode=0x08))
        except Exception:
            pass
        return

    # Send WebSocket upgrade to upstream
    import secrets

    ws_key = base64.b64encode(secrets.token_bytes(16)).decode()
    upgrade_request = (
        f"GET {upstream_path} HTTP/1.1\r\n"
        f"Host: localhost:{VOICE_PORT}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {ws_key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"\r\n"
    )
    upstream_sock.sendall(upgrade_request.encode())

    # Read upstream handshake response
    response_data = b""
    while b"\r\n\r\n" not in response_data:
        chunk = upstream_sock.recv(4096)
        if not chunk:
            upstream_sock.close()
            return
        response_data += chunk

    if b"101" not in response_data.split(b"\r\n")[0]:
        upstream_sock.close()
        try:
            client_sock.sendall(ws_encode_frame("voice server handshake failed", opcode=0x08))
        except Exception:
            pass
        return

    # Any extra data after the HTTP headers is WebSocket data
    extra = response_data.split(b"\r\n\r\n", 1)[1] if b"\r\n\r\n" in response_data else b""

    upstream_sock.setblocking(False)
    running = True

    def upstream_to_client():
        """Forward raw bytes from upstream to client."""
        nonlocal running
        buf = bytearray(extra) if extra else bytearray()  # noqa: F841
        while running:
            try:
                r, _, _ = select.select([upstream_sock], [], [], 0.1)
                if upstream_sock in r:
                    data = upstream_sock.recv(65536)
                    if not data:
                        running = False
                        break
                    try:
                        client_sock.sendall(data)
                    except Exception:
                        running = False
                        break
            except Exception:
                running = False
                break

    def client_to_upstream():
        """Forward raw bytes from client to upstream."""
        nonlocal running
        while running:
            try:
                r, _, _ = select.select([client_sock], [], [], 0.1)
                if client_sock not in r:
                    continue
                data = client_sock.recv(65536)
                if not data:
                    running = False
                    break
                upstream_sock.sendall(data)
            except Exception:
                running = False
                break

    # Send any extra data from handshake to client
    if extra:
        try:
            client_sock.sendall(extra)
        except Exception:
            running = False

    t1 = threading.Thread(target=upstream_to_client, daemon=True)
    t2 = threading.Thread(target=client_to_upstream, daemon=True)
    t1.start()
    t2.start()

    t1.join()
    t2.join()

    try:
        upstream_sock.close()
    except Exception:
        pass
