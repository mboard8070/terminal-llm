"""
Cloud model tool loops for the MAUDE Gateway.

Contains CloudMixin — methods for handling cloud/local model requests with
server-side tool execution loops. Mixed into GatewayHandler.
"""

import http.client
import json
import os
import re
import selectors
import shutil
import ssl
import string
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlparse

from .state import (
    PARALLEL_SAFE,
    TOOL_ADDENDUM,
    execute_tool,
    get_tools_for_message,
    logger,
    reset_rate_limits,
)


class CloudMixin:
    """Mixin providing cloud model tool loop methods for GatewayHandler."""

    CODEX_MAUDE_TOOL_BRIDGE = """MAUDE TOOL BRIDGE FOR CODEX:
You are running inside MAUDE on the DGX Spark. Do not use Codex's own imagegen skill for image generation.
Call MAUDE tools through the local gateway when the task needs Maude capabilities:

curl -s -X POST http://localhost:30080/api/tools/execute \
  -H 'Content-Type: application/json' \
  -d '{"name":"generate_image","arguments":{"prompt":"PROMPT","width":1024,"height":1024,"steps":28,"seed":-1}}'

IMAGE GENERATION:
When the user asks you to generate, draw, create, render, or make an image, call MAUDE's local Flux 1 / ComfyUI image tool through the local gateway using name "generate_image".
Use this local Flux 1 tool by default. If the user specifically asks for Flux 2, use name "generate_image_flux2" with arguments {"prompt":"PROMPT","model":"pro","aspect_ratio":"1:1","seed":-1}.
Do not start ComfyUI with "cd ~/nvidia-workbench/ComfyUI && ./start.sh" from inside MAUDE. ComfyUI must run as a separate user service: systemctl --user start maude-comfyui. The generate_image tool will start that service automatically if needed.
After the tool returns, include its markdown display link, usually like ![description](/download/file.png), so the mobile app can show the image.
Do not claim the image was generated until the MAUDE tool response says it succeeded.

HYPERFRAMES VIDEO:
Maude has a HyperFrames skill and native HyperFrames tools. Use these for HTML/CSS/JS programmatic video, motion graphics, or requests that mention HyperFrames. Do not use Mochi; it has been removed.

Check readiness:
curl -s -X POST http://localhost:30080/api/tools/execute \
  -H 'Content-Type: application/json' \
  -d '{"name":"skill_hyperframes","arguments":{"action":"doctor"}}'

Install/verify HyperFrames managed Chrome if doctor says Chrome is missing:
curl -s -X POST http://localhost:30080/api/tools/execute \
  -H 'Content-Type: application/json' \
  -d '{"name":"skill_hyperframes","arguments":{"action":"browser_ensure"}}'

Create a HyperFrames project:
curl -s -X POST http://localhost:30080/api/tools/execute \
  -H 'Content-Type: application/json' \
  -d '{"name":"skill_hyperframes","arguments":{"action":"init","name":"PROJECT_NAME","example":"blank"}}'

Render a HyperFrames project:
curl -s -X POST http://localhost:30080/api/tools/execute \
  -H 'Content-Type: application/json' \
  -d '{"name":"skill_hyperframes","arguments":{"action":"render","project_path":"/path/to/project","format":"mp4","quality":"standard","share":true}}'

HyperFrames is CLI-based; there is no long-running HyperFrames service to start. The rendered video is shared through /download/<file> when share is true.

YOUTUBE PUBLISHING:
Do not infer YouTube upload capability from shell environment variables. MAUDE uploads videos through its local gateway using Google OAuth credentials stored under ~/.config/maude, not a YouTube API key in the environment.
If the user asks to post, publish, or upload a video to YouTube, call the MAUDE youtube_upload tool through the local gateway. It defaults to public.

Example:
curl -s -X POST http://localhost:30080/api/tools/execute \
  -H 'Content-Type: application/json' \
  -d '{"name":"youtube_upload","arguments":{"file_path":"/path/to/video.mp4","title":"TITLE","description":"DESCRIPTION","tags":"AI, Shorts","privacy":"public","category":"28"}}'

If the tool returns an OAuth or credential error, report that exact tool result. Do not preemptively refuse because API keys are not visible in the shell environment."""

    @staticmethod
    def _codex_available_tools_addendum(messages: list[dict]) -> str:
        """Build a compact list of Maude tools Codex can call via the bridge."""
        user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        block.get("text", "") if isinstance(block, dict) else str(block) for block in content
                    )
                user_msg = str(content)
                break

        try:
            active_tools = (
                get_tools_for_message(user_msg, messages=messages) if get_tools_for_message else []
            )
        except Exception:
            active_tools = []

        if not active_tools:
            return ""

        lines = [
            "AVAILABLE MAUDE TOOLS FOR THIS REQUEST:",
            "Use any listed tool with:",
            "curl -s -X POST http://localhost:30080/api/tools/execute "
            "-H 'Content-Type: application/json' "
            '-d \'{"name":"TOOL_NAME","arguments":{...}}\'',
        ]
        for tool in active_tools:
            fn = tool.get("function", {})
            name = fn.get("name", "")
            description = " ".join((fn.get("description") or "").split())
            if not name:
                continue
            if len(description) > 180:
                description = description[:177] + "..."
            lines.append(f"- {name}: {description}")
        return "\n".join(lines)

    @staticmethod
    def _messages_to_codex_prompt(messages: list[dict]) -> str:
        """Flatten chat messages into a single prompt for `codex exec`."""
        parts = [CloudMixin.CODEX_MAUDE_TOOL_BRIDGE]
        tool_addendum = CloudMixin._codex_available_tools_addendum(messages)
        if tool_addendum:
            parts.append(tool_addendum)
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    block.get("text", "") if isinstance(block, dict) else str(block) for block in content
                )
            if content:
                parts.append(f"{role.upper()}:\n{content}")
        return "\n\n".join(parts).strip()

    def _codex_cli_response(self, req, resolved_name):
        """Handle a request by invoking the locally authenticated Codex CLI."""
        prompt = self._messages_to_codex_prompt(req.get("messages", []))
        if not prompt:
            self._json_response({"error": "No prompt provided for Codex CLI"}, 400)
            return

        stream = bool(req.get("stream"))
        model = os.environ.get("MAUDE_CODEX_MODEL", "")
        workdir = os.environ.get("MAUDE_CODEX_WORKDIR", "/home/mboard76")
        timeout = int(os.environ.get("MAUDE_CODEX_TIMEOUT", "3600"))

        with tempfile.NamedTemporaryFile(prefix="maude-codex-", suffix=".txt", delete=False) as tmp:
            output_path = tmp.name

        cmd = [
            os.environ.get("MAUDE_CODEX_BIN", shutil.which("codex") or "/home/mboard76/.npm-global/bin/codex"),
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--sandbox",
            os.environ.get("MAUDE_CODEX_SANDBOX", "danger-full-access"),
            "-C",
            workdir,
            "--output-last-message",
            output_path,
        ]
        if model:
            cmd.extend(["--model", model])
        cmd.append("-")

        started = time.time()
        try:
            if stream:
                self._start_sse_headers()
                self._send_trace_sse(
                    "tool_call",
                    {
                        "name": "codex_exec",
                        "args": json.dumps({"model": model or "default", "workdir": workdir}),
                    },
                )

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=workdir,
            )
            stdout, stderr = self._run_codex_json_process(proc, prompt, stream, started, timeout)
            try:
                with open(output_path, encoding="utf-8") as f:
                    content = f.read().strip()
            except OSError:
                content = ""

            if proc.returncode != 0:
                err = (stderr or stdout or "Codex CLI failed").strip()
                if stream:
                    self._send_trace_sse(
                        "tool_result",
                        {
                            "name": "codex_exec",
                            "preview": f"Error: {err[:160]}",
                            "elapsed": round(time.time() - started, 1),
                        },
                    )
                    self._close_sse_with_error(err)
                else:
                    self._json_response({"error": err}, 502)
                return

            if not content:
                content = self._last_codex_agent_message(stdout) or stdout.strip()

            elapsed = time.time() - started
            logger.info("Codex CLI: %.1fs, %d chars", elapsed, len(content))

            if stream:
                self._send_trace_sse(
                    "tool_result",
                    {
                        "name": "codex_exec",
                        "preview": f"Done in {elapsed:.1f}s",
                        "elapsed": round(elapsed, 1),
                    },
                )
                chunk = {
                    "id": f"chatcmpl-maude-codex-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": resolved_name,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                }
                line = f"data: {json.dumps(chunk)}\n\n".encode()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(line), line))
                finish = {
                    "id": chunk["id"],
                    "object": "chat.completion.chunk",
                    "created": chunk["created"],
                    "model": resolved_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                finish_line = f"data: {json.dumps(finish)}\n\n".encode()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(finish_line), finish_line))
                done_line = b"data: [DONE]\n\n"
                self.wfile.write(b"%x\r\n%s\r\n" % (len(done_line), done_line))
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                return

            self._json_response(
                {
                    "id": f"chatcmpl-maude-codex-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": resolved_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
        except subprocess.TimeoutExpired:
            msg = f"Codex CLI timed out after {timeout}s"
            try:
                proc.kill()
            except Exception:
                pass
            if stream:
                self._send_trace_sse(
                    "tool_result",
                    {
                        "name": "codex_exec",
                        "preview": f"Error: {msg}",
                        "elapsed": round(time.time() - started, 1),
                    },
                )
                self._close_sse_with_error(msg)
            else:
                self._json_response({"error": msg}, 504)
        except Exception as e:
            if stream:
                self._send_trace_sse(
                    "tool_result",
                    {
                        "name": "codex_exec",
                        "preview": f"Error: {str(e)[:160]}",
                        "elapsed": round(time.time() - started, 1),
                    },
                )
                self._close_sse_with_error(str(e))
            else:
                self._json_response({"error": f"Codex CLI error: {e}"}, 502)
        finally:
            try:
                os.unlink(output_path)
            except OSError:
                pass

    @staticmethod
    def _message_text(msg: dict) -> str:
        """Normalize a chat message content field to a plain string."""
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block) for block in content
            )
        return str(content or "").strip()

    @staticmethod
    def _summarize_dropped_messages(dropped: list[dict], max_entries: int = 12) -> str:
        """Build a short stand-in for conversation turns omitted before `grok -p`."""
        if not dropped:
            return ""
        lines = [f"[Earlier conversation summarized — {len(dropped)} messages omitted]"]
        for msg in dropped[:max_entries]:
            role = str(msg.get("role", "user")).upper()
            text = CloudMixin._message_text(msg).replace("\n", " ").strip()
            if not text:
                continue
            if len(text) > 140:
                text = text[:137] + "..."
            lines.append(f"- {role}: {text}")
        if len(dropped) > max_entries:
            lines.append(f"- ... and {len(dropped) - max_entries} more")
        return "\n".join(lines)

    @staticmethod
    def _prepare_grok_history(messages: list[dict]) -> tuple[list[dict], dict]:
        """Truncate / summarize chat history so `grok -p` context does not balloon.

        Controls (env):
          MAUDE_GROK_MAX_PROMPT_CHARS  overall prompt budget (default 48000)
          MAUDE_GROK_MAX_MSG_CHARS     per-message body cap (default 4000)
          MAUDE_GROK_KEEP_RECENT       recent messages kept in full (default 12)

        Returns (prepared_messages, meta) where meta includes removed/original counts.
        """
        max_prompt = int(os.environ.get("MAUDE_GROK_MAX_PROMPT_CHARS", "48000"))
        max_msg = int(os.environ.get("MAUDE_GROK_MAX_MSG_CHARS", "4000"))
        keep_recent = int(os.environ.get("MAUDE_GROK_KEEP_RECENT", "12"))
        max_prompt = max(2000, max_prompt)
        max_msg = max(200, max_msg)
        keep_recent = max(2, keep_recent)

        original = list(messages or [])
        normalized: list[dict] = []
        for msg in original:
            role = msg.get("role", "user")
            text = CloudMixin._message_text(msg)
            if not text and role != "system":
                continue
            if len(text) > max_msg:
                text = text[: max_msg - 20] + "\n... [truncated]"
            normalized.append({"role": role, "content": text})

        system_msgs = [m for m in normalized if m.get("role") == "system"]
        non_system = [m for m in normalized if m.get("role") != "system"]

        removed = 0
        dropped: list[dict] = []
        if len(non_system) > keep_recent:
            dropped = non_system[: -keep_recent]
            non_system = non_system[-keep_recent:]
            removed = len(dropped)

        prepared = list(system_msgs)
        if dropped:
            prepared.append(
                {
                    "role": "system",
                    "content": CloudMixin._summarize_dropped_messages(dropped),
                }
            )
        prepared.extend(non_system)

        def _prompt_len(msgs: list[dict]) -> int:
            return sum(len(CloudMixin._message_text(m)) + len(str(m.get("role", ""))) + 4 for m in msgs)

        # Shrink further from the oldest non-summary non-system message if over budget.
        while _prompt_len(prepared) > max_prompt and len(prepared) > 2:
            # Prefer dropping oldest non-system that is not the summary block.
            drop_idx = None
            for i, msg in enumerate(prepared):
                if msg.get("role") == "system" and i < len(system_msgs):
                    continue
                content = CloudMixin._message_text(msg)
                if content.startswith("[Earlier conversation summarized"):
                    continue
                # Don't drop the latest user/assistant turn if we can help it.
                if i >= len(prepared) - 1:
                    continue
                drop_idx = i
                break
            if drop_idx is None:
                # Last resort: truncate the longest remaining body.
                longest_i = max(
                    range(len(prepared)),
                    key=lambda i: len(CloudMixin._message_text(prepared[i])),
                )
                body = CloudMixin._message_text(prepared[longest_i])
                if len(body) <= 200:
                    break
                prepared[longest_i] = {
                    **prepared[longest_i],
                    "content": body[: max(200, len(body) // 2)] + "\n... [truncated]",
                }
                continue
            removed += 1
            prepared.pop(drop_idx)

        meta = {
            "removed": removed,
            "original": len(original),
            "kept": len(prepared),
            "max_prompt_chars": max_prompt,
            "prompt_chars": _prompt_len(prepared),
        }
        return prepared, meta

    @staticmethod
    def _messages_to_grok_prompt(messages: list[dict]) -> str:
        """Flatten (optionally pre-trimmed) chat messages into a single `grok -p` prompt."""
        parts = []
        for msg in messages:
            role = str(msg.get("role", "user")).upper()
            content = CloudMixin._message_text(msg)
            if content:
                parts.append(f"{role}:\n{content}")
        return "\n\n".join(parts).strip()

    def _grok_cli_response(self, req, resolved_name):
        """Handle a request by invoking the locally authenticated Grok CLI.

        Uses the grok.com / X Premium OAuth session rather than XAI_API_KEY, so
        no xAI API billing is involved. `grok -p` is single-turn: it prints the
        response to stdout and exits.

        When streaming, optionally uses `--output-format streaming-json` so tool
        start/end events surface as SSE traces in the TUI (same idea as Codex).
        """
        prepared, trim_meta = self._prepare_grok_history(req.get("messages", []))
        prompt = self._messages_to_grok_prompt(prepared)
        if not prompt:
            self._json_response({"error": "No prompt provided for Grok CLI"}, 400)
            return

        stream = bool(req.get("stream"))
        # Progress streaming default-on when the client asked for SSE.
        progress_env = os.environ.get("MAUDE_GROK_STREAM_PROGRESS", "").strip().lower()
        if progress_env in ("0", "false", "no", "off"):
            progress = False
        elif progress_env in ("1", "true", "yes", "on"):
            progress = True
        else:
            progress = stream

        workdir = os.environ.get("MAUDE_GROK_WORKDIR", "")
        if not workdir or not os.path.isdir(workdir):
            linux_home = "/home/mboard76"
            workdir = linux_home if os.path.isdir(linux_home) else os.path.expanduser("~")
        timeout = int(os.environ.get("MAUDE_GROK_TIMEOUT", "600"))
        grok_bin = os.environ.get("MAUDE_GROK_BIN", "")
        if not grok_bin or not os.path.isfile(grok_bin):
            grok_bin = (
                shutil.which("grok")
                or shutil.which("grok.exe")
                or os.path.join(os.path.expanduser("~"), ".grok", "bin", "grok.exe")
                or "/home/mboard76/.local/bin/grok"
            )
        if not os.path.isfile(grok_bin):
            if stream:
                self._start_sse_headers()
                self._send_trace_sse(
                    "error",
                    {"message": f"Grok CLI not found (looked for grok.exe). Set MAUDE_GROK_BIN."},
                )
                self._send_sse_done(resolved_name, f"chatcmpl-{int(time.time())}", int(time.time()))
                return
            self._json_response({"error": "Grok CLI not found. Set MAUDE_GROK_BIN to grok.exe"}, 500)
            return
        model = os.environ.get("MAUDE_GROK_MODEL", "") or (
            resolved_name if str(resolved_name).startswith("grok-") else "grok-4.6"
        )

        # Prefer --prompt-file for large prompts (avoids ARG_MAX issues).
        prompt_path = None
        cmd = [grok_bin, "--always-approve"]
        if model:
            cmd.extend(["--model", model])
        if progress:
            cmd.extend(["--output-format", "streaming-json"])
        if len(prompt) > 4000:
            with tempfile.NamedTemporaryFile(
                prefix="maude-grok-", suffix=".txt", delete=False, mode="w", encoding="utf-8"
            ) as tmp:
                tmp.write(prompt)
                prompt_path = tmp.name
            cmd.extend(["--prompt-file", prompt_path])
        else:
            cmd.extend(["-p", prompt])

        started = time.time()
        proc = None
        try:
            if stream:
                self._start_sse_headers()
                if trim_meta.get("removed"):
                    self._send_trace_sse(
                        "context_trim",
                        {
                            "removed": trim_meta["removed"],
                            "max_tokens": trim_meta.get("max_prompt_chars", 0) // 4,
                            "kept": trim_meta.get("kept"),
                            "prompt_chars": trim_meta.get("prompt_chars"),
                        },
                    )
                self._send_trace_sse(
                    "tool_call",
                    self._tool_call_trace_payload(
                        "grok_cli",
                        {
                            "model": model or resolved_name,
                            "progress": progress,
                            "prompt_chars": trim_meta.get("prompt_chars", len(prompt)),
                        },
                        json.dumps({"model": model or resolved_name}),
                    ),
                )

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=workdir,
            )
            stdout, stderr = self._run_grok_process(proc, stream, started, timeout, progress=progress)

            if proc.returncode != 0:
                err = (stderr or stdout or "Grok CLI failed").strip()
                # Prefer a compact stderr; streaming-json stdout is noisy on failure.
                if progress and stderr:
                    err = stderr.strip()
                elif progress and stdout:
                    err = self._grok_error_from_stdout(stdout) or err
                err = err[:2000]
                if stream:
                    self._send_trace_sse(
                        "tool_result",
                        {
                            "name": "grok_cli",
                            "preview": f"Error: {err[:160]}",
                            "elapsed": round(time.time() - started, 1),
                        },
                    )
                    self._close_sse_with_error(err)
                else:
                    self._json_response({"error": err}, 502)
                return

            if progress:
                content = self._last_grok_text(stdout)
            else:
                content = (stdout or "").strip()
            elapsed = time.time() - started
            logger.info(
                "Grok CLI: %.1fs, %d chars (progress=%s, trimmed=%s)",
                elapsed,
                len(content),
                progress,
                trim_meta.get("removed", 0),
            )

            if stream:
                self._send_trace_sse(
                    "tool_result",
                    {
                        "name": "grok_cli",
                        "preview": f"Done in {elapsed:.1f}s",
                        "elapsed": round(elapsed, 1),
                    },
                )
                chunk = {
                    "id": f"chatcmpl-maude-grok-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": resolved_name,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                }
                line = f"data: {json.dumps(chunk)}\n\n".encode()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(line), line))
                finish = {
                    "id": chunk["id"],
                    "object": "chat.completion.chunk",
                    "created": chunk["created"],
                    "model": resolved_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                finish_line = f"data: {json.dumps(finish)}\n\n".encode()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(finish_line), finish_line))
                done_line = b"data: [DONE]\n\n"
                self.wfile.write(b"%x\r\n%s\r\n" % (len(done_line), done_line))
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                return

            self._json_response(
                {
                    "id": f"chatcmpl-maude-grok-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": resolved_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
        except subprocess.TimeoutExpired:
            msg = f"Grok CLI timed out after {timeout}s"
            try:
                if proc:
                    proc.kill()
            except Exception:
                pass
            if stream:
                self._send_trace_sse(
                    "tool_result",
                    {
                        "name": "grok_cli",
                        "preview": f"Error: {msg}",
                        "elapsed": round(time.time() - started, 1),
                    },
                )
                self._close_sse_with_error(msg)
            else:
                self._json_response({"error": msg}, 504)
        except Exception as e:
            if stream:
                self._send_trace_sse(
                    "tool_result",
                    {
                        "name": "grok_cli",
                        "preview": f"Error: {str(e)[:160]}",
                        "elapsed": round(time.time() - started, 1),
                    },
                )
                self._close_sse_with_error(str(e))
            else:
                self._json_response({"error": f"Grok CLI error: {e}"}, 502)
        finally:
            if prompt_path:
                try:
                    os.unlink(prompt_path)
                except OSError:
                    pass

    def _run_grok_process(self, proc, stream: bool, started: float, timeout: int, progress: bool = False):
        """Wait for Grok CLI completion.

        When progress=True, parse streaming-json stdout and emit tool_call /
        tool_result SSE traces so the TUI shows agent activity. Otherwise drain
        stdout/stderr on background threads and send keepalives while streaming.
        """
        if progress:
            return self._run_grok_json_process(proc, stream, started, timeout)

        stdout_buf: list[str] = []
        stderr_buf: list[str] = []
        deadline = started + timeout
        last_keepalive = started

        def _drain(pipe, buf: list[str]):
            if not pipe:
                return
            try:
                data = pipe.read()
                if data:
                    buf.append(data)
            except Exception:
                pass

        stdout_thread = threading.Thread(target=_drain, args=(proc.stdout, stdout_buf), daemon=True)
        stderr_thread = threading.Thread(target=_drain, args=(proc.stderr, stderr_buf), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        while True:
            now = time.time()
            if now > deadline:
                try:
                    proc.kill()
                except Exception:
                    pass
                stdout_thread.join(timeout=0.5)
                stderr_thread.join(timeout=0.5)
                raise subprocess.TimeoutExpired(proc.args, timeout)

            if stream and now - last_keepalive >= 15:
                self._send_trace_sse(
                    "keepalive",
                    {"name": "grok_cli", "elapsed": round(now - started, 1)},
                )
                last_keepalive = now

            if proc.poll() is not None:
                break
            time.sleep(0.5)

        remaining = max(0.1, deadline - time.time())
        stdout_thread.join(timeout=remaining)
        stderr_thread.join(timeout=0.5)
        return "".join(stdout_buf), "".join(stderr_buf)

    def _run_grok_json_process(self, proc, stream: bool, started: float, timeout: int):
        """Line-read Grok streaming-json stdout and translate into Maude traces."""
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        active_tools: dict = {}
        deadline = started + timeout
        last_keepalive = started

        def _read_stderr():
            if not proc.stderr:
                return
            try:
                for line in proc.stderr:
                    stderr_lines.append(line)
            except Exception:
                pass

        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        if proc.stdout:
            # Do not use selectors on Windows — pipes are not sockets (WinError 10038).
            stdout_q: list[str] = []
            stdout_done = threading.Event()

            def _read_stdout():
                try:
                    for line in proc.stdout:
                        stdout_q.append(line)
                except Exception:
                    pass
                finally:
                    stdout_done.set()

            stdout_thread = threading.Thread(target=_read_stdout, daemon=True)
            stdout_thread.start()
            emitted = 0
            try:
                while True:
                    now = time.time()
                    if now > deadline:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        raise subprocess.TimeoutExpired(proc.args, timeout)
                    if stream and now - last_keepalive >= 15:
                        self._send_trace_sse(
                            "keepalive",
                            {"name": "grok_cli", "elapsed": round(now - started, 1)},
                        )
                        last_keepalive = now

                    while emitted < len(stdout_q):
                        line = stdout_q[emitted]
                        emitted += 1
                        stdout_lines.append(line)
                        if stream:
                            self._emit_grok_json_trace(line, active_tools, started)

                    if stdout_done.is_set() and emitted >= len(stdout_q):
                        break
                    time.sleep(0.05)
            finally:
                stdout_thread.join(timeout=0.5)

        remaining = max(0.1, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            raise
        stderr_thread.join(timeout=0.5)
        return "".join(stdout_lines), "".join(stderr_lines)

    def _emit_grok_json_trace(self, line: str, active_tools: dict, started: float):
        """Map one Grok streaming-json event to a Maude SSE tool/llm trace."""
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict):
            return

        event_type = event.get("type", "")

        if event_type == "tool_call":
            tool_id = event.get("toolCallId") or f"grok-tool-{len(active_tools)}"
            name = event.get("toolName") or event.get("title") or "tool"
            # Namespace so nested Grok tools don't collide with gateway tool names.
            display = name if str(name).startswith("grok_") else f"grok_{name}"
            active_tools[tool_id] = {"name": display, "started": time.time()}
            args = event.get("rawInput") if isinstance(event.get("rawInput"), dict) else {}
            if not args and event.get("title"):
                args = {"title": event.get("title")}
            self._send_trace_sse(
                "tool_call",
                self._tool_call_trace_payload(display, args, json.dumps(args) if args else "{}"),
            )
            return

        if event_type == "tool_call_update":
            status = event.get("status")
            if status not in ("completed", "failed", "error", "cancelled"):
                return
            tool_id = event.get("toolCallId")
            info = active_tools.pop(tool_id, None) if tool_id else None
            name = (info or {}).get("name") or "grok_tool"
            tool_started = (info or {}).get("started") or started
            preview = self._grok_tool_preview(event)
            if status != "completed" and not preview.startswith("Error"):
                preview = f"Error: {preview}" if preview else f"Error: {status}"
            self._send_trace_sse(
                "tool_result",
                {
                    "name": name,
                    "preview": preview,
                    "elapsed": round(time.time() - tool_started, 1),
                },
            )
            return

        if event_type == "usage":
            usage = event.get("usage") or {}
            self._send_trace_sse(
                "llm_call",
                {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
                    "elapsed": round(time.time() - started, 1),
                },
            )
            return

        if event_type in ("error",):
            msg = event.get("message") or event.get("error") or json.dumps(event)[:160]
            self._send_trace_sse("error", {"message": str(msg)[:200]})

    @staticmethod
    def _grok_tool_preview(event: dict) -> str:
        """Compact preview string for a completed/failed Grok tool_call_update."""
        raw = event.get("rawOutput")
        if isinstance(raw, dict):
            ofp = raw.get("output_for_prompt")
            if ofp:
                return str(ofp).replace("\n", " ").strip()[:160]
            if raw.get("exit_code") is not None:
                out = raw.get("output_for_prompt") or ""
                base = f"exit: {raw.get('exit_code')}"
                if out:
                    return f"{base} {str(out).replace(chr(10), ' ').strip()}"[:160]
                return base
            if raw.get("error"):
                return f"Error: {str(raw.get('error'))[:140]}"
            if raw.get("type"):
                return str(raw.get("type"))[:160]
        content = event.get("content")
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict):
                    inner = block.get("content")
                    if isinstance(inner, dict) and inner.get("text"):
                        texts.append(str(inner["text"]))
                    elif block.get("text"):
                        texts.append(str(block["text"]))
            if texts:
                return " ".join(texts).replace("\n", " ").strip()[:160]
        if event.get("status"):
            return str(event.get("status"))
        return "done"

    @staticmethod
    def _last_grok_text(stdout: str) -> str:
        """Reconstruct the final assistant text from streaming-json text events."""
        parts: list[str] = []
        for line in (stdout or "").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "text":
                parts.append(str(event.get("data") or ""))
        text = "".join(parts).strip()
        if text:
            return text
        # Fallback: some formats put the final answer on a result event.
        for line in reversed((stdout or "").splitlines()):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("type") == "result" and event.get("result"):
                return str(event.get("result")).strip()
        return ""

    @staticmethod
    def _grok_error_from_stdout(stdout: str) -> str:
        """Pull a useful error message out of streaming-json stdout on failure."""
        for line in reversed((stdout or "").splitlines()):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") in ("error", "end") and event.get("stopReason") not in (None, "end_turn"):
                return str(event.get("message") or event.get("stopReason") or event)[:500]
            if event.get("type") == "error":
                return str(event.get("message") or event.get("error") or event)[:500]
        return ""

    def _run_codex_json_process(self, proc, prompt: str, stream: bool, started: float, timeout: int):
        """Feed Codex and translate its JSONL stdout into Maude trace events."""
        if proc.stdin:
            proc.stdin.write(prompt)
            proc.stdin.close()

        stdout_lines = []
        stderr_lines = []
        active_items = {}
        deadline = started + timeout
        last_keepalive = started

        def _read_stderr():
            if not proc.stderr:
                return
            for line in proc.stderr:
                stderr_lines.append(line)

        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        if proc.stdout:
            selector = selectors.DefaultSelector()
            selector.register(proc.stdout, selectors.EVENT_READ)
            try:
                while True:
                    now = time.time()
                    if now > deadline:
                        raise subprocess.TimeoutExpired(proc.args, timeout)
                    if stream and now - last_keepalive >= 15:
                        self._send_trace_sse(
                            "keepalive",
                            {"name": "codex_exec", "elapsed": round(now - started, 1)},
                        )
                        last_keepalive = now

                    events = selector.select(timeout=0.5)
                    if not events:
                        if proc.poll() is not None:
                            break
                        continue

                    line = proc.stdout.readline()
                    if line:
                        stdout_lines.append(line)
                        if stream:
                            self._emit_codex_json_trace(line, active_items, started)
                        continue
                    if proc.poll() is not None:
                        break
            finally:
                selector.unregister(proc.stdout)

        remaining = deadline - time.time()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(proc.args, timeout)
        proc.wait(timeout=remaining)
        stderr_thread.join(timeout=0.2)
        return "".join(stdout_lines), "".join(stderr_lines)

    def _emit_codex_json_trace(self, line: str, active_items: dict, started: float):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return

        event_type = event.get("type", "")
        item = event.get("item") or {}

        if event_type == "turn.started":
            return

        if event_type == "thread.started":
            self._send_trace_sse("keepalive", {"name": "codex_exec", "elapsed": round(time.time() - started, 1)})
            return

        if event_type == "turn.completed":
            usage = event.get("usage") or {}
            self._send_trace_sse(
                "llm_call",
                {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "cache_read_tokens": usage.get("cached_input_tokens", 0),
                    "elapsed": round(time.time() - started, 1),
                },
            )
            return

        if event_type == "item.started":
            item_id = item.get("id") or f"codex-{len(active_items)}"
            name = self._codex_item_name(item, event_type)
            active_items[item_id] = name
            args = self._codex_item_args(item, event_type)
            self._send_trace_sse("tool_call", self._tool_call_trace_payload(name, args, json.dumps(args)))
            return

        if event_type == "item.completed":
            if item.get("type") == "agent_message":
                return
            item_id = item.get("id")
            name = active_items.pop(item_id, None) or self._codex_item_name(item, event_type)
            self._send_trace_sse(
                "tool_result",
                {
                    "name": name,
                    "preview": self._codex_item_preview(item, event_type),
                    "elapsed": round(time.time() - started, 1),
                },
            )
            return

        if event_type.endswith(".started"):
            name = event_type.removesuffix(".started").replace(".", "_")
            active_items[event_type] = name
            args = self._codex_event_args(event)
            self._send_trace_sse("tool_call", self._tool_call_trace_payload(name, args, json.dumps(args)))
            return

        if event_type.endswith(".completed"):
            start_type = event_type.removesuffix(".completed") + ".started"
            name = active_items.pop(start_type, event_type.removesuffix(".completed").replace(".", "_"))
            self._send_trace_sse(
                "tool_result",
                {
                    "name": name,
                    "preview": self._codex_event_preview(event),
                    "elapsed": round(time.time() - started, 1),
                },
            )
            return

        if event_type in ("error", "turn.failed"):
            self._send_trace_sse("error", {"message": self._codex_event_preview(event)})

    @staticmethod
    def _codex_item_name(item: dict, event_type: str) -> str:
        item_type = str(item.get("type") or event_type or "item").replace(".", "_")
        if item_type in ("command_execution", "exec_command", "shell_command"):
            return "codex_shell"
        return f"codex_{item_type}"

    @staticmethod
    def _codex_item_args(item: dict, event_type: str) -> dict:
        for key in ("command", "cmd", "args", "path", "text"):
            if key in item:
                return {key: item[key]}
        return {"event": event_type, "type": item.get("type", "unknown")}

    @staticmethod
    def _codex_item_preview(item: dict, event_type: str) -> str:
        for key in ("aggregated_output", "output", "stdout", "stderr", "text", "result", "message"):
            value = item.get(key)
            if value:
                return str(value).replace("\n", " ").strip()[:160]
        return event_type

    @staticmethod
    def _codex_event_args(event: dict) -> dict:
        for key in ("command", "cmd", "args", "path"):
            if key in event:
                return {key: event[key]}
        return {"event": event.get("type", "unknown")}

    @staticmethod
    def _codex_event_preview(event: dict) -> str:
        for key in ("message", "error", "output", "text", "stderr", "stdout"):
            value = event.get(key)
            if value:
                return str(value).replace("\n", " ").strip()[:160]
        return json.dumps(event)[:160]

    @staticmethod
    def _last_codex_agent_message(stdout: str) -> str:
        last = ""
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = event.get("item") or {}
            if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                last = item.get("text", "") or last
        return last.strip()

    def _start_sse_headers(self):
        """Send SSE response headers for streaming. Call once before any SSE writes."""
        self.send_response(200)
        self._add_cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _send_trace_sse(self, trace_type: str, data: dict):
        """Send a trace event via SSE.

        Normal mode: SSE comment (ignored by OpenAI SDK / fetch parser).
        EventSource mode: named event (visible to EventSource on iOS).
        Headers must already be sent."""
        payload = json.dumps({"type": trace_type, **data})
        if getattr(self, "_eventstream_mode", False):
            line = f"event: trace\ndata: {payload}\n\n".encode()
        else:
            line = f": trace {payload}\n\n".encode()
        try:
            self.wfile.write(b"%x\r\n%s\r\n" % (len(line), line))
            self.wfile.flush()
        except Exception:
            pass

    @staticmethod
    def _tool_task_label(name: str, args: dict | None = None) -> str:
        """Return a short user-facing description of what the tool is doing."""
        args = args or {}
        command = str(args.get("command", "")).strip()
        query = str(args.get("query", "")).strip()
        prompt = str(args.get("prompt", "")).strip()
        file_path = str(args.get("file_path") or args.get("path") or args.get("local_path") or "").strip()

        if name == "run_agent":
            agent_name = str(args.get("agent", "")).strip()
            return f"Spawned 1 agent: {agent_name}" if agent_name else "Spawned 1 agent"
        if name == "run_agents":
            tasks = args.get("tasks", [])
            if isinstance(tasks, list):
                agent_names = [
                    str(task.get("agent", "")).strip()
                    for task in tasks
                    if isinstance(task, dict) and str(task.get("agent", "")).strip()
                ]
                count = len(tasks)
                if agent_names:
                    return f"Spawned {count} agents: {', '.join(agent_names[:4])}"
                return f"Spawned {count} agents"
            return "Spawned agents"
        if name == "execute_plan":
            stages = args.get("stages", [])
            if isinstance(stages, list):
                count = len(stages)
                stage_word = "stage" if count == 1 else "stages"
                return f"Plan mode: executing {count} {stage_word}"
            return "Plan mode: executing tool plan"
        if name in {"run_command", "run_terminal_command", "grok_run_terminal_command", "grok_run_command"}:
            cmd = command.lower()
            if not cmd and args.get("description"):
                return str(args.get("description"))[:80]
            if "comfyui" in cmd or "8188" in cmd:
                return "Checking or starting ComfyUI"
            if "ffmpeg" in cmd:
                return "Rendering the video"
            if "hyperframes" in cmd:
                return "Building the video scene"
            if "youtube" in cmd:
                return "Checking YouTube state"
            if any(part in cmd for part in ("ls ", "find ", "rg ", "grep ", "tail ", "cat ", "ss ", "lsof ")):
                return "Inspecting local files and services"
            if any(part in cmd for part in ("git ", "npm ", "pytest", "py_compile")):
                return "Verifying code changes"
            return "Running a local command"
        if name in {"web_search", "web_browse", "web_image_search"}:
            return f"Researching: {query[:70]}" if query else "Researching online"
        if name in {"generate_image", "generate_image_flux2", "generate_image_muse"}:
            return f"Creating image asset: {prompt[:70]}" if prompt else "Creating image asset"
        if name == "youtube_upload":
            return "Uploading the video to YouTube"
        if name.startswith("youtube_"):
            return "Checking YouTube"
        if name.startswith("gmail_"):
            return "Working with Gmail"
        if name.startswith("drive_"):
            return "Working with Google Drive"
        if name.startswith("calendar_"):
            return "Working with Calendar"
        if name in {"read_file", "write_file", "list_directory"}:
            return f"Working with files: {file_path[:70]}" if file_path else "Working with files"
        if name in {"share_file", "pull_shared"}:
            return "Moving the finished file"
        if name.startswith("codex_"):
            return "Delegating work to Codex"
        if name == "grok_cli":
            return "Running Grok agent"
        if name.startswith("grok_"):
            base = name[len("grok_") :]
            # Recurse once on the unprefixed tool name for a friendlier label.
            if base and base != name:
                return CloudMixin._tool_task_label(base, args)
            return f"Grok: {base.replace('_', ' ')}"
        return name.replace("_", " ").capitalize()

    def _tool_call_trace_payload(self, name: str, args: dict | None, args_preview: str) -> dict:
        return {
            "name": name,
            "args": args_preview,
            "task": self._tool_task_label(name, args),
        }

    @staticmethod
    def _model_route_trace_payload(route_trace: dict | None, route: dict, resolved_name: str) -> dict:
        """Return a user-facing trace payload that explains model routing."""
        route_trace = route_trace or {}
        requested = route_trace.get("requested_model") or resolved_name
        provider = route.get("provider", "unknown")
        parsed_url = urlparse(route.get("base_url") or "")
        endpoint = parsed_url.netloc or parsed_url.path or "local"
        route_kind = "alias" if requested != resolved_name else "direct"
        tool_mode = route_trace.get("tool_mode") or "server"
        return {
            "requested_model": requested,
            "resolved_model": resolved_name,
            "provider": provider,
            "endpoint": endpoint,
            "max_context": route.get("max_context", 0),
            "route_kind": route_kind,
            "tool_mode": tool_mode,
            "summary": f"{requested} -> {resolved_name}" if route_kind == "alias" else resolved_name,
        }

    def _close_sse_with_error(self, error_msg: str):
        """Send an error trace and close the SSE stream cleanly."""
        try:
            self._send_trace_sse("error", {"message": error_msg})
            # Send [DONE] and close chunked encoding
            done_line = b"data: [DONE]\n\n"
            self.wfile.write(b"%x\r\n%s\r\n" % (len(done_line), done_line))
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception:
            pass

    @staticmethod
    def _compact_tool_result(name: str, result: str) -> str:
        """Truncate tool results to prevent context bloat across loop iterations.

        The full result is already sent to the client via SSE trace — this only
        affects what the LLM sees when planning its next step."""
        try:
            from maude_core.context_hygiene import compact_tool_result

            return compact_tool_result(name, result)
        except Exception:
            if not result:
                return result
            n = len(result)
            if n > 4000:
                return result[:3500] + f"\n... (truncated, {n} chars total)"
            return result

    def _try_fast_dispatch(self, user_msg: str, *, sse_started: bool = False):
        """Skip first LLM tool-selection for high-confidence single-tool intents.

        Used by phone/web gateway path (OpenAI + Claude tool loops). Returns
        (tool_name, args, compact_result) or None. Emits tool_call / tool_result
        traces so the phone UI shows the fast path the same as a normal tool step.
        """
        text = (user_msg or "").strip()
        if not text or len(text) > 240:
            return None
        # Don't hijack when client already pre-scoped tools for a multi-tool task
        try:
            from maude_core.fast_dispatch import fast_dispatch

            hit = fast_dispatch(text)
        except Exception as exc:
            logger.debug("fast_dispatch unavailable: %s", exc)
            return None
        if not hit:
            return None
        tool_name, args, tool_result = hit
        compact = self._compact_tool_result(tool_name, tool_result or "")
        if sse_started:
            try:
                self._send_trace_sse(
                    "tool_call",
                    self._tool_call_trace_payload(tool_name, args or {}, json.dumps(args or {})),
                )
                self._send_trace_sse(
                    "tool_result",
                    {
                        "name": tool_name,
                        "preview": (tool_result or "")[:200],
                        "elapsed": 0,
                        "fast_dispatch": True,
                    },
                )
            except Exception:
                pass
        logger.info("gateway fast_dispatch hit: %s", tool_name)
        return tool_name, args or {}, compact

    @staticmethod
    def _estimate_tokens(messages):
        """Rough token estimate: chars / 4. Handles both string and list content (Claude blocks)."""
        try:
            from maude_core.context_hygiene import estimate_tokens

            return estimate_tokens(messages)
        except Exception:
            total = 0
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    total += len(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            total += len(json.dumps(block))
                        elif isinstance(block, str):
                            total += len(block)
                tc = msg.get("tool_calls")
                if tc:
                    total += len(json.dumps(tc))
            return total // 4

    @staticmethod
    def _trim_messages(messages, max_tokens, format="openai"):
        """Context hygiene: drop old tool payloads, sliding summary, token trim.

        Mutates `messages` in place. Returns number of messages removed/summarized away.
        """
        try:
            from maude_core.context_hygiene import (
                drop_old_tool_payloads,
                sliding_window_with_summary,
                trim_to_token_budget,
            )

            _, tool_dropped = drop_old_tool_payloads(messages, format=format, in_place=True)
            prepared, win_meta = sliding_window_with_summary(messages, in_place=False)
            # Replace list contents with windowed version
            if win_meta.get("removed", 0) > 0 or len(prepared) != len(messages):
                messages[:] = prepared
            token_removed = trim_to_token_budget(messages, max_tokens, format=format)
            return int(win_meta.get("removed", 0)) + int(token_removed) + int(tool_dropped > 0)
        except Exception:
            # Fallback: legacy middle-trim
            threshold = int(max_tokens * 0.8)
            est = CloudMixin._estimate_tokens(messages)
            if est <= threshold:
                return 0
            removed = 0
            while CloudMixin._estimate_tokens(messages) > threshold and len(messages) > 4:
                idx = 2
                if idx >= len(messages) - 2:
                    break
                if format == "openai":
                    if messages[idx].get("role") == "assistant":
                        messages.pop(idx)
                        removed += 1
                        while idx < len(messages) - 2 and messages[idx].get("role") == "tool":
                            messages.pop(idx)
                            removed += 1
                    else:
                        messages.pop(idx)
                        removed += 1
                else:
                    if messages[idx].get("role") == "assistant":
                        messages.pop(idx)
                        removed += 1
                        if idx < len(messages) - 2 and messages[idx].get("role") == "user":
                            content = messages[idx].get("content", "")
                            is_tool_result = isinstance(content, list) and any(
                                isinstance(b, dict) and b.get("type") == "tool_result" for b in content
                            )
                            if is_tool_result:
                                messages.pop(idx)
                                removed += 1
                    else:
                        messages.pop(idx)
                        removed += 1
            return removed

    def _send_content_chunks(self, content: str, model_name: str, chunk_id: str, created: int):
        """Send content as word-boundary SSE chunks for typewriter effect.
        Headers must already be sent."""
        words = content.split(" ") if content else [""]
        chunks = []
        current = ""
        for word in words:
            current += (" " if current else "") + word
            if len(current) > 20:
                chunks.append(current)
                current = ""
        if current:
            chunks.append(current)

        for i, chunk_text in enumerate(chunks):
            spacer = " " if i < len(chunks) - 1 else ""
            sse_data = json.dumps(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk_text + spacer},
                            "finish_reason": None,
                        }
                    ],
                }
            )
            line = f"data: {sse_data}\n\n".encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(line), line))
            self.wfile.flush()

    def _send_sse_done(self, model_name: str, chunk_id: str, created: int):
        """Send finish + [DONE] events and close chunked encoding."""
        finish_data = json.dumps(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        finish_line = f"data: {finish_data}\n\n".encode()
        self.wfile.write(b"%x\r\n%s\r\n" % (len(finish_line), finish_line))

        done_line = b"data: [DONE]\n\n"
        self.wfile.write(b"%x\r\n%s\r\n" % (len(done_line), done_line))

        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _cloud_model_with_tools(self, req, route, resolved_name):
        """Handle cloud model request with server-side tool execution loop.

        Sends non-streaming requests to the cloud API in a loop, executing
        tool calls locally via maude_core, until a final text response is
        produced. The final response is streamed back to the client as SSE.
        """
        tool_retries = 0
        api_key = os.environ.get(route["api_key_env"], "") if route.get("api_key_env") else ""

        # Get the user's latest message for tool selection / fast_dispatch
        user_msg = ""
        for msg in reversed(req.get("messages", [])):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_msg = content
                elif isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(str(block.get("text", "")))
                        elif isinstance(block, str):
                            parts.append(block)
                    user_msg = "\n".join(parts)
                else:
                    user_msg = str(content or "")
                break

        # Use pre-scoped tools if provided, otherwise select by message + session sticky domains
        session_id = req.get("session_id") or os.environ.get("MAUDE_SESSION_ID", "default")
        tools_pre_scoped = bool(req.get("tools"))

        def _select_active_tools(msgs):
            if tools_pre_scoped:
                return req.get("tools")
            return get_tools_for_message(user_msg, session_id=session_id, messages=msgs)

        active_tools = _select_active_tools(req.get("messages", []))

        # Enhance system prompt so Mistral knows it has tool access
        tool_addendum = TOOL_ADDENDUM
        # Inject device location if provided by mobile app
        location = req.get("location")
        if location and isinstance(location, dict):
            lat = location.get("lat")
            lng = location.get("lng")
            if lat is not None and lng is not None:
                tool_addendum += (
                    f"\n\nDEVICE LOCATION: The user's phone is at latitude {lat:.6f}, "
                    f"longitude {lng:.6f} (accuracy: {location.get('accuracy', 'unknown')}m). "
                    "Use this for location-aware responses — nearby places, weather, directions, etc. "
                    "You do NOT need to ask the user where they are."
                )

        messages = list(req.get("messages", []))
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] = msg["content"] + tool_addendum
                break

        # Connection details for the cloud API
        parsed_url = urlparse(route["base_url"])
        use_ssl = parsed_url.scheme == "https"
        host = parsed_url.hostname
        port = parsed_url.port or (443 if use_ssl else 80)
        api_path = parsed_url.path.rstrip("/") + "/v1/chat/completions"

        reset_rate_limits()
        max_iterations = 40
        recent_tool_calls = []
        pending_images = []  # Collect image URLs from tool results for auto-injection
        is_streaming = req.get("stream", False)
        sse_started = False

        # Start SSE headers immediately for streaming so client gets trace events
        if is_streaming:
            self._start_sse_headers()
            sse_started = True
            self._send_trace_sse(
                "model_route",
                self._model_route_trace_payload(req.get("_route_trace"), route, resolved_name),
            )

        # Fast path: list/read/shell/memory/image/URL — skip first tool-selection LLM hop
        # (phone + web clients hit this path; Mac client has its own local fast_dispatch)
        if not tools_pre_scoped:
            fd = self._try_fast_dispatch(user_msg if isinstance(user_msg, str) else "", sse_started=sse_started)
            if fd:
                tool_name, args, compact = fd
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
                                    "arguments": json.dumps(args),
                                },
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": "fast_dispatch_1",
                        "name": tool_name,
                        "content": compact,
                    }
                )

        for iteration in range(max_iterations):
            # On last iteration, drop tools to force a summary response
            is_final = iteration == max_iterations - 1
            if is_final:
                messages.append(
                    {
                        "role": "user",
                        "content": "(System: You've used many tool calls. Wrap up now — summarize what you've done and what remains.)",
                    }
                )

            # Re-select tools each iteration so activate_tool_domain / history stickiness applies
            active_tools = _select_active_tools(messages)

            # Context trimming — keep messages within model's context window
            max_ctx = route.get("max_context", 128000)
            tool_schema_overhead = len(json.dumps(active_tools)) // 4 if active_tools else 0
            effective_max = max_ctx - tool_schema_overhead
            trimmed = self._trim_messages(messages, effective_max, format="openai")
            if trimmed:
                logger.info("Trimmed %d messages to fit %s context (%d tokens)", trimmed, resolved_name, effective_max)
                if sse_started:
                    self._send_trace_sse("context_trim", {"removed": trimmed, "max_tokens": effective_max})

            # Build non-streaming request for tool loop
            loop_req = {
                "model": resolved_name,
                "messages": messages,
                "stream": False,
                "max_tokens": req.get("max_tokens", 4096),
                "temperature": req.get("temperature", 0.7),
            }
            if not is_final:
                loop_req["tools"] = active_tools
                loop_req["tool_choice"] = "auto"

            body = json.dumps(loop_req).encode()
            llm_start = time.time()

            try:
                # Fresh connection per iteration — avoids stale sockets after long tool calls
                if use_ssl:
                    ctx = ssl.create_default_context()
                    # Disable older versions of SSL/TLS that are known to have issues
                    ctx.options |= ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3 | ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
                    conn = http.client.HTTPSConnection(host, port, timeout=300, context=ctx)
                else:
                    conn = http.client.HTTPConnection(host, port, timeout=300)

                headers = {
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                }
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                # Run LLM call in thread with keepalive pings
                llm_result_box = [None, None, None]  # [status, body, exception]

                def _llm_call():
                    try:
                        conn.request("POST", api_path, body=body, headers=headers)
                        resp = conn.getresponse()
                        llm_result_box[0] = resp.status
                        llm_result_box[1] = resp.read()
                        conn.close()
                    except Exception as exc:
                        llm_result_box[2] = exc

                t = threading.Thread(target=_llm_call)
                t.start()
                while t.is_alive():
                    t.join(timeout=15)
                    if t.is_alive() and sse_started:
                        elapsed_so_far = time.time() - llm_start
                        self._send_trace_sse(
                            "keepalive",
                            {
                                "name": "llm_call",
                                "elapsed": round(elapsed_so_far, 1),
                            },
                        )
                t.join()

                if llm_result_box[2] is not None:
                    raise llm_result_box[2]

                resp_status = llm_result_box[0]
                resp_body = llm_result_box[1]

                if resp_status != 200:
                    try:
                        err = json.loads(resp_body)
                    except Exception:
                        err = {"error": resp_body.decode(errors="replace")}
                    if sse_started:
                        self._close_sse_with_error(f"LLM error: {resp_status}")
                    else:
                        self._json_response(err, resp_status)
                    return

                result = json.loads(resp_body)
                choice = result.get("choices", [{}])[0]
                message = choice.get("message", {})
                finish_reason = choice.get("finish_reason", "")

            except ConnectionRefusedError:
                err_msg = f"Provider {route['provider']} connection refused"
                if sse_started:
                    self._close_sse_with_error(err_msg)
                else:
                    self._json_response({"error": err_msg}, 503)
                return
            except Exception as e:
                err_msg = str(e)
                transient = any(
                    k in err_msg.lower()
                    for k in (
                        "ssl",
                        "chunked",
                        "name resolution",
                        "connection",
                        "timed out",
                        "reset by peer",
                        "broken pipe",
                        "bad gateway",
                        "502",
                        "503",
                    )
                )
                if transient and tool_retries < 3:
                    tool_retries += 1
                    wait = 2 ** (tool_retries - 1)
                    logger.warning("Transient error in tool loop, retry %d/3 in %ds: %s", tool_retries, wait, err_msg)
                    if sse_started:
                        self._send_trace_sse("error", {"message": f"Connection error, retrying ({tool_retries}/3)..."})
                    time.sleep(wait)
                    continue
                err_msg = f"Tool loop error: {e}"
                if sse_started:
                    self._close_sse_with_error(err_msg)
                else:
                    self._json_response({"error": err_msg}, 502)
                return

            # Emit LLM call trace
            llm_elapsed = time.time() - llm_start
            usage = result.get("usage", {})
            prompt_tok = usage.get("prompt_tokens", 0)
            compl_tok = usage.get("completion_tokens", 0)
            if sse_started:
                self._send_trace_sse(
                    "llm_call",
                    {
                        "prompt_tokens": prompt_tok,
                        "completion_tokens": compl_tok,
                        "elapsed": round(llm_elapsed, 2),
                    },
                )
            logger.info("LLM: %d+%d tokens in %.1fs", prompt_tok, compl_tok, llm_elapsed)

            # Check for tool calls
            tool_calls = message.get("tool_calls")
            if tool_calls and finish_reason in ("tool_calls", "stop"):
                # Add assistant message with tool_calls to conversation
                # Strip non-standard fields (e.g. reasoning from Nemotron)
                # that may cause errors on subsequent API calls
                clean_msg = {
                    "role": message.get("role", "assistant"),
                    "content": message.get("content") or "",
                    "tool_calls": message.get("tool_calls", []),
                }
                messages.append(clean_msg)

                # Normalize tool_call IDs — local Mistral Nemo requires exactly
                # 9 alphanumeric chars, but cloud providers (OpenRouter, Mistral API)
                # send longer IDs that must be preserved for result matching.
                is_local = route.get("provider") == "local"
                if is_local:
                    for _idx, tc in enumerate(tool_calls):
                        tc_id = tc.get("id", "")
                        clean_id = "".join(c for c in tc_id if c in string.ascii_letters + string.digits)
                        if len(clean_id) < 9:
                            clean_id = clean_id + "x" * (9 - len(clean_id))
                        tc["id"] = clean_id[:9]

                # Parse all tool calls
                parsed_tc = []  # [(tc, func_name, func_args)]
                for tc in tool_calls:
                    func_name = tc["function"]["name"]
                    raw_args = tc["function"].get("arguments", "{}")
                    if isinstance(raw_args, dict):
                        func_args = raw_args
                    elif isinstance(raw_args, str):
                        try:
                            func_args = json.loads(raw_args)
                        except (json.JSONDecodeError, ValueError):
                            logger.warning("Failed to parse tool args for %s: %s", func_name, raw_args[:200])
                            func_args = {}
                    else:
                        logger.warning("Unexpected args type for %s: %s", func_name, type(raw_args))
                        func_args = {}

                    # Emit tool_call trace
                    args_preview = json.dumps(func_args, ensure_ascii=False)
                    if len(args_preview) > 80:
                        args_preview = args_preview[:80] + "..."
                    if sse_started:
                        self._send_trace_sse(
                            "tool_call",
                            self._tool_call_trace_payload(func_name, func_args, args_preview),
                        )

                    # Duplicate detection
                    call_sig = (func_name, json.dumps(func_args, sort_keys=True))
                    if call_sig in recent_tool_calls:
                        tool_result = "(Already called with same arguments. Respond with the best answer you have.)"
                        messages.append(
                            {
                                "role": "tool",
                                "name": func_name,
                                "content": tool_result,
                                "tool_call_id": tc["id"],
                            }
                        )
                    else:
                        recent_tool_calls.append(call_sig)
                        parsed_tc.append((tc, func_name, func_args))

                def _gw_exec_tool(tc, func_name, func_args):
                    """Execute one tool, return (tc, func_name, result, elapsed)."""
                    logger.info("tool %s(%s)", func_name, func_args)
                    t0 = time.time()
                    res = execute_tool(func_name, func_args)
                    return tc, func_name, res, time.time() - t0

                def _gw_emit_result(tc, func_name, tool_result, tool_elapsed):
                    """Emit trace and append result to messages."""
                    preview = (tool_result or "")[:80].replace("\n", " ").strip()
                    if len(tool_result or "") > 80:
                        preview += "..."
                    if sse_started:
                        self._send_trace_sse(
                            "tool_result",
                            {
                                "name": func_name,
                                "preview": preview,
                                "elapsed": round(tool_elapsed, 2),
                            },
                        )
                    messages.append(
                        {
                            "role": "tool",
                            "name": func_name,
                            "content": self._compact_tool_result(func_name, tool_result or ""),
                            "tool_call_id": tc["id"],
                        }
                    )
                    for _m in re.finditer(r"!\[([^\]]*)\]\(([^)]+)\)", tool_result or ""):
                        pending_images.append((_m.group(1), _m.group(2)))

                # Split into parallel-safe and sequential
                parallel_batch = [(tc, fn, args) for tc, fn, args in parsed_tc if fn in PARALLEL_SAFE]
                sequential_batch = [(tc, fn, args) for tc, fn, args in parsed_tc if fn not in PARALLEL_SAFE]

                # Run parallel-safe tools concurrently with keepalive pings
                if len(parallel_batch) > 1:
                    from concurrent.futures import ThreadPoolExecutor

                    par_names = [fn for _, fn, _ in parallel_batch]
                    logger.info("Parallel execution: %d tools (%s)", len(parallel_batch), ", ".join(par_names))
                    if sse_started:
                        self._send_trace_sse(
                            "parallel_start",
                            {
                                "count": len(parallel_batch),
                                "tools": par_names,
                            },
                        )
                    parallel_results = {}
                    tool_start = time.time()
                    with ThreadPoolExecutor(max_workers=min(len(parallel_batch), 6)) as pool:
                        futures = {
                            pool.submit(_gw_exec_tool, tc, fn, args): (tc, fn, args) for tc, fn, args in parallel_batch
                        }
                        # Wait with keepalive pings
                        while futures:
                            done_set = set()
                            for future in list(futures):
                                if future.done():
                                    tc_out, fn_out, res_out, elapsed_out = future.result()
                                    parallel_results[tc_out["id"]] = (tc_out, fn_out, res_out, elapsed_out)
                                    done_set.add(future)
                            for f in done_set:
                                del futures[f]
                            if futures and sse_started:
                                elapsed_so_far = time.time() - tool_start
                                self._send_trace_sse(
                                    "keepalive",
                                    {
                                        "name": "parallel_tools",
                                        "elapsed": round(elapsed_so_far, 1),
                                    },
                                )
                            if futures:
                                time.sleep(2)
                    # Append in original order
                    for tc, _fn, _args in parallel_batch:
                        tc_out, fn_out, res_out, elapsed_out = parallel_results[tc["id"]]
                        _gw_emit_result(tc_out, fn_out, res_out, elapsed_out)
                elif len(parallel_batch) == 1:
                    tc, fn, args = parallel_batch[0]
                    tool_start = time.time()
                    tool_result_box = [None]

                    def _run_tool(fn=fn, args=args):
                        tool_result_box[0] = execute_tool(fn, args)

                    t = threading.Thread(target=_run_tool)
                    t.start()
                    while t.is_alive():
                        t.join(timeout=15)
                        if t.is_alive() and sse_started:
                            self._send_trace_sse(
                                "keepalive",
                                {
                                    "name": fn,
                                    "elapsed": round(time.time() - tool_start, 1),
                                },
                            )
                    _gw_emit_result(tc, fn, tool_result_box[0], time.time() - tool_start)

                # Run state-mutating tools sequentially with keepalive
                for tc, func_name, func_args in sequential_batch:
                    logger.info("tool %s(%s)", func_name, func_args)
                    tool_start = time.time()
                    tool_result_box = [None]

                    def _run_tool(fn=func_name, a=func_args):
                        tool_result_box[0] = execute_tool(fn, a)

                    t = threading.Thread(target=_run_tool)
                    t.start()
                    while t.is_alive():
                        t.join(timeout=15)
                        if t.is_alive() and sse_started:
                            self._send_trace_sse(
                                "keepalive",
                                {
                                    "name": func_name,
                                    "elapsed": round(time.time() - tool_start, 1),
                                },
                            )
                    _gw_emit_result(tc, func_name, tool_result_box[0], time.time() - tool_start)

                continue  # Loop back to get next response from model

            # No tool calls — final text response
            final_content = message.get("content", "")

            # Auto-inject images from tool results that the LLM didn't include
            if pending_images:
                for alt, url in pending_images:
                    if url not in final_content:
                        final_content += f"\n\n![{alt}]({url})"

            if not is_streaming:
                self._json_response(
                    {
                        "id": f"chatcmpl-tool-{int(time.time())}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": resolved_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": final_content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": result.get("usage", {}),
                    }
                )
            elif sse_started:
                # SSE headers already sent — just send content + done
                chunk_id = f"chatcmpl-tool-{int(time.time())}"
                created = int(time.time())
                self._send_content_chunks(final_content, resolved_name, chunk_id, created)
                self._send_sse_done(resolved_name, chunk_id, created)
            else:
                self._send_as_sse(final_content, resolved_name)
            conn.close()
            return

        # Max iterations reached (shouldn't normally happen — last iteration drops tools)
        fallback = "I've completed as many steps as I can in one go. Ask me to continue if there's more to do."
        if not is_streaming:
            self._json_response(
                {
                    "id": f"chatcmpl-tool-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": resolved_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": fallback},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
        elif sse_started:
            chunk_id = f"chatcmpl-tool-{int(time.time())}"
            created = int(time.time())
            self._send_content_chunks(fallback, resolved_name, chunk_id, created)
            self._send_sse_done(resolved_name, chunk_id, created)
        else:
            self._send_as_sse(fallback, resolved_name)
        conn.close()

    def _send_as_sse(self, content, model_name):
        """Send a text response to the client as SSE, matching Mistral's streaming format."""
        self.send_response(200)
        self._add_cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        chunk_id = f"chatcmpl-tool-{int(time.time())}"
        created = int(time.time())

        # Split content into small chunks at word boundaries for streaming feel
        words = content.split(" ") if content else [""]
        chunks = []
        current = ""
        for word in words:
            current += (" " if current else "") + word
            if len(current) > 20:
                chunks.append(current)
                current = ""
        if current:
            chunks.append(current)

        for i, chunk_text in enumerate(chunks):
            spacer = " " if i < len(chunks) - 1 else ""
            sse_data = json.dumps(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk_text + spacer},
                            "finish_reason": None,
                        }
                    ],
                }
            )
            line = f"data: {sse_data}\n\n".encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(line), line))
            self.wfile.flush()

        # Send finish event
        finish_data = json.dumps(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        finish_line = f"data: {finish_data}\n\n".encode()
        self.wfile.write(b"%x\r\n%s\r\n" % (len(finish_line), finish_line))

        done_line = b"data: [DONE]\n\n"
        self.wfile.write(b"%x\r\n%s\r\n" % (len(done_line), done_line))

        # End chunked encoding
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _claude_tool_loop(self, req, route, resolved_name):
        """Handle Claude (Anthropic) model request with server-side tool execution loop.

        Claude's API differs from OpenAI/Mistral:
        - Auth via x-api-key header (not Bearer token)
        - Endpoint: /v1/messages (not /v1/chat/completions)
        - System prompt is a top-level field (not a message)
        - Tool schema uses { name, description, input_schema } (not function wrapper)
        - Tool results go in user messages as { type: "tool_result", tool_use_id }
        - Stop reasons: "tool_use" and "end_turn" (not "tool_calls" and "stop")
        """
        claude_retries = 0
        api_key = os.environ.get(route["api_key_env"], "")

        # Get the user's latest message for tool selection / fast_dispatch
        user_msg = ""
        for msg in reversed(req.get("messages", [])):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_msg = content
                elif isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(str(block.get("text", "")))
                        elif isinstance(block, str):
                            parts.append(block)
                    user_msg = "\n".join(parts)
                else:
                    user_msg = str(content or "")
                break

        # Use pre-scoped tools if provided, otherwise select by message + session sticky domains
        session_id = req.get("session_id") or os.environ.get("MAUDE_SESSION_ID", "default")
        tools_pre_scoped = bool(req.get("tools"))

        def _openai_to_claude_tools(openai_tools):
            converted = []
            for tool in openai_tools or []:
                func = tool.get("function", {})
                converted.append(
                    {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                    }
                )
            if converted:
                converted[-1]["cache_control"] = {"type": "ephemeral"}
            return converted

        def _select_claude_tools(msgs):
            if tools_pre_scoped:
                return _openai_to_claude_tools(req.get("tools"))
            return _openai_to_claude_tools(
                get_tools_for_message(user_msg, session_id=session_id, messages=msgs)
            )

        claude_tools = _select_claude_tools(req.get("messages", []))

        # Extract system prompt from messages and convert to Claude format
        system_text = ""
        claude_messages = []
        for msg in req.get("messages", []):
            if msg.get("role") == "system":
                system_text = msg.get("content", "")
            else:
                claude_messages.append(
                    {
                        "role": msg.get("role"),
                        "content": msg.get("content", ""),
                    }
                )

        # Enhance system prompt with tool context (same addendum as Mistral loop)
        tool_addendum = TOOL_ADDENDUM

        # Inject device location if provided by mobile app
        location = req.get("location")
        if location and isinstance(location, dict):
            lat = location.get("lat")
            lng = location.get("lng")
            if lat is not None and lng is not None:
                tool_addendum += (
                    f"\n\nDEVICE LOCATION: The user's phone is at latitude {lat:.6f}, "
                    f"longitude {lng:.6f} (accuracy: {location.get('accuracy', 'unknown')}m). "
                    "Use this for location-aware responses — nearby places, weather, directions, etc. "
                    "You do NOT need to ask the user where they are."
                )

        system_text += tool_addendum

        # Use block format for system prompt with cache_control
        # This caches the full system prompt + tool addendum across loop iterations
        # and across requests within the 5-minute TTL window
        system_blocks = [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        # Connection details for Claude API
        parsed_url = urlparse(route["base_url"])
        use_ssl = parsed_url.scheme == "https"
        host = parsed_url.hostname
        port = parsed_url.port or (443 if use_ssl else 80)
        api_path = "/v1/messages"

        reset_rate_limits()
        max_iterations = 40
        recent_tool_calls = []
        pending_images = []  # Collect image URLs from tool results for auto-injection
        is_streaming = req.get("stream", False)
        sse_started = False

        # Start SSE headers immediately for streaming so client gets trace events
        if is_streaming:
            self._start_sse_headers()
            sse_started = True
            self._send_trace_sse(
                "model_route",
                self._model_route_trace_payload(req.get("_route_trace"), route, resolved_name),
            )

        # Fast path for phone/web Claude routes (same patterns as OpenAI loop)
        if not tools_pre_scoped:
            fd = self._try_fast_dispatch(user_msg if isinstance(user_msg, str) else "", sse_started=sse_started)
            if fd:
                tool_name, args, compact = fd
                claude_messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "fast_dispatch_1",
                                "name": tool_name,
                                "input": args,
                            }
                        ],
                    }
                )
                claude_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "fast_dispatch_1",
                                "content": compact,
                            }
                        ],
                    }
                )

        for iteration in range(max_iterations):
            # On last iteration, drop tools to force a summary response
            is_final = iteration == max_iterations - 1
            if is_final:
                claude_messages.append(
                    {
                        "role": "user",
                        "content": "(System: You've used many tool calls. Wrap up now — summarize what you've done and what remains.)",
                    }
                )

            # Re-select tools each iteration so domain activation expands schemas mid-loop
            claude_tools = _select_claude_tools(claude_messages)

            # Context trimming — keep messages within model's context window
            max_ctx = route.get("max_context", 200000)
            system_overhead = len(json.dumps(system_blocks)) // 4
            tool_schema_overhead = len(json.dumps(claude_tools)) // 4 if claude_tools else 0
            effective_max = max_ctx - system_overhead - tool_schema_overhead
            trimmed = self._trim_messages(claude_messages, effective_max, format="claude")
            if trimmed:
                logger.info(
                    "Trimmed %d Claude messages to fit %s context (%d tokens)", trimmed, resolved_name, effective_max
                )
                if sse_started:
                    self._send_trace_sse("context_trim", {"removed": trimmed, "max_tokens": effective_max})

            # Build non-streaming request for tool loop
            loop_req = {
                "model": resolved_name,
                "max_tokens": req.get("max_tokens", 4096),
                "system": system_blocks,
                "messages": claude_messages,
            }
            if not is_final:
                loop_req["tools"] = claude_tools

            body = json.dumps(loop_req).encode()
            llm_start = time.time()

            try:
                # Fresh connection per iteration — avoids stale sockets after long tool calls
                if use_ssl:
                    ctx = ssl.create_default_context()
                    conn = http.client.HTTPSConnection(host, port, timeout=300, context=ctx)
                else:
                    conn = http.client.HTTPConnection(host, port, timeout=300)

                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Length": str(len(body)),
                }

                # Run LLM call in thread with keepalive pings
                llm_result_box = [None, None, None]  # [status, body, exception]

                def _llm_call():
                    try:
                        conn.request("POST", api_path, body=body, headers=headers)
                        resp = conn.getresponse()
                        llm_result_box[0] = resp.status
                        llm_result_box[1] = resp.read()
                        conn.close()
                    except Exception as exc:
                        llm_result_box[2] = exc

                t = threading.Thread(target=_llm_call)
                t.start()
                while t.is_alive():
                    t.join(timeout=15)
                    if t.is_alive() and sse_started:
                        elapsed_so_far = time.time() - llm_start
                        self._send_trace_sse(
                            "keepalive",
                            {
                                "name": "llm_call",
                                "elapsed": round(elapsed_so_far, 1),
                            },
                        )
                t.join()

                if llm_result_box[2] is not None:
                    raise llm_result_box[2]

                resp_status = llm_result_box[0]
                resp_body = llm_result_box[1]

                if resp_status != 200:
                    try:
                        err = json.loads(resp_body)
                    except Exception:
                        err = {"error": resp_body.decode(errors="replace")}
                    if not sse_started:
                        self._json_response(err, resp_status)
                    else:
                        err_msg = (
                            err.get("error", {}).get("message", str(err))
                            if isinstance(err.get("error"), dict)
                            else str(err)
                        )
                        self._close_sse_with_error(f"Claude error: {err_msg}")
                    return

                result = json.loads(resp_body)
                stop_reason = result.get("stop_reason", "")

            except ConnectionRefusedError:
                err_msg = f"Provider {route['provider']} connection refused"
                if sse_started:
                    self._close_sse_with_error(err_msg)
                else:
                    self._json_response({"error": err_msg}, 503)
                return
            except Exception as e:
                err_msg = str(e)
                transient = any(
                    k in err_msg.lower()
                    for k in (
                        "ssl",
                        "chunked",
                        "name resolution",
                        "connection",
                        "timed out",
                        "reset by peer",
                        "broken pipe",
                        "bad gateway",
                        "502",
                        "503",
                        "overloaded",
                    )
                )
                if transient and claude_retries < 3:
                    claude_retries += 1
                    wait = 2 ** (claude_retries - 1)
                    logger.warning(
                        "Transient error in Claude tool loop, retry %d/3 in %ds: %s", claude_retries, wait, err_msg
                    )
                    if sse_started:
                        self._send_trace_sse(
                            "error", {"message": f"Connection error, retrying ({claude_retries}/3)..."}
                        )
                    time.sleep(wait)
                    continue
                err_msg = f"Claude tool loop error: {e}"
                if sse_started:
                    self._close_sse_with_error(err_msg)
                else:
                    self._json_response({"error": err_msg}, 502)
                return

            # Emit LLM call trace (include cache stats)
            llm_elapsed = time.time() - llm_start
            usage = result.get("usage", {})
            prompt_tok = usage.get("input_tokens", 0)
            compl_tok = usage.get("output_tokens", 0)
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_create = usage.get("cache_creation_input_tokens", 0)
            trace_data = {
                "prompt_tokens": prompt_tok,
                "completion_tokens": compl_tok,
                "elapsed": round(llm_elapsed, 2),
            }
            if cache_read:
                trace_data["cache_read_tokens"] = cache_read
            if cache_create:
                trace_data["cache_create_tokens"] = cache_create
            if sse_started:
                self._send_trace_sse("llm_call", trace_data)
            cache_info = ""
            if cache_read:
                cache_info += f" cache_read={cache_read}"
            if cache_create:
                cache_info += f" cache_write={cache_create}"
            logger.info("Claude LLM: %d+%d tokens in %.1fs%s", prompt_tok, compl_tok, llm_elapsed, cache_info)

            # Parse content blocks from Claude response
            content_blocks = result.get("content", [])
            text_parts = []
            tool_use_blocks = []
            for block in content_blocks:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_use_blocks.append(block)

            # Check for tool use
            if tool_use_blocks and stop_reason == "tool_use":
                # Add assistant message (full content blocks) to conversation
                claude_messages.append(
                    {
                        "role": "assistant",
                        "content": content_blocks,
                    }
                )

                # Parse all tool blocks, emit traces, detect duplicates
                tool_results = []
                parsed_tu = []  # [(tu_block, func_name, func_args, tool_use_id)]
                for tu in tool_use_blocks:
                    func_name = tu.get("name", "")
                    func_args = tu.get("input", {})
                    tool_use_id = tu.get("id", "")

                    # Emit tool_call trace
                    args_preview = json.dumps(func_args, ensure_ascii=False)
                    if len(args_preview) > 80:
                        args_preview = args_preview[:80] + "..."
                    if sse_started:
                        self._send_trace_sse(
                            "tool_call",
                            self._tool_call_trace_payload(func_name, func_args, args_preview),
                        )

                    # Duplicate detection
                    call_sig = (func_name, json.dumps(func_args, sort_keys=True))
                    if call_sig in recent_tool_calls:
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": "(Already called with same arguments. Respond with the best answer you have.)",
                            }
                        )
                    else:
                        recent_tool_calls.append(call_sig)
                        parsed_tu.append((tu, func_name, func_args, tool_use_id))

                def _cl_exec_tool(tu, func_name, func_args, tool_use_id):
                    """Execute one Claude tool call."""
                    logger.info("claude tool %s(%s)", func_name, func_args)
                    t0 = time.time()
                    res = execute_tool(func_name, func_args)
                    return tu, func_name, func_args, tool_use_id, res, time.time() - t0

                def _cl_emit_and_collect(func_name, func_args, tool_use_id, tool_result, tool_elapsed):
                    """Emit trace, collect result, gather images."""
                    preview = (tool_result or "")[:80].replace("\n", " ").strip()
                    if len(tool_result or "") > 80:
                        preview += "..."
                    if sse_started:
                        self._send_trace_sse(
                            "tool_result",
                            {
                                "name": func_name,
                                "preview": preview,
                                "elapsed": round(tool_elapsed, 2),
                            },
                        )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": self._compact_tool_result(func_name, tool_result or ""),
                        }
                    )
                    for _m in re.finditer(r"!\[([^\]]*)\]\(([^)]+)\)", tool_result or ""):
                        pending_images.append((_m.group(1), _m.group(2)))

                # Split into parallel-safe and sequential
                par_batch = [(tu, fn, fa, tid) for tu, fn, fa, tid in parsed_tu if fn in PARALLEL_SAFE]
                seq_batch = [(tu, fn, fa, tid) for tu, fn, fa, tid in parsed_tu if fn not in PARALLEL_SAFE]

                # Run parallel-safe tools concurrently with keepalive
                if len(par_batch) > 1:
                    from concurrent.futures import ThreadPoolExecutor

                    par_names = [fn for _, fn, _, _ in par_batch]
                    logger.info("Parallel execution: %d tools (%s)", len(par_batch), ", ".join(par_names))
                    if sse_started:
                        self._send_trace_sse(
                            "parallel_start",
                            {
                                "count": len(par_batch),
                                "tools": par_names,
                            },
                        )
                    par_results = {}
                    tool_start = time.time()
                    with ThreadPoolExecutor(max_workers=min(len(par_batch), 6)) as pool:
                        futures = {pool.submit(_cl_exec_tool, tu, fn, fa, tid): tid for tu, fn, fa, tid in par_batch}
                        while futures:
                            done_set = set()
                            for future in list(futures):
                                if future.done():
                                    _tu_o, fn_o, fa_o, tid_o, res_o, el_o = future.result()
                                    par_results[tid_o] = (fn_o, fa_o, tid_o, res_o, el_o)
                                    done_set.add(future)
                            for f in done_set:
                                del futures[f]
                            if futures and sse_started:
                                self._send_trace_sse(
                                    "keepalive",
                                    {
                                        "name": "parallel_tools",
                                        "elapsed": round(time.time() - tool_start, 1),
                                    },
                                )
                            if futures:
                                time.sleep(2)
                    # Emit in original order
                    for _tu, _fn, _fa, tid in par_batch:
                        fn_o, fa_o, tid_o, res_o, el_o = par_results[tid]
                        _cl_emit_and_collect(fn_o, fa_o, tid_o, res_o, el_o)
                elif len(par_batch) == 1:
                    tu, fn, fa, tid = par_batch[0]
                    tool_start = time.time()
                    tool_result_box = [None]

                    def _run_tool(fn=fn, fa=fa):
                        tool_result_box[0] = execute_tool(fn, fa)

                    t = threading.Thread(target=_run_tool)
                    t.start()
                    while t.is_alive():
                        t.join(timeout=15)
                        if t.is_alive() and sse_started:
                            self._send_trace_sse(
                                "keepalive",
                                {
                                    "name": fn,
                                    "elapsed": round(time.time() - tool_start, 1),
                                },
                            )
                    _cl_emit_and_collect(fn, fa, tid, tool_result_box[0], time.time() - tool_start)

                # Run state-mutating tools sequentially with keepalive
                for _tu, func_name, func_args, tool_use_id in seq_batch:
                    logger.info("claude tool %s(%s)", func_name, func_args)
                    tool_start = time.time()
                    tool_result_box = [None]

                    def _run_tool(fn=func_name, fa=func_args):
                        tool_result_box[0] = execute_tool(fn, fa)

                    t = threading.Thread(target=_run_tool)
                    t.start()
                    while t.is_alive():
                        t.join(timeout=15)
                        if t.is_alive() and sse_started:
                            self._send_trace_sse(
                                "keepalive",
                                {
                                    "name": func_name,
                                    "elapsed": round(time.time() - tool_start, 1),
                                },
                            )
                    _cl_emit_and_collect(
                        func_name, func_args, tool_use_id, tool_result_box[0], time.time() - tool_start
                    )

                # Add tool results as a user message (Claude API format)
                claude_messages.append(
                    {
                        "role": "user",
                        "content": tool_results,
                    }
                )

                continue  # Loop back to get next response from model

            # No tool calls — final text response
            final_content = "\n\n".join(text_parts) if text_parts else ""

            # Auto-inject images from tool results that the LLM didn't include
            if pending_images:
                for alt, url in pending_images:
                    if url not in final_content:
                        final_content += f"\n\n![{alt}]({url})"

            if not is_streaming:
                self._json_response(
                    {
                        "id": f"chatcmpl-claude-{int(time.time())}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": resolved_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": final_content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tok,
                            "completion_tokens": compl_tok,
                            "total_tokens": prompt_tok + compl_tok,
                        },
                    }
                )
            elif sse_started:
                chunk_id = f"chatcmpl-claude-{int(time.time())}"
                created = int(time.time())
                self._send_content_chunks(final_content, resolved_name, chunk_id, created)
                self._send_sse_done(resolved_name, chunk_id, created)
            else:
                self._send_as_sse(final_content, resolved_name)
            conn.close()
            return

        # Max iterations reached (shouldn't normally happen — last iteration drops tools)
        fallback = "I've completed as many steps as I can in one go. Ask me to continue if there's more to do."
        if not is_streaming:
            self._json_response(
                {
                    "id": f"chatcmpl-claude-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": resolved_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": fallback},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
        elif sse_started:
            chunk_id = f"chatcmpl-claude-{int(time.time())}"
            created = int(time.time())
            self._send_content_chunks(fallback, resolved_name, chunk_id, created)
            self._send_sse_done(resolved_name, chunk_id, created)
        else:
            self._send_as_sse(fallback, resolved_name)
        conn.close()
