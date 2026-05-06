"""
Cloud model tool loops for the MAUDE Gateway.

Contains CloudMixin — methods for handling cloud/local model requests with
server-side tool execution loops. Mixed into GatewayHandler.
"""

import http.client
import json
import os
import re
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

HyperFrames is CLI-based; there is no long-running HyperFrames service to start. The rendered video is shared through /download/<file> when share is true."""

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
            active_tools = get_tools_for_message(user_msg) if get_tools_for_message else []
        except Exception:
            active_tools = []

        if not active_tools:
            return ""

        lines = [
            "AVAILABLE MAUDE TOOLS FOR THIS REQUEST:",
            "Use any listed tool with:",
            "curl -s -X POST http://localhost:30080/api/tools/execute "
            "-H 'Content-Type: application/json' "
            "-d '{\"name\":\"TOOL_NAME\",\"arguments\":{...}}'",
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
        timeout = int(os.environ.get("MAUDE_CODEX_TIMEOUT", "900"))

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

    def _run_codex_json_process(self, proc, prompt: str, stream: bool, started: float, timeout: int):
        """Feed Codex and translate its JSONL stdout into Maude trace events."""
        if proc.stdin:
            proc.stdin.write(prompt)
            proc.stdin.close()

        stdout_lines = []
        stderr_lines = []
        active_items = {}
        deadline = started + timeout

        def _read_stderr():
            if not proc.stderr:
                return
            for line in proc.stderr:
                stderr_lines.append(line)

        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        if proc.stdout:
            while True:
                if time.time() > deadline:
                    raise subprocess.TimeoutExpired(proc.args, timeout)
                line = proc.stdout.readline()
                if line:
                    stdout_lines.append(line)
                    if stream:
                        self._emit_codex_json_trace(line, active_items, started)
                    continue
                if proc.poll() is not None:
                    break
                time.sleep(0.05)

        proc.wait(timeout=max(0.1, deadline - time.time()))
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
            self._send_trace_sse("tool_call", {"name": name, "args": json.dumps(self._codex_item_args(item, event_type))})
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
            self._send_trace_sse("tool_call", {"name": name, "args": json.dumps(self._codex_event_args(event))})
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
        if not result:
            return result
        n = len(result)
        # write_file / edit_file results are already short
        if name in ("write_file", "edit_file", "change_directory", "get_working_directory"):
            return result
        # read_file: keep first 80 + last 20 lines, drop middle
        if name == "read_file" and n > 3000:
            lines = result.split("\n")
            if len(lines) > 100:
                head = "\n".join(lines[:80])
                tail = "\n".join(lines[-20:])
                return f"{head}\n\n... ({len(lines) - 100} lines omitted) ...\n\n{tail}"
            return result[:3000] + f"\n... (truncated, {n} chars total)"
        # run_command: keep first 2000 + last 800
        if name == "run_command" and n > 3000:
            return result[:2000] + f"\n\n... ({n - 2800} chars omitted) ...\n\n" + result[-800:]
        # list_directory: keep first 60 entries
        if name == "list_directory" and n > 2000:
            lines = result.split("\n")
            if len(lines) > 65:
                return "\n".join(lines[:65]) + f"\n... ({len(lines) - 65} more entries)"
            return result[:2000] + "\n... (truncated)"
        # Everything else: hard cap at 4000
        if n > 4000:
            return result[:3500] + f"\n... (truncated, {n} chars total)"
        return result

    @staticmethod
    def _estimate_tokens(messages):
        """Rough token estimate: chars / 4. Handles both string and list content (Claude blocks)."""
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
            # Also count tool_calls if present
            tc = msg.get("tool_calls")
            if tc:
                total += len(json.dumps(tc))
        return total // 4

    @staticmethod
    def _trim_messages(messages, max_tokens, format="openai"):
        """Remove oldest middle messages to fit within max_tokens (80% threshold).

        Preserves first 2 messages (system/user) and the most recent messages.
        In openai format: removes assistant+tool pairs together.
        In claude format: removes assistant+user-tool-result pairs together.
        Returns number of messages removed.
        """
        threshold = int(max_tokens * 0.8)

        # Check if we even need to trim
        est = CloudMixin._estimate_tokens(messages)
        if est <= threshold:
            return 0

        removed = 0
        # Keep trimming from position 2 (after system/first-user) until under threshold
        while CloudMixin._estimate_tokens(messages) > threshold and len(messages) > 4:
            idx = 2  # Start removing from after the first 2 messages

            if idx >= len(messages) - 2:
                break  # Don't remove the most recent messages

            if format == "openai":
                # Remove assistant message + any following tool messages
                if messages[idx].get("role") == "assistant":
                    messages.pop(idx)
                    removed += 1
                    # Remove consecutive tool messages that followed it
                    while idx < len(messages) - 2 and messages[idx].get("role") == "tool":
                        messages.pop(idx)
                        removed += 1
                else:
                    messages.pop(idx)
                    removed += 1
            elif format == "claude":
                # Remove assistant message + following user message (tool results)
                if messages[idx].get("role") == "assistant":
                    messages.pop(idx)
                    removed += 1
                    # Remove following user message with tool results
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

        # Get the user's latest message for tool selection
        user_msg = ""
        for msg in reversed(req.get("messages", [])):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        # Use pre-scoped tools if provided, otherwise select by message content
        active_tools = req.get("tools") or get_tools_for_message(user_msg)

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
                            {
                                "name": func_name,
                                "args": args_preview,
                            },
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

        # Get the user's latest message for tool selection
        user_msg = ""
        for msg in reversed(req.get("messages", [])):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        # Use pre-scoped tools if provided, otherwise select by message content
        active_tools_openai = req.get("tools") or get_tools_for_message(user_msg)

        # Convert OpenAI tool format to Claude format
        claude_tools = []
        for tool in active_tools_openai:
            func = tool.get("function", {})
            claude_tools.append(
                {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                }
            )

        # Mark last tool with cache_control so tools + system get cached
        if claude_tools:
            claude_tools[-1]["cache_control"] = {"type": "ephemeral"}

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
                            {
                                "name": func_name,
                                "args": args_preview,
                            },
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
