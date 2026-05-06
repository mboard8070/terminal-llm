"""
Agent Executor — runs subagents with tool access through the gateway.

The gateway already handles tool loops (Mistral and Claude). This module
simply sends a chat/completions request with scoped tools and stream=False,
letting the gateway execute tools and return the final response.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import httpx

from maude_core.tool_defs import TOOLS
from subagents import AGENT_TOOL_SCOPES, SUBAGENTS

GATEWAY_URL = "http://localhost:30080/v1"

# Default model for agents (gateway resolves short names)
_AGENT_MODELS = {
    "code": "codestral",
    "research": "mistral",
    "writer": "mistral",
    "reasoning": "devstral",
    "search": "mistral",
}


@dataclass
class AgentTask:
    """A task to dispatch to a subagent."""

    agent_name: str
    task: str
    context: str = ""
    max_iterations: int = 20


@dataclass
class AgentResult:
    """Result from a subagent execution."""

    agent_name: str
    output: str
    tool_calls_made: int = 0
    elapsed: float = 0.0
    error: str | None = None


class SharedContext:
    """Thread-safe shared context for parallel agents to exchange findings."""

    def __init__(self):
        self._data: dict[str, str] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: str):
        with self._lock:
            self._data[key] = value

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._data.get(key)

    def get_all(self) -> dict[str, str]:
        with self._lock:
            return dict(self._data)

    def summary(self) -> str:
        """Return a text summary of all shared findings so far."""
        with self._lock:
            if not self._data:
                return ""
            parts = []
            for agent, result in self._data.items():
                parts.append(f"[{agent}]: {result[:500]}")
            return "\n\n".join(parts)


def _get_scoped_tools(agent_name: str) -> list[dict]:
    """Filter TOOLS to only those allowed for this agent."""
    allowed = set(AGENT_TOOL_SCOPES.get(agent_name, []))
    if not allowed:
        return []
    return [t for t in TOOLS if t["function"]["name"] in allowed]


def execute_agent_with_tools(
    task: AgentTask,
    shared_context: SharedContext | None = None,
) -> AgentResult:
    """Execute a single agent task through the gateway.

    Sends a non-streaming request to the gateway with scoped tools.
    The gateway runs its internal tool loop and returns the final response.
    """
    start = time.time()
    agent_name = task.agent_name.lower()

    # Validate agent exists
    agent = SUBAGENTS.get(agent_name)
    if not agent:
        return AgentResult(
            agent_name=agent_name,
            output="",
            error=f"Unknown agent: {agent_name}. Available: {', '.join(SUBAGENTS.keys())}",
        )

    # Build scoped tools
    scoped_tools = _get_scoped_tools(agent_name)

    # Build system prompt
    system_prompt = agent.system_prompt
    if shared_context:
        shared_summary = shared_context.summary()
        if shared_summary:
            system_prompt += (
                f"\n\n--- Findings from other agents working in parallel ---\n"
                f"{shared_summary}\n"
                f"--- End of shared findings ---"
            )

    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    if task.context:
        messages.append({"role": "user", "content": f"Context:\n{task.context}\n\nTask: {task.task}"})
    else:
        messages.append({"role": "user", "content": task.task})

    # Build request
    model = _AGENT_MODELS.get(agent_name, "mistral")
    request_body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": agent.temperature,
        "max_tokens": agent.max_tokens,
    }
    if scoped_tools:
        request_body["tools"] = scoped_tools

    try:
        resp = httpx.post(
            f"{GATEWAY_URL}/chat/completions",
            json=request_body,
            timeout=300.0,
        )
        resp.raise_for_status()
        result = resp.json()

        # Extract response
        choice = result.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")

        elapsed = time.time() - start

        # Store result in shared context if available
        if shared_context:
            shared_context.set(agent_name, content[:500])

        return AgentResult(
            agent_name=agent_name,
            output=content,
            elapsed=round(elapsed, 1),
        )

    except httpx.HTTPStatusError as e:
        return AgentResult(
            agent_name=agent_name,
            output="",
            elapsed=round(time.time() - start, 1),
            error=f"Gateway returned {e.response.status_code}: {e.response.text[:200]}",
        )
    except Exception as e:
        return AgentResult(
            agent_name=agent_name,
            output="",
            elapsed=round(time.time() - start, 1),
            error=str(e),
        )


def execute_agents_parallel(tasks: list[AgentTask]) -> list[AgentResult]:
    """Execute multiple agent tasks in parallel.

    Earlier finishers' results are available to later ones via SharedContext.
    """
    shared = SharedContext()
    results: list[AgentResult] = [None] * len(tasks)

    with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as executor:
        future_to_idx = {executor.submit(execute_agent_with_tools, task, shared): i for i, task in enumerate(tasks)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = AgentResult(
                    agent_name=tasks[idx].agent_name,
                    output="",
                    error=str(e),
                )

    return results
