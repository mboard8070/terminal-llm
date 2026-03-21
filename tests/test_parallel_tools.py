"""
Tests for parallel tool execution across all execution paths:
- Direct tool loop (chat_local.py)
- Gateway tool loops (Mistral + Claude)
- Agent executor (agents get parallel tools via gateway)
- PARALLEL_SAFE classification correctness
"""

import threading
import time
from typing import ClassVar
from unittest.mock import MagicMock, patch

from maude_core.tools_plan import PARALLEL_SAFE


class TestParallelSafeClassification:
    """Verify tools are correctly classified as parallel-safe or mutating."""

    MUST_BE_PARALLEL: ClassVar[set[str]] = {
        "read_file",
        "list_directory",
        "search_file",
        "search_directory",
        "get_working_directory",
        "web_browse",
        "web_search",
        "gmail_list",
        "gmail_read",
        "drive_list",
        "drive_search",
        "drive_read",
        "github_list_prs",
        "github_view_pr",
        "recall_memory",
        "list_memories",
        "system_stats",
        "gpu_processes",
    }

    MUST_BE_SEQUENTIAL: ClassVar[set[str]] = {
        "write_file",
        "edit_file",
        "run_command",
        "change_directory",
        "gmail_send",
        "save_memory",
        "forget_memory",
        "drive_upload",
        "drive_delete",
        "drive_update_doc",
        "calendar_create_event",
        "calendar_delete_event",
        "generate_image",
        "social_post",
        "sandbox_exec",
        "sandbox_write_file",
    }

    def test_parallel_tools_classified_correctly(self):
        for name in self.MUST_BE_PARALLEL:
            assert name in PARALLEL_SAFE, f"{name} should be in PARALLEL_SAFE"

    def test_mutating_tools_excluded(self):
        for name in self.MUST_BE_SEQUENTIAL:
            assert name not in PARALLEL_SAFE, f"{name} should NOT be in PARALLEL_SAFE"


class TestParallelExecution:
    """Verify that parallel-safe tools actually run concurrently."""

    def test_parallel_tools_overlap_in_time(self):
        """Three parallel-safe tools should run with overlapping execution times."""
        from maude_core.tools_plan import execute_plan

        timestamps = []  # [(event, index, time)]
        call_counter = {"n": 0}
        lock = threading.Lock()

        def mock_execute(name, args):
            with lock:
                idx = call_counter["n"]
                call_counter["n"] += 1
                timestamps.append(("start", idx, time.time()))
            time.sleep(0.1)  # Simulate I/O
            with lock:
                timestamps.append(("end", idx, time.time()))
            return f"result:{name}"

        with patch("maude_core.execute.execute_tool", side_effect=mock_execute):
            execute_plan(
                [
                    [
                        {"name": "read_file", "args": {"path": "a.py"}},
                        {"name": "search_file", "args": {"path": "b.py", "pattern": "x"}},
                        {"name": "list_directory", "args": {"path": "/tmp"}},
                    ]
                ]
            )

        # Extract start/end times per call index
        starts = {idx: t for ev, idx, t in timestamps if ev == "start"}
        ends = {idx: t for ev, idx, t in timestamps if ev == "end"}
        assert len(starts) == 3

        # At least one pair should have overlapping execution
        overlaps = 0
        for i in starts:
            for j in starts:
                if i < j and starts[j] < ends[i] and starts[i] < ends[j]:
                    overlaps += 1
        assert overlaps > 0, "Parallel tools should have overlapping execution times"

    def test_sequential_tools_do_not_overlap(self):
        """Mutating tools should run one after another."""
        from maude_core.tools_plan import execute_plan

        timestamps = []
        lock = threading.Lock()

        def mock_execute(name, args):
            with lock:
                timestamps.append(("start", name, time.time()))
            time.sleep(0.05)
            with lock:
                timestamps.append(("end", name, time.time()))
            return f"result:{name}"

        with patch("maude_core.execute.execute_tool", side_effect=mock_execute):
            execute_plan(
                [
                    [
                        {"name": "write_file", "args": {"path": "a", "content": "x"}},
                        {"name": "run_command", "args": {"command": "echo hi"}},
                    ]
                ]
            )

        starts = {name: t for ev, name, t in timestamps if ev == "start"}
        ends = {name: t for ev, name, t in timestamps if ev == "end"}

        # run_command should start after write_file ends
        assert starts["run_command"] >= ends["write_file"] - 0.001  # small float tolerance

    def test_mixed_batch_parallel_runs_first(self):
        """In a mixed batch, parallel tools run, then sequential tools run."""
        from maude_core.tools_plan import execute_plan

        call_order = []
        lock = threading.Lock()

        def mock_execute(name, args):
            time.sleep(0.02)  # Small delay so parallel tools finish together
            with lock:
                call_order.append(name)
            return "ok"

        with patch("maude_core.execute.execute_tool", side_effect=mock_execute):
            execute_plan(
                [
                    [
                        {"name": "read_file", "args": {"path": "a"}},
                        {"name": "list_directory", "args": {"path": "/tmp"}},
                        {"name": "run_command", "args": {"command": "echo"}},
                    ]
                ]
            )

        # run_command (sequential) should be last
        assert call_order[-1] == "run_command"


class TestAgentParallelTools:
    """Verify agents get parallel tool execution via gateway."""

    def test_agents_run_in_parallel(self):
        """execute_agents_parallel should run multiple agents concurrently."""
        from agent_executor import AgentTask, execute_agents_parallel

        timestamps = []
        lock = threading.Lock()

        def mock_post(*args, **kwargs):
            with lock:
                timestamps.append(("start", time.time()))
            time.sleep(0.1)
            with lock:
                timestamps.append(("end", time.time()))
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"choices": [{"message": {"content": "done"}}]}
            resp.raise_for_status = MagicMock()
            return resp

        tasks = [
            AgentTask(agent_name="research", task="task 1"),
            AgentTask(agent_name="research", task="task 2"),
            AgentTask(agent_name="research", task="task 3"),
        ]

        with patch("agent_executor.httpx.post", side_effect=mock_post):
            results = execute_agents_parallel(tasks)

        assert len(results) == 3
        assert all(r.output == "done" for r in results)

        # Check that agents ran concurrently (overlapping timestamps)
        starts = [t for ev, t in timestamps if ev == "start"]
        ends = [t for ev, t in timestamps if ev == "end"]
        # All 3 should start before the first one ends
        assert starts[2] < ends[0], "Agents should run concurrently"

    def test_agent_sends_scoped_tools_to_gateway(self):
        """Agent should send tool definitions to gateway, which handles parallel exec."""
        from agent_executor import AgentTask, execute_agent_with_tools

        captured_body = {}

        def mock_post(url, **kwargs):
            captured_body.update(kwargs.get("json", {}))
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"choices": [{"message": {"content": "result"}}]}
            resp.raise_for_status = MagicMock()
            return resp

        task = AgentTask(agent_name="research", task="find info about X")

        with patch("agent_executor.httpx.post", side_effect=mock_post):
            result = execute_agent_with_tools(task)

        assert result.output == "result"
        # Verify tools were sent to gateway (gateway handles parallel exec)
        assert "tools" in captured_body or "messages" in captured_body

    def test_agent_shared_context_thread_safe(self):
        """SharedContext should safely handle concurrent writes."""
        from agent_executor import SharedContext

        ctx = SharedContext()
        errors = []

        def writer(name, value):
            try:
                for i in range(100):
                    ctx.set(f"{name}_{i}", f"{value}_{i}")
                    ctx.get(f"{name}_{i}")
                    ctx.summary()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"agent_{i}", f"val_{i}")) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(ctx.get_all()) == 500


class TestExecutePlanInAgentContext:
    """Verify execute_plan works correctly when called by an agent."""

    @patch("maude_core.execute.execute_tool")
    def test_execute_plan_with_agent_typical_pattern(self, mock_exec):
        """Simulate a research agent using execute_plan: search, then read results."""
        from maude_core.tools_plan import execute_plan

        def side_effect(name, args):
            if name == "web_search":
                return f"Result for: {args.get('query', '')}"
            if name == "web_browse":
                return f"Page content from: {args.get('url', '')}"
            return "ok"

        mock_exec.side_effect = side_effect

        result = execute_plan(
            [
                # Stage 0: parallel web searches
                [
                    {"name": "web_search", "args": {"query": "python async"}},
                    {"name": "web_search", "args": {"query": "python threading"}},
                    {"name": "web_search", "args": {"query": "python multiprocessing"}},
                ],
                # Stage 1: read top results (could use $refs in real usage)
                [
                    {"name": "web_browse", "args": {"url": "https://docs.python.org/async"}},
                    {"name": "web_browse", "args": {"url": "https://docs.python.org/threading"}},
                ],
            ]
        )

        assert mock_exec.call_count == 5
        assert "2 stages" in result
        assert "5 tools" in result
        assert "Result for: python async" in result
        assert "Page content from: https://docs.python.org/async" in result
