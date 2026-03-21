"""
Planned Execution — multi-stage tool execution with dependency resolution.

The model emits a plan with stages. Each stage's tools run in parallel.
Stages run sequentially. Later stages can reference earlier results via $N.M syntax.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tool_registry import register_tool

from .log import log

# Read-only tools safe to run concurrently — shared across all tool loops
PARALLEL_SAFE = frozenset(
    {
        "read_file",
        "list_directory",
        "search_file",
        "search_directory",
        "get_working_directory",
        "web_browse",
        "web_search",
        "web_image_search",
        "web_view",
        "view_image",
        "gmail_list",
        "gmail_read",
        "drive_list",
        "drive_search",
        "drive_read",
        "sheets_read",
        "sheets_list_sheets",
        "calendar_list_events",
        "calendar_search_events",
        "calendar_list_calendars",
        "contacts_list",
        "contacts_get",
        "contacts_search",
        "youtube_search",
        "youtube_get_video",
        "youtube_get_channel",
        "youtube_list_playlists",
        "youtube_get_playlist_items",
        "youtube_get_comments",
        "youtube_my_channel",
        "github_list_prs",
        "github_view_pr",
        "github_pr_diff",
        "github_pr_comments",
        "github_list_issues",
        "github_view_issue",
        "github_list_repos",
        "github_view_repo",
        "github_list_branches",
        "github_list_commits",
        "github_list_runs",
        "github_view_run",
        "github_list_releases",
        "github_search",
        "github_notifications",
        "slides_get_presentation",
        "slides_get_slide",
        "substack_list_drafts",
        "substack_list_posts",
        "substack_get_post",
        "substack_get_stats",
        "recall_memory",
        "list_memories",
        "mesh_status",
        "list_tasks",
        "list_projects",
        "system_stats",
        "gpu_processes",
        "session_list",
        "activity_feed",
        "scheduler_status",
        "node_status",
        "sandbox_status",
        "sandbox_list",
        "sandbox_read_file",
        "ask_frontier",
    }
)

# Regex to match $N.M references in argument values
_REF_PATTERN = re.compile(r"\$(\d+)\.(\d+)")


def _resolve_refs(value, results_by_stage):
    """Replace $N.M references in a string value with actual results.

    $N.M = result from stage N, tool index M.
    """
    if not isinstance(value, str):
        return value

    def _replacer(match):
        stage_idx = int(match.group(1))
        tool_idx = int(match.group(2))
        stage_results = results_by_stage.get(stage_idx, [])
        if tool_idx < len(stage_results):
            return stage_results[tool_idx]
        return match.group(0)  # leave unreplaced if missing

    return _REF_PATTERN.sub(_replacer, value)


def _resolve_args(args: dict, results_by_stage: dict) -> dict:
    """Resolve all $N.M references in a tool's arguments."""
    resolved = {}
    for key, value in args.items():
        if isinstance(value, str):
            resolved[key] = _resolve_refs(value, results_by_stage)
        elif isinstance(value, list):
            resolved[key] = [_resolve_refs(v, results_by_stage) if isinstance(v, str) else v for v in value]
        else:
            resolved[key] = value
    return resolved


def execute_plan(stages: list) -> str:
    """Execute a multi-stage tool plan.

    Args:
        stages: list of stages, each stage is a list of tool calls:
                [{"name": "tool_name", "args": {…}}, …]

    Returns:
        Formatted string with all results organized by stage.
    """
    from .execute import execute_tool

    results_by_stage = {}  # stage_idx -> [result_str, ...]
    output_parts = []
    total_start = time.time()

    for stage_idx, stage in enumerate(stages):
        if not isinstance(stage, list):
            output_parts.append(
                f"## Stage {stage_idx}: ERROR — expected list of tool calls, got {type(stage).__name__}"
            )
            continue

        stage_start = time.time()
        calls = []
        for tool_idx, call in enumerate(stage):
            name = call.get("name", "")
            raw_args = call.get("args", {})
            if not name:
                continue
            # Resolve $N.M references from previous stages
            resolved_args = _resolve_args(raw_args, results_by_stage)
            calls.append((tool_idx, name, resolved_args))

        if not calls:
            results_by_stage[stage_idx] = []
            continue

        # Determine which calls can run in parallel vs sequential
        parallel = [(idx, n, a) for idx, n, a in calls if n in PARALLEL_SAFE]
        sequential = [(idx, n, a) for idx, n, a in calls if n not in PARALLEL_SAFE]

        stage_results = {}  # tool_idx -> result string

        # Run parallel-safe tools concurrently
        if len(parallel) > 1:
            with ThreadPoolExecutor(max_workers=min(len(parallel), 6)) as pool:
                futures = {pool.submit(execute_tool, name, args): (idx, name) for idx, name, args in parallel}
                for future in as_completed(futures):
                    idx, name = futures[future]
                    try:
                        stage_results[idx] = future.result() or ""
                    except Exception as e:
                        stage_results[idx] = f"Error: {e}"
        elif len(parallel) == 1:
            idx, name, args = parallel[0]
            try:
                stage_results[idx] = execute_tool(name, args) or ""
            except Exception as e:
                stage_results[idx] = f"Error: {e}"

        # Run sequential tools in order
        for idx, name, args in sequential:
            try:
                stage_results[idx] = execute_tool(name, args) or ""
            except Exception as e:
                stage_results[idx] = f"Error: {e}"

        # Store results in order for reference by later stages
        ordered_results = (
            [stage_results.get(idx, "") for idx in range(max(stage_results.keys()) + 1)] if stage_results else []
        )
        results_by_stage[stage_idx] = ordered_results
        stage_elapsed = time.time() - stage_start

        # Build output for this stage
        output_parts.append(f"## Stage {stage_idx} ({stage_elapsed:.1f}s)")
        for idx, name, args in calls:
            result = stage_results.get(idx, "")
            # Truncate individual results to keep output manageable
            if len(result) > 2000:
                result = result[:1800] + f"\n... (truncated, {len(result)} chars total)"
            arg_hint = ""
            for key in ("command", "query", "path", "local_path", "name", "url"):
                if key in args:
                    val = str(args[key])
                    if len(val) > 60:
                        val = val[:60] + "…"
                    arg_hint = f" ({val})"
                    break
            output_parts.append(f"### [{stage_idx}.{idx}] {name}{arg_hint}\n{result}")

    total_elapsed = time.time() - total_start
    tool_count = sum(len(stage) for stage in stages if isinstance(stage, list))
    header = f"Plan executed: {len(stages)} stages, {tool_count} tools, {total_elapsed:.1f}s total"
    log(f"[execute_plan] {header}")

    return header + "\n\n" + "\n\n".join(output_parts)


@register_tool("execute_plan")
def _dispatch_execute_plan(args):
    stages = args.get("stages", [])
    if not stages:
        return "Error: No stages provided"
    return execute_plan(stages)
