"""
Agent dispatch tool implementations — run_agent, run_agents.

These tools let the main LLM dispatch tasks to specialized subagents
that have their own tool access through the gateway.
"""

from maude_core.log import log
from tool_registry import register_tool


@register_tool("run_agent")
def _dispatch_run_agent(args):
    from agent_executor import AgentTask, execute_agent_with_tools

    agent_name = args.get("agent", "")
    task = args.get("task", "")
    context = args.get("context", "")

    if not agent_name or not task:
        return "Error: 'agent' and 'task' are required"

    log(f"Dispatching to {agent_name} agent: {task[:60]}...")

    agent_task = AgentTask(
        agent_name=agent_name,
        task=task,
        context=context,
    )
    result = execute_agent_with_tools(agent_task)

    if result.error:
        return f"[{agent_name} agent error] {result.error}"

    return f"[{agent_name} agent — {result.elapsed}s]\n\n{result.output}"


@register_tool("run_agents")
def _dispatch_run_agents(args):
    from agent_executor import AgentTask, execute_agents_parallel

    tasks_data = args.get("tasks", [])
    if not tasks_data:
        return "Error: 'tasks' array is required"

    agent_tasks = []
    for t in tasks_data:
        agent_tasks.append(
            AgentTask(
                agent_name=t.get("agent", ""),
                task=t.get("task", ""),
                context=t.get("context", ""),
            )
        )

    agent_names = [t.agent_name for t in agent_tasks]
    log(f"Dispatching {len(agent_tasks)} agents in parallel: {', '.join(agent_names)}")

    results = execute_agents_parallel(agent_tasks)

    # Format results
    parts = []
    for r in results:
        if r.error:
            parts.append(f"### {r.agent_name} agent (error)\n{r.error}")
        else:
            parts.append(f"### {r.agent_name} agent ({r.elapsed}s)\n\n{r.output}")

    return "\n\n---\n\n".join(parts)
