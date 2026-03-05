"""
Memory tools — LLM-callable persistent memory operations.
"""

from tool_registry import register_tool
from .memory_utils import get_memory
from .log import log


@register_tool("save_memory")
def _dispatch_save_memory(args):
    mem = get_memory()
    if not mem:
        return "Error: Memory system unavailable."

    key = args.get("key", "").strip()
    value = args.get("value", "").strip()
    category = args.get("category", "fact")

    if not key or not value:
        return "Error: Both 'key' and 'value' are required."

    mem.remember(key, value, category)
    log(f"Memory saved: [{category}] {key}")
    return f"Remembered [{category}] {key}: {value}"


@register_tool("recall_memory")
def _dispatch_recall_memory(args):
    mem = get_memory()
    if not mem:
        return "Error: Memory system unavailable."

    query = args.get("query", "").strip()
    category = args.get("category")

    if not query:
        return "Error: 'query' is required."

    results = mem.search(query, limit=5, category=category)
    if not results:
        return f"No memories found matching '{query}'."

    lines = [f"Found {len(results)} relevant memories:"]
    for m in results:
        lines.append(f"- [{m.category}] **{m.key}**: {m.value}")
    return "\n".join(lines)


@register_tool("list_memories")
def _dispatch_list_memories(args):
    mem = get_memory()
    if not mem:
        return "Error: Memory system unavailable."

    category = args.get("category")
    limit = args.get("limit", 20)

    memories = mem.list_memories(category=category, limit=limit)
    if not memories:
        return "No memories stored." if not category else f"No memories in category '{category}'."

    lines = [f"Stored memories ({len(memories)}):"]
    for m in memories:
        lines.append(f"- [{m.category}] **{m.key}**: {m.value}")
    return "\n".join(lines)


@register_tool("forget_memory")
def _dispatch_forget_memory(args):
    mem = get_memory()
    if not mem:
        return "Error: Memory system unavailable."

    key = args.get("key", "").strip()
    if not key:
        return "Error: 'key' is required."

    if mem.forget(key):
        log(f"Memory forgotten: {key}")
        return f"Forgot: {key}"
    return f"No memory found for '{key}'."
