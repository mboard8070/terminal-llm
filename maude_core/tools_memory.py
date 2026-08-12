"""
Memory tools — LLM-callable persistent memory operations.
"""

from tool_registry import register_tool

from .log import log
from .memory_utils import get_memory
from .mempalace_utils import get_palace_kg, get_palace_stack


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

    try:
        from .context_hygiene import clip_memory_value, memory_top_k

        limit = int(args.get("limit", memory_top_k()))
        limit = max(1, min(limit, 10))
    except Exception:
        clip_memory_value = lambda v: v if len(v) <= 400 else v[:397] + "..."  # noqa: E731
        limit = 5

    results = mem.search(query, limit=limit, category=category)
    if not results:
        return f"No memories found matching '{query}'."

    lines = [f"Found {len(results)} relevant memories (top-{limit}):"]
    for m in results:
        lines.append(f"- [{m.category}] **{m.key}**: {clip_memory_value(m.value)}")
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


@register_tool("palace_search")
def _dispatch_palace_search(args):
    stack = get_palace_stack()
    if stack is None:
        return "Error: MemPalace unavailable."

    query = args.get("query", "").strip()
    if not query:
        return "Error: 'query' is required."

    wing = args.get("wing") or None
    room = args.get("room") or None
    n_results = int(args.get("n_results", 5))

    try:
        return stack.search(query, wing=wing, room=room, n_results=n_results)
    except Exception as e:
        log(f"palace_search failed: {e}")
        return f"Error: palace search failed ({e})."


@register_tool("palace_recall")
def _dispatch_palace_recall(args):
    stack = get_palace_stack()
    if stack is None:
        return "Error: MemPalace unavailable."

    wing = args.get("wing") or None
    room = args.get("room") or None
    n_results = int(args.get("n_results", 10))

    try:
        return stack.recall(wing=wing, room=room, n_results=n_results)
    except Exception as e:
        log(f"palace_recall failed: {e}")
        return f"Error: palace recall failed ({e})."


@register_tool("palace_status")
def _dispatch_palace_status(args):
    stack = get_palace_stack()
    if stack is None:
        return "Error: MemPalace unavailable."
    try:
        status = stack.status()
    except Exception as e:
        return f"Error: palace status failed ({e})."
    lines = [f"Palace: {status.get('palace_path', '?')}", f"Drawers: {status.get('total_drawers', 0)}"]
    l0 = status.get("L0_identity") or {}
    if l0.get("exists"):
        lines.append(f"L0 identity: {l0.get('path')} ({l0.get('tokens', 0)} tokens)")
    else:
        lines.append("L0 identity: not configured")
    return "\n".join(lines)


@register_tool("palace_kg_add_fact")
def _dispatch_palace_kg_add_fact(args):
    kg = get_palace_kg()
    if kg is None:
        return "Error: MemPalace knowledge graph unavailable."

    subject = args.get("subject", "").strip()
    predicate = args.get("predicate", "").strip()
    obj = args.get("object", "").strip()
    if not subject or not predicate or not obj:
        return "Error: 'subject', 'predicate', and 'object' are all required."

    confidence = float(args.get("confidence", 1.0))
    try:
        kg.add_triple(subject=subject, predicate=predicate, obj=obj, confidence=confidence)
    except Exception as e:
        log(f"palace_kg_add_fact failed: {e}")
        return f"Error: add_triple failed ({e})."
    log(f"KG fact added: {subject} --{predicate}--> {obj}")
    return f"Recorded: {subject} --{predicate}--> {obj} (confidence={confidence})"


@register_tool("palace_kg_query")
def _dispatch_palace_kg_query(args):
    kg = get_palace_kg()
    if kg is None:
        return "Error: MemPalace knowledge graph unavailable."

    entity = args.get("entity", "").strip()
    if not entity:
        return "Error: 'entity' is required."

    direction = args.get("direction", "outgoing")
    try:
        results = kg.query_entity(name=entity, direction=direction)
    except Exception as e:
        log(f"palace_kg_query failed: {e}")
        return f"Error: query_entity failed ({e})."

    if not results:
        return f"No facts recorded for '{entity}'."

    lines = [f"Facts about {entity}:"]
    for row in results:
        if isinstance(row, dict):
            subj = row.get("subject", entity)
            pred = row.get("predicate", "?")
            obj = row.get("object", "?")
            lines.append(f"- {subj} --{pred}--> {obj}")
        else:
            lines.append(f"- {row}")
    return "\n".join(lines)


# ── Mission scratch (ephemeral working memory for a task) ─────────────


@register_tool("scratch_set")
def _dispatch_scratch_set(args):
    """Store a note on the current mission scratch pad (does not bloat chat history)."""
    from .context_hygiene import get_mission_scratch, save_mission_scratch

    key = (args.get("key") or "").strip()
    value = (args.get("value") or "").strip()
    if not key or not value:
        return "Error: both 'key' and 'value' are required."
    mission_id = (args.get("mission_id") or "").strip() or None
    scratch = get_mission_scratch(mission_id)
    scratch.set_note(key, value)
    if args.get("title"):
        scratch.title = str(args["title"])[:120]
    if args.get("objective"):
        scratch.objective = str(args["objective"])[:400]
    save_mission_scratch(scratch)
    return f"Scratch note set: {key}"


@register_tool("scratch_add_finding")
def _dispatch_scratch_add_finding(args):
    """Append a finding to the mission scratch pad."""
    from .context_hygiene import get_mission_scratch, save_mission_scratch

    text = (args.get("text") or args.get("finding") or "").strip()
    if not text:
        return "Error: 'text' is required."
    mission_id = (args.get("mission_id") or "").strip() or None
    scratch = get_mission_scratch(mission_id)
    scratch.add_finding(text)
    save_mission_scratch(scratch)
    return f"Finding recorded ({len(scratch.findings)} total)."


@register_tool("scratch_show")
def _dispatch_scratch_show(args):
    """Show the current mission scratch pad."""
    from .context_hygiene import get_mission_scratch

    mission_id = (args.get("mission_id") or "").strip() or None
    scratch = get_mission_scratch(mission_id)
    block = scratch.prompt_block(max_findings=20)
    return block or f"Scratch pad '{scratch.mission_id}' is empty."


@register_tool("scratch_clear")
def _dispatch_scratch_clear(args):
    """Clear the mission scratch pad."""
    from .context_hygiene import clear_mission_scratch

    mission_id = (args.get("mission_id") or "").strip() or None
    clear_mission_scratch(mission_id)
    return "Scratch pad cleared."
