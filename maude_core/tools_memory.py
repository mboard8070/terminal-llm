"""
Memory tools — LLM-callable persistent memory operations.
"""

from tool_registry import register_tool

from .log import log
from .memory_ledger import LEDGER_CATEGORIES, get_ledger
from .memory_utils import get_memory
from .mempalace_utils import get_palace_kg, get_palace_stack


@register_tool("save_memory")
def _dispatch_save_memory(args):
    key = args.get("key", "").strip()
    value = args.get("value", "").strip()
    category = args.get("category", "fact")

    if not key or not value:
        return "Error: Both 'key' and 'value' are required."

    ledger = get_ledger()
    record = ledger.save(key, value, category if category in LEDGER_CATEGORIES else "fact")

    mem = get_memory()
    if mem:
        mem.remember(key, value, record.category, metadata={"ledger": True})

    log(f"Memory ledger saved: [{record.category}] {key}")
    return f"Remembered [{record.category}] {key}: {value}"


@register_tool("recall_memory")
def _dispatch_recall_memory(args):
    query = args.get("query", "").strip()
    category = args.get("category")

    if not query:
        return "Error: 'query' is required."

    ledger = get_ledger()
    ledger_results = ledger.search(query, category=category, limit=5)

    mem_results = []
    mem = get_memory()
    if mem:
        mem_results = mem.search(query, limit=5, category=category)

    seen = {(r.category, r.key) for r in ledger_results}
    for result in mem_results:
        if (result.category, result.key) not in seen:
            ledger_results.append(result)
            seen.add((result.category, result.key))

    if not ledger_results:
        return f"No memories found matching '{query}'."

    lines = [f"Found {len(ledger_results[:5])} relevant memories:"]
    for m in ledger_results[:5]:
        lines.append(f"- [{m.category}] **{m.key}**: {m.value}")
    return "\n".join(lines)


@register_tool("list_memories")
def _dispatch_list_memories(args):
    category = args.get("category")
    limit = args.get("limit", 20)

    ledger = get_ledger()
    memories = ledger.latest_records(category=category)[:limit]

    if not memories:
        mem = get_memory()
        if mem:
            memories = mem.list_memories(category=category, limit=limit)

    if not memories:
        return "No memories stored." if not category else f"No memories in category '{category}'."

    lines = [f"Stored memories ({len(memories)}):"]
    for m in memories:
        lines.append(f"- [{m.category}] **{m.key}**: {m.value}")
    return "\n".join(lines)


@register_tool("memory_ledger_status")
def _dispatch_memory_ledger_status(args):
    ledger = get_ledger()
    status = ledger.status()
    lines = [
        f"Memory ledger: {status['path']}",
        f"Records: {status['records']}",
        "Front door: save_memory / recall_memory",
        "Deep archive: MemPalace remains optional, not the default memory model.",
    ]
    if status["categories"]:
        lines.append("Categories: " + ", ".join(f"{k}={v}" for k, v in status["categories"].items()))
    if status["files"]:
        lines.append("Files: " + ", ".join(status["files"]))
    return "\n".join(lines)


@register_tool("forget_memory")
def _dispatch_forget_memory(args):
    key = args.get("key", "").strip()
    if not key:
        return "Error: 'key' is required."

    ledger_removed = get_ledger().forget(key)
    mem_removed = False
    mem = get_memory()
    if mem:
        mem_removed = mem.forget(key)

    if ledger_removed or mem_removed:
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
