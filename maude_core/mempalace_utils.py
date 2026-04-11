"""
MemPalace instance management — lazy-loaded shared palace components.

Pattern follows memory_utils.py: singletons initialized on first access,
shared across all tool handlers and context injection.
"""

import os

from .log import log

_palace_config = None
_palace_stack = None
_palace_kg = None


def get_palace_config():
    """Get or create the shared MempalaceConfig instance."""
    global _palace_config
    if _palace_config is None:
        try:
            from mempalace.config import MempalaceConfig

            _palace_config = MempalaceConfig()
            log(f"MemPalace config loaded — palace at {_palace_config.palace_path}")
        except Exception as e:
            log(f"MemPalace config init failed: {e}")
            return None
    return _palace_config


def get_palace_stack():
    """Get or create the shared MemoryStack instance (for wake-up context)."""
    global _palace_stack
    if _palace_stack is None:
        try:
            from mempalace.layers import MemoryStack

            cfg = get_palace_config()
            if cfg:
                _palace_stack = MemoryStack(palace_path=cfg.palace_path)
        except Exception as e:
            log(f"MemPalace stack init failed: {e}")
            return None
    return _palace_stack


def get_palace_kg():
    """Get or create the shared KnowledgeGraph instance."""
    global _palace_kg
    if _palace_kg is None:
        try:
            from mempalace.knowledge_graph import KnowledgeGraph

            cfg = get_palace_config()
            if cfg:
                kg_path = os.path.join(cfg.palace_path, "knowledge_graph.sqlite3")
                _palace_kg = KnowledgeGraph(db_path=kg_path)
        except Exception as e:
            log(f"MemPalace KG init failed: {e}")
            return None
    return _palace_kg


def get_palace_collection():
    """Get or create the shared ChromaDB collection."""
    try:
        import chromadb

        cfg = get_palace_config()
        if not cfg:
            return None
        client = chromadb.PersistentClient(path=cfg.palace_path)
        return client.get_collection("mempalace_drawers")
    except Exception as e:
        log(f"MemPalace collection access failed: {e}")
        return None


_EMPTY_MARKERS = ("No palace found", "No results", "No drawers")


def _is_empty_result(text: str) -> bool:
    if not text:
        return True
    return any(marker in text for marker in _EMPTY_MARKERS)


def get_palace_context_for_prompt(user_input: str, n_results: int = 5) -> str | None:
    """Return a palace context block for system-prompt injection, or None if nothing relevant."""
    stack = get_palace_stack()
    if stack is None:
        return None
    try:
        hits = stack.search(user_input, n_results=n_results)
    except Exception as e:
        log(f"MemPalace search failed: {e}")
        return None
    if _is_empty_result(hits):
        return None
    return f"## Relevant memories from MemPalace\n{hits}"
