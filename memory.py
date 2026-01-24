"""
Persistent Memory System for MAUDE.

Provides semantic search across memories using nomic-embed-text,
conversation history storage, and context injection for prompts.
"""

import json
import math
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from openai import OpenAI


@dataclass
class Memory:
    """A single memory item."""
    key: str
    value: str
    category: str  # "fact", "preference", "conversation", "task", "person"
    created_at: str
    updated_at: str
    access_count: int
    metadata: Optional[Dict[str, Any]] = None


class MaudeMemory:
    """Persistent memory with semantic search capabilities."""

    DB_PATH = Path.home() / ".config" / "maude" / "memory.db"

    def __init__(self, embed_url: str = None):
        """
        Initialize memory system.

        Args:
            embed_url: URL for embedding model (default: local Ollama)
        """
        import os
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Use environment variable or default to local Ollama
        self.embed_url = embed_url or os.environ.get(
            "EMBED_AGENT_URL", "http://localhost:11434/v1"
        )
        self.embed_client = OpenAI(base_url=self.embed_url, api_key="not-needed")
        self.embed_model = "nomic-embed-text:latest"

        self._init_db()

    def _init_db(self):
        """Initialize SQLite database."""
        self.conn = sqlite3.connect(str(self.DB_PATH), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Memories table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                category TEXT DEFAULT 'fact',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                embedding TEXT,
                metadata TEXT
            )
        """)

        # Conversations table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                channel TEXT DEFAULT 'cli',
                tokens INTEGER DEFAULT 0
            )
        """)

        # Indexes
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_session
            ON conversations(session_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_category
            ON memories(category)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_updated
            ON memories(updated_at)
        """)

        self.conn.commit()

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text using nomic-embed-text."""
        try:
            response = self.embed_client.embeddings.create(
                model=self.embed_model,
                input=text[:8000]  # Truncate to avoid token limits
            )
            return response.data[0].embedding
        except Exception as e:
            # Silently fail - semantic search will fall back to text search
            return []

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    # ─────────────────────────────────────────────────────────────────
    # Core Memory Operations
    # ─────────────────────────────────────────────────────────────────

    def remember(
        self,
        key: str,
        value: str,
        category: str = "fact",
        metadata: dict = None
    ) -> bool:
        """
        Store or update a memory.

        Args:
            key: Unique identifier for the memory
            value: The content to remember
            category: Type of memory (fact, preference, person, task, conversation)
            metadata: Additional structured data
        """
        now = datetime.now().isoformat()

        # Generate embedding for semantic search
        embedding = self._get_embedding(f"{key}: {value}")

        self.conn.execute("""
            INSERT INTO memories (key, value, category, created_at, updated_at,
                                  access_count, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, 0, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                category = excluded.category,
                updated_at = excluded.updated_at,
                embedding = excluded.embedding,
                metadata = excluded.metadata,
                access_count = access_count + 1
        """, (
            key, value, category, now, now,
            json.dumps(embedding) if embedding else None,
            json.dumps(metadata) if metadata else None
        ))
        self.conn.commit()
        return True

    def recall(self, key: str) -> Optional[str]:
        """
        Retrieve a memory by exact key.

        Args:
            key: The memory key to look up

        Returns:
            The memory value, or None if not found
        """
        cursor = self.conn.execute(
            "SELECT value FROM memories WHERE key = ?", (key,)
        )
        row = cursor.fetchone()

        if row:
            # Update access count
            self.conn.execute("""
                UPDATE memories
                SET access_count = access_count + 1, updated_at = ?
                WHERE key = ?
            """, (datetime.now().isoformat(), key))
            self.conn.commit()
            return row[0]
        return None

    def forget(self, key: str) -> bool:
        """
        Remove a memory.

        Args:
            key: The memory key to remove

        Returns:
            True if memory was removed, False if not found
        """
        cursor = self.conn.execute(
            "DELETE FROM memories WHERE key = ?", (key,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def search(
        self,
        query: str,
        limit: int = 5,
        category: str = None,
        threshold: float = 0.3
    ) -> List[Memory]:
        """
        Semantic search across memories.

        Args:
            query: Search query
            limit: Maximum results to return
            category: Filter by category (optional)
            threshold: Minimum similarity score (0-1)

        Returns:
            List of matching memories, sorted by relevance
        """
        query_embedding = self._get_embedding(query)

        if not query_embedding:
            # Fall back to text search if embedding fails
            return self._text_search(query, limit, category)

        # Get all memories with embeddings
        sql = "SELECT * FROM memories WHERE embedding IS NOT NULL"
        params = []
        if category:
            sql += " AND category = ?"
            params.append(category)

        cursor = self.conn.execute(sql, params)

        # Calculate cosine similarity for each memory
        results = []
        for row in cursor.fetchall():
            mem_embedding = json.loads(row["embedding"]) if row["embedding"] else []
            if mem_embedding:
                similarity = self._cosine_similarity(query_embedding, mem_embedding)
                if similarity >= threshold:
                    results.append((similarity, Memory(
                        key=row["key"],
                        value=row["value"],
                        category=row["category"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        access_count=row["access_count"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else None
                    )))

        # Sort by similarity (descending) and return top results
        results.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in results[:limit]]

    def _text_search(
        self,
        query: str,
        limit: int,
        category: str = None
    ) -> List[Memory]:
        """Fallback text search when embeddings unavailable."""
        sql = "SELECT * FROM memories WHERE (key LIKE ? OR value LIKE ?)"
        params = [f"%{query}%", f"%{query}%"]

        if category:
            sql += " AND category = ?"
            params.append(category)

        sql += " ORDER BY access_count DESC, updated_at DESC"
        sql += f" LIMIT {limit}"

        cursor = self.conn.execute(sql, params)
        return [Memory(
            key=r["key"],
            value=r["value"],
            category=r["category"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            access_count=r["access_count"],
            metadata=json.loads(r["metadata"]) if r["metadata"] else None
        ) for r in cursor.fetchall()]

    def list_memories(
        self,
        category: str = None,
        limit: int = 20
    ) -> List[Memory]:
        """List memories, optionally filtered by category."""
        sql = "SELECT * FROM memories"
        params = []

        if category:
            sql += " WHERE category = ?"
            params.append(category)

        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(sql, params)
        return [Memory(
            key=r["key"],
            value=r["value"],
            category=r["category"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            access_count=r["access_count"],
            metadata=json.loads(r["metadata"]) if r["metadata"] else None
        ) for r in cursor.fetchall()]

    # ─────────────────────────────────────────────────────────────────
    # Conversation History
    # ─────────────────────────────────────────────────────────────────

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        channel: str = "cli",
        tokens: int = 0
    ):
        """Save a conversation message."""
        self.conn.execute("""
            INSERT INTO conversations (session_id, role, content, timestamp, channel, tokens)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, role, content, datetime.now().isoformat(), channel, tokens))
        self.conn.commit()

    def get_conversation(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a session."""
        cursor = self.conn.execute("""
            SELECT role, content, timestamp, channel
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit))

        messages = [
            {"role": r["role"], "content": r["content"],
             "timestamp": r["timestamp"], "channel": r["channel"]}
            for r in cursor.fetchall()
        ]
        return list(reversed(messages))

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation sessions."""
        cursor = self.conn.execute("""
            SELECT session_id,
                   MIN(timestamp) as started,
                   MAX(timestamp) as last_message,
                   COUNT(*) as message_count,
                   channel
            FROM conversations
            GROUP BY session_id
            ORDER BY last_message DESC
            LIMIT ?
        """, (limit,))

        return [dict(r) for r in cursor.fetchall()]

    def summarize_and_archive(
        self,
        session_id: str,
        summary: str
    ):
        """Archive a conversation with its summary as a memory."""
        self.remember(
            key=f"conversation:{session_id}",
            value=summary,
            category="conversation",
            metadata={
                "session_id": session_id,
                "archived_at": datetime.now().isoformat()
            }
        )

    # ─────────────────────────────────────────────────────────────────
    # Context Building for Prompts
    # ─────────────────────────────────────────────────────────────────

    def get_context_for_prompt(
        self,
        query: str,
        max_memories: int = 5,
        include_preferences: bool = True
    ) -> str:
        """
        Get relevant memories to inject into prompt context.

        Args:
            query: The user's current query
            max_memories: Maximum memories to include
            include_preferences: Whether to include user preferences

        Returns:
            Formatted context string for injection into system prompt
        """
        sections = []

        # Search for relevant memories
        memories = self.search(query, limit=max_memories)
        if memories:
            mem_lines = [f"- **{m.key}**: {m.value}" for m in memories]
            sections.append("## Relevant Context\n" + "\n".join(mem_lines))

        # Get user preferences
        if include_preferences:
            prefs = self.list_memories(category="preference", limit=5)
            if prefs:
                pref_lines = [f"- {p.key}: {p.value}" for p in prefs]
                sections.append("## User Preferences\n" + "\n".join(pref_lines))

        # Get facts about people mentioned
        people = self.list_memories(category="person", limit=3)
        if people:
            people_lines = [f"- {p.key}: {p.value}" for p in people]
            sections.append("## People\n" + "\n".join(people_lines))

        return "\n\n".join(sections) if sections else ""

    # ─────────────────────────────────────────────────────────────────
    # Statistics and Management
    # ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        # Count by category
        cursor = self.conn.execute("""
            SELECT category, COUNT(*) as count
            FROM memories
            GROUP BY category
        """)
        categories = {r["category"]: r["count"] for r in cursor.fetchall()}

        # Total memories
        cursor = self.conn.execute("SELECT COUNT(*) as total FROM memories")
        total = cursor.fetchone()["total"]

        # Conversation stats
        cursor = self.conn.execute("""
            SELECT COUNT(DISTINCT session_id) as sessions,
                   COUNT(*) as messages
            FROM conversations
        """)
        conv_stats = cursor.fetchone()

        # Most accessed memories
        cursor = self.conn.execute("""
            SELECT key, access_count
            FROM memories
            ORDER BY access_count DESC
            LIMIT 5
        """)
        top_accessed = [(r["key"], r["access_count"]) for r in cursor.fetchall()]

        # Database size
        db_size = self.DB_PATH.stat().st_size if self.DB_PATH.exists() else 0

        return {
            "total_memories": total,
            "by_category": categories,
            "conversation_sessions": conv_stats["sessions"],
            "total_messages": conv_stats["messages"],
            "top_accessed": top_accessed,
            "database_size_kb": db_size // 1024
        }

    def clear_category(self, category: str) -> int:
        """Clear all memories in a category. Returns count deleted."""
        cursor = self.conn.execute(
            "DELETE FROM memories WHERE category = ?", (category,)
        )
        self.conn.commit()
        return cursor.rowcount

    def clear_old_conversations(self, days: int = 30) -> int:
        """Clear conversations older than specified days."""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        cursor = self.conn.execute(
            "DELETE FROM conversations WHERE timestamp < ?", (cutoff,)
        )
        self.conn.commit()
        return cursor.rowcount

    def export_memories(self) -> List[Dict[str, Any]]:
        """Export all memories as JSON-serializable list."""
        cursor = self.conn.execute("SELECT * FROM memories")
        return [
            {
                "key": r["key"],
                "value": r["value"],
                "category": r["category"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "access_count": r["access_count"],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else None
            }
            for r in cursor.fetchall()
        ]

    def import_memories(self, memories: List[Dict[str, Any]]) -> int:
        """Import memories from export. Returns count imported."""
        count = 0
        for mem in memories:
            self.remember(
                key=mem["key"],
                value=mem["value"],
                category=mem.get("category", "fact"),
                metadata=mem.get("metadata")
            )
            count += 1
        return count


# ─────────────────────────────────────────────────────────────────
# Command Handlers
# ─────────────────────────────────────────────────────────────────

def handle_memory_command(args: list, memory: MaudeMemory) -> str:
    """Handle /memory command."""
    if not args:
        # Show stats
        stats = memory.get_stats()
        lines = [
            "Memory Statistics:",
            f"  Total memories: {stats['total_memories']}",
            f"  Categories: {stats['by_category']}",
            f"  Conversation sessions: {stats['conversation_sessions']}",
            f"  Total messages: {stats['total_messages']}",
            f"  Database size: {stats['database_size_kb']} KB",
        ]
        if stats['top_accessed']:
            lines.append("  Most accessed:")
            for key, count in stats['top_accessed']:
                lines.append(f"    - {key} ({count}x)")
        return "\n".join(lines)

    action = args[0].lower()

    if action == "list":
        category = args[1] if len(args) > 1 else None
        memories = memory.list_memories(category=category)
        if not memories:
            return "No memories found."
        lines = ["Memories:"]
        for m in memories:
            lines.append(f"  [{m.category}] {m.key}: {m.value[:50]}...")
        return "\n".join(lines)

    elif action == "search" and len(args) > 1:
        query = " ".join(args[1:])
        results = memory.search(query)
        if not results:
            return f"No memories matching '{query}'"
        lines = [f"Search results for '{query}':"]
        for m in results:
            lines.append(f"  [{m.category}] {m.key}: {m.value[:60]}...")
        return "\n".join(lines)

    elif action == "clear" and len(args) > 1:
        category = args[1]
        count = memory.clear_category(category)
        return f"Cleared {count} memories from category '{category}'"

    elif action == "export":
        data = memory.export_memories()
        export_path = Path.home() / ".config" / "maude" / "memory_export.json"
        export_path.write_text(json.dumps(data, indent=2))
        return f"Exported {len(data)} memories to {export_path}"

    return """Usage: /memory [command]

Commands:
  /memory              Show statistics
  /memory list [cat]   List memories (optional category filter)
  /memory search <q>   Search memories
  /memory clear <cat>  Clear category
  /memory export       Export to JSON"""


def handle_remember_command(args: list, memory: MaudeMemory) -> str:
    """Handle /remember command."""
    if len(args) < 2:
        return "Usage: /remember <key> <value> [category]\n\nCategories: fact, preference, person, task"

    key = args[0]

    # Check if last arg is a category
    categories = {"fact", "preference", "person", "task", "conversation"}
    if args[-1].lower() in categories:
        value = " ".join(args[1:-1])
        category = args[-1].lower()
    else:
        value = " ".join(args[1:])
        category = "fact"

    memory.remember(key, value, category)
    return f"Remembered [{category}] {key}: {value}"


def handle_recall_command(args: list, memory: MaudeMemory) -> str:
    """Handle /recall command."""
    if not args:
        return "Usage: /recall <key>"

    key = args[0]
    value = memory.recall(key)

    if value:
        return f"{key}: {value}"

    # Try semantic search as fallback
    results = memory.search(key, limit=3)
    if results:
        lines = [f"No exact match for '{key}', but found similar:"]
        for m in results:
            lines.append(f"  {m.key}: {m.value[:60]}...")
        return "\n".join(lines)

    return f"No memory found for '{key}'"


def handle_forget_command(args: list, memory: MaudeMemory) -> str:
    """Handle /forget command."""
    if not args:
        return "Usage: /forget <key>"

    key = args[0]
    if memory.forget(key):
        return f"Forgot: {key}"
    return f"No memory found for '{key}'"
