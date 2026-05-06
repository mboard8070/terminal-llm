import { useState, useCallback, useEffect } from "react";
import { uuid } from "./uuid";
import {
  Conversation,
  loadConversations,
  loadConversationsFromServer,
  saveConversations,
  loadActiveId,
  saveActiveId,
  deleteMessages,
} from "./storage";

export type { Conversation } from "./storage";

function generateTitle(firstMessage: string): string {
  const trimmed = firstMessage.trim().replace(/\s+/g, " ");
  if (trimmed.length <= 40) return trimmed;
  return trimmed.slice(0, 37) + "...";
}

export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>(loadConversations);
  const [activeId, setActiveId] = useState<string | null>(loadActiveId);

  // Hydrate from server on mount (merges with any local-only conversations)
  useEffect(() => {
    loadConversationsFromServer().then((remote) => {
      if (remote.length > 0) {
        setConversations(remote);
      }
    });
  }, []);

  const persist = useCallback((convs: Conversation[]) => {
    // Sort newest-first before saving
    const sorted = [...convs].sort((a, b) => b.updatedAt - a.updatedAt);
    setConversations(sorted);
    saveConversations(sorted);
  }, []);

  const createConversation = useCallback(
    (firstMessage: string, model: string): string => {
      const id = uuid();
      const now = Date.now();
      const conv: Conversation = {
        id,
        title: generateTitle(firstMessage),
        createdAt: now,
        updatedAt: now,
        model,
      };
      const updated = [conv, ...conversations];
      persist(updated);
      setActiveId(id);
      saveActiveId(id);
      return id;
    },
    [conversations, persist],
  );

  const switchConversation = useCallback((id: string) => {
    setActiveId(id);
    saveActiveId(id);
  }, []);

  const deleteConversation = useCallback(
    (id: string) => {
      const updated = conversations.filter((c) => c.id !== id);
      persist(updated);
      deleteMessages(id);
      if (activeId === id) {
        const next = updated.length > 0 ? updated[0].id : null;
        setActiveId(next);
        saveActiveId(next);
      }
    },
    [conversations, activeId, persist],
  );

  const updateTitle = useCallback(
    (id: string, title: string) => {
      const updated = conversations.map((c) =>
        c.id === id ? { ...c, title: generateTitle(title) } : c,
      );
      persist(updated);
    },
    [conversations, persist],
  );

  const touchConversation = useCallback(
    (id: string) => {
      const updated = conversations.map((c) =>
        c.id === id ? { ...c, updatedAt: Date.now() } : c,
      );
      persist(updated);
    },
    [conversations, persist],
  );

  const startNewChat = useCallback(() => {
    setActiveId(null);
    saveActiveId(null);
  }, []);

  return {
    conversations,
    activeId,
    createConversation,
    switchConversation,
    deleteConversation,
    updateTitle,
    touchConversation,
    startNewChat,
  };
}
