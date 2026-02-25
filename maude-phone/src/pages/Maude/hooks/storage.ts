import type { ChatMessage } from "./useChat";

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  model: string;
}

const KEYS = {
  index: "maude-conversations",
  messages: (id: string) => `maude-conv-msgs:${id}`,
  active: "maude-active-conv",
} as const;

// ── Conversation index ──────────────────────────────────────────────

export function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(KEYS.index);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function saveConversations(convs: Conversation[]): void {
  localStorage.setItem(KEYS.index, JSON.stringify(convs));
}

// ── Per-conversation messages ───────────────────────────────────────

export function loadMessages(id: string): ChatMessage[] {
  try {
    const raw = localStorage.getItem(KEYS.messages(id));
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function saveMessages(id: string, messages: ChatMessage[]): void {
  localStorage.setItem(KEYS.messages(id), JSON.stringify(messages));
}

export function deleteMessages(id: string): void {
  localStorage.removeItem(KEYS.messages(id));
}

// ── Active conversation ID ──────────────────────────────────────────

export function loadActiveId(): string | null {
  return localStorage.getItem(KEYS.active);
}

export function saveActiveId(id: string | null): void {
  if (id === null) {
    localStorage.removeItem(KEYS.active);
  } else {
    localStorage.setItem(KEYS.active, id);
  }
}
