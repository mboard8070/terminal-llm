import type { ChatMessage } from "./useChat";
import { getGatewayUrl } from "../../../lib/gateway";

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  model: string;
}

function apiBase(): string { return getGatewayUrl(); }

const KEYS = {
  index: "maude-conversations",
  messages: (id: string) => `maude-conv-msgs:${id}`,
  active: "maude-active-conv",
} as const;

// ── Server sync helpers (fire-and-forget for saves) ─────────────

async function serverGet<T>(path: string): Promise<T | null> {
  try {
    const r = await fetch(`${apiBase()}${path}`);
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

function serverPost(path: string, data: unknown): void {
  fetch(`${apiBase()}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  }).catch(() => {});
}

// ── Conversation index ──────────────────────────────────────────

export function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(KEYS.index);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export async function loadConversationsFromServer(): Promise<Conversation[]> {
  const remote = await serverGet<Conversation[]>("/api/conversations");
  if (remote && remote.length > 0) {
    localStorage.setItem(KEYS.index, JSON.stringify(remote));
    return remote;
  }
  return loadConversations();
}

export function saveConversations(convs: Conversation[]): void {
  localStorage.setItem(KEYS.index, JSON.stringify(convs));
  serverPost("/api/conversations", convs);
}

// ── Per-conversation messages ───────────────────────────────────

export function loadMessages(id: string): ChatMessage[] {
  try {
    const raw = localStorage.getItem(KEYS.messages(id));
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export async function loadMessagesFromServer(id: string): Promise<ChatMessage[]> {
  const remote = await serverGet<ChatMessage[]>(`/api/conversations/${id}/messages`);
  if (remote && remote.length > 0) {
    localStorage.setItem(KEYS.messages(id), JSON.stringify(remote));
    return remote;
  }
  return loadMessages(id);
}

export function saveMessages(id: string, messages: ChatMessage[]): void {
  localStorage.setItem(KEYS.messages(id), JSON.stringify(messages));
  serverPost(`/api/conversations/${id}/messages`, messages);
}

export function deleteMessages(id: string): void {
  localStorage.removeItem(KEYS.messages(id));
  serverPost(`/api/conversations/${id}/delete`, {});
}

// ── Active conversation ID (local only — per-device) ────────────

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
